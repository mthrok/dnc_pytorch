import numpy as np
import torch
import torch.nn.functional as F
from dnc.addressing import weighted_softmax, CosineWeights, TemporalLinkage, Freeness

from .util import one_hot


def _test_weighted_softmax(strength_op):
    batch_size, num_heads, memory_size = 5, 3, 7
    activations = torch.randn(batch_size, num_heads, memory_size)
    weights = torch.ones((batch_size, num_heads))

    observed = weighted_softmax(activations, weights, strength_op)
    expected = torch.stack(
        [
            F.softmax(a * strength_op(w).unsqueeze(-1))
            for a, w in zip(activations, weights)
        ],
        dim=0,
    )

    torch.testing.assert_close(observed, expected)


def test_weighted_softmax_identity():
    _test_weighted_softmax(lambda x: x)


def test_weighted_softmax_softplus():
    _test_weighted_softmax(F.softplus)


def test_cosine_weights_shape():
    batch_size, num_heads, memory_size, word_size = 5, 3, 7, 2

    module = CosineWeights(num_heads, word_size)
    mem = torch.randn([batch_size, memory_size, word_size])
    keys = torch.randn([batch_size, num_heads, word_size])
    strengths = torch.randn([batch_size, num_heads])
    weights = module(mem, keys, strengths)

    assert weights.shape == torch.Size([batch_size, num_heads, memory_size])


def test_cosine_weights_values():
    batch_size, num_heads, memory_size, word_size = 5, 4, 10, 2

    mem = torch.randn((batch_size, memory_size, word_size))
    mem[0, 0, 0] = 1
    mem[0, 0, 1] = 2
    mem[0, 1, 0] = 3
    mem[0, 1, 1] = 4
    mem[0, 2, 0] = 5
    mem[0, 2, 1] = 6

    keys = torch.randn((batch_size, num_heads, word_size))
    keys[0, 0, 0] = 5
    keys[0, 0, 1] = 6
    keys[0, 1, 0] = 1
    keys[0, 1, 1] = 2
    keys[0, 2, 0] = 5
    keys[0, 2, 1] = 6
    keys[0, 3, 0] = 3
    keys[0, 3, 1] = 4

    strengths = torch.randn((batch_size, num_heads))

    module = CosineWeights(num_heads, word_size)
    result = module(mem, keys, strengths)

    # Manually checks results.
    strengths_softplus = np.log(1 + np.exp(strengths.numpy()))
    similarity = np.zeros((memory_size))

    for b in range(batch_size):
        for h in range(num_heads):
            key = keys[b, h]
            key_norm = np.linalg.norm(key)

            for m in range(memory_size):
                row = mem[b, m]
                similarity[m] = np.dot(key, row) / (key_norm * np.linalg.norm(row))

            similarity = np.exp(similarity * strengths_softplus[b, h])
            similarity /= similarity.sum()
            ref = torch.from_numpy(similarity).to(dtype=torch.float32)
            torch.testing.assert_close(result[b, h], ref, atol=1e-4, rtol=1e-4)


def test_cosine_weights_divide_by_zero():
    batch_size, num_heads, memory_size, word_size = 5, 4, 10, 2

    module = CosineWeights(num_heads, word_size)
    keys = torch.randn([batch_size, num_heads, word_size], requires_grad=True)
    strengths = torch.randn([batch_size, num_heads], requires_grad=True)

    # First row of memory is non-zero to concentrate attention on this location.
    # Remaining rows are all zero.
    mem = torch.zeros([batch_size, memory_size, word_size])
    mem[:, 0, :] = 1
    mem.requires_grad = True

    output = module(mem, keys, strengths)
    output.sum().backward()

    assert torch.all(~output.isnan())
    assert torch.all(~mem.grad.isnan())
    assert torch.all(~keys.grad.isnan())
    assert torch.all(~strengths.grad.isnan())


def test_temporal_linkage():
    batch_size, memory_size, num_reads, num_writes = 7, 4, 11, 5

    module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

    state = None
    num_steps = 5
    for i in range(num_steps):
        write_weights = torch.rand([batch_size, num_writes, memory_size])
        write_weights /= write_weights.sum(2, keepdim=True) + 1

        # Simulate (in final steps) link 0-->1 in head 0 and 3-->2 in head 1
        if i == num_steps - 2:
            write_weights[0, 0, :] = one_hot(memory_size, 0)
            write_weights[0, 1, :] = one_hot(memory_size, 3)
        elif i == num_steps - 1:
            write_weights[0, 0, :] = one_hot(memory_size, 1)
            write_weights[0, 1, :] = one_hot(memory_size, 2)

        state = module(write_weights, state)

        # link should be bounded in range [0, 1]
        assert torch.all(0 <= state.link.min() <= 1)

        # link diagonal should be zero
        torch.testing.assert_close(
            state.link[:, :, range(memory_size), range(memory_size)],
            torch.zeros([batch_size, num_writes, memory_size]),
        )

        # link rows and columns should sum to at most 1
        assert torch.all(state.link.sum(2) <= 1)
        assert torch.all(state.link.sum(3) <= 1)

    # records our transitions in batch 0: head 0: 0->1, and head 1: 3->2
    torch.testing.assert_close(
        state.link[0, 0, :, 0], one_hot(memory_size, 1, dtype=torch.float32)
    )
    torch.testing.assert_close(
        state.link[0, 1, :, 3], one_hot(memory_size, 2, dtype=torch.float32)
    )

    # Now test calculation of forward and backward read weights
    prev_read_weights = torch.randn((batch_size, num_reads, memory_size))
    prev_read_weights[0, 5, :] = one_hot(memory_size, 0)  # read 5, posn 0
    prev_read_weights[0, 6, :] = one_hot(memory_size, 2)  # read 6, posn 2

    forward_read_weights = module.directional_read_weights(
        state.link, prev_read_weights, is_forward=True
    )
    backward_read_weights = module.directional_read_weights(
        state.link, prev_read_weights, is_forward=False
    )

    # Check directional weights calculated correctly.
    torch.testing.assert_close(
        forward_read_weights[0, 5, 0, :],  # read=5, write=0
        one_hot(memory_size, 1, dtype=torch.float32),
    )
    torch.testing.assert_close(
        backward_read_weights[0, 6, 1, :],  # read=6, write=1
        one_hot(memory_size, 3, dtype=torch.float32),
    )


def test_temporal_linkage_precedence_weights():
    batch_size, memory_size, num_writes = 7, 3, 5

    module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

    prev_precedence_weights = torch.rand(batch_size, num_writes, memory_size)
    write_weights = torch.rand(batch_size, num_writes, memory_size)

    # These should sum to at most 1 for each write head in each batch.
    write_weights /= write_weights.sum(2, keepdim=True) + 1
    prev_precedence_weights /= prev_precedence_weights.sum(2, keepdim=True) + 1

    write_weights[0, 1, :] = 0  # batch 0 head 1: no writing
    write_weights[1, 2, :] /= write_weights[1, 2, :].sum()  # b1 h2: all writing

    precedence_weights = module._precedence_weights(
        prev_precedence_weights=prev_precedence_weights, write_weights=write_weights
    )

    # precedence weights should be bounded in range [0, 1]
    assert torch.all(0 <= precedence_weights)
    assert torch.all(precedence_weights <= 1)

    # no writing in batch 0, head 1
    torch.testing.assert_close(
        precedence_weights[0, 1, :], prev_precedence_weights[0, 1, :]
    )

    # all writing in batch 1, head 2
    torch.testing.assert_close(precedence_weights[1, 2, :], write_weights[1, 2, :])


def test_freeness():
    batch_size, memory_size, num_reads, num_writes = 5, 11, 3, 7

    module = Freeness(memory_size)

    free_gate = torch.rand((batch_size, num_reads))

    # Produce read weights that sum to 1 for each batch and head.
    prev_read_weights = torch.rand((batch_size, num_reads, memory_size))
    prev_read_weights[1, :, 3] = 0
    prev_read_weights /= prev_read_weights.sum(2, keepdim=True)
    prev_write_weights = torch.rand((batch_size, num_writes, memory_size))
    prev_write_weights /= prev_write_weights.sum(2, keepdim=True)
    prev_usage = torch.rand((batch_size, memory_size))

    # Add some special values that allows us to test the behaviour:
    prev_write_weights[1, 2, 3] = 1
    prev_read_weights[2, 0, 4] = 1
    free_gate[2, 0] = 1

    usage = module(prev_write_weights, free_gate, prev_read_weights, prev_usage)

    # Check all usages are between 0 and 1.
    assert torch.all(0 <= usage)
    assert torch.all(usage <= 1)

    # Check that the full write at batch 1, position 3 makes it fully used.
    assert usage[1][3] == 1

    # Check that the full free at batch 2, position 4 makes it fully free.
    assert usage[2][4] == 0


def test_freeness_write_allocation_weights():
    batch_size, memory_size, num_writes = 7, 23, 5

    module = Freeness(memory_size)

    usage = torch.rand((batch_size, memory_size))
    write_gates = torch.rand((batch_size, num_writes))

    # Turn off gates for heads 1 and 3 in batch 0. This doesn't scaling down the
    # weighting, but it means that the usage doesn't change, so we should get
    # the same allocation weightings for: (1, 2) and (3, 4) (but all others
    # being different).
    write_gates[0, 1] = 0
    write_gates[0, 3] = 0
    # and turn heads 0 and 2 on for full effect.
    write_gates[0, 0] = 1
    write_gates[0, 2] = 1

    # In batch 1, make one of the usages 0 and another almost 0, so that these
    # entries get most of the allocation weights for the first and second heads.
    usage[1] = usage[1] * 0.9 + 0.1  # make sure all entries are in [0.1, 1]
    usage[1][4] = 0  # write head 0 should get allocated to position 4
    usage[1][3] = 1e-4  # write head 1 should get allocated to position 3
    write_gates[1, 0] = 1  # write head 0 fully on
    write_gates[1, 1] = 1  # write head 1 fully on

    weights = module.write_allocation_weights(
        usage=usage, write_gates=write_gates, num_writes=num_writes
    )

    assert torch.all(0 <= weights)
    assert torch.all(weights <= 1)

    # Check that weights sum to close to 1
    torch.testing.assert_close(
        weights.sum(dim=2), torch.ones((batch_size, num_writes)), atol=1e-3, rtol=0
    )

    # Check the same / different allocation weight pairs as described above.
    assert (weights[0, 0, :] - weights[0, 1, :]).abs().max() > 0.1
    torch.testing.assert_close(weights[0, 1, :], weights[0, 2, :])
    assert (weights[0, 2, :] - weights[0, 3, :]).abs().max() > 0.1
    torch.testing.assert_close(weights[0, 3, :], weights[0, 4, :])

    torch.testing.assert_close(
        weights[1][0], one_hot(memory_size, 4, dtype=torch.float32), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        weights[1][1], one_hot(memory_size, 3, dtype=torch.float32), atol=1e-3, rtol=0
    )


def test_freeness_write_allocation_weights_gradient():
    batch_size, memory_size, num_writes = 7, 5, 3

    module = Freeness(memory_size).to(torch.float64)

    usage = torch.rand(
        (batch_size, memory_size), dtype=torch.float64, requires_grad=True
    )
    write_gates = torch.rand(
        (batch_size, num_writes), dtype=torch.float64, requires_grad=True
    )

    def func(usage, write_gates):
        return module.write_allocation_weights(usage, write_gates, num_writes)

    torch.autograd.gradcheck(func, (usage, write_gates))


def test_freeness_allocation():
    batch_size, memory_size = 7, 13

    usage = torch.rand((batch_size, memory_size))
    module = Freeness(memory_size)
    allocation = module._allocation(usage)

    # 1. Test that max allocation goes to min usage, and vice versa.
    assert torch.all(usage.argmin(dim=1) == allocation.argmax(dim=1))
    assert torch.all(usage.argmax(dim=1) == allocation.argmin(dim=1))

    # 2. Test that allocations sum to almost 1.
    torch.testing.assert_close(
        allocation.sum(dim=1), torch.ones(batch_size), atol=0.01, rtol=0
    )


def test_freeness_allocation_gradient():
    batch_size, memory_size = 1, 5

    usage = torch.rand(
        (batch_size, memory_size), dtype=torch.float64, requires_grad=True
    )
    module = Freeness(memory_size).to(torch.float64)

    torch.autograd.gradcheck(module._allocation, (usage,))
