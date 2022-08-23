import torch

from dnc.access import MemoryAccess, AccessState
from dnc.addressing import TemporalLinkageState

from .util import one_hot

BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
INPUT_SIZE = 10


def test_memory_access_build_and_train():
    module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    inputs = torch.randn((TIME_STEPS, BATCH_SIZE, INPUT_SIZE))

    outputs = []
    state = None
    for input in inputs:
        output, state = module(input, state)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=-1)

    targets = torch.rand((TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE))
    loss = (output - targets).square().mean()

    optim = torch.optim.SGD(module.parameters(), lr=1)
    optim.zero_grad()
    loss.backward()
    optim.step()


def test_memory_access_valid_read_mode():
    module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    inputs = module._read_inputs(torch.randn((BATCH_SIZE, INPUT_SIZE)))

    # Check that the read modes for each read head constitute a probability
    # distribution.
    torch.testing.assert_close(
        inputs["read_mode"].sum(2), torch.ones([BATCH_SIZE, NUM_READS])
    )
    assert torch.all(inputs["read_mode"] >= 0)


def test_memory_access_write_weights():
    memory = 10 * (torch.rand((BATCH_SIZE, MEMORY_SIZE, WORD_SIZE)) - 0.5)
    usage = torch.rand((BATCH_SIZE, MEMORY_SIZE))

    allocation_gate = torch.rand((BATCH_SIZE, NUM_WRITES))
    write_gate = torch.rand((BATCH_SIZE, NUM_WRITES))

    write_gate = torch.rand((BATCH_SIZE, NUM_WRITES))
    write_content_keys = torch.rand((BATCH_SIZE, NUM_WRITES, WORD_SIZE))
    write_content_strengths = torch.rand((BATCH_SIZE, NUM_WRITES))

    # Check that turning on allocation gate fully brings the write gate to
    # the allocation weighting (which we will control by controlling the usage).
    usage[:, 3] = 0
    allocation_gate[:, 0] = 1
    write_gate[:, 0] = 1

    inputs = {
        "allocation_gate": allocation_gate,
        "write_gate": write_gate,
        "write_content_keys": write_content_keys,
        "write_content_strengths": write_content_strengths,
    }

    module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    weights = module._write_weights(inputs, memory, usage)

    # Check the weights sum to their target gating.
    torch.testing.assert_close(weights.sum(dim=2), write_gate, atol=5e-2, rtol=0)

    # Check that we fully allocated to the third row.
    weights_0_0_target = one_hot(MEMORY_SIZE, 3, dtype=torch.float32)
    torch.testing.assert_close(weights[0, 0], weights_0_0_target, atol=1e-3, rtol=0)


def test_memory_access_read_weights():
    memory = 10 * (torch.rand((BATCH_SIZE, MEMORY_SIZE, WORD_SIZE)) - 0.5)
    prev_read_weights = torch.rand((BATCH_SIZE, NUM_READS, MEMORY_SIZE))
    prev_read_weights /= prev_read_weights.sum(2, keepdim=True) + 1

    link = torch.rand((BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE))
    # Row and column sums should be at most 1:
    link /= torch.maximum(link.sum(2, keepdim=True), torch.tensor(1))
    link /= torch.maximum(link.sum(3, keepdim=True), torch.tensor(1))

    # We query the memory on the third location in memory, and select a large
    # strength on the query. Then we select a content-based read-mode.
    read_content_keys = torch.rand((BATCH_SIZE, NUM_READS, WORD_SIZE))
    read_content_keys[0, 0] = memory[0, 3]
    read_content_strengths = torch.full(
        (BATCH_SIZE, NUM_READS), 100.0, dtype=torch.float64
    )
    read_mode = torch.rand((BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES))
    read_mode[0, 0, :] = one_hot(1 + 2 * NUM_WRITES, 2 * NUM_WRITES)
    inputs = {
        "read_content_keys": read_content_keys,
        "read_content_strengths": read_content_strengths,
        "read_mode": read_mode,
    }

    module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    read_weights = module._read_weights(inputs, memory, prev_read_weights, link)

    # read_weights for batch 0, read head 0 should be memory location 3
    ref = one_hot(MEMORY_SIZE, 3, dtype=torch.float64)
    torch.testing.assert_close(read_weights[0, 0, :], ref, atol=1e-3, rtol=0)


def test_memory_access_gradients():
    kwargs = {"dtype": torch.float64, "requires_grad": True}

    inputs = torch.randn((BATCH_SIZE, INPUT_SIZE), **kwargs)
    memory = torch.zeros((BATCH_SIZE, MEMORY_SIZE, WORD_SIZE), **kwargs)
    read_weights = torch.zeros((BATCH_SIZE, NUM_READS, MEMORY_SIZE), **kwargs)
    precedence_weights = torch.zeros((BATCH_SIZE, NUM_WRITES, MEMORY_SIZE), **kwargs)
    link = torch.zeros((BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE), **kwargs)

    module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inputs, memory, read_weights, link, precedence_weights):
            write_weights = torch.zeros((BATCH_SIZE, NUM_WRITES, MEMORY_SIZE))
            usage = torch.zeros((BATCH_SIZE, MEMORY_SIZE))

            state = AccessState(
                memory=memory,
                read_weights=read_weights,
                write_weights=write_weights,
                linkage=TemporalLinkageState(
                    link=link,
                    precedence_weights=precedence_weights,
                ),
                usage=usage,
            )
            output, _ = self.module(inputs, state)
            return output.sum()

    module = Wrapper(module).to(torch.float64)

    torch.autograd.gradcheck(
        module,
        (inputs, memory, read_weights, link, precedence_weights),
    )
