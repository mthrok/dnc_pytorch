from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch import Tensor

from .addressing import (
    CosineWeights,
    Freeness,
    TemporalLinkage,
    TemporalLinkageState,
)


@dataclass
class AccessState:
    memory: Tensor  # [batch_size, memory_size, word_size]
    read_weights: Tensor  # [batch_size, num_reads, memory_size]
    write_weights: Tensor  # [batch_size, num_writes, memory_size]
    linkage: TemporalLinkageState
    # link: [batch_size, num_writes, memory_size, memory_size]
    # precedence_weights: [batch_size, num_writes, memory_size]
    usage: Tensor  # [batch_size, memory_size]


def _erase_and_write(memory, address, reset_weights, values):
    """Module to erase and write in the external memory.

    Erase operation:
        M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

    Add operation:
        M_t(i) = M_t'(i) + w_t(i) * a_t

    where e are the reset_weights, w the write weights and a the values.

    Args:
        memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
        address: 3-D tensor `[batch_size, num_writes, memory_size]`.
        reset_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
        values: 3-D tensor `[batch_size, num_writes, word_size]`.

    Returns:
        3-D tensor of shape `[batch_size, num_writes, word_size]`.
    """
    weighted_resets = address.unsqueeze(3) * reset_weights.unsqueeze(2)
    reset_gate = torch.prod(1 - weighted_resets, dim=1)
    return memory * reset_gate + torch.matmul(address.transpose(-1, -2), values)


class Reshape(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, input):
        return input.reshape(self.dims)


class MemoryAccess(torch.nn.Module):
    """Access module of the Differentiable Neural Computer.

    This memory module supports multiple read and write heads. It makes use of:

    *   `addressing.TemporalLinkage` to track the temporal ordering of writes in
        memory for each write head.
    *   `addressing.FreenessAllocator` for keeping track of memory usage, where
        usage increase when a memory location is written to, and decreases when
        memory is read from that the controller says can be freed.

    Write-address selection is done by an interpolation between content-based
    lookup and using unused memory.

    Read-address selection is done by an interpolation of content-based lookup
    and following the link graph in the forward or backwards read direction.
    """

    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1):
        """Creates a MemoryAccess module.

        Args:
            memory_size: The number of memory slots (N in the DNC paper).
            word_size: The width of each memory slot (W in the DNC paper)
            num_reads: The number of read heads (R in the DNC paper).
            num_writes: The number of write heads (fixed at 1 in the paper).
            name: The name of the module.
        """
        super().__init__()
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes

        self._write_content_weights_mod = CosineWeights(num_writes, word_size)
        self._read_content_weights_mod = CosineWeights(num_reads, word_size)

        self._linkage = TemporalLinkage(memory_size, num_writes)
        self._freeness = Freeness(memory_size)

        def _linear(first_dim, second_dim, pre_act=None, post_act=None):
            """Returns a linear transformation of `inputs`, followed by a reshape."""
            mods = []
            mods.append(torch.nn.LazyLinear(first_dim * second_dim))
            if pre_act is not None:
                mods.append(pre_act)
            mods.append(Reshape([-1, first_dim, second_dim]))
            if post_act is not None:
                mods.append(post_act)
            return torch.nn.Sequential(*mods)

        self._write_vectors = _linear(num_writes, word_size)
        self._erase_vectors = _linear(num_writes, word_size, pre_act=torch.nn.Sigmoid())
        self._free_gate = torch.nn.Sequential(
            torch.nn.LazyLinear(num_reads),
            torch.nn.Sigmoid(),
        )
        self._alloc_gate = torch.nn.Sequential(
            torch.nn.LazyLinear(num_writes),
            torch.nn.Sigmoid(),
        )
        self._write_gate = torch.nn.Sequential(
            torch.nn.LazyLinear(num_writes),
            torch.nn.Sigmoid(),
        )
        num_read_modes = 1 + 2 * num_writes
        self._read_mode = _linear(
            num_reads, num_read_modes, post_act=torch.nn.Softmax(dim=-1)
        )
        self._write_keys = _linear(num_writes, word_size)
        self._write_strengths = torch.nn.LazyLinear(num_writes)

        self._read_keys = _linear(num_reads, word_size)
        self._read_strengths = torch.nn.LazyLinear(num_reads)

    def _read_inputs(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Applies transformations to `inputs` to get control for this module."""
        # v_t^i - The vectors to write to memory, for each write head `i`.
        write_vectors = self._write_vectors(inputs)

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vectors = self._erase_vectors(inputs)

        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        free_gate = self._free_gate(inputs)

        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        allocation_gate = self._alloc_gate(inputs)

        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate = self._write_gate(inputs)

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        read_mode = self._read_mode(inputs)

        # Parameters for the (read / write) "weights by content matching" modules.
        write_keys = self._write_keys(inputs)
        write_strengths = self._write_strengths(inputs)

        read_keys = self._read_keys(inputs)
        read_strengths = self._read_strengths(inputs)

        result = {
            "read_content_keys": read_keys,
            "read_content_strengths": read_strengths,
            "write_content_keys": write_keys,
            "write_content_strengths": write_strengths,
            "write_vectors": write_vectors,
            "erase_vectors": erase_vectors,
            "free_gate": free_gate,
            "allocation_gate": allocation_gate,
            "write_gate": write_gate,
            "read_mode": read_mode,
        }
        return result

    def _write_weights(
        self,
        inputs: Tensor,
        memory: Tensor,
        usage: Tensor,
    ) -> Tensor:
        """Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
            inputs: Collection of inputs to the access module, including controls for
                how to chose memory writing, such as the content to look-up and the
                weighting between content-based and allocation-based addressing.
            memory: A tensor of shape  `[batch_size, memory_size, word_size]`
                containing the current memory contents.
            usage: Current memory usage, which is a tensor of shape `[batch_size,
                memory_size]`, used for allocation-based addressing.

        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` indicating where
                to write to (if anywhere) for each write head.
        """
        # c_t^{w, i} - The content-based weights for each write head.
        write_content_weights = self._write_content_weights_mod(
            memory, inputs["write_content_keys"], inputs["write_content_strengths"]
        )

        # a_t^i - The allocation weights for each write head.
        write_allocation_weights = self._freeness.write_allocation_weights(
            usage=usage,
            write_gates=(inputs["allocation_gate"] * inputs["write_gate"]),
            num_writes=self._num_writes,
        )

        # Expands gates over memory locations.
        allocation_gate = inputs["allocation_gate"].unsqueeze(-1)
        write_gate = inputs["write_gate"].unsqueeze(-1)

        # w_t^{w, i} - The write weightings for each write head.
        return write_gate * (
            allocation_gate * write_allocation_weights
            + (1 - allocation_gate) * write_content_weights
        )

    def _read_weights(
        self,
        inputs: Tensor,
        memory: Tensor,
        prev_read_weights: Tensor,
        link: Tensor,
    ) -> Tensor:
        """Calculates read weights for each read head.

        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.

        Args:
            inputs: Controls for this access module. This contains the content-based
                keys to lookup, and the weightings for the different read modes.
            memory: A tensor of shape `[batch_size, memory_size, word_size]`
                containing the current memory contents to do content-based lookup.
            prev_read_weights: A tensor of shape `[batch_size, num_reads,
                memory_size]` containing the previous read locations.
            link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` containing the temporal write transition graphs.

        Returns:
            A tensor of shape `[batch_size, num_reads, memory_size]` containing the
            read weights for each read head.
        """
        # c_t^{r, i} - The content weightings for each read head.
        content_weights = self._read_content_weights_mod(
            memory, inputs["read_content_keys"], inputs["read_content_strengths"]
        )

        # Calculates f_t^i and b_t^i.
        forward_weights = self._linkage.directional_read_weights(
            link, prev_read_weights, is_forward=True
        )
        backward_weights = self._linkage.directional_read_weights(
            link, prev_read_weights, is_forward=False
        )

        m = self._num_writes
        backward_mode = inputs["read_mode"][:, :, :m, None]
        forward_mode = inputs["read_mode"][:, :, m : 2 * m, None]
        content_mode = inputs["read_mode"][:, :, None, 2 * m]

        read_weights = (
            content_mode * content_weights
            + (forward_mode * forward_weights).sum(dim=2)
            + (backward_mode * backward_weights).sum(dim=2)
        )
        return read_weights

    def forward(
        self,
        inputs: Tensor,
        prev_state: Optional[AccessState] = None,
    ) -> Tuple[Tensor, AccessState]:
        """Connects the MemoryAccess module into the graph.

        Args:
            inputs: tensor of shape `[batch_size, input_size]`. This is used to
                control this access module.
            prev_state: Instance of `AccessState` containing the previous state.

        Returns:
            A tuple `(output, next_state)`, where `output` is a tensor of shape
            `[batch_size, num_reads, word_size]`, and `next_state` is the new
            `AccessState` named tuple at the current time t.
        """
        if inputs.ndim != 2:
            raise ValueError("Expected `inputs` to be 2D. Found: {inputs.ndim}.")
        if prev_state is None:
            B, device = inputs.shape[0], inputs.device
            prev_state = AccessState(
                memory=torch.zeros(
                    (B, self._memory_size, self._word_size), device=device
                ),
                read_weights=torch.zeros(
                    (B, self._num_reads, self._memory_size), device=device
                ),
                write_weights=torch.zeros(
                    (B, self._num_writes, self._memory_size), device=device
                ),
                linkage=None,
                usage=torch.zeros((B, self._memory_size), device=device),
            )

        inputs = self._read_inputs(inputs)

        # Update usage using inputs['free_gate'] and previous read & write weights.
        usage = self._freeness(
            write_weights=prev_state.write_weights,
            free_gate=inputs["free_gate"],
            read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage,
        )

        # Write to memory.
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = _erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=inputs["erase_vectors"],
            values=inputs["write_vectors"],
        )

        linkage_state = self._linkage(write_weights, prev_state.linkage)

        # Read from memory.
        read_weights = self._read_weights(
            inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weights,
            link=linkage_state.link,
        )
        read_words = torch.matmul(read_weights, memory)

        return (
            read_words,
            AccessState(
                memory=memory,
                read_weights=read_weights,
                write_weights=write_weights,
                linkage=linkage_state,
                usage=usage,
            ),
        )
