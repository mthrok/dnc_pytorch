from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class TemporalLinkageState:
    link: Tensor  # [batch_size, num_writes, memory_size, memory_size]
    precedence_weights: Tensor  # [batch_size, num_writes, memory_size]


_EPSILON = 1e-6


def _vector_norms(m: Tensor) -> Tensor:
    norm = torch.sum(m * m, axis=2, keepdim=True)
    return torch.sqrt(norm + _EPSILON)


def weighted_softmax(activations: Tensor, strengths: Tensor, strengths_op=F.softplus):
    """Returns softmax over activations multiplied by positive strengths.

    Args:
        activations: A tensor of shape `[batch_size, num_heads, memory_size]`, of
            activations to be transformed. Softmax is taken over the last dimension.
        strengths: A tensor of shape `[batch_size, num_heads]` containing strengths to
            multiply by the activations prior to the softmax.
        strengths_op: An operation to transform strengths before softmax.

    Returns:
        A tensor of same shape as `activations` with weighted softmax applied.
    """
    transformed_strengths = strengths_op(strengths).unsqueeze(-1)
    sharp_activations = activations * transformed_strengths
    softmax = F.softmax(sharp_activations, dim=-1)
    return softmax


class CosineWeights(torch.nn.Module):
    def __init__(self, num_heads, word_size, strength_op=F.softplus):
        """
        Args:
            num_heads: number of memory heads.
            word_size: memory word size.
            strength_op: operation to apply to strengths (default softplus).
        """
        super().__init__()
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op

    def forward(self, memory: Tensor, keys: Tensor, strengths: Tensor) -> Tensor:
        """

        Args:
            memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
            strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

        Returns:
            Weights tensor of shape `[batch_size, num_heads, memory_size]`.
        """
        dot = torch.matmul(keys, memory.transpose(-1, -2))  # <B, H, M>
        memory_norms = _vector_norms(memory)  # <B, M, 1>
        key_norms = _vector_norms(keys)  # <B, H, 1>
        norm = torch.matmul(key_norms, memory_norms.transpose(-1, -2))  # <B, H, M>

        similarity = dot / (norm + _EPSILON)

        return weighted_softmax(similarity, strengths, self._strength_op)


class TemporalLinkage(torch.nn.Module):
    def __init__(self, memory_size, num_writes):
        super().__init__()
        self._memory_size = memory_size
        self._num_writes = num_writes

    def _link(
        self,
        prev_link: Tensor,
        prev_precedence_weights: Tensor,
        write_weights: Tensor,
    ) -> Tensor:
        """Calculates the new link graphs.

        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.

        Args:
            prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` representing the previous link graphs for each write
                head.
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` which is the previous "aggregated" write weights for
                each write head.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new locations in memory written to.

        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
           containing the new link graphs for each write head.
        """
        batch_size = prev_link.size(0)
        write_weights_i = write_weights.unsqueeze(3)  # <B, W, M, 1>
        write_weights_j = write_weights.unsqueeze(2)  # <B, W, 1, M>
        prev_precedence_weights_j = prev_precedence_weights.unsqueeze(2)  # <B, W, 1, M>

        prev_link_scale = 1 - write_weights_i - write_weights_j  # <B, W, M, M>
        new_link = write_weights_i * prev_precedence_weights_j  # <B, W, M, M>
        link = prev_link_scale * prev_link + new_link  # <B, W, M, M>

        # Return the link with the diagonal set to zero, to remove self-looping
        # edges.
        mask = (
            torch.eye(self._memory_size)
            .repeat(batch_size, self._num_writes, 1, 1)
            .bool()
        )
        link[mask] = 0
        return link

    def _precedence_weights(
        self,
        prev_precedence_weights: Tensor,
        write_weights: Tensor,
    ) -> Tensor:
        """Calculates the new precedence weights given the current write weights.

        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.

        Args:
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` containing the previous precedence weights.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new write weights.

        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size]` containing the
            new precedence weights.
        """
        write_sum = write_weights.sum(dim=2, keepdim=True)
        return (1 - write_sum) * prev_precedence_weights + write_weights

    def forward(
        self,
        write_weights: Tensor,
        prev_state: Optional[TemporalLinkageState] = None,
    ) -> TemporalLinkageState:
        """Calculate the updated linkage state given the write weights.

        Args:
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the memory addresses of the different write heads.
            prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
                shape `[batch_size, num_writes, memory_size, memory_size]`, and a
                tensor `precedence_weights` of shape `[batch_size, num_writes,
                memory_size]` containing the aggregated history of recent writes.

        Returns:
            A `TemporalLinkageState` tuple `next_state`, which contains the updated
            link and precedence weights.
        """
        if write_weights.ndim != 3:
            raise ValueError(
                f"Expected `write_weights` to be 3D. Found: {write_weights.ndim}"
            )
        if (
            write_weights.size(1) != self._num_writes
            or write_weights.size(2) != self._memory_size
        ):
            raise ValueError(
                "Expected the shape of `write_weights` to be "
                f"[batch, {self._num_writes}, {self._memory_size}]. "
                f"Found: {write_weights.shape}."
            )

        if prev_state is None:
            B, W, M = write_weights.shape
            prev_state = TemporalLinkageState(
                link=torch.zeros((B, W, M, M), device=write_weights.device),
                precedence_weights=torch.zeros((B, W, M), device=write_weights.device),
            )

        link = self._link(prev_state.link, prev_state.precedence_weights, write_weights)
        precedence_weights = self._precedence_weights(
            prev_state.precedence_weights, write_weights
        )
        return TemporalLinkageState(link=link, precedence_weights=precedence_weights)

    def directional_read_weights(
        self,
        link: Tensor,
        prev_read_weights: Tensor,
        is_forward: bool,
    ) -> Tensor:
        """Calculates the forward or the backward read weights.

        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.

        Args:
            link: tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the link graphs L_t.
            prev_read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read weights w_{t-1}^r.
            forward: Boolean indicating whether to follow the "future" direction in
              the link graph (True) or the "past" direction (False).

        Returns:
            tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """
        # <B, W, R, M>
        expanded_read_weights = torch.stack(
            [prev_read_weights for _ in range(self._num_writes)], dim=1
        )
        if is_forward:
            link = link.transpose(-1, -2)
        result = torch.matmul(expanded_read_weights, link)  # <B, W, R, M>
        return result.permute((0, 2, 1, 3))  # <B, R, W, M>


class Freeness(torch.nn.Module):
    def __init__(self, memory_size):
        super().__init__()
        self._memory_size = memory_size

    def _usage_after_write(
        self,
        prev_usage: Tensor,
        write_weights: Tensor,
    ) -> Tensor:
        """Calcualtes the new usage after writing to memory.

        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.

        Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        write_weights = 1 - torch.prod(1 - write_weights, 1)
        return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(
        self, prev_usage: Tensor, free_gate: Tensor, read_weights: Tensor
    ) -> Tensor:
        """Calcualtes the new usage after reading and freeing from memory.

        Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            free_gate: tensor of shape `[batch_size, num_reads]` with entries in the
                range [0, 1] indicating the amount that locations read from can be
                freed.
            read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.

        Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        free_gate = free_gate.unsqueeze(-1)
        free_read_weights = free_gate * read_weights
        phi = torch.prod(1 - free_read_weights, 1)
        return prev_usage * phi

    def forward(
        self,
        write_weights: Tensor,
        free_gate: Tensor,
        read_weights: Tensor,
        prev_usage: Tensor,
    ) -> Tensor:
        """Calculates the new memory usage u_t.

        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.

        Args:
            write_weights: tensor of shape `[batch_size, num_writes,
                memory_size]` giving write weights at previous time step.
            free_gate: tensor of shape `[batch_size, num_reads]` which indicates
                which read heads read memory that can now be freed.
            read_weights: tensor of shape `[batch_size, num_reads,
                memory_size]` giving read weights at previous time step.
            prev_usage: tensor of shape `[batch_size, memory_size]` giving
                usage u_{t - 1} at the previous time step, with entries in range
                [0, 1].

        Returns:
            tensor of shape `[batch_size, memory_size]` representing updated memory
            usage.
        """
        with torch.no_grad():
            usage = self._usage_after_write(prev_usage, write_weights)
        usage = self._usage_after_read(usage, free_gate, read_weights)
        return usage

    def _allocation(self, usage: Tensor) -> Tensor:
        """Computes allocation by sorting `usage`.

        This corresponds to the value a = a_t[\phi_t[j]] in the paper.

        Args:
            usage: tensor of shape `[batch_size, memory_size]` indicating current
                memory usage. This is equal to u_t in the paper when we only have one
                write head, but for multiple write heads, one should update the usage
                while iterating through the write heads to take into account the
                allocation returned by this function.

        Returns:
            Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        usage = _EPSILON + (1 - _EPSILON) * usage

        nonusage = 1 - usage
        sorted_nonusage, indices = torch.topk(nonusage, k=self._memory_size)
        sorted_usage = 1 - sorted_nonusage

        # emulate tf.cumprod(exclusive=True)
        sorted_usage = F.pad(sorted_usage, (1, 0), mode="constant", value=1)
        prod_sorted_usage = torch.cumprod(sorted_usage, dim=1)
        prod_sorted_usage = prod_sorted_usage[:, :-1]

        sorted_allocation = sorted_nonusage * prod_sorted_usage
        inverse_indices = torch.argsort(indices)
        return torch.gather(sorted_allocation, 1, inverse_indices)

    def write_allocation_weights(
        self,
        usage: Tensor,
        write_gates: Tensor,
        num_writes: Tensor,
    ) -> Tensor:
        """Calculates freeness-based locations for writing to.

        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)

        Args:
            usage: A tensor of shape `[batch_size, memory_size]` representing
                current memory usage.
            write_gates: A tensor of shape `[batch_size, num_writes]` with values in
                the range [0, 1] indicating how much each write head does writing
                based on the address returned here (and hence how much usage
                increases).
            num_writes: The number of write heads to calculate write weights for.

        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` containing the
                freeness-based write locations. Note that this isn't scaled by
                `write_gate`; this scaling must be applied externally.
        """
        write_gates = write_gates.unsqueeze(-1)
        allocation_weights = []
        for i in range(num_writes):
            allocation_weights.append(self._allocation(usage))
            usage = usage + (1 - usage) * write_gates[:, i, :] * allocation_weights[i]
        return torch.stack(allocation_weights, dim=1)
