from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor

from .access import MemoryAccess, AccessState


@dataclass
class DNCState:
    access_output: Tensor  # [batch_size, num_reads, word_size]
    access_state: AccessState
    #   memory: Tensor [batch_size, memory_size, word_size]
    #   read_weights: Tensor  # [batch_size, num_reads, memory_size]
    #   write_weights: Tensor  # [batch_size, num_writes, memory_size]
    #   linkage: TemporalLinkageState
    #       link: [batch_size, num_writes, memory_size, memory_size]
    #       precedence_weights: [batch_size, num_writes, memory_size]
    #   usage: Tensor  # [batch_size, memory_size]
    controller_state: Tuple[Tensor, Tensor]
    #   h_n: [num_layers, batch_size, projection_size]
    #   c_n: [num_layers, batch_size, hidden_size]


class DNC(torch.nn.Module):
    """DNC core module.

    Contains controller and memory access module.
    """

    def __init__(
        self,
        access_config,
        controller_config,
        output_size,
        clip_value=None,
    ):
        """Initializes the DNC core.

        Args:
            access_config: dictionary of access module configurations.
            controller_config: dictionary of controller (LSTM) module configurations.
            output_size: output dimension size of core.
            clip_value: clips controller and core output values to between
                `[-clip_value, clip_value]` if specified.
        Raises:
            TypeError: if direct_input_size is not None for any access module other
            than KeyValueMemory.
        """
        super().__init__()

        self._controller = torch.nn.LSTMCell(**controller_config)
        self._access = MemoryAccess(**access_config)
        self._output = torch.nn.LazyLinear(output_size)
        if clip_value is None:
            self._clip = lambda x: x
        else:
            self._clip = lambda x: torch.clamp(x, min=-clip_value, max=clip_value)

    def forward(self, inputs: Tensor, prev_state: Optional[DNCState] = None):
        """Connects the DNC core into the graph.

        Args:
            inputs: Tensor input.
            prev_state: A `DNCState` tuple containing the fields `access_output`,
                `access_state` and `controller_state`. `access_state` is a 3-D Tensor
                of shape `[batch_size, num_reads, word_size]` containing read words.
                `access_state` is a tuple of the access module's state, and
                `controller_state` is a tuple of controller module's state.

        Returns:
            A tuple `(output, next_state)` where `output` is a tensor and `next_state`
            is a `DNCState` tuple containing the fields `access_output`,
            `access_state`, and `controller_state`.
        """
        if inputs.ndim != 2:
            raise ValueError(f"Expected `inputs` to be 2D: Found {inputs.ndim}.")
        if prev_state is None:
            B, device = inputs.shape[0], inputs.device
            num_reads = self._access._num_reads
            word_size = self._access._word_size
            prev_state = DNCState(
                access_output=torch.zeros((B, num_reads, word_size), device=device),
                access_state=None,
                controller_state=None,
            )

        def batch_flatten(x):
            return torch.reshape(x, [x.size(0), -1])

        controller_input = torch.concat(
            [
                batch_flatten(inputs),  # [batch_size, num_input_feats]
                batch_flatten(
                    prev_state.access_output
                ),  # [batch_size, num_reads*word_size]
            ],
            dim=1,
        )  # [batch_size, num_input_feats + num_reads * word_size]

        controller_state = self._controller(
            controller_input, prev_state.controller_state
        )
        controller_state = tuple(self._clip(t) for t in controller_state)
        controller_output = controller_state[0]

        access_output, access_state = self._access(
            controller_output, prev_state.access_state
        )

        output = torch.concat(
            [
                controller_output,  # [batch_size, num_ctrl_feats]
                batch_flatten(access_output),  # [batch_size, num_reads*word_size]
            ],
            dim=1,
        )
        output = self._output(output)
        output = self._clip(output)

        return (
            output,
            DNCState(
                access_output=access_output,
                access_state=access_state,
                controller_state=controller_state,
            ),
        )
