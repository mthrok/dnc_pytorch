#!/usr/bin/env python3
"""Run DNC through repeat-copy task and visualize things"""
import os
import argparse
import logging

import torch
from dnc.repeat_copy import RepeatCopy
from dnc.dnc import DNC
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


_LG = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    model_opts = parser.add_argument_group("Model Parameters")
    model_opts.add_argument(
        "--hidden-size", type=int, default=64, help="Size of LSTM hidden layer."
    )
    model_opts.add_argument(
        "--memory-size", type=int, default=16, help="The number of memory slots."
    )
    model_opts.add_argument(
        "--word-size", type=int, default=16, help="The width of each memory slot."
    )
    model_opts.add_argument(
        "--num-write-heads", type=int, default=1, help="Number of memory write heads."
    )
    model_opts.add_argument(
        "--num-read-heads", type=int, default=4, help="Number of memory read heads."
    )
    model_opts.add_argument(
        "--clip-value",
        type=float,
        default=20,
        help="Maximum absolute value of controller and dnc outputs.",
    )

    task_opts = parser.add_argument_group("Task Parameters")
    task_opts.add_argument(
        "--num-bits", type=int, default=4, help="Dimensionality of each vector to copy"
    )
    task_opts.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Lower limit on number of vectors in the observation pattern to copy",
    )
    task_opts.add_argument(
        "--max-length",
        type=int,
        default=2,
        help="Upper limit on number of vectors in the observation pattern to copy",
    )
    task_opts.add_argument(
        "--min-repeats",
        type=int,
        default=1,
        help="Lower limit on number of copy repeats.",
    )
    task_opts.add_argument(
        "--max-repeats",
        type=int,
        default=2,
        help="Upper limit on number of copy repeats.",
    )

    train_opts = parser.add_argument_group("Training Options")
    train_opts.add_argument(
        "--device",
        type=torch.device,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to perform the training.",
    )
    train_opts.add_argument(
        "--checkpoint-path", required=True, help="Checkpoint path."
    )
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError("Checkpoint file does not exist.")
    return args


def _main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    dataset = RepeatCopy(
        args.num_bits,
        1,  # batch_size
        args.min_length,
        args.max_length,
        args.min_repeats,
        args.max_repeats,
    )

    dnc = DNC(
        access_config={
            "memory_size": args.memory_size,
            "word_size": args.word_size,
            "num_reads": args.num_read_heads,
            "num_writes": args.num_write_heads,
        },
        controller_config={
            "input_size": args.num_bits + 2 + args.num_read_heads * args.word_size,
            "hidden_size": args.hidden_size,
        },
        output_size=dataset.target_size,
        clip_value=args.clip_value,
    ).to(args.device)

    dnc.load_state_dict(torch.load(args.checkpoint_path))

    input, states, output = _test(dnc, dataset, args.device)

    _visualize(input, states, output)


def _test(dnc, dataset, device):
    batch = dataset(device=device)

    state = None
    outputs = []
    states = []
    for inputs in batch.observations:
        output, state = dnc(inputs, state)
        states.append(state)
        outputs.append(output)
    return batch, states, torch.stack(outputs, dim=0)


def _visualize(input, states, output):
    # print(input)
    # print(states)
    # print(output)

    def _fmt(tensors, normalize=False):
        tensors = [torch.zeros_like(tensors[0])] + tensors
        tensors = torch.stack(tensors, dim=0)
        if normalize:
            tensors -= tensors.min()
            tensors /= (tensors.max() + torch.finfo(tensors.dtype).eps)
        return tensors.detach().numpy()

    memory = _fmt([s.access_state.memory[0] for s in states])
    usage = _fmt([s.access_state.usage[0][..., None] for s in states])
    readout = _fmt([s.access_output[0] for s in states], normalize=True)

    ani = _animate_tensors(input.observations, input.target, memory, usage, readout)
    plt.show()
    ani.save("memory.gif", dpi=300, writer=PillowWriter(fps=25))


def _animate_tensors(input, target, memory, usage, readout):
    num_frames, num_memory, _ = memory.shape

    print(input[:, 0, :], target[:, 0, :])
    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [4, 1, 4]})

    kwargs = {'interpolation': 'nearest', 'vmin': 0, 'vmax': 1, 'animated': True}
    mem = ax[0].imshow(memory[0], **kwargs, aspect="auto")
    ax[0].set_ylabel("Memory Content")
    ax[0].set_ylabel("Memory Slot")
    usg = ax[1].imshow(usage[0], **kwargs, aspect="auto")
    rdo = ax[2].imshow(readout[0], **kwargs, aspect=1)
    # plt.colorbar(mem)

    def _animate(i):
        fig.suptitle(f"Step {i}: {input}")
        mem.set_array(memory[i])
        usg.set_array(usage[i])
        rdo.set_array(readout[i])
        return [mem, usg, rdo]

    ani = FuncAnimation(fig, _animate, interval=300, blit=True, repeat=True, frames=num_frames)
    return ani

if __name__ == '__main__':
    _main()
