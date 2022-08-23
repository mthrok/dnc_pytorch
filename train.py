"""Example script to train the DNC on a repeated copy task."""
import os
import argparse
import logging

import torch
from dnc.repeat_copy import RepeatCopy
from dnc.dnc import DNC

_LG = logging.getLogger(__name__)


def _main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    dataset = RepeatCopy(
        args.num_bits,
        args.batch_size,
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

    optimizer = torch.optim.RMSprop(dnc.parameters(), lr=args.lr, eps=args.eps)

    _run_train_loop(
        dnc,
        dataset,
        optimizer,
        args.num_training_iterations,
        args.report_interval,
        args.checkpoint_interval,
        args.checkpoint_dir,
        args.device,
    )


def _run_train_loop(
    dnc,
    dataset,
    optimizer,
    num_training,
    report_interval,
    checkpoint_interval,
    checkpoint_dir,
    device,
):
    total_loss = 0
    for i in range(num_training):
        batch = dataset(device=device)
        state = None
        outputs = []
        for inputs in batch.observations:
            output, state = dnc(inputs, state)
            outputs.append(output)
        outputs = torch.stack(outputs, 0)
        loss = dataset.cost(outputs, batch.target, batch.mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % report_interval == 0:
            outputs = torch.round(batch.mask.unsqueeze(-1) * torch.sigmoid(outputs))
            dataset_string = dataset.to_human_readable(batch, outputs)
            _LG.info(f"{i}: Avg training loss {total_loss / report_interval}")
            _LG.info(dataset_string)
            total_loss = 0
        if checkpoint_interval is not None and (i + 1) % checkpoint_interval == 0:
            path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(dnc.state_dict(), path)


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

    optim_opts = parser.add_argument_group("Optimizer Parameters")
    optim_opts.add_argument(
        "--max-grad-norm", type=float, default=50, help="Gradient clipping norm limit."
    )
    optim_opts.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-4,
        dest="lr",
        help="Optimizer learning rate.",
    )
    optim_opts.add_argument(
        "--optimizer-epsilon",
        type=float,
        default=1e-10,
        dest="eps",
        help="Epsilon used for RMSProp optimizer.",
    )

    task_opts = parser.add_argument_group("Task Parameters")
    task_opts.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
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
        "--num-training-iterations",
        type=int,
        default=100_000,
        help="Number of iterations to train for.",
    )
    train_opts.add_argument(
        "--report-interval",
        type=int,
        default=100,
        help="Iterations between reports (samples, valid loss).",
    )
    train_opts.add_argument(
        "--checkpoint-dir", default=None, help="Checkpointing directory."
    )
    train_opts.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Checkpointing step interval.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is None and args.checkpoint_interval is not None:
        raise RuntimeError(
            "`--checkpoint-dir` is provided but `--checkpoint-interval` is not provided."
        )
    if args.checkpoint_dir is not None and args.checkpoint_interval is None:
        raise RuntimeError(
            "`--checkpoint-interval` is provided but `--checkpoint-dir` is not provided."
        )
    return args


if __name__ == "__main__":
    _main()
