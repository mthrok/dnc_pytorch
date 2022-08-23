import torch

from dnc.dnc import DNC


def test_dnc():
    """smoke test"""
    memory_size = 16
    word_size = 16
    num_reads = 4
    num_writes = 1

    clip_value = 20

    input_size = 4
    hidden_size = 64
    output_size = input_size

    batch_size = 16
    time_steps = 64

    access_config = {
        "memory_size": memory_size,
        "word_size": word_size,
        "num_reads": num_reads,
        "num_writes": num_writes,
    }
    controller_config = {
        "input_size": input_size + num_reads * word_size,
        "hidden_size": hidden_size,
    }

    dnc = DNC(
        access_config=access_config,
        controller_config=controller_config,
        output_size=output_size,
        clip_value=clip_value,
    )
    inputs = torch.randn((time_steps, batch_size, input_size))
    state = None
    for input in inputs:
        output, state = dnc(input, state)
