import torch
import torch.nn.functional as F


def one_hot(length, index, dtype=None):
    val = F.one_hot(torch.tensor(index), num_classes=length)
    if dtype is not None:
        val = val.to(dtype=dtype)
    return val
