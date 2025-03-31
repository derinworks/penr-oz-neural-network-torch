import torch
from torch import Tensor

def batch_norm(x: Tensor, epsilon = 1e-5) -> Tensor:
    """
    Applies batch normalization to given input
    :param x: an input
    :param epsilon: the smallest normalization step
    :return: batch normalized result
    """
    mean = x.mean()
    variance = x.var(unbiased=False)  # Use unbiased=False to match population variance
    scale = 1.0 / torch.sqrt(variance + epsilon)
    return (x - mean) * scale
