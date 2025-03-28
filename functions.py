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

def cross_entropy_loss(x: Tensor, y: Tensor, epsilon = 1e-9) -> Tensor:
    """
    Compute the cross-entropy loss
    :param x: an input (1D softmax activated).
    :param y: an expected output (1D expected output, a probability distribution).
    :param epsilon: small offset for numerical stability of log of x
    :return: Cross entropy loss.
    """
    return -torch.sum(y * torch.log(x + epsilon))
