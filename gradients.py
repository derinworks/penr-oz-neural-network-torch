from __future__ import annotations
from torch import Tensor
import torch
import torch.nn as nn

def matrix(values: list[list[float]]) -> Tensor:
    """
    :param values: nested list of floats
    :return: a 2D Tensor of type double requiring gradient
    """
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)

def vector(values: list[float]) -> Tensor:
    """
    :param values: list of floats
    :return: a 2D Tensor as column vector of type double requiring gradient
    """
    return matrix([values])

def activate(tensor: Tensor, algo: str) -> Tensor:
    """
    Apply activation function based on the given algorithm.
    :param tensor: a 1D or 2D Tensor to activate
    :param algo: The activation algorithm ("sigmoid", "relu", "tanh", "softmax").
    :return: Activated tensor.
    """
    if algo == "sigmoid":
        return tensor.sigmoid()
    elif algo == "relu":
        return tensor.relu()
    elif algo == "tanh":
        return tensor.tanh()
    elif algo == "softmax" and tensor.dim() == 2:
        return tensor.softmax(dim=1)
    raise ValueError(f"Unsupported activation algorithm: {algo}")

def calculate_cost(algo: str, logits: Tensor, activation: Tensor, target: list[list[float]]):
    """
    Calculates lost based on activation algorithm given
    :param algo: The activation algorithm ("sigmoid", "relu", "tanh", "softmax").
    :param logits: pre-activation tensor before softmax activation
    :param activation: activated tensor
    :param target: target to compare with (same shape as activation or label index for softmax)
    :return: calculated cost
    """
    if target is None or any(tgt is None for tgt in target):
        return torch.empty(0)
    elif algo == "softmax":
        label_tensor = torch.tensor([tgt[0] for tgt in target], dtype=torch.int64)
        return nn.functional.cross_entropy(logits, label_tensor)
    else:
        target_tensor = torch.tensor(target, dtype=torch.float64)
        return nn.functional.mse_loss(activation, target_tensor)
