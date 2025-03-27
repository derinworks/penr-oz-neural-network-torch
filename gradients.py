from __future__ import annotations
from torch import Tensor
import torch
import torch.nn as nn
import functions as func

def vector(values: list[float]) -> Tensor:
    """
    :param values: list of floats
    :return: a 1D Tensor of type double requiring gradient
    """
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)

def scalar(value: float) -> Tensor:
    """
    :param value: a float
    :return: a scalar Tensor of type double requiring gradient
    """
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)

def activate(tensor: Tensor, algo: str) -> Tensor:
    """
    Apply activation function based on the given algorithm.
    :param tensor: a Tensor to activate
    :param algo: The activation algorithm ("sigmoid", "relu", "tanh", "softmax").
    :return: Activated tensor.
    """
    if algo == "sigmoid":
        return tensor.sigmoid()
    elif algo == "relu":
        return tensor.relu()
    elif algo == "tanh":
        return tensor.tanh()
    elif algo == "softmax":
        return tensor.softmax(dim=0)
    else:
        raise ValueError(f"Unsupported activation algorithm: {algo}")

class Activation:
    def __init__(self, tensor: Tensor):
        self.tensor = tensor

    def activate(self, algo: str):
        """
        Activates vector
        :param algo: The activation algorithm ("softmax").
        :return: activated vector
        """
        if algo == "softmax":
            return SoftmaxActivation(self.tensor)
        else:
            return Activation(activate(self.tensor, algo))

    def batch_norm(self) -> Activation:
        """
        Applies batch normalization to the values of this vector
        """
        return Activation(func.batch_norm(self.tensor))

    def apply_dropout(self, rate: float) -> Activation:
        """
        Drops out values of this vector by given rate
        :param rate: drop out rate
        """
        return Activation(nn.functional.dropout(self.tensor, p=rate))

    def calculate_cost(self, target: Tensor) -> Tensor:
        """
        Calculates cost between this vector and target
        :param target: a vector
        :return: cost between this and target
        """
        return nn.functional.mse_loss(self.tensor, target)

class SoftmaxActivation(Activation):
    def __init__(self, pre_activation: Tensor):
        """
        Initializes a softmax activation vector for given values
        :param pre_activation: tensor to be activated
        """
        super().__init__(pre_activation.softmax(dim=0))

    def calculate_cost(self, target: Tensor) -> Tensor:
        """
        Calculates cross entropy cost between this vector and target
        :param target: vector
        :return: cost between this and target
        """
        return func.cross_entropy_loss(self.tensor, target)
