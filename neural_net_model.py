import json
import logging
import os
import math
import random
from typing import Tuple
import time
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch import Tensor
import functions as func


log = logging.getLogger(__name__)

class Layer:
    def __init__(self, input_size: int = 0, output_size: int = 0, weight_algo="xavier", bias_algo="zeros"):
        """
        Initialize a layer of neurons
        :param input_size: represents the input size of layer
        :param output_size: represents the output size of layer
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        """
        self.weights: Tensor = (torch.randn(input_size, output_size) * (
            math.sqrt(1 / input_size) if weight_algo == "xavier"
            else math.sqrt(2 / input_size) if weight_algo == "he"
            else 1 # if weight_algo == gaussian
        )).double().requires_grad_() if all(sz > 0 for sz in (input_size, output_size)) else torch.empty(0)
        self.biases: Tensor = (torch.zeros(output_size) if bias_algo == "zeros"
                       else torch.randn(output_size) # if bias_algo == "random"
        ).double().requires_grad_() if output_size > 0 else torch.empty(0)
        self.hidden = False

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]: # pragma: no cover
        """
        :param input_tensor: an input tensor (1D vector or 2D batch matrix both supported)
        :return: a forwarded tuple of pre-activation and activated tensors (shape corresponding to input)
        """
        pass

class EmbeddingLayer(Layer):
    def __init__(self, input_size: int, output_size: int, weight_algo: str, _bias_algo):
        """
        Initialize a layer of neurons that input can be embedded into
        :param _bias_algo: ignored (always zeros)
        """
        super().__init__(input_size, output_size, weight_algo, "zeros")

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.weights[input_tensor.long()]
        return embedded, embedded

class ActivationLayer(Layer):
    def __init__(self, input_size: int, output_size: int, weight_algo: str, bias_algo="random"):
        """
        Initialize a layer of neurons that can be activated
        :param bias_algo: Initialization algorithm for bias (default: "random")
        """
        super().__init__(input_size, output_size, weight_algo, bias_algo)

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        # if input is 3D (coming from embedding), then concatenated view is needed which can be achieved
        # by (-1, input_size of weights)
        # check if last two dimensions of the input can be viewed as input dimension of weights, if needed
        weight_input_size = self.weights.size(0)
        if input_tensor.dim() == 2 and input_tensor.numel() == weight_input_size:
            input_view = input_tensor.view(-1)
        elif input_tensor.dim() >= 2 and math.prod(input_tensor.shape[-2:]) == weight_input_size:
            input_view = input_tensor.view(-1, weight_input_size)
        else:
            input_view = input_tensor
        # PyTorch internally treats 1D vector of shape (input_size, ) as (1, input_size)
        # to achieve dot product of (1, input_size) x (input_size, output_size) = (1, output_size)
        # then automatically squeezes it to (output_size,) which matches biases shape (output_size,)
        # on flip side for batch 2D vectors of shape (batch_size, input_size) dot product result is
        # of shape (batch_size, output_size) then biases are automatically un-squeezed to shape
        # (batch_size, output_size) to repeatedly added for an output of shape (batch_size, output_size)
        forwarded = input_view @ self.weights + self.biases
        return forwarded, forwarded

class SigmoidLayer(ActivationLayer):
    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        pre_activation, _ = super().forward(input_tensor)
        return pre_activation, pre_activation.sigmoid()

class ReluLayer(ActivationLayer):
    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        pre_activation, _ = super().forward(input_tensor)
        if self.hidden:
            # stabilize output in hidden layers prevent overflow with ReLU activations
            pre_activation = func.batch_norm(pre_activation)
        return pre_activation, pre_activation.relu()

class TanhLayer(ActivationLayer):
    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        pre_activation, _ = super().forward(input_tensor)
        return pre_activation, pre_activation.tanh()

class SoftmaxLayer(ActivationLayer):
    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        logits, _ = super().forward(input_tensor)
        return logits, logits.softmax(dim=logits.ndim - 1)

class MultiLayerPerceptron:
    _layer_map: dict[str, type(ActivationLayer)] = {
        "embedding": EmbeddingLayer,
        "relu": ReluLayer,
        "sigmoid": SigmoidLayer,
        "softmax": SoftmaxLayer,
        "tanh": TanhLayer,
    }

    def __init__(self, layer_sizes: list[int], weight_algo: str = "xavier", bias_algo: str = "random",
                 forward_algos: list[str] = None):
        """
        Initialize a multi-layer perceptron
        :param layer_sizes: List of integers where each integer represents the input size of the corresponding layer
        and the output size of the next layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param forward_algos: Forwarding algorithms per layer (default: "relu")
        """
        self.algos = forward_algos or ["relu"] * (len(layer_sizes) - 1)
        self.layers: list[Layer] = []
        size_idx = 0
        for algo in self.algos:
            in_sz = layer_sizes[size_idx] if size_idx < len(layer_sizes) else 0
            out_sz = layer_sizes[size_idx + 1] if size_idx < len(layer_sizes) -1 else 0
            if algo not in self._layer_map.keys():
                raise ValueError(f"Unsupported activation algorithm: {algo}")
            self.layers.append(self._layer_map.get(algo)(in_sz, out_sz, weight_algo, bias_algo))
            size_idx += 2 if algo == "embedding" else 1
        for i, layer in enumerate(self.layers):
            layer.hidden = (0 < i < len(self.layers) - 1)


class NeuralNetworkModel(MultiLayerPerceptron):
    def __init__(self, model_id, layer_sizes: list[int], weight_algo="xavier", bias_algo="random", activation_algos=None):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms (default: "relu")
        """
        super().__init__(layer_sizes, weight_algo, bias_algo, activation_algos)
        self.model_id = model_id
        self.optimizer = torch.optim.Adam(self.params) if len(self.params) > 0 else None
        self.progress = []
        self.training_data_buffer: list[Tuple] = []
        self.training_buffer_size: int = self.num_params
        self.avg_cost = None

    @property
    def weights(self) -> list[Tensor]:
        """
        :return: Model weights
        """
        return [l.weights for l in self.layers]

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Model parameters
        """
        return self.weights + [l.biases for l in self.layers]

    @property
    def num_params(self) -> int:
        """
        :return: Number of model parameters
        """
        return sum([p.numel() for p in self.params])

    def get_model_data(self) -> dict:
        return {
            "algos": self.algos,
            "layers": [{
                "weights": l.weights.tolist(),
                "biases": l.biases.tolist(),
            } for l in self.layers],
            "progress": self.progress,
            "training_data_buffer": self.training_data_buffer,
            "average_cost": self.avg_cost,
        }

    def set_model_data(self, model_data: dict):
        for layer, layer_state in zip(self.layers, model_data["layers"]):
            layer.weights = torch.tensor(layer_state["weights"], dtype=torch.float64, requires_grad=True)
            layer.biases = torch.tensor(layer_state["biases"], dtype=torch.float64, requires_grad=True)

        self.optimizer = torch.optim.Adam(self.params)
        self.progress = model_data["progress"]
        self.training_data_buffer = model_data["training_data_buffer"]
        self.training_buffer_size = self.num_params
        self.avg_cost = model_data["average_cost"]

    def serialize(self):
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"model_{self.model_id}.json")
        model_data = self.get_model_data()
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        log.info(f"Model saved successfully: {model_path}")
        optimizer_path = os.path.join("models", f"optimizer_{self.model_id}.pth")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        log.info(f"Optimizer saved successfully: {optimizer_path}")

    @classmethod
    def deserialize(cls, model_id: str):
        try:
            model_path = os.path.join("models", f"model_{model_id}.json")
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            model = cls(model_id, [], activation_algos=model_data["algos"])
            model.set_model_data(model_data)
            optimizer_path = os.path.join("models", f"optimizer_{model_id}.pth")
            model.optimizer.load_state_dict(torch.load(optimizer_path))
            return model
        except FileNotFoundError as e:
            log.error(f"File not found error occurred: {str(e)}")
            raise KeyError(f"Model {model_id} not created yet.")

    @classmethod
    def delete(cls, model_id: str):
        try:
            model_path = os.path.join("models", f"model_{model_id}.json")
            os.remove(model_path)
            optimizer_path = os.path.join("models", f"optimizer_{model_id}.pth")
            os.remove(optimizer_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    def compute_output(self, input_vector: list[float], target: list[float] = None) -> Tuple[list[float], float]:
        """
        Compute activated output and optionally also cost compared to the provided target vector.
        :param input_vector: Input vector
        :param target: Target vector (optional)
        :return: activation, cost (optional)
        """
        # forward pass
        input_tensor = torch.tensor(input_vector, dtype=torch.float64)
        activation, cost = self._forward(input_tensor, target)
        # activation same shape list and a float cost is returned, if any
        return activation.tolist(), cost.item() if cost.numel() > 0 else None

    def _forward(self, input_tensor: Tensor, target: list, dropout_rate=0.0) -> Tuple[Tensor, Tensor]:
        activation = input_tensor
        logits = None
        for layer in self.layers:
            logits, activation = layer.forward(activation)
            if layer.hidden and target is None:
                # Apply dropout only to hidden layers during training
                activation = nn.functional.dropout(activation, p=dropout_rate)

        if target is None or any(tgt is None for tgt in target):
            cost = torch.empty(0)
        elif self.algos[-1] == "softmax":
            label_data = target[0] if logits.ndim == 1 else [tgt[0] for tgt in target]
            label_tensor = torch.tensor(label_data, dtype=torch.int64)
            cost = nn.functional.cross_entropy(logits, label_tensor)
        else:
            target_tensor = torch.tensor(target, dtype=torch.float64)
            cost = nn.functional.mse_loss(activation, target_tensor)

        return activation, cost

    def train(self, training_data: list[Tuple[list[float], list[float]]], epochs=100, learning_rate=0.01, decay_rate=0.9,
              dropout_rate=0.2, l2_lambda=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Train the neural network using the provided training data.
        :param training_data: list of tuples of input and target
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param decay_rate: Decay rate of learning rate for finer gradient descent
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers
        :param l2_lambda: L2 regularization strength
        :param beta1: adam optimizer first moment parameter
        :param beta2: adam optimizer second moment parameter
        :param epsilon: adam optimizer the smallest step parameter
        """
        # Combine incoming training data with buffered data
        self.training_data_buffer.extend(training_data)

        # Check if buffer size is sufficient
        if len(self.training_data_buffer) < self.training_buffer_size:
            print(f"Model {self.model_id}: Insufficient training data. "
                  f"Current buffer size: {len(self.training_data_buffer)}, "
                  f"required: {self.training_buffer_size}")
            self.serialize() # serialize model with partial training data for next time
            return

        # Proceed with training using combined data if buffer size is sufficient
        training_data = self.training_data_buffer
        self.training_data_buffer = []  # Clear buffer

        # Calculate sample size
        training_sample_size = int(len(training_data) / epochs)  # sample equally per epoch

        # Adjust optimizer hyperparameters
        for param_group in self.optimizer.param_groups:
            param_group["betas"] = (beta1, beta2)
            param_group["eps"] = epsilon

        self.progress = []
        last_serialized = time.time()
        for epoch in range(epochs):
            random.shuffle(training_data)
            training_sample = training_data[:training_sample_size]

            # decay learning rate
            current_learning_rate = learning_rate * (decay_rate ** epoch)

            # Adjust optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_learning_rate

            # organize training data for forward pass
            input_tensor = torch.tensor([inpt for inpt, _ in training_sample], dtype=torch.float64)
            target = [tgt for _, tgt in training_sample]
            # calculate cost
            _, cost = self._forward(input_tensor, target, dropout_rate)
            # apply L2 regularization, if any
            if l2_lambda > 0.0:
                cost += l2_lambda * sum((w ** 2).sum() for w in self.weights)
            # zero out gradients
            self.optimizer.zero_grad()
            # back propagate to populate gradients
            cost.backward()
            # apply optimizer gradient descent
            self.optimizer.step()

            # Record progress
            progress_dt, progress_cost = dt.now().isoformat(), cost.item()
            self.progress.append({
                "dt": progress_dt,
                "epoch": epoch + 1,
                "cost": progress_cost
            })
            print(f"Model {self.model_id}: {progress_dt} - Epoch {epoch + 1}, Cost: {progress_cost:.4f}")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize() # pragma: no cover

        # Calculate current average progress cost
        avg_progress_cost = sum([progress["cost"] for progress in self.progress]) / len(self.progress)
        # Update overall average cost
        self.avg_cost = ((self.avg_cost or avg_progress_cost) + avg_progress_cost) / 2.0
        # Log training result
        training_dt = dt.now().isoformat()
        print(f"Model {self.model_id}: {training_dt} - Done training for {epochs} epochs, "
              f"Cost: {avg_progress_cost:.4f} Overall Cost: {self.avg_cost:.4f}")

        # Serialize model after training
        self.serialize()
