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


log = logging.getLogger(__name__)

class Layer:
    algo = ""
    def __init__(self):
        self.hidden = False
        self.training = False
        self.weights = torch.empty(0)
        self.bias = torch.empty(0)

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Layer parameters
        """
        return []

    @params.setter
    def params(self, new_params: list): # pragma: no cover
        pass

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        :param input_tensor: an input tensor (1D vector or 2D batch matrix both supported)
        :return: a forwarded tensor (shape corresponding to input)
        """
        return input_tensor

class EmbeddingLayer(Layer):
    algo = "embedding"
    def __init__(self, vocab_size: int = 0, embedding_size: int = 0):
        """
        Initialize a layer of neurons
        :param vocab_size: represents the vocabulary size of layer
        :param embedding_size: represents the embedding size of layer
        """
        super().__init__()
        if all(sz > 0 for sz in (vocab_size, embedding_size)):
            self.weights: Tensor = torch.randn(vocab_size, embedding_size).double()

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Layer parameters
        """
        return [self.weights]

    @params.setter
    def params(self, new_params: list):
        self.weights = torch.tensor(new_params[0], dtype=torch.float64)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.weights[input_tensor.long()]

class LinearLayer(Layer):
    algo = "linear"
    def __init__(self, input_size: int = 0, output_size: int = 0, bias_algo="zeros"):
        """
        Initialize a layer of neurons
        :param input_size: represents the input size of layer
        :param output_size: represents the output size of layer
        :param bias_algo: Initialization algorithm for bias (default: "zeros")
        """
        super().__init__()
        if all(sz > 0 for sz in (input_size, output_size)):
            self.weights: Tensor = torch.randn(input_size, output_size).double()

        self.bias: Tensor = (torch.randn(output_size) if bias_algo == "random"
                            else torch.zeros(output_size)).double() # if bias_algo == "zeros"

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Layer parameters
        """
        return [self.weights, self.bias]

    @params.setter
    def params(self, new_params: list):
        self.weights = torch.tensor(new_params[0], dtype=torch.float64)
        self.bias = torch.tensor(new_params[1], dtype=torch.float64)

    def forward(self, input_tensor: Tensor) -> Tensor:
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
        return input_view @ self.weights + self.bias

class BatchNormLayer(Layer):
    algo = "batchnorm"
    def __init__(self, dim_size: int):
        super().__init__()
        self.gain = torch.ones(dim_size, dtype=torch.float64)
        self.bias = torch.zeros(dim_size, dtype=torch.float64)
        self.mean = torch.zeros(dim_size, dtype=torch.float64)
        self.variance = torch.ones(dim_size, dtype=torch.float64)

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Layer parameters
        """
        return [self.gain, self.bias]

    @params.setter
    def params(self, new_params: list):
        self.gain = torch.tensor(new_params[0], dtype=torch.float64)
        self.bias = torch.tensor(new_params[1], dtype=torch.float64)
        self.mean = torch.zeros_like(self.bias, dtype=torch.float64)
        self.variance = torch.ones_like(self.gain, dtype=torch.float64)

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.training: # calculate current batch norm statistics during training
            mean = input_tensor.mean(0, keepdim=True)
            variance = input_tensor.var(0, keepdim=True)
            # update overall statistics
            with torch.no_grad():
                self.mean = 0.9 * self.mean + 0.1 * mean
                self.variance = 0.9 * self.variance + 0.1 * variance
        else: # using overall statistics during inference
            mean, variance = self.mean, self.variance
        # forward pass
        return self.gain * (input_tensor - mean) / torch.sqrt(variance + 1e-5) + self.bias

class SigmoidLayer(Layer):
    algo = "sigmoid"
    def forward(self, pre_activation: Tensor) -> Tensor: return pre_activation.sigmoid()

class ReluLayer(Layer):
    algo = "relu"
    weight_gain = math.sqrt(2.0)
    def forward(self, pre_activation: Tensor) -> Tensor: return pre_activation.relu()

class TanhLayer(Layer):
    algo = "tanh"
    weight_gain = 5.0 / 3.0
    def forward(self, pre_activation: Tensor) -> Tensor: return pre_activation.tanh()

class SoftmaxLayer(Layer):
    algo = "softmax"
    def forward(self, logits: Tensor) -> Tensor: return logits.softmax(dim=logits.ndim - 1)

class MultiLayerPerceptron:
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
        algos = forward_algos or ["relu"] * (len(layer_sizes) - 1)
        # ensure linear sandwich
        self.algos = []
        for i, algo in reversed(list(enumerate(algos))):
            self.algos.insert(0, algo)
            if algo in ["relu", "sigmoid", "softmax", "tanh"] and (
                    i == 0 or algos[i - 1] not in ["batchnorm", "linear"]
            ):
                self.algos.insert(0, "linear")
        # build layers
        self.layers: list[Layer] = []
        size_idx = 0
        linear_layer_for_gain = None
        num_algos = len(self.algos)
        num_layer_sizes = len(layer_sizes)
        for i, algo in enumerate(self.algos):
            # prep input, output sizes
            in_sz = layer_sizes[size_idx] if size_idx < num_layer_sizes else 0
            out_sz = layer_sizes[size_idx + 1] if size_idx + 1 < num_layer_sizes else 0
            # create layer
            if algo == "embedding":
                layer = EmbeddingLayer(in_sz, out_sz)
            elif algo == "linear":
                layer = LinearLayer(in_sz, out_sz, bias_algo)
            elif algo == "batchnorm":
                layer = BatchNormLayer(in_sz)
            elif algo == "relu":
                layer = ReluLayer()
            elif algo == "sigmoid":
                layer = SigmoidLayer()
            elif algo == "softmax":
                layer = SoftmaxLayer()
            elif algo == "tanh":
                layer = TanhLayer()
            else:
                raise ValueError(f"Unsupported activation algorithm: {algo}")
            # scale linear layer weights based on init weight algo
            if algo == "linear" and in_sz > 0 and weight_algo in ["xavier", "he"]:
                layer.weights /= math.sqrt(in_sz)
            # store linear layer for later scaling with gain based on activation algo, if any
            if algo == "linear":
                linear_layer_for_gain = layer if weight_algo == "he" else None
            # apply gain to linear layer weights based on activation algo
            if linear_layer_for_gain is not None:
                if algo == "relu":
                    linear_layer_for_gain.weights *= math.sqrt(2.0)
                elif algo == "tanh":
                    linear_layer_for_gain.weights *= 5.0 / 3.0
            # set hidden flag
            layer.hidden = (0 < i < num_algos - 1)
            # add layer
            self.layers.append(layer)
            # shift to next layer sizes
            if algo == "embedding":
                size_idx += 2
            elif algo == "linear":
                size_idx += 1

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Model parameters
        """
        return [p for l in self.layers for p in l.params]

class NeuralNetworkModel(MultiLayerPerceptron):
    def __init__(self, model_id, layer_sizes: list[int], weight_algo="xavier", bias_algo="random",
                 activation_algos=None, optimizer_algo="adam"):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms (default: "relu")
        :param optimizer_algo: Optimization algorithm (default: "adam")
        """
        super().__init__(layer_sizes, weight_algo, bias_algo, activation_algos)
        self.model_id = model_id
        self.optimizer = torch.optim.Adam(self.params) if optimizer_algo == "adam" and len(self.params) > 0 else None
        self.progress = []
        self.training_data_buffer: list[Tuple] = []
        self.training_buffer_size: int = self.num_params
        self.avg_cost = None
        self.stats = None

    @property
    def weights(self) -> list[Tensor]:
        """
        :return: Model weights
        """
        return [l.weights for l in self.layers if l.weights.numel() > 0]

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
                "params": [p.tolist() for p in l.params],
            } for l in self.layers],
            "progress": self.progress,
            "training_data_buffer": self.training_data_buffer,
            "average_cost": self.avg_cost,
            "stats": self.stats
        }

    def set_model_data(self, model_data: dict):
        for layer, layer_state in zip(self.layers, model_data["layers"]):
            layer.params = layer_state["params"]

        self.progress = model_data["progress"]
        self.training_data_buffer = model_data["training_data_buffer"]
        self.training_buffer_size = self.num_params
        self.avg_cost = model_data["average_cost"]
        self.stats = model_data["stats"]

    def serialize(self):
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"model_{self.model_id}.json")
        model_data = self.get_model_data()
        with open(model_path, 'w', encoding='utf-8') as f:
            # noinspection PyTypeChecker
            json.dump(model_data, f, indent=4)
        log.info(f"Model saved successfully: {model_path}")
        if self.optimizer is not None:
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
            if os.path.exists(optimizer_path):
                model.optimizer = torch.optim.Adam(model.params)
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
        activations, cost = self._forward(input_tensor, target)
        # last activation same shape list and a float cost is returned, if any
        return activations[-1].tolist(), cost.item() if cost.numel() > 0 else None

    def _forward(self, input_tensor: Tensor, target: list, dropout_rate=0.0) -> Tuple[list[Tensor], Tensor]:
        target_specified = target is not None and all(tgt is not None for tgt in target)
        forwarded_tensors = []
        forwarded_tensor = input_tensor
        logits = input_tensor
        for layer in self.layers:
            layer.training = target_specified
            logits = forwarded_tensor
            forwarded_tensor = layer.forward(logits)
            if layer.hidden and layer.training:
                # Apply dropout only to hidden layers during training
                forwarded_tensor = nn.functional.dropout(forwarded_tensor, p=dropout_rate)
            forwarded_tensors.append(forwarded_tensor)

        if not target_specified:
            cost = torch.empty(0)
        elif self.algos[-1] == "softmax":
            label_data = target[0] if logits.ndim == 1 else [tgt[0] for tgt in target]
            label_tensor = torch.tensor(label_data, dtype=torch.int64)
            cost = nn.functional.cross_entropy(logits, label_tensor)
        else:
            target_tensor = torch.tensor(target, dtype=torch.float64)
            cost = nn.functional.mse_loss(forwarded_tensor, target_tensor)

        return forwarded_tensors, cost

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
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["betas"] = (beta1, beta2)
                param_group["eps"] = epsilon

        self.progress = []
        activations = None
        last_serialized = time.time()
        for epoch in range(epochs):
            random.shuffle(training_data)
            training_sample = training_data[:training_sample_size]

            # decay learning rate
            current_learning_rate = learning_rate * (decay_rate ** epoch)

            # Adjust optimizer learning rate
            if self.optimizer is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_learning_rate

            # organize training data for forward pass
            input_tensor = torch.tensor([inpt for inpt, _ in training_sample], dtype=torch.float64)
            target = [tgt for _, tgt in training_sample]
            # require grad
            for p in self.params:
                p.requires_grad_()
            # calculate cost
            activations, cost = self._forward(input_tensor, target, dropout_rate)
            # apply L2 regularization, if any
            if l2_lambda > 0.0:
                cost += l2_lambda * sum((w ** 2).sum() for w in self.weights)
            # clear gradients
            for p in self.params:
                p.grad = None
            # on last epoch retain final activation gradients to collect stats
            if epoch + 1 == epochs:
                for a in activations:
                    a.retain_grad()
            # back propagate to populate gradients
            cost.backward()
            if self.optimizer is not None: # apply optimizer gradient descent
                self.optimizer.step()
            else: # apply stochastic gradient descent
                for p in self.params:
                    p.data -= current_learning_rate * p.grad

            # Record progress
            progress_dt, progress_cost = dt.now().isoformat(), cost.item()
            self.progress.append({
                "dt": progress_dt,
                "epoch": epoch + 1,
                "cost": progress_cost,
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
        # Update stats
        activation_hist = [torch.histogram(a, density=True) for a in activations]
        activation_grad_hist = [None if a.grad is None else torch.histogram(a.grad, density=True)
                                for a in activations]
        weight_grad_hist = [torch.histogram(w.grad, density=True) for w in self.weights]
        self.stats = {
            "layers": [{
                "algo": l.algo,
                "activation": {
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "saturated": (a.abs() > 0.97).float().mean().item(),
                    "histogram": {
                        "x": ah.bin_edges[:-1].tolist(),
                        "y": ah.hist.tolist()
                    },
                },
                "gradient": {
                    "mean": a.grad.mean().item(),
                    "std": a.grad.std().item(),
                    "histogram": {
                        "x": agh.bin_edges[:-1].tolist(),
                        "y": agh.hist.tolist()
                    },
                } if a.grad is not None else None,
            } for l, a, ah, agh in zip(self.layers, activations, activation_hist, activation_grad_hist)],
            "weights": [{
                "shape": str(tuple(w.shape)),
                "data": {
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                },
                "gradient": {
                    "mean": w.grad.mean().item(),
                    "std": w.grad.std().item(),
                    "histogram": {
                        "x": wgh.bin_edges[:-1].tolist(),
                        "y": wgh.hist.tolist()
                    },
                },
            } for w, wgh in zip(self.weights, weight_grad_hist)],
        }
        # Serialize model after training
        self.serialize()
