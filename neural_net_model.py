import json
import logging
import os
import math
import random
from typing import Tuple, Callable
import time
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

log = logging.getLogger(__name__)

class Layer:
    algo = ""
    def __init__(self):
        self.hidden = False
        self.training = False
        self.weights: Tensor | None = None
        self.bias: Tensor | None = None

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Layer parameters
        """
        p  = [] if self.weights is None else [self.weights]
        p += [] if self.bias is None else [self.bias]
        return p

    @property
    def state_dict(self) -> dict:
        return {
            "params": [p.tolist() for p in self.params],
        }

    @state_dict.setter
    def state_dict(self, new_state: dict):
        new_params: list = new_state["params"]
        if len(new_params) > 0:
            self.weights = torch.tensor(new_params[0], dtype=torch.float64)
        if len(new_params) > 1:
            self.bias = torch.tensor(new_params[1], dtype=torch.float64)


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

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.weights[input_tensor.long()]

class FlattenLayer(Layer):
    algo = "flatten"
    def forward(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.dim() > 2: # batch input
            flattened = input_tensor.view(input_tensor.shape[0], -1)
        else:
            flattened = input_tensor.view(-1)
        return flattened

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
            self.weights = torch.randn(input_size, output_size).double()
            if bias_algo:
                self.bias = (torch.zeros(output_size) if bias_algo == "zeros"
                             else torch.randn(output_size) # if bias_algo == "random"
                            ).double()

    def forward(self, input_tensor: Tensor) -> Tensor:
        # PyTorch internally treats 1D vector of shape (input_size, ) as (1, input_size)
        # to achieve dot product of (1, input_size) x (input_size, output_size) = (1, output_size)
        # then automatically squeezes it to (output_size,) which matches biases shape (output_size,)
        # on flip side for batch 2D vectors of shape (batch_size, input_size) dot product result is
        # of shape (batch_size, output_size) then biases are automatically un-squeezed to shape
        # (batch_size, output_size) to repeatedly added for an output of shape (batch_size, output_size)
        forwarded = input_tensor @ self.weights
        if self.bias is not None:
            forwarded += self.bias
        return forwarded

class BatchNormLayer(Layer):
    algo = "batchnorm"
    def __init__(self, dim_size: int, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gain = torch.ones(dim_size, dtype=torch.float64) if dim_size > 0 else None
        self.bias = torch.zeros(dim_size, dtype=torch.float64) if dim_size > 0 else None
        self.variance = torch.ones(dim_size, dtype=torch.float64) if dim_size > 0 else None
        self.mean = torch.zeros(dim_size, dtype=torch.float64) if dim_size > 0 else None

    @property
    def params(self) -> list[Tensor]:
        p  = [] if self.gain is None else [self.gain]
        p += [] if self.bias is None else [self.bias]
        return p

    @property
    def state_dict(self) -> dict:
        return super().state_dict | {
            "eps": self.eps,
            "momentum": self.momentum,
        }

    @state_dict.setter
    def state_dict(self, new_state: dict):
        new_params: list = new_state["params"]
        if len(new_params) > 0:
            self.gain = torch.tensor(new_params[0], dtype=torch.float64)
            self.variance = torch.ones_like(self.gain, dtype=torch.float64)
        if len(new_params) > 1:
            self.bias = torch.tensor(new_params[1], dtype=torch.float64)
            self.mean = torch.zeros_like(self.bias, dtype=torch.float64)
        self.eps = new_state["eps"]
        self.momentum = new_state["momentum"]

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.training: # calculate current batch norm statistics during training
            mean = input_tensor.mean(0, keepdim=True)
            variance = input_tensor.var(0, keepdim=True)
            # update overall statistics
            with torch.no_grad():
                self.mean = (1 - self.momentum) * self.mean + self.momentum * mean
                self.variance = (1 - self.momentum) * self.variance + self.momentum * variance
        else: # using overall statistics during inference
            mean, variance = self.mean, self.variance
        # forward pass
        return self.gain * (input_tensor - mean) / torch.sqrt(variance + self.eps) + self.bias

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
    def __init__(self, layer_sizes: list[int], weight_algo = "xavier", bias_algo = "zeros",
                 activation_algos: list[str] = None, batchnorm=(1e-5, 0.1)):
        """
        Initialize a multi-layer perceptron
        :param layer_sizes: List of integers where each integer represents the input size of the corresponding layer
        and the output size of the next layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "zeros")
        :param activation_algos: Forwarding algorithms per layer (default: ["relu"] * num_layers)
        :param batchnorm: Batch normalization configuration a tuple of epsilon and momentum
        """
        algos = activation_algos or ["relu"] * (len(layer_sizes) - 1)
        # ensure linear sandwich
        self.algos = []
        for i, algo in reversed(list(enumerate(algos))):
            self.algos.insert(0, algo)
            if algo in ["relu", "sigmoid", "softmax", "tanh"] and (
                    i == 0 or algos[i - 1] not in ["batchnorm", "linear"]
            ):
                self.algos.insert(0, "linear")
            elif algo in ["embedding"] and algos[i + 1] not in ["flatten"]:
                self.algos.insert(i + 1, "flatten")
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
            elif algo == "flatten":
                layer = FlattenLayer()
            elif algo == "linear":
                layer = LinearLayer(in_sz, out_sz, bias_algo)
            elif algo == "batchnorm":
                layer = BatchNormLayer(in_sz, *batchnorm)
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
                    linear_layer_for_gain.weights *= ReluLayer.weight_gain
                elif algo == "tanh":
                    linear_layer_for_gain.weights *= TanhLayer.weight_gain
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
    def __init__(self, model_id, layer_sizes: list[int] = None, weight_algo="xavier", bias_algo="zeros",
                 activation_algos=None, optimizer_algo="adam", batchnorm=(1e-5, 0.1)):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents a dimension of a layer.
        :param optimizer_algo: Optimization algorithm (default: "adam")
        """
        super().__init__(layer_sizes or [], weight_algo, bias_algo, activation_algos, batchnorm)
        self.model_id = model_id
        self.optimizer: Optimizer | None = None
        if len(self.params) > 0:
            if optimizer_algo == "adam":
                self.optimizer = torch.optim.Adam(self.params)
        self.progress = []
        self.training_data_buffer: list[Tuple] = []
        self.training_buffer_size: int = self.num_params
        self.avg_cost = None
        self.avg_cost_history = []
        self.stats = None
        self.status = "Created"

    @property
    def weights(self) -> list[Tensor]:
        """
        :return: Model weights
        """
        return [l.weights for l in self.layers if l.weights is not None]

    @property
    def num_params(self) -> int:
        """
        :return: Number of model parameters
        """
        return sum([p.numel() for p in self.params])

    def get_model_data(self) -> dict:
        return {
            "algos": self.algos,
            "layers": [l.state_dict for l in self.layers],
            "progress": self.progress,
            "training_data_buffer": self.training_data_buffer,
            "average_cost": self.avg_cost,
            "average_cost_history": self.avg_cost_history,
            "stats": self.stats,
            "status": self.status,
        }

    def set_model_data(self, model_data: dict):
        for layer, layer_state in zip(self.layers, model_data["layers"]):
            layer.state_dict = layer_state

        self.progress = model_data["progress"]
        self.training_data_buffer = model_data["training_data_buffer"]
        self.training_buffer_size = self.num_params
        self.avg_cost = model_data["average_cost"]
        self.avg_cost_history = model_data["average_cost_history"]
        self.stats = model_data["stats"]
        self.status = model_data["status"]

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
            model = cls(model_id, activation_algos=model_data["algos"])
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
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    def compute_output(self, input_data: list, target: list = None) -> Tuple[list, float]:
        """
        Compute activated output and optionally also cost compared to the provided target vector.
        :param input_data: Input data 1D or 2D list
        :param target: Target data 1D or 2D list (optional)
        :return: activated output, cost (optional)
        """
        # forward pass
        input_tensor = torch.tensor(input_data, dtype=torch.float64)
        activations, cost = self._forward(input_tensor, target)
        # last activation  and a float cost is returned, if any
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

    def train(self, training_data: list[Tuple[list[float], list[float]]], epochs=100, learning_rate=0.01, batch_size=None,
              decay_rate=0.9, dropout_rate=0.2, l2_lambda=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Train the neural network using the provided training data.
        :param training_data: list of tuples of input and target
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param batch_size: Batch size override for training sample.
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
            log.info(f"Model {self.model_id}: Insufficient training data. "
                  f"Current buffer size: {len(self.training_data_buffer)}, "
                  f"required: {self.training_buffer_size}")
            self.serialize() # serialize model with partial training data for next time
            return

        # Proceed with training using combined data if buffer size is sufficient
        training_data = self.training_data_buffer
        self.training_data_buffer = []  # Clear buffer

        # Calculate sample size
        training_sample_size = batch_size or int(len(training_data) / epochs)  # explicit or sample equally per epoch
        log.info(f"Training sample size: {training_sample_size}")

        # Adjust optimizer hyperparameters
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["betas"] = (beta1, beta2)
                param_group["eps"] = epsilon

        # Reset model for training prep and save
        self.progress = []
        self.stats = None
        self.status = "Training"
        self.serialize()

        # Start training
        activations = None
        last_serialized = time.time()
        for epoch in range(epochs):
            random_indices = torch.randint(0, len(training_data), (training_sample_size,))
            training_sample = [training_data[i] for i in random_indices]

            # decay learning rate
            current_learning_rate = learning_rate * (decay_rate ** epoch)

            # Adjust optimizer learning rate
            if self.optimizer is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_learning_rate

            # organize training data for forward pass
            input_tensor = torch.tensor([inpt for inpt, _ in training_sample], dtype=torch.float64)
            target = [tgt for _, tgt in training_sample]
            # copy weights for later update ratio calc
            prev_weights: list[Tensor] = [w.clone().detach() for w in self.weights]
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
            # check if training taking long
            long_training = time.time() - last_serialized >= 10
            # on last epoch or for long training intervals
            # retain final activation gradients to collect stats
            if epoch + 1 == epochs or long_training:
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
            if epoch % max(1, epochs // 100) == 0: # only 100 progress points or less stored
                with torch.no_grad():
                    weight_upd_ratio = [((w - pw).data.std() / (w.data.std() + 1e-8)).item()
                                        for pw, w in zip(prev_weights, self.weights)]
                self.progress.append({
                    "dt": progress_dt,
                    "epoch": epoch + 1,
                    "cost": progress_cost,
                    "weight_upd_ratio": [weight_upd_ratio.pop(0) if l.weights is not None and len(weight_upd_ratio) > 0
                                         else None for l in self.layers],
                })
            # Log each
            log.info(f"Model {self.model_id}: Epoch {epoch + 1}, Cost: {progress_cost:.4f}")

            # Serialize model while long training intervals
            if long_training: # pragma: no cover
                self._record_training_overall_progress(activations)
                self.serialize()
                last_serialized = time.time()

        # Mark training finished
        self.status = "Trained"
        # Log training is done
        log.info(f"Model {self.model_id}: Done training for {epochs} epochs.")
        # Serialize model after training
        self._record_training_overall_progress(activations)
        self.serialize()

    def _record_training_overall_progress(self, activations):
        # Calculate current average progress cost
        progress_cost = [progress["cost"] for progress in self.progress]
        avg_progress_cost = sum(progress_cost) / len(self.progress)
        # Update overall average cost
        self.avg_cost = ((self.avg_cost or avg_progress_cost) + avg_progress_cost) / 2.0
        self.avg_cost_history.append(self.avg_cost)
        if len(self.avg_cost_history) > 100: #
            self.avg_cost_history.pop(random.randint(1, 98))
        # Update stats
        hist_f: Callable[[torch.return_types.histogram], Tuple[list, list]] = (
            lambda h: (h.bin_edges[:-1].tolist(), h.hist.tolist()))
        act_hist = [hist_f(torch.histogram(a, density=True)) for a in activations]
        act_grad_hist = [([], []) if a.grad is None else hist_f(torch.histogram(a.grad, density=True))
                         for a in activations]
        weight_grad_hist = [([], []) if l.weights is None else hist_f(torch.histogram(l.weights.grad, density=True))
                            for l in self.layers]
        self.stats = {
            "layers": [{
                "algo": l.algo,
                "activation": {
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "saturated": (a.abs() > 0.97).float().mean().item(),
                    "histogram": {"x": ahx, "y": ahy},
                },
                "gradient": {
                    "mean": a.grad.mean().item(),
                    "std": a.grad.std().item(),
                    "histogram": {"x": ghx, "y": ghy},
                } if a.grad is not None else None,
            } for l, a, (ahx, ahy), (ghx, ghy) in zip(self.layers, activations, act_hist, act_grad_hist)],
            "weights": [{
                "shape": str(tuple(l.weights.shape)),
                "data": {
                    "mean": l.weights.mean().item(),
                    "std": l.weights.std().item(),
                },
                "gradient": {
                    "mean": l.weights.grad.mean().item(),
                    "std": l.weights.grad.std().item(),
                    "histogram": {"x": ghx, "y": ghy},
                },
            } if l.weights is not None else None for l, (ghx, ghy) in zip(self.layers, weight_grad_hist)],
        }
        # Log training progress
        log.info(f"Model {self.model_id} - Cost: {avg_progress_cost:.4f} Overall Cost: {self.avg_cost:.4f}")
