import json
import logging
import os
import math
import random
from typing import Tuple
import time
from datetime import datetime as dt
import torch
from torch import Tensor
from gradients import scalar, vector, Activation


log = logging.getLogger(__name__)

class Neuron:
    def __init__(self, input_size: int = 0, weight_algo="xavier", bias_algo="random", activation_algo="relu"):
        """
        Initialize a neuron
        :param input_size: represents the input size of neuron
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for bias (default: "random")
        :param activation_algo: Activation algorithm (default: "relu")
        """
        self.weights = vector([
            random.uniform(-math.sqrt(1 / input_size), math.sqrt(1 / input_size)) if weight_algo == "xavier"
            else random.uniform(-math.sqrt(2 / input_size), math.sqrt(2 / input_size)) if weight_algo == "he"
            else random.uniform(-1, 1) # gaussian
            for _ in range(input_size)
        ])
        self.bias = scalar(0) if bias_algo == "zeros" else scalar(random.uniform(-1, 1))
        self.activation_algo = activation_algo

    def output(self, input_vector: Tensor) -> Tensor:
        """
        Gives output of this neuron pre-activation for given input.
        :param input_vector: an input vector
        :return: output
        """
        return self.weights.vdot(input_vector) + self.bias

    def activate(self, input_vector: Tensor) -> Activation:
        """
        Activates this neuron with given input.
        :param input_vector: an input vector
        :return: activated output scalar
        """
        return Activation(self.output(input_vector)).activate(self.activation_algo)

class Layer:
    def __init__(self, input_size: int = 0, output_size: int = 0, weight_algo="xavier", bias_algo="random", activation_algo="relu"):
        """
        Initialize a layer of neurons
        :param input_size: represents the input size of layer
        :param output_size: represents the output size of layer
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for bias (default: "random")
        :param activation_algo: Activation algorithm (default: "relu")
        """
        self.neurons = [Neuron(input_size, weight_algo, bias_algo, activation_algo)
                        for _ in range(output_size)]
        self.activation_algo = activation_algo

    def output(self, input_vector: Tensor) -> Activation:
        """
        Gives output of this layer of neurons pre-activation for given input
        :param input_vector: an input vector
        :return: output vector pre-activation
        """
        return Activation(torch.stack([neuron.output(input_vector) for neuron in self.neurons]))

class MultiLayerPerceptron:
    def __init__(self, layer_sizes: list[int],
                 weight_algo: str = "xavier",
                 bias_algo: str = "random",
                 activation_algos: list[str] = None):
        """
        Initialize a multi-layer perceptron
        :param layer_sizes: List of integers where each integer represents the input size of the corresponding layer
        and the output size of the next layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms per layer (default: "relu")
        """
        input_sizes = layer_sizes[:-1]
        output_sizes = layer_sizes[1:]
        if activation_algos is None:
            activation_algos = ["relu"] * len(input_sizes)
        self.layers = [Layer(input_size, output_size, weight_algo, bias_algo, activation_algo)
                       for input_size, output_size, activation_algo in
                       zip(input_sizes, output_sizes, activation_algos)]

class NeuralNetworkModel(MultiLayerPerceptron):
    def __init__(self, model_id, layer_sizes: list[int], weight_algo="xavier", bias_algo="random", activation_algos=None):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms (default: "sigmoid")
        """
        super().__init__(layer_sizes, weight_algo, bias_algo, activation_algos)
        self.model_id = model_id
        self.optimizer = torch.optim.Adam(self.params) if len(self.params) > 0 else None
        self.progress = []
        self.training_data_buffer: list[Tuple[Tensor, Tensor]] = []
        self.training_buffer_size = self.num_params

    @property
    def weights(self) -> list[Tensor]:
        """
        :return: Model weights
        """
        return [n.weights for l in self.layers for n in l.neurons]

    @property
    def params(self) -> list[Tensor]:
        """
        :return: Model parameters
        """
        return self.weights + [n.bias for l in self.layers for n in l.neurons]

    @property
    def num_params(self) -> int:
        """
        :return: Number of model parameters
        """
        return sum([p.numel() for p in self.params])

    def get_model_data(self) -> dict:
        return {
            "layers": [{
                "algo": l.activation_algo,
                "neurons": [{
                    "weights": [w.tolist() for w in n.weights],
                    "bias": n.bias.item()
                } for n in l.neurons]
            } for l in self.layers],
            "progress": self.progress,
            "training_data_buffer": [tuple(tv.tolist() for tv in ttv) for ttv in self.training_data_buffer],
        }

    def set_model_data(self, model_data: dict):
        for layer_state in model_data["layers"]:
            layer = Layer(activation_algo=layer_state["algo"])
            self.layers.append(layer)
            for neuron_state in layer_state["neurons"]:
                neuron = Neuron(activation_algo=layer_state["algo"])
                layer.neurons.append(neuron)
                neuron.weights = vector([w for w in neuron_state["weights"]])
                neuron.bias = scalar(neuron_state["bias"])

        self.optimizer = torch.optim.Adam(self.params)
        self.progress = model_data["progress"]
        self.training_data_buffer = [tuple(vector(t) for t in tt) for tt in model_data["training_data_buffer"]]
        self.training_buffer_size = self.num_params

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
            model = cls(model_id, [])
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

    def compute_output(self, input_vector: Tensor, target: Tensor = None, dropout_rate=0.0) -> Tuple[Activation, Tensor]:
        """
        Compute activated output and optionally also cost compared to the provided target vector.
        :param input_vector: Input vector
        :param target: Target vector (optional)
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers (optional)
        :return: activation tensor, cost, gradients
        """
        activation = Activation(input_vector)
        num_layers = len(self.layers)
        for l in range(num_layers):
            algo = self.layers[l].activation_algo
            pre_activation: Activation = self.layers[l].output(activation.tensor)
            if algo == "relu" and l < num_layers - 1:
                # stabilize output in hidden layers prevent overflow with ReLU activations
                pre_activation.batch_norm()
            activation = pre_activation.activate(algo)
            if l < num_layers - 1:  # Hidden layers only
                # Apply dropout only to hidden layers
                activation.apply_dropout(dropout_rate)

        cost = torch.empty(0)
        if target is not None:
            cost = activation.calculate_cost(target)

        return activation, cost

    def train(self, training_data: list[Tuple[Tensor, Tensor]], epochs=100, learning_rate=0.01, decay_rate=0.9,
              dropout_rate=0.2, l2_lambda=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Train the neural network using the provided training data.
        :param training_data: List of tuples [(input_vector, target_vector), ...].
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

        # Calculate sample size
        training_sample_size = int(self.training_buffer_size * 0.1)  # sample 10% of buffer

        # Proceed with training using combined data if buffer size is sufficient
        training_data = self.training_data_buffer
        self.training_data_buffer = []  # Clear buffer

        # Adjust optimizer hyperparameters
        for param_group in self.optimizer.param_groups:
            param_group["betas"] = (beta1, beta2)
            param_group["eps"] = epsilon

        self.progress = []
        last_serialized = time.time()
        for epoch in range(epochs):
            random.shuffle(training_data)
            training_data_sample = training_data[:training_sample_size]

            # decay learning rate
            current_learning_rate = learning_rate * (decay_rate ** epoch)

            # Adjust optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_learning_rate

            # calculate costs
            costs: list[Tensor] = []
            for input_vector, target_vector in training_data_sample:
                _, cost = self.compute_output(input_vector, target_vector, dropout_rate)
                costs.append(cost)
            # average cost
            avg_cost = torch.stack(costs).mean()
            # apply L2 regularization, if any
            if l2_lambda > 0.0:
                avg_cost += l2_lambda * sum((w ** 2).sum() for w in self.weights)
            # zero out gradients
            self.optimizer.zero_grad()
            # back propagate to populate gradients
            avg_cost.backward()
            # apply optimizer gradient descent
            self.optimizer.step()

            # Record progress
            progress_dt, progress_cost = dt.now().isoformat(), avg_cost.item()
            self.progress.append({
                "dt": progress_dt,
                "epoch": epoch + 1,
                "cost": progress_cost
            })
            print(f"Model {self.model_id}: {progress_dt} - Epoch {epoch + 1}, Cost: {progress_cost:.4f}")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize()

        # Serialize model after training
        self.serialize()
