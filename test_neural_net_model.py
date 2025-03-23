import unittest
from parameterized import parameterized
from torch import Tensor
from neural_net_model import NeuralNetworkModel, Neuron, Layer, MultiLayerPerceptron
from gradients import vector, Activation

class TestNeuralNetModel(unittest.TestCase):

    @parameterized.expand([
        (9, "xavier", "random", "sigmoid"),
        (18, "xavier", "zeros", "relu"),
        (9, "he", "zeros", "tanh"),
        (4, "he", "random", "relu"),
        (3, "gaussian", "random", "sigmoid"),
    ])
    def test_neuron_initialization(self, input_size, weight_algo, bias_algo, activation_algo):
        neuron = Neuron(input_size, weight_algo, bias_algo, activation_algo)

        self.assertIsNotNone(neuron)
        self.assertEqual(input_size, len(neuron.weights.tolist()))
        self.assertIsNotNone(neuron.bias)
        self.assertEqual(activation_algo, neuron.activation_algo)

    def test_neuron_activate(self):
        neuron = Neuron(2)
        input_vector = vector([1.0, 2.0])

        activated_scalar: Activation = neuron.activate(input_vector)

        self.assertIsNotNone(activated_scalar.tensor)

    @parameterized.expand([
        (9, 9,),
        (9, 18,),
        (18, 9, ),
    ])
    def test_layer_initialization(self, input_size, output_size):
        layer = Layer(input_size, output_size)

        self.assertIsNotNone(layer)
        self.assertEqual(output_size, len(layer.neurons))
        self.assertEqual(input_size, len(layer.neurons[0].weights.tolist()))

    @parameterized.expand([
        ([9, 9, 9], "xavier", "random",),
        ([18, 9, 3], "xavier", "zeros",),
        ([9, 18, 9], "he", "zeros",),
        ([4, 8, 16], "he", "random",),
        ([3, 3, 3, 3], "gaussian", "random",),
    ])
    def test_multi_layer_perceptron_initialization(self, layer_sizes, weight_algo, bias_algo):
        multi_layer_perceptron = MultiLayerPerceptron(layer_sizes, weight_algo, bias_algo)

        self.assertIsNotNone(multi_layer_perceptron)
        self.assertEqual(len(multi_layer_perceptron.layers), len(layer_sizes) - 1)

    @parameterized.expand([
        ([9, 9, 9], "xavier", "random",),
        ([18, 9, 3], "xavier", "zeros",),
        ([9, 18, 9], "he", "zeros",),
        ([4, 8, 16], "he", "random",),
        ([3, 3, 3, 3], "gaussian", "random",),
    ])
    def test_model_initialization(self, layer_sizes: list[int], weight_algo: str, bias_algo: str):
        model = NeuralNetworkModel("test", layer_sizes, weight_algo, bias_algo)
        expected_buffer_size = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )

        self.assertEqual("test", model.model_id)
        for i in range(len(layer_sizes) - 1):
            self.assertEqual(layer_sizes[i], len(model.layers[i].neurons[0].weights.tolist()))
            self.assertEqual(layer_sizes[i + 1], len(model.layers[i].neurons))
        self.assertEqual(0, len(model.progress))
        self.assertEqual(expected_buffer_size, model.training_buffer_size)

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2,),
        ([18, 9, 3], ["relu", "softmax"],),
        ([9, 18, 9], ["tanh"] * 2,),
        ([4, 8, 16], ["softmax"] * 2,),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"],),
    ])
    def test_compute_output(self, layer_sizes: list[int], algos: list[str]):
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos)
        input_sizes = layer_sizes[:-1]
        output_sizes = layer_sizes[1:]
        sample_input = vector([0.5] * input_sizes[0])

        output, cost = model.compute_output(sample_input)

        self.assertEqual(output_sizes[-1], len(output.tensor.tolist()))
        self.assertEqual(0, cost.numel())

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2,),
        ([9, 9, 9], ["relu", "softmax"],),
        ([9, 9, 9], ["tanh"] * 2,),
        ([18, 9, 3], ["relu", "sigmoid"],),
        ([9, 18, 9], ["sigmoid", "softmax"],),
        ([4, 8, 16], ["sigmoid"] * 2,),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"],),
        ([18, 9, 3], ["relu"] * 2,),
        ([9, 18, 9], ["relu", "tanh"],),
    ])
    def test_train(self, layer_sizes: list[int], algos: list[str]):
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos)
        sample_input = vector([0.5] * layer_sizes[0])
        sample_target = Tensor([1.0] + [0.0] * (layer_sizes[-1] - 1)).double()

        initial_weights = [[n.weights.tolist() for n in l.neurons] for l in model.layers]
        initial_biases = [[n.bias.item() for n in l.neurons] for l in model.layers]
        _, initial_cost = model.compute_output(sample_input, sample_target)
        # Add enough data to meet the training buffer size
        training_data = [(sample_input, sample_target)] * model.training_buffer_size

        model.train(training_data, epochs=1)

        updated_weights = [[n.weights.tolist() for n in l.neurons] for l in model.layers]
        updated_biases = [[n.bias.item() for n in l.neurons] for l in model.layers]

        # Check that the model data is still valid
        self.assertEqual(len(initial_weights), len(updated_weights))
        self.assertEqual(len(initial_weights[0]), len(updated_weights[0]))
        self.assertEqual(len(initial_weights[0][0]), len(updated_weights[0][0]))
        self.assertEqual(len(initial_biases), len(updated_biases))
        self.assertEqual(len(initial_biases[0]), len(updated_biases[0]))

        # Ensure training progress
        self.assertGreater(len(model.progress), 0)
        self.assertNotEqual(model.progress[0]["cost"], initial_cost)
        self.assertEqual(len(model.training_data_buffer), 0)

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        # Verify model parameters correctly deserialized
        for i in range(len(layer_sizes) - 1):
            self.assertEqual(layer_sizes[i], len(persisted_model.layers[i].neurons[0].weights.tolist()))
            self.assertEqual(layer_sizes[i + 1], len(persisted_model.layers[i].neurons))

        persisted_weights = [[n.weights.tolist() for n in l.neurons] for l in persisted_model.layers]
        persisted_biases = [[n.bias.item() for n in l.neurons] for l in persisted_model.layers]

        self.assertEqual(len(updated_weights), len(persisted_weights))
        for i, (uwl, pwl) in enumerate(zip(updated_weights, persisted_weights)):
            for j, (uwv, pwv) in enumerate(zip(uwl, pwl)):
                for k, (uw, pw) in enumerate(zip(uwv, pwv)):
                    self.assertAlmostEqual(uw, pw, 4, f"Element at loc {i},{j},{k}")
        self.assertEqual(len(updated_biases), len(persisted_biases))
        for i, (ubl, pbl) in enumerate(zip(updated_biases, persisted_biases)):
            for j, (ub, pb) in enumerate(zip(ubl, pbl)):
                self.assertAlmostEqual(ub, pb, 4, f"Element at loc {i}, {j}")
        self.assertEqual(len(persisted_model.progress), len(model.progress))
        self.assertEqual(len(persisted_model.training_data_buffer), 0)

    def test_train_with_insufficient_data(self):
        model = NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9], activation_algos=["relu"] * 2)

        # Test that training does not proceed when data is less than the buffer size
        input_size = 9
        output_size = 9

        sample_input = vector([0.5] * input_size)  # Example input as a list of numbers
        sample_target = vector([1.0] * output_size)  # Example target as a list of numbers

        # Add insufficient data
        training_data = [(sample_input, sample_target)] * (model.training_buffer_size - 1)

        model.train(training_data=training_data, epochs=1)

        # Ensure no training progress and buffering
        self.assertEqual(len(model.progress), 0)
        self.assertGreaterEqual(len(model.training_data_buffer), len(training_data))

        # Deserialize and check if recorded training buffer
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        self.assertEqual(len(persisted_model.training_data_buffer), len(model.training_data_buffer))

    def test_invalid_activation_algo(self):
        model = NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9], activation_algos=["unknown_algo"] * 2)

        input_size = 9
        output_size = 9

        sample_input = vector([0.5] * input_size)
        sample_target = vector([1.0] * output_size)

        # Add enough data to meet the training buffer size
        training_data = [(sample_input, sample_target)] * model.training_buffer_size

        # Test that setting an invalid activation algorithm raises a ValueError
        with self.assertRaises(ValueError) as context:
            model.train(training_data=training_data, epochs=1)

        # Assert the error message
        self.assertEqual(str(context.exception), "Unsupported activation algorithm: unknown_algo")

    def test_invalid_model_deserialization(self):
        # Test that deserializing a nonexistent model raises a KeyError
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("nonexistent_model")

    def test_delete(self):
        NeuralNetworkModel.delete("test")
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("test")

    def test_invalid_delete(self):
        # No error raised for failing to delete
        NeuralNetworkModel.delete("nonexistent")

if __name__ == '__main__':
    unittest.main()
