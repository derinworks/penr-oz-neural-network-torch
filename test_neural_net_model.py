import unittest
from parameterized import parameterized
import numpy as np
import torch
from neural_net_model import NeuralNetworkModel, MultiLayerPerceptron, Layer, EmbeddingLayer


class TestNeuralNetModel(unittest.TestCase):

    @parameterized.expand([
        (9, 9, "xavier", "random"),
        (18, 9, "xavier", "zeros"),
        (9, 3, "he", "zeros"),
        (4, 4, "he", "random"),
        (3, 9, "gaussian", "random"),
    ])
    def test_layer_initialization(self, input_size: int, output_size: int, weight_algo: str, bias_algo: str):
        layer = Layer(input_size, output_size, weight_algo, bias_algo)

        self.assertEqual((input_size, output_size), tuple(layer.weights.shape))
        self.assertEqual((output_size,), tuple(layer.biases.shape))

    @parameterized.expand([
        ("random",),
        ("zeros",),
    ])
    def test_embedding_layer(self, bias_algo: str):
        layer = EmbeddingLayer(27, 2, "gaussian", bias_algo)
        _, single_embedded = layer.forward(torch.tensor([0.0, 5.0, 13.0]))
        _, batch_embedded = layer.forward(torch.tensor([[0, 5, 13]] * 10))

        self.assertEqual((27, 2), tuple(layer.weights.shape))
        self.assertListEqual([0] * 2, layer.biases.tolist())
        self.assertEqual((3, 2), tuple(single_embedded.shape))
        self.assertEqual((10, 3, 2), tuple(batch_embedded.shape))

    @parameterized.expand([
        ([9, 9, 9], "xavier", "random", None, [(9, 9)] * 2,),
        ([18, 9, 3], "xavier", "zeros", ["relu", "softmax"], [(18, 9), (9, 3)],),
        ([9, 18, 9], "he", "zeros", ["sigmoid"] * 2, [(9, 18), (18, 9)],),
        ([4, 8, 16], "he", "random", ["tanh"] * 2, [(4, 8), (8, 16)]),
        ([3, 3, 3, 3], "gaussian", "random", ["relu", "tanh", "softmax"], [(3, 3)] * 3,),
        ([18, 2, 6, 20, 18], "gaussian", "random", ["embedding", "tanh", "softmax"], [(18, 2), (6, 20), (20, 18)],)
    ])
    def test_multi_layer_perceptron_init(self, l_sizes: list[int], w_algo: str, b_algo: str, fwd_algos: list[str],
                                         expected_layer_shapes: list[tuple]):
        multi_layer_perceptron = MultiLayerPerceptron(l_sizes, w_algo, b_algo, fwd_algos)

        self.assertEqual(len(multi_layer_perceptron.layers), len(expected_layer_shapes))
        for (input_size, output_size), layer in zip(expected_layer_shapes, multi_layer_perceptron.layers):
            self.assertEqual((input_size, output_size), tuple(layer.weights.shape))
            self.assertEqual((output_size,), tuple(layer.biases.shape))
        self.assertFalse(multi_layer_perceptron.layers[0].hidden)
        for hidden_layer in multi_layer_perceptron.layers[1:-2]:
            self.assertTrue(hidden_layer.hidden)
        self.assertFalse(multi_layer_perceptron.layers[-1].hidden)

    @parameterized.expand([
        ([3, 3], None, 12,),
        ([9, 9, 9], None, 180,),
        ([18, 9, 3], ["sigmoid"] * 2, 201,),
        ([10, 3, 6, 20, 10], ["embedding", "tanh", "softmax"], 383,),
    ])
    def test_model_initialization(self, layer_sizes: list[int], algos: list[str], expected_buffer_size: int):
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos)

        self.assertEqual("test", model.model_id)
        self.assertEqual(0, len(model.progress))
        self.assertEqual(expected_buffer_size, model.training_buffer_size)
        self.assertEqual(expected_buffer_size, model.num_params)

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2, [0.5] * 9, None,),
        ([9, 9], ["softmax"], [1] + [0] * 8, [4]),
        ([18, 9, 3], ["relu", "softmax"], [1] + [0] * 17, None,),
        ([9, 18, 9], ["tanh"] * 2, [0.5] * 9, [0.5] * 9,),
        ([4, 8, 16], ["tanh", "softmax"], [0.5] * 4, [13],),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"], [0.5] * 3, None,),
        ([9, 2, 6, 18, 9], ["embedding", "tanh", "softmax"], [0, 5, 8], [2],)
    ])
    def test_compute_output(self, layer_sizes: list[int], algos: list[str], sample_input: list[float], target):
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos)
        output_sizes = layer_sizes[1:]

        output, cost = model.compute_output(sample_input, target)

        self.assertEqual(output_sizes[-1], len(output))
        if target is None:
            self.assertIsNone(cost)
        else:
            self.assertIsNotNone(cost)

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2, [0.5] * 9, [1.0] + [0.0] * 8,),
        ([9, 9, 9], ["relu", "softmax"], [0.5] * 9, [1]),
        ([9, 9, 9], ["tanh"] * 2, [0.5] * 9, [1.0] + [0.0] * 8,),
        ([18, 9, 3], ["relu", "sigmoid"], [0.5] * 18, [1.0] + [0.0] * 2,),
        ([9, 18, 9], ["sigmoid", "softmax"], [0.5] * 9, [1]),
        ([4, 8, 16], ["sigmoid"] * 2, [0.5] * 4, [1.0] + [0.0] * 15,),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"], [0.5] * 3, [0]),
        ([18, 9, 3], ["relu"] * 2, [0.5] * 18, [1.0] + [0.0] * 2,),
        ([9, 18, 9], ["relu", "tanh"], [0.5] * 9, [1.0] + [0.0] * 8,),
        ([9, 2, 6, 18, 9], ["embedding", "tanh", "softmax"], [0, 5, 7], [5],),
    ])
    def test_train(self, layer_sizes: list[int], algos: list[str], sample_input: list[float], target: list[float]):
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos)

        initial_weights = [l.weights.tolist() for l in model.layers]
        initial_biases = [l.biases.tolist() for l in model.layers]
        _, initial_cost = model.compute_output(sample_input, target)
        # Add enough data to meet the training buffer size
        training_data = [(sample_input, target)] * model.training_buffer_size

        model.train(training_data, epochs=1, dropout_rate=0.001)

        updated_weights = [l.weights.tolist() for l in model.layers]
        updated_biases = [l.biases.tolist() for l in model.layers]

        # Check that the model data is still valid
        for a, e in zip(updated_weights, initial_weights):
            np.testing.assert_array_almost_equal(a, e, 1)
        for a, e in zip(updated_biases, initial_biases):
            np.testing.assert_array_almost_equal(a, e, 1)

        # Ensure training progress
        self.assertGreater(len(model.progress), 0)
        self.assertNotEqual(model.progress[0]["cost"], initial_cost)
        self.assertEqual(sum([p["cost"] for p in model.progress]) / len(model.progress), model.avg_cost)
        self.assertEqual(len(model.training_data_buffer), 0)

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        # Verify model parameters correctly deserialized
        persisted_weights = [l.weights.tolist() for l in persisted_model.layers]
        persisted_biases = [l.biases.tolist() for l in persisted_model.layers]

        for a, e in zip(persisted_weights, updated_weights):
            np.testing.assert_array_almost_equal(a, e, 8)
        for a, e in zip(persisted_biases, updated_biases):
            np.testing.assert_array_almost_equal(a, e, 8)
        self.assertEqual(len(persisted_model.progress), len(model.progress))
        self.assertEqual(len(persisted_model.training_data_buffer), 0)
        self.assertEqual(persisted_model.avg_cost, model.avg_cost)

    def test_train_with_insufficient_data(self):
        model = NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9], activation_algos=["relu"] * 2)

        # Test that training does not proceed when data is less than the buffer size
        input_size = 9
        output_size = 9

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [1.0] * output_size  # Example target as a list of numbers

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
        with self.assertRaises(ValueError) as context:
            NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9], activation_algos=["relu", "unknown_algo"])

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
