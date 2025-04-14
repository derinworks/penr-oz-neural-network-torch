import unittest
from parameterized import parameterized
import numpy as np
import torch
from neural_net_model import (Layer,
                              LinearLayer, EmbeddingLayer, BatchNormLayer,
                              SigmoidLayer, SoftmaxLayer, ReluLayer, TanhLayer,
                              NeuralNetworkModel, MultiLayerPerceptron)

class TestNeuralNetModel(unittest.TestCase):

    @parameterized.expand([
        (True, True, [0.0] * 9,),
        (True, False, [0.1] * 3,),
        (False, True, [0.2] * 18,),
        (False, False, [1.0],),
    ])
    def test_layer_initialization(self, hidden: bool, training: bool, input_vector: list[float]):
        layer = Layer()
        layer.hidden = hidden
        layer.training = training
        output = layer.forward(torch.tensor(input_vector).double())

        self.assertEqual(hidden, layer.hidden)
        self.assertEqual(training, layer.training)
        np.testing.assert_array_almost_equal(input_vector, output.tolist(), 1)

    @parameterized.expand([
        (9, 9, "random"),
        (18, 9, "zeros"),
        (9, 3, "zeros"),
        (4, 4, "random"),
        (3, 9, "random"),
    ])
    def test_linear_layer_initialization(self, input_size: int, output_size: int, bias_algo: str):
        layer = LinearLayer(input_size, output_size, bias_algo)
        output = layer.forward(torch.tensor([0.0] * input_size).double())

        self.assertEqual((input_size, output_size), tuple(layer.weights.shape))
        self.assertEqual((output_size,), tuple(layer.bias.shape))
        self.assertEqual((output_size,), tuple(output.shape))

    def test_embedding_layer(self):
        layer = EmbeddingLayer(27, 2)
        single_embedded = layer.forward(torch.tensor([0.0, 5.0, 13.0]))
        batch_embedded = layer.forward(torch.tensor([[0, 5, 13]] * 10))

        self.assertEqual((27, 2), tuple(layer.weights.shape))
        self.assertEqual((3, 2), tuple(single_embedded.shape))
        self.assertEqual((10, 3, 2), tuple(batch_embedded.shape))

    @parameterized.expand([
        (False,),
        (True,),
    ])
    def test_batchnorm_layer(self, training: bool):
        layer = BatchNormLayer(9)
        layer.training = training
        normalized = layer.forward(torch.tensor([0.2] * 9))

        self.assertEqual((9,), tuple(layer.gain.shape))
        self.assertEqual((9,), tuple(layer.bias.shape))
        self.assertEqual((9,), tuple(layer.mean.shape))
        self.assertEqual((9,), tuple(layer.variance.shape))
        self.assertEqual((9,), tuple(normalized.shape))

    @parameterized.expand([
        ([9, 9, 9], "xavier", "random", None,
         [LinearLayer, ReluLayer, LinearLayer, ReluLayer],
         [[(9,9),(9,)],[],[(9,9),(9,)],[]],),
        ([18, 9, 3], "xavier", "zeros", ["relu", "softmax"],
         [LinearLayer, ReluLayer, LinearLayer, SoftmaxLayer],
         [[(18,9),(9,)],[],[(9,3),(3,)],[]],),
        ([9, 18, 9], "he", "zeros", ["sigmoid"] * 2,
         [LinearLayer, SigmoidLayer, LinearLayer, SigmoidLayer],
         [[(9,18),(18,)],[],[(18,9),(9,)],[]]),
        ([4, 8, 16], "he", "random", ["tanh"] * 2,
         [LinearLayer, TanhLayer, LinearLayer, TanhLayer],
         [[(4,8),(8,)],[],[(8,16),(16,)],[]]),
        ([3, 3, 3, 3], "he", "random", ["relu", "linear", "tanh", "softmax"],
         [LinearLayer, ReluLayer, LinearLayer, TanhLayer, LinearLayer, SoftmaxLayer],
         [[(3,3),(3,)],[],[(3,3),(3,)],[],[(3,3),(3,)],[]],),
        ([18, 2, 6, 20, 18], "gaussian", "random", ["embedding", "tanh", "linear", "softmax"],
         [EmbeddingLayer, LinearLayer, TanhLayer, LinearLayer, SoftmaxLayer],
         [[(18,2)],[(6,20),(20,)],[],[(20,18),(18,)],[]],),
        ([18, 2, 6, 20, 18], "he", "zeros", ["embedding", "linear", "batchnorm", "tanh", "softmax"],
         [EmbeddingLayer, LinearLayer, BatchNormLayer, TanhLayer, LinearLayer, SoftmaxLayer],
         [[(18, 2)], [(6, 20), (20,)], [(20,), (20,)], [], [(20, 18), (18,)], []],),
    ])
    def test_multi_layer_perceptron_init(self, l_sizes: list[int], w_algo: str, b_algo: str, fwd_algos: list[str],
                                         expected_layers: list[str],
                                         expected_layer_shapes: list[list[tuple]]):
        mlp = MultiLayerPerceptron(l_sizes, w_algo, b_algo, fwd_algos)

        self.assertListEqual(expected_layers, [l.__class__ for l in mlp.layers])
        self.assertEqual(len(expected_layer_shapes), len(mlp.layers))
        for expected_layer_shape, layer in zip(expected_layer_shapes, mlp.layers):
            self.assertListEqual(expected_layer_shape, [tuple(p.shape) for p in layer.params])
        self.assertFalse(mlp.layers[0].hidden)
        for hidden_layer in mlp.layers[1:-2]:
            self.assertTrue(hidden_layer.hidden)
        self.assertFalse(mlp.layers[-1].hidden)

    @parameterized.expand([
        ([3, 3], None, None, 12,),
        ([9, 9, 9], None, "adam", 180,),
        ([18, 9, 3], ["sigmoid"] * 2, None, 201,),
        ([10, 3, 6, 20, 10], ["embedding", "tanh", "softmax"], "adam", 380,),
        ([10, 3, 6, 20, 10], ["embedding", "linear", "batchnorm", "tanh", "softmax"], None, 420,),
    ])
    def test_model_initialization(self, layer_sizes: list[int], algos: list[str], optimizer: str,
                                  expected_buffer_size: int):
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos)

        self.assertEqual("test", model.model_id)
        self.assertEqual(0, len(model.progress))
        self.assertTrue(optimizer is None or model.optimizer is not None)
        self.assertEqual(expected_buffer_size, model.training_buffer_size)
        self.assertEqual(expected_buffer_size, model.num_params)

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2, [0.5] * 9, None,),
        ([9, 9], ["softmax"], [1] + [0] * 8, [4]),
        ([18, 9, 3], ["relu", "softmax"], [1] + [0] * 17, None,),
        ([9, 18, 9], ["tanh"] * 2, [0.5] * 9, [0.5] * 9,),
        ([4, 8, 16], ["tanh", "softmax"], [0.5] * 4, [13],),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"], [0.5] * 3, None,),
        ([9, 2, 6, 18, 9], ["embedding", "tanh", "softmax"], [0, 5, 8], [2],),
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
        ([9, 9, 9], ["sigmoid"] * 2, None, [0.5] * 9, [1.0] + [0.0] * 8,),
        ([9, 9, 9], ["relu", "softmax"], "adam", [0.5] * 9, [1]),
        ([9, 9, 9], ["tanh"] * 2, None, [0.5] * 9, [1.0] + [0.0] * 8,),
        ([18, 9, 3], ["relu", "sigmoid"], "adam", [0.5] * 18, [1.0] + [0.0] * 2,),
        ([9, 18, 9], ["sigmoid", "linear", "softmax"], None, [0.5] * 9, [1]),
        ([4, 8, 16], ["sigmoid"] * 2, None, [0.5] * 4, [1.0] + [0.0] * 15,),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"], "adam", [0.5] * 3, [0]),
        ([18, 9, 3], ["relu"] * 2, None, [0.5] * 18, [1.0] + [0.0] * 2,),
        ([9, 18, 9], ["relu", "tanh"], "adam", [0.5] * 9, [1.0] + [0.0] * 8,),
        ([9, 2, 6, 18, 9], ["embedding", "tanh", "softmax"], "adam", [0, 5, 7], [5],),
        ([9, 2, 6, 18, 9], ["embedding", "linear", "batchnorm", "tanh", "softmax"], None, [0, 5, 7], [5],),
    ])
    def test_train(self, layer_sizes: list[int], algos: list[str], optimizer: str, sample_input: list[float],
                   target: list[float]):

        # clean up any persisted previous test model
        NeuralNetworkModel.delete("test")

        # create model
        model = NeuralNetworkModel("test", layer_sizes, activation_algos=algos, optimizer_algo=optimizer)

        initial_params = [p.tolist() for p in model.params]
        _, initial_cost = model.compute_output(sample_input, target)
        # Add enough data to meet the training buffer size
        training_data = [(sample_input, target)] * model.training_buffer_size

        model.train(training_data, epochs=1, dropout_rate=0.001)

        updated_params = [p.tolist() for p in model.params]

        # Check that the model data is still valid
        for a, e in zip(updated_params, initial_params):
            np.testing.assert_array_almost_equal(a, e, 1)

        # Ensure training progress
        self.assertGreater(len(model.progress), 0)
        self.assertNotEqual(model.progress[0]["cost"], initial_cost)
        self.assertEqual(sum([p["cost"] for p in model.progress]) / len(model.progress), model.avg_cost)
        self.assertEqual(len(model.training_data_buffer), 0)

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        # Verify model parameters correctly deserialized
        persisted_params = [p.tolist() for p in persisted_model.params]

        for a, e in zip(persisted_params, updated_params):
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
