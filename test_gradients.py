import unittest
from parameterized import parameterized
from torch import Tensor
from gradients import activate, vector, scalar, Activation

class TestGradients(unittest.TestCase):

    @parameterized.expand([
        ([0],),
        ([0.0],),
        ([1, 2.0],),
        ([1.0, 2.0, -1],),
    ])
    def test_init_vector(self, values):
        v = vector(values)

        self.assertEqual(len(values), len(v.tolist()))

    @parameterized.expand([
        (1.0, "sigmoid", 0.7311),
        (-2, "relu", 0.0),
        (3.0, "tanh", 0.9951),
        (1.1, "softmax", 1.0),
    ])
    def test_activation(self, value, algo, expected):
        actual = activate(Tensor([value]).double(), algo)

        self.assertAlmostEqual(expected, actual.item(), 4)

    @parameterized.expand([
        ([2.0, 1.0], "softmax", [0.7311, 0.2689]),
        ([2.0, 1.0, -1.0], "softmax", [0.7054, 0.2595, 0.0351]),
    ])
    def test_vector_activation(self, values, algo, expected):
        actual = Activation(vector(values)).activate(algo)

        self.assertEqual(len(expected), len(actual.tensor.tolist()))
        for i, (e, a) in enumerate(zip(expected, actual.tensor.tolist())):
            self.assertAlmostEqual(e, a, 4, f"Elements at index {i}")

    def test_activation_unsupported(self):
        with self.assertRaises(ValueError) as ve:
            activate(Tensor([1.0]).double(), "b0gU2")

        self.assertEqual("Unsupported activation algorithm: b0gU2", str(ve.exception))

    @parameterized.expand([
        (Activation(vector([-1.5, 0.2, 2.0])).batch_norm(), [-1.2129, -0.0233, 1.2362]),
        (Activation(vector([-1.5, 0.2, 2.0])).apply_dropout(0.99999), [0.0, 0.0, 0.0]),
    ])
    def test_vector_apply_func(self, actual: Activation, expected: list[float]):
        self.assertEqual(len(expected), len(actual.tensor.tolist()))
        for i, (e, a) in enumerate(zip(expected, actual.tensor.tolist())):
            self.assertAlmostEqual(e, a, 4, f"Element at index {i}")

    @parameterized.expand([
        (Activation(vector([-1.5, 0.2, 2.0])), [0.0, 0.5, 1.0], 1.1133),
        (Activation(vector([2.0, 1.0, -1.0])).activate("softmax"), [0.2, 0.3, 0.5], 2.149),
    ])
    def test_vector_calculate_cost(self, activation: Activation, target: list[float], expected: float):
        actual = activation.calculate_cost(vector(target))

        self.assertAlmostEqual(expected, actual.item(), 4)

    @parameterized.expand([
        ((scalar(1.0),), lambda a: a, [1.0]),
        ((scalar(1.0), scalar(2.0)), lambda a, b: a + b, [1.0, 1.0]),
        ((scalar(-2.0), scalar(4.0)), lambda a, b: a * b, [-2.0, 4.0]),
        ((scalar(-2.0), scalar(-6.0), scalar(10.0)), lambda a, b, c: a * (b + c), [-2.0, -2.0, 4.0]),
        ((scalar(2), scalar(1)), lambda a, b: a**3 + b, [1.0, 12.0]),
        ((scalar(1.0),), lambda a: activate(a, "sigmoid"), [0.1966]),
        ((scalar(-2),), lambda a: activate(3 * a + 1,"relu"), [0.0]),
        ((scalar(2),), lambda a: activate(a + 1, "relu"), [1.0]),
        ((scalar(1.0),), lambda a: activate(a, "tanh"), [0.42]),
        ((vector([-1.5, 0.2]),),
         lambda a: Activation(a).activate("sigmoid").calculate_cost(Tensor([0.2, 0.8]).double()),
         [-0.0619, -0.0026]),
        ((vector([-1.5, 0.2]),),
         lambda a: Activation(a).activate("softmax").calculate_cost(Tensor([0.2, 0.8]).double()),
         [ 0.0455, -0.0455]),
    ])
    def test_scalar_back_propagate(self, tensors: tuple[Tensor], f, expected: list[float]):
        result: Tensor = f(*tensors)
        result.backward()
        gradients = []
        for t in tensors:
            gradients += t.grad.tolist() if t.ndim > 0 else [t.grad.item()]

        self.assertEqual(len(expected), len(gradients))
        for i, (e, g) in enumerate(zip(expected, reversed(gradients))):
            self.assertAlmostEqual(e, g, 4, f"Element at index {i}")

if __name__ == "__main__":
    unittest.main()
