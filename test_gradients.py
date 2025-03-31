import unittest
from parameterized import parameterized
import numpy as np
from torch import Tensor
from gradients import activate, vector, matrix, calculate_cost

class TestGradients(unittest.TestCase):

    @parameterized.expand([
        ([0],),
        ([0.0],),
        ([1, 2.0],),
        ([1.0, 2.0, -1],),
    ])
    def test_init_vector(self, values):
        v = vector(values)

        self.assertListEqual([values], v.tolist())
        self.assertTrue(v.requires_grad)

    @parameterized.expand([
        ([[0]],),
        ([[0.0], [1.1]],),
        ([[1, 2.0], [-2.0, 0.0]],),
        ([[1.0, 2.0, -1], [-1.0, 3.0, 0], [-3.0, 0.0, 1.5]],),
    ])
    def test_init_matrix(self, values):
        m = matrix(values)

        self.assertListEqual(values, m.tolist())
        self.assertTrue(m.requires_grad)

    @parameterized.expand([
        (1.0, "sigmoid", 0.7311),
        (-2, "relu", 0.0),
        (3.0, "tanh", 0.9951),
        (1.1, "softmax", 1.0),
    ])
    def test_activation(self, value, algo, expected):
        actual = activate(vector([value]), algo)

        self.assertAlmostEqual(expected, actual.squeeze(0).item(), 4)

    @parameterized.expand([
        ([2.0, 1.0], "softmax", [0.7311, 0.2689]),
        ([2.0, 1.0, -1.0], "softmax", [0.7054, 0.2595, 0.0351]),
    ])
    def test_vector_activation(self, values, algo, expected):
        actual = activate(vector(values), algo)

        np.testing.assert_array_almost_equal(actual.tolist(), [expected], 4)

    def test_activation_unsupported(self):
        with self.assertRaises(ValueError) as ve:
            activate(vector([1.0]), "b0gU2")

        self.assertEqual("Unsupported activation algorithm: b0gU2", str(ve.exception))

    @parameterized.expand([
        ("sigmoid", [1.0, 1.0, 1.0], [0.7311, 0.7311, 0.7311], [0.0, 0.5, 1.0], 0.2201),
        ("softmax", [2.0, 1.0, -1.0], [0.2, 0.3, 0.5], [1.0], 1.349),
    ])
    def test_vector_calculate_cost(self, algo, logits, activation, target, expected):
        actual = calculate_cost(algo, vector(logits), vector(activation), [target])

        self.assertAlmostEqual(expected, actual.item(), 4)

    @parameterized.expand([
        ((vector([1.0]),), lambda a: a, [1.0]),
        ((vector([1.0]), vector([2.0]),), lambda a, b: a + b, [1.0, 1.0]),
        ((vector([-2.0]), vector([4.0])), lambda a, b: a * b, [-2.0, 4.0]),
        ((vector([-2.0]), vector([-6.0]), vector([10.0])), lambda a, b, c: a * (b + c), [-2.0, -2.0, 4.0]),
        ((vector([2]), vector([1])), lambda a, b: a**3 + b, [1.0, 12.0]),
        ((vector([1.0]),), lambda a: activate(a, "sigmoid"), [0.1966]),
        ((vector([-2]),), lambda a: activate(3 * a + 1,"relu"), [0.0]),
        ((vector([2]),), lambda a: activate(a + 1, "relu"), [1.0]),
        ((vector([1.0]),), lambda a: activate(a, "tanh"), [0.42]),
    ])
    def test_back_propagate(self, tensors: tuple[Tensor], f, expected):
        result = f(*tensors)
        result.backward()
        gradients = [g for t in reversed(tensors) for g in t.grad.squeeze(0).tolist()]

        np.testing.assert_array_almost_equal(gradients, expected, 4)

if __name__ == "__main__":
    unittest.main()
