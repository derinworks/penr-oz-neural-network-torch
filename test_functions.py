from torch import Tensor
import unittest
from parameterized import parameterized
import functions as func

class TestFunctions(unittest.TestCase):

    @parameterized.expand([
        (func.cross_entropy_loss(Tensor([0.2, 0.3, 0.5]), Tensor([0.3, 0.4, 0.4])), 1.2417),
    ])
    def test_func_scalar(self, actual: Tensor, expected: float):
        self.assertAlmostEqual(expected, actual.item(), 4)

    @parameterized.expand([
        (func.batch_norm(Tensor([-1.5, 0.2, 2.0]).double()), [-1.2129, -0.0233, 1.2362]),
        (func.batch_norm(Tensor([0.0, 0.0, 0.0]).double()), [0.0, 0.0, 0.0]),
    ])
    def test_func_list(self, actual: Tensor, expected: list[float]):
        self.assertEqual(len(expected), len(actual.tolist()))
        for i, (e, a) in enumerate(zip(expected, actual.tolist())):
            self.assertAlmostEqual(e, a, 4, f"Element at index {i}")

if __name__ == "__main__":
    unittest.main()
