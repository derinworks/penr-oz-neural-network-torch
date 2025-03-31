import unittest
from parameterized import parameterized
import torch
import functions as func

class TestFunctions(unittest.TestCase):

    @parameterized.expand([
        (-1.5, 0.0),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([-1.5, 0.2, 2.0], [-1.2129, -0.0233, 1.2362]),
        ([[-1.5, 0.2, 2.0]], [[-1.2129, -0.0233, 1.2362]]),
        ([[1.0, 0.5], [-1.1, 2.2]], [[0.2955, -0.1267], [-1.4777, 1.3088]]),
    ])
    def test_batch_norm(self, data, expected):
        actual = func.batch_norm(torch.tensor(data))

        torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=1e-4)

if __name__ == "__main__":
    unittest.main()
