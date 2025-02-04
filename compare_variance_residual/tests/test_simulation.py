import random
import unittest

import numpy as np

from compare_variance_residual.simulation import create_orthogonal_feature_spaces


class TestSimulationFunctions(unittest.TestCase):
    feature_spaces = None
    num_samples = None
    d_list = None
    atol = 1e-10

    def setUp(self):
        # Test the function
        self.d_list = [random.randint(1, 100) for _ in range(3)]
        self.num_samples = random.randint(sum(self.d_list) + 1, sum(self.d_list) + 100)
        self.feature_spaces = create_orthogonal_feature_spaces(self.num_samples, self.d_list)

    def test_feature_means(self):
        # make sure means are close to 0
        for space in self.feature_spaces:
            assert np.allclose(space.mean(axis=0), 0, atol=self.atol)

    def test_orthogonality_between_feature_spaces(self):
        # Verify orthogonality between feature spaces
        for i in range(len(self.feature_spaces) - 1):
            for j in range(i + 1, len(self.feature_spaces)):
                product = self.feature_spaces[i].T @ self.feature_spaces[j]
                assert np.allclose(product, np.zeros_like(self.feature_spaces[i].T @ self.feature_spaces[j]), atol=self.atol)


if __name__ == '__main__':
    unittest.main()
