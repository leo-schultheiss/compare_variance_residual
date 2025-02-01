import random
import unittest

import numpy as np

from compare_variance_residual.simulated.simulation import create_orthogonal_feature_spaces


class TestSimulationFunctions(unittest.TestCase):
    feature_spaces = None

    def tearDown(self):
        self.feature_spaces = None

    def setUp(self):
        # Test the function
        num_samples = random.randint(1, 100)
        d_list = [random.randint(1, 100) for _ in range(3)]
        self.feature_spaces = create_orthogonal_feature_spaces(num_samples, d_list)

    def test_orthogonality_within_feature_space(self):
        for i, space in enumerate(self.feature_spaces):
            # Verify orthogonality within each space
            assert np.allclose(space.T @ space, np.eye(space.shape[1]))

    def test_orthogonality_between_feature_spaces(self):
        # Verify orthogonality between feature spaces
        for i in range(len(self.feature_spaces) - 1):
            for j in range(i + 1, len(self.feature_spaces)):
                product = self.feature_spaces[i].T @ self.feature_spaces[j]
                assert np.allclose(product, np.zeros_like(self.feature_spaces[i].T @ self.feature_spaces[j]), atol=1e-2)

    def test_feature_means(self):
        # make sure means are close to 0
        for space in self.feature_spaces:
            assert np.allclose(space.mean(axis=0), 0, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
