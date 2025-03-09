"""
dataset.py: Synthetic data generation and experiment execution.

This module provides functionality to generate synthetic datasets
with customizable feature spaces and target variables, and perform
machine learning experiments to study the impact of various data
parameters. Includes:

- Dataset generation with structured feature spaces (`generate_dataset`)
- Experimentation with variance partitioning and residual methods
  (`run_experiment`)
- Utility functions for feature space stacking and orthogonalization
"""

import numpy as np
from scipy.stats import zscore


def generate_dataset(d_list=None, scalars=None, n_targets=10000, n_samples=10000, noise_target=0.1, noise_features=0.01,
                     construction_method="orthogonal", random_state=42):
    """
    Generate synthetic datasets with customizable feature spaces for training and testing machine learning models.

    This function creates synthetic datasets with specified configurations of feature spaces, target variables,
    and sample sizes. The function allows for flexible dataset construction using various methods
    such as stacking feature spaces or applying Singular Value Decomposition (SVD) based strategies. The user
    can control noise levels, distribution of random target values, and scalars of feature spaces. The resulting
    datasets are returned as structured arrays for training and testing purposes.

    Parameters:
        d_list: list of int, optional
            A list specifying the dimensions of the feature spaces. Defaults to [100, 100, 100].
        scalars: list of float, optional
            A list indicating the relative scalars for each feature space. Must sum to 1. Defaults to [1/3, 1/3, 1/3].
        n_targets: int, optional
            The number of target variables in the dataset.
        n_samples: int, optional
            The number of samples (rows) in the dataset.
        noise_target: float, optional
            The standard deviation of the Gaussian noise to be added to the targets. Defaults to 0.1.
        noise_features: float, optional
            The standard deviation of the Gaussian noise to be added to the features. Defaults to 0.01.
        construction_method: str, optional
            The method to use for constructing the feature spaces. Can be either "orthogonal" or "random". Defaults to "orthogonal".
        random_state: int, optional
            Seed for the random number generator to ensure reproducibility. Defaults to 42.

    Returns:
        tuple:
            A tuple containing:
            - Xs: list of ndarray, Training feature spaces as arrays.
            - Xs_test: list of ndarray, Testing feature spaces as arrays.
            - Y_train: ndarray, Target variables for the training set.
            - Y_test: ndarray, Target variables for the testing set.

    Raises:
        ValueError:
            If an unknown construction_method is provided.
    """

    if random_state is not None:
        np.random.seed(random_state)

    if d_list is None:
        d_list = [100] * 3

    if scalars is None:
        scalars = [1 / 3] * 3

    if construction_method == "random":
        # generate feature spaces
        feature_spaces = [zscore(np.random.randn(n_samples, dim), axis=0) for dim in d_list]

        # concatenate the first feature with all other feature spaces
        # [0, 1], [0, 2], [0, 3], ...
        Xs = [np.hstack([feature_spaces[0], feature_space]) for feature_space in feature_spaces[1:]]

        # generate weights
        betas = [np.random.randn(d, n_targets) for d in d_list]
    elif construction_method == "orthogonal":
        feature_spaces = create_orthogonal_feature_spaces(n_samples, d_list)

        # add the first feature with all other feature spaces
        # [0 + 1, 0 + 2, 0 + 3, ...]
        Xs = [np.hstack([feature_spaces[0], feature_space]) for feature_space in feature_spaces[1:]]

        # generate weights
        betas = [np.random.randn(sum(d_list), n_targets) for _ in d_list]
    else:
        raise ValueError(f"Unknown construction_method {construction_method}.")

    Xs = [zscore(x) for x in Xs]
    for i in range(len(Xs)):
        noise = zscore(np.random.randn(n_samples, Xs[i].shape[1]), axis=0)
        Xs[i] = ((1 - noise_features) ** 0.5) * Xs[i] + (noise_features ** 0.5) * noise

    betas = [zscore(beta) for beta in betas]

    # generate target
    Y = sum([(alpha ** 0.5) * zscore(np.dot(feature_space, beta)) for alpha, feature_space, beta in
             zip(scalars, feature_spaces, betas)])
    Y = zscore(Y)

    # add noise
    noise = zscore(np.random.randn(n_samples, n_targets))
    Y = ((1 - noise_target) ** 0.5) * Y + noise * (noise_target ** 0.5)
    Y = zscore(Y)

    from himalaya.backend import get_backend

    backend = get_backend()

    Xs = [backend.asarray(X, dtype="float32") for X in Xs]
    Y = backend.asarray(Y, dtype="float32")

    return Xs, Y


def create_orthogonal_feature_spaces(num_samples, d_list):
    """
        Creates a set of orthogonal feature spaces from a random matrix using Singular Value
        Decomposition (SVD). The function ensures the output feature spaces are mutually
        orthogonal and have dimensions specified by the input rank list.

        The input matrix is generated using a specified random distribution, normalized
        using z-score normalization, and processed using SVD. The singular values
        are optionally shuffled to allow for randomness in how the orthogonal spaces
        are constructed.

        Parameters:
            num_samples: int
                The number of samples (rows) in the generated feature spaces. Must be
                greater than the sum of ranks specified in the rank list.
            d_list: list[int]
                A list defining the ranks (columns) for each orthogonal feature space.
                The sum of all ranks should not exceed the number of samples.

        Returns:
            list[np.ndarray]
                A list of orthogonal feature spaces, each being a numpy array of
                shape (num_samples, rank) corresponding to the ranks defined in
                the input `d_list`.

        Raises:
            AssertionError
                If `num_samples` is not greater than the sum of ranks in `d_list`.
    """
    assert num_samples > sum(d_list), "Number of samples must be greater than the sum of ranks."

    feature_spaces = []

    # Generate a random matrix of shape (dim, rank)
    M = np.random.randn(num_samples, sum(d_list))
    M = zscore(M)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # normalize S
    S = np.ones_like(S)

    start = 0
    for rank in d_list:
        _S = np.zeros(len(S))
        _S[start:start + rank] = S[start:start + rank]

        # create rectangular diagonal sigma matrix
        diag_S = np.diag(_S)

        feature_space = U @ diag_S @ Vt
        feature_spaces.append(feature_space)
        start += rank
    return feature_spaces


if __name__ == "__main__":
    from himalaya.ridge import GroupRidgeCV
    from himalaya.backend import get_backend, set_backend

    set_backend("cupy")
    backend = get_backend()
    d_list = [100, 100, 100]
    scalars = [0.6, 0.3, 0.1]
    n_targets = 1000
    n_samples_train = 10000
    n_samples_test = 1000
    noise_scalar = 0.1
    random_distribution = "normal"

    (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(d_list, scalars, n_targets, n_samples_train,
                                                            n_samples_test, noise_scalar, random_distribution,
                                                            42)

    # import matplotlib.pyplot as plt
    # plt.plot(Xs_train[:][0], Y_train[:][0])
    # plt.show()

    Xs_train, Xs_test, Y_train, Y_test = [backend.to_numpy(X) for X in Xs_train], [backend.to_numpy(X) for X in
                                                                                   Xs_test], backend.to_numpy(
        Y_train), backend.to_numpy(Y_test)

    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(Xs_train[0], Y_train)
    Y_pred = model.predict(Xs_test[0])
    print(fr"R^2 score for $X_0$: {r2_score(Y_test, Y_pred) ** 2} vs expected {scalars[0] + scalars[1]}")

    model = LinearRegression()
    model.fit(Xs_train[1], Y_train)
    Y_pred = model.predict(Xs_test[1])
    print(fr"R^2 score for $X_1$: {r2_score(Y_test, Y_pred)} vs expected {scalars[0] + scalars[2]}")

    X_train_stack = np.hstack(Xs_train)
    X_test_stack = np.hstack(Xs_test)
    model = LinearRegression()
    model.fit(X_train_stack, Y_train)
    Y_pred = model.predict(X_test_stack)
    print(f"R^2 score on stacked features: {r2_score(Y_test, Y_pred)}")

    model = GroupRidgeCV(groups="input", solver_params=dict(progress_bar=False))
    model.fit(Xs_train, Y_train)
    Y_pred = model.predict(Xs_test)
    print(f"R^2 score for banded: {r2_score(Y_test, Y_pred)}")

    # check if Xs are random
    assert not np.allclose(Xs_train[0], Xs_train[1], atol=1e-5)

    # check if Xs are demeaned
    for X in Xs_train:
        assert np.allclose(X.mean(0), 0, atol=1e-5)
    for X in Xs_test:
        assert np.allclose(X.mean(0), 0, atol=1e-5)

    # check if Ys are demeaned
    assert np.allclose(Y_train.mean(0), 0, atol=noise_scalar)
    assert np.allclose(Y_test.mean(0), 0, atol=noise_scalar)
