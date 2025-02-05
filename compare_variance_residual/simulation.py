"""
simulation.py: Synthetic data generation and experiment execution.

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
from himalaya.progress_bar import bar
from scipy.stats import zscore

from compare_variance_residual.residual import residual_method
from compare_variance_residual.variance_partitioning import variance_partitioning


def sample_random_distribution(shape, distribution) -> np.ndarray:
    """Create a random distribution.

    Parameters
    ----------
    shape : array of shape (n_samples,)
        x coordinates.
    distribution : str in {"normal", "uniform", "exponential", "gamma", "beta", "lognormal", "pareto"}
        Distribution to generate.

    Returns
    -------
    array of shape (n_samples, )
        Generated distribution.
    """
    if distribution == "normal":
        return np.random.randn(*shape)
    elif distribution == "uniform":
        return np.random.uniform(-1, 1, size=shape)
    elif distribution == "exponential":
        return np.random.exponential(size=shape)
    elif distribution == "gamma":
        return np.random.gamma(shape=1, size=shape)
    elif distribution == "beta":
        return np.random.beta(a=1, b=1, size=shape)
    elif distribution == "lognormal":
        return np.random.lognormal(size=shape)
    elif distribution == "pareto":
        return np.random.pareto(a=1, size=shape)
    else:
        raise ValueError(f"Unknown distribution {distribution}.")


def generate_dataset(d_list=None, scalars=None, n_targets=10000, n_samples_train=1000, n_samples_test=100,
                     noise_scalar=0.1, random_distribution="normal", construction_method="orthogonal",
                     split_train_and_test=True, random_state=42):
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
        n_samples_train: int, optional
            The number of samples in the training dataset.
        n_samples_test: int, optional
            The number of samples in the testing dataset.
        noise_scalar: float, optional
            The standard deviation of the Gaussian noise to be added to the targets.
        random_distribution: str, optional
            The type of random distribution to use for generating target values. Defaults to "normal".
        construction_method: str, optional
            The method to use for constructing the feature spaces. Can be either "random" or "svd". Defaults to "random".
        split_train_and_test: bool, optional
            If train and test set should be created out of the
        random_state: int, optional
            Seed for the random number generator to ensure reproducibility. Defaults to None.

    Returns:
        tuple:
            A tuple containing:
            - Xs_train: list of ndarray, Training feature spaces as arrays.
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
        if split_train_and_test:
            # generate feature spaces
            feature_spaces = [
                zscore(sample_random_distribution((n_samples_train + n_samples_test, dim), random_distribution)) for dim
                in
                d_list]

            # split into train and test set
            feature_spaces_train = [feature_space[:n_samples_train] for feature_space in feature_spaces]
            feature_spaces_test = [feature_space[n_samples_train:] for feature_space in feature_spaces]
        else:

            feature_spaces_train = [zscore(sample_random_distribution((n_samples_train, dim), random_distribution)) for
                                    dim in
                                    d_list]
            feature_spaces_test = [zscore(sample_random_distribution((n_samples_test, dim), random_distribution)) for
                                   dim in
                                   d_list]

        # concatenate the first feature with all other feature spaces
        # [0, 1], [0, 2], [0, 3], ...
        Xs_train = [np.hstack([feature_spaces_train[0], feature_space]) for feature_space in feature_spaces_train[1:]]
        Xs_test = [np.hstack([feature_spaces_test[0], feature_space]) for feature_space in feature_spaces_test[1:]]

        # generate weights
        betas = [sample_random_distribution([d, n_targets], "normal") for d in d_list]
    elif construction_method == "orthogonal":
        if split_train_and_test:
            feature_spaces = create_orthogonal_feature_spaces(n_samples_train + n_samples_test, d_list,
                                                              random_distribution)
            feature_spaces_train = [feature_space[:n_samples_train] for feature_space in feature_spaces]
            feature_spaces_test = [feature_space[n_samples_train:] for feature_space in feature_spaces]
        else:
            feature_spaces_train = create_orthogonal_feature_spaces(n_samples_train, d_list, random_distribution)
            feature_spaces_test = create_orthogonal_feature_spaces(n_samples_test, d_list, random_distribution)

        # add the first feature with all other feature spaces
        # [0 + 1, 0 + 2, 0 + 3, ...]
        Xs_train = [1 / 2 * (feature_spaces_train[0] + feature_space) for feature_space in feature_spaces_train[1:]]
        Xs_test = [1 / 2 * (feature_spaces_test[0] + feature_space) for feature_space in feature_spaces_test[1:]]

        # generate weights
        betas = [sample_random_distribution([sum(d_list), n_targets], "normal") for _ in d_list]
    else:
        raise ValueError(f"Unknown construction_method {construction_method}.")

    Xs_train = [zscore(x) for x in Xs_train]
    Xs_test = [zscore(x) for x in Xs_test]
    betas = [zscore(beta) for beta in betas]

    # generate targets
    Y_train = sum(
        [alpha * zscore(feature_space @ beta) for alpha, feature_space, beta in
         zip(scalars, feature_spaces_train, betas)])
    Y_test = sum(
        [alpha * zscore(feature_space @ beta) for alpha, feature_space, beta in
         zip(scalars, feature_spaces_test, betas)])
    Y_train = zscore(Y_train)
    Y_test = zscore(Y_test)
    # add noise
    noise_train = zscore(sample_random_distribution([n_samples_train, n_targets], "normal"))
    noise_test = zscore(sample_random_distribution([n_samples_test, n_targets], "normal"))
    Y_train += noise_train * noise_scalar
    Y_test += noise_test * noise_scalar

    from himalaya.backend import get_backend

    backend = get_backend()

    Xs_train = [backend.asarray(X, dtype="float32") for X in Xs_train]
    Xs_test = [backend.asarray(X, dtype="float32") for X in Xs_test]
    Y_train = backend.asarray(Y_train, dtype="float32")
    Y_test = backend.asarray(Y_test, dtype="float32")

    return Xs_train, Xs_test, Y_train, Y_test


def create_orthogonal_feature_spaces(num_samples, d_list, random_distribution="normal"):
    """
    Generates three orthogonal feature spaces with ranks specified by d_list.

    Parameters:
        num_samples (int): The number of dimensions for each feature space.
        d_list (list): A list containing rank/dimension for each feature space.

    Returns:
        list: A list of numpy arrays representing orthogonal feature spaces.
    """
    assert num_samples > sum(d_list), "Number of samples must be greater than the sum of ranks."

    # backend = get_backend()

    feature_spaces = []

    # Generate a random matrix of shape (dim, rank)
    M = sample_random_distribution(shape=(num_samples, sum(d_list)), distribution=random_distribution)
    M = zscore(M)

    # may reduce rank
    U, S, Vt = np.linalg.svd(M, full_matrices=True)

    start = 0
    for rank in d_list:
        _S = np.zeros(len(S))
        _S[start:start + rank] = S[start:start + rank]

        # create rectangular diagonal sigma matrix
        diag_S = np.diag(_S)
        diag_S = np.pad(diag_S, ((0, U.shape[0] - diag_S.shape[0]), (0, Vt.shape[0] - diag_S.shape[1])))

        feature_space = U @ diag_S @ Vt
        feature_spaces.append(feature_space)
        start += rank
    return feature_spaces


def run_experiment(variable_name, variable_values, n_runs, n_observations, d_list, scalars, n_targets, n_samples_train,
                   n_samples_test, noise_scalar_level, construction_method, random_distribution, alphas, cv, use_ols):
    """
    Execute machine learning experiments by varying specified parameters.

    Parameters:
    -----------
    variable_name : str
        Name of the variable to vary, e.g., 'Number of Features'.
    variable_values : list
        Values for the variable being tested.
    n_runs : int
        Number of experiment runs for each value.
    n_observations : int
        Number of observations in the dataset.
    d_list : list of int
        Dimensions of the feature spaces.
    scalars : list of float
        Scalars for feature spaces, must sum to 1.
    n_targets : int
        Number of target variables.
    n_samples_train : int
        Number of training samples.
    n_samples_test : int
        Number of testing samples.
    noise_scalar_level : float
        Proportion of noise added to target variables.
    construction_method : str
        'stack' or 'orthogonal' for the dataset construction method.
    random_distribution : str
        Distribution to use, e.g., 'normal', 'uniform'.
    alphas : ndarray
        Regularization parameters for ridge regression.
    cv : int
        Number of cross-validation folds.
    use_ols : bool
        Whether to use Ordinary Least Squares (OLS) regression.

    Returns:
    --------
    list
        Nested list containing variance results for different metrics (R-squared, rho).
    """

    predicted_results = [[] for _ in range(6)]

    for value in bar(variable_values, title=f"Varying {variable_name}"):
        if variable_name == "Number of Samples Training":
            n_samples_train = int(value)
        elif variable_name == "Number of Samples Testing":
            n_samples_test = int(value)
        elif variable_name == "Number of Features":
            d_list = [int(value)] * len(d_list)
        elif variable_name == "Number of Targets":
            n_targets = int(value)
        elif variable_name == "Proportion of Noise Added to Target":
            noise_scalar_level = value
        elif variable_name == "Sampling Distribution":
            random_distribution = value
        elif variable_name == "Unique Variance Explained":
            scalars = value
        else:
            raise ValueError(f"Unknown variable_name {variable_name}.")

        variances_r2, variances_direct_r2, residuals_r2 = [], [], []
        variances_rho, variances_direct_rho, residuals_rho = [], [], []

        for run in range(n_runs):
            (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(d_list=d_list,
                                                                    scalars=scalars,
                                                                    n_targets=n_targets,
                                                                    n_samples_train=n_samples_train,
                                                                    n_samples_test=n_samples_test,
                                                                    noise_scalar=noise_scalar_level,
                                                                    construction_method=construction_method,
                                                                    random_distribution=random_distribution,
                                                                    random_state=run)
            variance_r2 = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv,
                                                False)
            variance_direct_r2 = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, True)
            residual_r2 = residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, use_ols)

            variance_rho = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, False)
            variance_direct_rho = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, True)
            residual_rho = residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, use_ols, score_func=False)

            variance_r2 = np.nan_to_num(variance_r2)
            variance_direct_r2 = np.nan_to_num(variance_direct_r2)
            residual_r2 = np.nan_to_num(residual_r2)
            variance_rho = np.nan_to_num(variance_rho)
            variance_direct_rho = np.nan_to_num(variance_direct_rho)
            residual_rho = np.nan_to_num(residual_rho)

            variances_r2.append(variance_r2)
            variances_direct_r2.append(variance_direct_r2)
            residuals_r2.append(residual_r2)
            variances_rho.append(variance_rho)
            variances_direct_rho.append(variance_direct_rho)
            residuals_rho.append(residual_rho)

        predicted_results[0].append(variances_r2)
        predicted_results[1].append(variances_direct_r2)
        predicted_results[2].append(residuals_r2)
        predicted_results[3].append(variances_rho)
        predicted_results[4].append(variances_direct_rho)
        predicted_results[5].append(residuals_rho)

    return predicted_results


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
                                                            "orthogonal",
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
