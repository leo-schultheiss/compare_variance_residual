import numpy as np
from himalaya.backend import get_backend
from himalaya.progress_bar import bar
from scipy.linalg import orth
from scipy.stats import zscore

from compare_variance_residual.simulated.residual import residual_method
from compare_variance_residual.simulated.variance_partitioning import variance_partitioning


def create_random_distribution(shape, distribution) -> np.ndarray:
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


def generate_dataset(d_list=None, scalars=None, n_targets=100, n_samples_train=100, n_samples_test=100, noise=0.1,
                     random_distribution="normal", construction_method="stack", random_state=None):
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
        n_targets: int
            The number of target variables in the dataset.
        n_samples_train: int
            The number of samples in the training dataset.
        n_samples_test: int
            The number of samples in the testing dataset.
        noise: float
            The standard deviation of the Gaussian noise to be added to the targets.
        random_distribution: str
            The type of random distribution to use for generating target values. Defaults to "normal".
        construction_method: str
            The method to use for constructing the feature spaces. Can be either "stack" or "svd". Defaults to "stack".
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
    backend = get_backend()

    if d_list is None:
        d_list = [100] * 3

    if scalars is None:
        scalars = [1 / 3] * 3

    if construction_method == "stack":
        Xs_train, Xs_test, Y_train, Y_test = stacked_feature_spaces(d_list, scalars, n_samples_train,
                                                                    n_samples_test, n_targets, noise,
                                                                    random_distribution)
    elif construction_method == "svd":
        Xs_train, Xs_test, Y_train, Y_test = svd_feature_spaces(d_list, scalars, n_samples_train,
                                                                n_samples_test, n_targets, noise, random_distribution)
    else:
        raise ValueError(f"Unknown construction_method {construction_method}.")

    Xs_train = [backend.asarray(X, dtype="float32") for X in Xs_train]
    Xs_test = [backend.asarray(X, dtype="float32") for X in Xs_test]
    Y_train = backend.asarray(Y_train, dtype="float32")
    Y_test = backend.asarray(Y_test, dtype="float32")

    return Xs_train, Xs_test, Y_train, Y_test


def stacked_feature_spaces(d_list, scalars, n_samples_train, n_samples_test, n_targets, noise, random_distribution):
    # generate feature spaces
    feature_spaces_train = [create_random_distribution([n_samples_train, d], random_distribution) for d in d_list]
    feature_spaces_test = [create_random_distribution([n_samples_test, d], random_distribution) for d in d_list]

    # concatenate the first feature with all other feature spaces
    # [0, 1], [0, 2], [0, 3], ...
    Xs_train = [np.hstack([feature_spaces_train[0], feature_space]) for feature_space in feature_spaces_train[1:]]
    Xs_test = [np.hstack([feature_spaces_test[0], feature_space]) for feature_space in feature_spaces_test[1:]]

    Xs_train = [zscore(X) for X in Xs_train]
    Xs_test = [zscore(X) for X in Xs_test]

    # generate scalars
    thetas = [create_random_distribution([d, n_targets], "normal") for d in d_list]

    # generate targets
    Y_train = sum(
        [alpha * zscore(feature_space @ theta) for alpha, feature_space, theta in
         zip(scalars, feature_spaces_train, thetas)])
    Y_test = sum(
        [alpha * zscore(feature_space @ theta) for alpha, feature_space, theta in
         zip(scalars, feature_spaces_test, thetas)])

    Y_train = zscore(Y_train)
    Y_test = zscore(Y_test)

    # add noise
    Y_train += create_random_distribution([n_samples_train, n_targets], "normal") * noise
    Y_test += create_random_distribution([n_samples_test, n_targets], "normal") * noise

    return Xs_train, Xs_test, Y_train, Y_test


def svd_feature_spaces(d_list, scalars, n_samples_train, n_samples_test, n_targets, noise, random_distribution):
    # create orthonormal S matrix
    S_train = orth(create_random_distribution([n_samples_train, d_shared], random_distribution))
    S_test = orth(create_random_distribution([n_samples_test, d_shared], random_distribution))

    Us_train, Us_test = [], []

    # create orthonormal U_i matrices and ensure they are orthogonal to S and each other
    for d_unique in d_unique_list:
        U_train = orth(create_random_distribution([n_samples_train, d_unique], random_distribution))
        U_test = orth(create_random_distribution([n_samples_test, d_unique], random_distribution))

        # remove projections onto S and all previously created U_i matrices
        for prev_U_train, prev_U_test in zip(Us_train, Us_test):
            U_train -= prev_U_train @ (prev_U_train.T @ U_train)
            U_test -= prev_U_test @ (prev_U_test.T @ U_test)
        U_train -= S_train @ (S_train.T @ U_train)
        U_test -= S_test @ (S_test.T @ U_test)

        # re-orthonormalize after removing projections
        U_train = orth(U_train)
        U_test = orth(U_test)

        Us_train.append(U_train)
        Us_test.append(U_test)

    # combine S and Ui to each feature
    Xs_train = [np.hstack([S_train, U_train]) for U_train in Us_train]
    Xs_test = [np.hstack([S_test, U_test]) for U_test in Us_test]

    # combine target out of S and U_i
    theta_S = create_random_distribution([d_shared, n_targets], "normal")
    thetas_U = [create_random_distribution([U_train.shape[1], n_targets], "normal") for U_train in Us_train]
    Y_train = S_train @ theta_S + sum(U_train @ theta_U for U_train, theta_U in zip(Us_train, thetas_U))
    Y_test = S_test @ theta_S + sum(U_test @ theta_U for U_test, theta_U in zip(Us_test, thetas_U))

    # add noise
    Y_train += create_random_distribution([n_samples_train, n_targets], "normal") * noise
    Y_test += create_random_distribution([n_samples_test, n_targets], "normal") * noise

    return Xs_test, Xs_train, Y_test, Y_train


def run_experiment(variable_name, variable_values, n_runs, n_observations, d_list, scalars, n_targets, n_samples_train,
                   n_samples_test, noise_level, construction_method, random_distribution, alphas, cv, use_ols):
    predicted_results = [[] for _ in range(6)]

    for value in bar(variable_values, title=f"Varying {variable_name}"):
        if variable_name == "sample size training":
            n_samples_train = int(value)
        elif variable_name == "sample size testing":
            n_samples_test = int(value)
        elif variable_name == "number of features":
            d_list = [int(value)] * len(d_list)
        elif variable_name == "number of targets":
            n_targets = int(value)
        elif variable_name == "scalar amount of noise in the target":
            noise_level = value
        elif variable_name == "sampling distribution":
            random_distribution = value
        elif variable_name == "unique variance explained":
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
                                                                    n_samples_test=n_samples_test, noise=noise_level,
                                                                    construction_method=construction_method,
                                                                    random_distribution=random_distribution,
                                                                    random_state=run)
            variance_r2 = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv,
                                                False)
            variance_direct_r2 = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, True)
            residual_r2 = residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, use_ols)

            variance_rho = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, False, use_r2=False)
            variance_direct_rho = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, True,
                                                        use_r2=False)
            residual_rho = residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, use_ols, use_r2=False)

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
    d_list = [100, 100, 100]
    scalars = [1 / 3, 1 / 3, 1 / 3]
    n_targets = 100
    n_samples_train = 100
    n_samples_test = 50
    noise = 0.1
    random_distribution = "normal"

    (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(d_list, scalars, n_targets, n_samples_train,
                                                            n_samples_test, noise, random_distribution, "stack", 42)

    import matplotlib.pyplot as plt

    plt.scatter(Xs_train[0][:, 0], Y_train[:, 0], label="train")
    plt.show()

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression()
    model.fit(Xs_train[0], Y_train)

    Y_pred = model.predict(Xs_test[0])
    print(f"R^2 score on one train: {r2_score(Y_test, Y_pred)}")

    X_stack = np.hstack(Xs_train)
    model = LinearRegression()
    model.fit(X_stack, Y_train)

    X_test = np.hstack(Xs_test)
    Y_pred = model.predict(X_test)
    print(f"R^2 score on all train: {r2_score(Y_test, Y_pred)}")

    from himalaya.ridge import BandedRidgeCV
    from himalaya.backend import set_backend

    set_backend("cupy")

    model = BandedRidgeCV(groups="input")
    model.fit(Xs_train, Y_train)
    Y_pred = model.predict(Xs_test)
    print(f"R^2 score on banded: {r2_score(Y_test, Y_pred)}")

    # check if Xs are random
    assert not np.allclose(Xs_train[0], Xs_train[1], atol=1e-5)

    # check if Xs are demeaned
    for X in Xs_train:
        assert np.allclose(X.mean(0), 0, atol=1e-5)
    for X in Xs_test:
        assert np.allclose(X.mean(0), 0, atol=1e-5)

    # check if Ys are demeaned
    assert np.allclose(Y_train.mean(0), 0, atol=noise)
    assert np.allclose(Y_test.mean(0), 0, atol=noise)
