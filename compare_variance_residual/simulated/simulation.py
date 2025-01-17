import numpy as np
from himalaya.backend import get_backend
from himalaya.progress_bar import bar
from scipy.linalg import orth

from compare_variance_residual.simulated.residual import residual_method
from compare_variance_residual.simulated.variance_partitioning import variance_partitioning


def create_random_distribution(shape, distribution) -> np.ndarray:
    """Create a random distribution.

    Parameters
    ----------
    shape : array of shape (n_samples,)
        x coordinates.
    distribution : str in {"normal", "uniform", "exponential", "gamma", "beta", "poisson", "lognormal", "pareto"}
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


def generate_dataset(d_shared=50, d_unique_list=None, n_targets=100, n_samples_train=100, n_samples_test=100, noise=0.1,
                     random_distribution="normal", construction_method="svd", random_state=None):
    """Utility to generate dataset.

    Parameters
    ----------
    d_shared : int
        Dimension of shared component between feature spaces.
    d_unique_list : list of int
        Dimension of unique component in each feature space.
    n_targets : int
        Number of targets.
    n_samples_train : int
        Number of samples in the training set.
    n_samples_test : int
        Number of sample in the testing set.
    noise : float >= 0
        Scale of the Gaussian white noise added to the targets.
    random_distribution : str in {"normal", "uniform"}
        Function to generate random features.
        Should support the signature (n_samples, n_features) -> array of shape (n_samples, n_features).
    random_state : int, or None
        Random generator seed use to generate the true kernel weights.

    Returns
    -------
    Xs_train : array of shape (n_feature_spaces, n_samples_train, n_features)
        Training features.
    Xs_test : array of shape (n_feature_spaces, n_samples_test, n_features)
        Testing features.
    Y_train : array of shape (n_samples_train, n_targets)
        Training targets.
    Y_test : array of shape (n_samples_test, n_targets)
        Testing targets.
    kernel_weights : array of shape (n_targets, n_features)
        Kernel weights in the prediction of the targets.
    n_features_list : list of int of length (n_features, )
        Number of features in each kernel.
    """

    if random_state is not None:
        np.random.seed(random_state)
    backend = get_backend()

    if d_unique_list is None:
        d_unique_list = [50, 50]

    if construction_method == "stack":
        Xs_test, Xs_train, Y_test, Y_train = stacked_feature_spaces(d_shared, d_unique_list, n_samples_train,
                                                                    n_samples_test, n_targets, random_distribution)
    elif construction_method == "svd":
        Xs_train, Xs_test = [], []
        Y_train, Y_test = [], []
        Us_train, Us_test = [], []

        # Dimensions and ranks
        d_s, d_0, d_1 = 10, 10, 10  # Dimensions of S, U_0, U_1
        r_s, r_0, r_1 = 5, 5, 5  # Ranks of S, U_0, U_1

        # Generate orthogonal subspaces
        S_train = orth(generate_dataset([d_s, r_s], n_samples_train))
        S_test = orth(generate_dataset([d_s, r_s], n_samples_test))

        for i in range(2):
            U_train = orth(generate_dataset([d_s, r_s], n_samples_train))
            U_test = orth(generate_dataset([d_s, r_s], n_samples_test))

            unique_component = 0.5
            shared_component = 1 - unique_component

            # Construct feature space
            X_train = shared_component * S_train @ np.random.randn(r_s, n_samples_train) + unique_component * U_train @ np.random.randn(r_0, n_samples_train)
            X_test =  shared_component * S_test @ np.random.randn(r_s, n_samples_test) + unique_component * U_test @ np.random.randn(r_0, n_samples_test)

            Xs_train.append(X_train)
            Xs_test.append(X_test)
        # Define target Y
        # Define weights for Y computation
        w_s = np.random.randn(r_s, 1)  # Shared weight
        w_u0 = np.random.randn(r_0, 1)  # Unique weight for U_0
        w_u1 = np.random.randn(r_1, 1)  # Unique weight for U_1

        # Compute Y
        Y_shared = S.T @ w_s @ np.random.randn(1, n_samples)  # Shared contribution
        Y_u0 = U_0.T @ w_u0 @ np.random.randn(1, n_samples)  # Unique contribution from U_0
        Y_u1 = U_1.T @ w_u1 @ np.random.randn(1, n_samples)  # Unique contribution from U_1

        Y_train = S_train.T @ np.random.randn(d_s, 1) + np.random.randn(n_samples_train, 1)
        Y_test = S_test.T @ np.random.randn(d_s, 1) + np.random.randn(n_samples_test, 1)

        Y += noise * np.random.randn(*Y.shape)
    else:
        raise ValueError(f"Unknown construction_method {construction_method}.")

    # demean features across all feature spaces
    mean_train = np.mean(np.concatenate(Xs_train), axis=0)
    mean_test = np.mean(np.concatenate(Xs_test), axis=0)
    Xs_train = [X - mean_train for X in Xs_train]
    Xs_test = [X - mean_test for X in Xs_test]

    # normalize targets
    std = Y_train.std(0)[None]
    Y_train /= std
    Y_test /= std

    # add noise
    Y_train += create_random_distribution([n_samples_train, n_targets], "normal") * noise
    Y_test += create_random_distribution([n_samples_test, n_targets], "normal") * noise

    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    Xs_train = [backend.asarray(X, dtype="float32") for X in Xs_train]
    Xs_test = [backend.asarray(X, dtype="float32") for X in Xs_test]
    Y_train = backend.asarray(Y_train, dtype="float32")
    Y_test = backend.asarray(Y_test, dtype="float32")

    return Xs_train, Xs_test, Y_train, Y_test


def stacked_feature_spaces(d_shared, d_unique_list, n_samples_train, n_samples_test, n_targets, random_distribution):
    # generate shared component
    S_train = create_random_distribution([n_samples_train, d_shared], random_distribution)
    S_test = create_random_distribution([n_samples_test, d_shared], random_distribution)
    S_train -= S_train.mean(0)
    S_test -= S_test.mean(0)
    # generate shared weights
    beta_S = create_random_distribution([d_shared, n_targets], "normal")
    Us_train, Us_test = [], []
    betas_U = []
    Xs_train, Xs_test = [], []
    for ii, d_unique in enumerate(d_unique_list):
        # generate unique component
        U_train = create_random_distribution([n_samples_train, d_unique], random_distribution)
        U_test = create_random_distribution([n_samples_test, d_unique], random_distribution)
        U_train -= U_train.mean(0)
        U_test -= U_test.mean(0)

        # generate unique weights
        beta_U = create_random_distribution([d_unique, n_targets], "normal")
        betas_U.append(beta_U)

        # concatenate shared and unique components
        X_train = np.hstack([S_train, U_train])
        X_test = np.hstack([S_test, U_test])

        Us_train.append(U_train)
        Us_test.append(U_test)
        Xs_train.append(X_train)
        Xs_test.append(X_test)
    Y_train = S_train @ beta_S + sum([U @ beta_U for U, beta_U in zip(Us_train, betas_U)])
    Y_test = S_test @ beta_S + sum([U @ beta_U for U, beta_U in zip(Us_test, betas_U)])
    return Xs_test, Xs_train, Y_test, Y_train


def run_experiment(variable_values, variable_name, n_runs, d_shared, d_unique_list, n_targets,
                   n_samples_train, n_samples_test, noise_level, random_distribution, alphas, cv,
                   direct_variance_partitioning, ignore_negative_r2, use_ols):
    predicted_variance = []
    predicted_residual = []

    for value in bar(variable_values, title=f"Varying {variable_name}"):
        variance_runs = []
        residual_runs = []

        if variable_name == "sample size training":
            n_samples_train = int(value)
        elif variable_name == "sample size testing":
            n_samples_test = int(value)
        elif variable_name == "number of features $X_{0,1}$":
            d_shared = int(value)
            d_unique_list = [int(value), int(value)]
        elif variable_name == "number of targets":
            n_targets = int(value)
        elif variable_name == "relative amount of noise in the target":
            noise_level = value
        elif variable_name == "sampling distribution":
            random_distribution = value
        else:
            raise ValueError(f"Unknown variable_name {variable_name}.")

        for run in range(n_runs):
            (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(
                d_shared=d_shared, d_unique_list=d_unique_list, n_targets=n_targets, n_samples_train=n_samples_train,
                n_samples_test=n_samples_test, noise=noise_level, random_distribution=random_distribution,
                random_state=run + 100)
            variance = variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas, cv,
                                             direct_variance_partitioning, ignore_negative_r2)
            residual = residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas, cv, use_ols, ignore_negative_r2)
            variance = np.nan_to_num(variance)
            residual = np.nan_to_num(residual)
            variance_runs.append(variance)
            residual_runs.append(residual)

        predicted_variance.append(variance_runs)
        predicted_residual.append(residual_runs)
    return predicted_variance, predicted_residual


if __name__ == "__main__":
    d_shared = 50
    d_unique_list = [50, 50]
    n_targets = 100
    n_samples_train = 100
    n_samples_test = 50
    noise = 0.1
    random_distribution = "normal"

    (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(d_shared, d_unique_list, n_targets, n_samples_train,
                                                            n_samples_test, noise, random_distribution, 42)

    import matplotlib.pyplot as plt

    plt.scatter(Xs_train[0][:, 0], Y_train[:, 0], label="train")
    plt.show()

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression()
    model.fit(Xs_train[0], Y_train)

    Y_pred = model.predict(Xs_test[0])
    print(f"R^2 score on one train: {r2_score(Y_test, Y_pred)}")

    X_train = np.hstack(Xs_train)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    X_test = np.hstack(Xs_test)
    Y_pred = model.predict(X_test)
    print(f"R^2 score on all train: {r2_score(Y_test, Y_pred)}")

    # check if Xs are random
    assert not np.allclose(Xs_train[0], Xs_train[1])

    # check if Xs are demeaned
    mean_train = np.mean(np.concatenate(Xs_train, axis=0), axis=0)
    mean_test = np.mean(np.concatenate(Xs_test, axis=0), axis=0)
    assert np.allclose(mean_train, 0)
    assert np.allclose(mean_test, 0)

    # check if Ys are demeaned
    assert np.allclose(Y_train.mean(0), 0)
    assert np.allclose(Y_test.mean(0), 0)
