import numpy as np
from himalaya.backend import get_backend
from himalaya.progress_bar import bar

from simulated.residual import residual_method
from simulated.variance_partitioning import variance_partitioning


def generate_dataset(n_targets=100,
                     n_samples_train=100, n_samples_test=100,
                     noise=0, unique_contributions=None,
                     n_features_list=None, random_distribution="normal", random_state=None):
    """Utility to generate dataset.

    Parameters
    ----------
    n_targets : int
        Number of targets.
    n_samples_train : int
        Number of samples in the training set.
    n_samples_test : int
        Number of sample in the testing set.
    noise : float > 0
        Scale of the Gaussian white noise added to the targets.
    unique_contributions : list of floats
        Proportion of the target variance explained by each feature space.
    n_features_list : list of int of length (n_features, ) or None
        Number of features in each kernel. If None, use 1000 features for each.
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

    def generate_distribution(shape, distribution):
        """Generate a distribution.

        Parameters
        ----------
        shape : array of shape (n_samples, )
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
        elif distribution == "poisson":
            return np.random.poisson(size=shape)
        elif distribution == "lognormal":
            return np.random.lognormal(size=shape)
        elif distribution == "pareto":
            return np.random.pareto(a=1, size=shape)
        else:
            raise ValueError(f"Unknown distribution {distribution}.")

    if random_state is not None:
        np.random.seed(random_state)
    backend = get_backend()

    if unique_contributions is None:
        unique_contributions = [0.5, 0.5]

    if n_features_list is None:
        n_features_list = np.full(len(unique_contributions), fill_value=1000)

    Xs_train, Xs_test = [], []
    Y_train, Y_test = np.zeros((n_samples_train, n_targets)), np.zeros((n_samples_test, n_targets))

    # generate shared component
    S_train = generate_distribution([n_samples_train, 1], random_distribution)
    S_test = generate_distribution([n_samples_test, 1], random_distribution)

    for ii, unique_contribution in enumerate(unique_contributions):
        n_features = n_features_list[ii]

        # generate random features
        X_train = generate_distribution([n_samples_train, n_features], random_distribution)
        X_test = generate_distribution([n_samples_test, n_features], random_distribution)

        # add shared component
        unique_component = unique_contribution
        shared_component = (1 - sum(unique_contributions)) / len(unique_contributions)

        X_train = shared_component * S_train + unique_component * X_train
        X_test = shared_component * S_test + unique_component * X_test

        # demean
        X_train -= X_train.mean(0)
        X_test -= X_test.mean(0)

        Xs_train.append(X_train)
        Xs_test.append(X_test)

        weights = generate_distribution([n_features, n_targets], random_distribution) / n_features

        if ii == 0:
            Y_train = X_train @ weights
            Y_test = X_test @ weights
        else:
            Y_train += X_train @ weights
            Y_test += X_test @ weights

    std = Y_train.std(0)[None]
    Y_train /= std
    Y_test /= std

    Y_train += generate_distribution([n_samples_train, n_targets], random_distribution) * noise
    Y_test += generate_distribution([n_samples_test, n_targets], random_distribution) * noise
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    Xs_train = [backend.asarray(X, dtype="float32") for X in Xs_train]
    Xs_test = [backend.asarray(X, dtype="float32") for X in Xs_test]
    Y_train = backend.asarray(Y_train, dtype="float32")
    Y_test = backend.asarray(Y_test, dtype="float32")

    return Xs_train, Xs_test, Y_train, Y_test


def run_experiment(variable_values, variable_name, n_runs, unique_contributions, n_features_list, n_targets,
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
            n_features_list = [int(value), int(value)]
        elif variable_name == "number of features $X_{0}$":
            n_features_list = [int(value), n_features_list[1]]
        elif variable_name == "number of targets":
            n_targets = int(value)
        elif variable_name == "relative amount of noise in the target":
            noise_level = value
        elif variable_name == "proportions of unique contribution":
            unique_contributions = value
        elif variable_name == "sampling distribution":
            random_distribution = value

        for run in range(n_runs):
            (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(
                n_features_list=n_features_list, n_targets=n_targets,
                n_samples_train=n_samples_train, n_samples_test=n_samples_test,
                noise=noise_level, unique_contributions=unique_contributions,
                random_distribution=random_distribution, random_state=run + 100)
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
