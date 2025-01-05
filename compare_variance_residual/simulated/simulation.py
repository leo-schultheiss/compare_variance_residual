import numpy as np
from himalaya.backend import get_backend
from matplotlib import pyplot as plt


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


def generate_dataset(n_targets=500,
                     n_samples_train=1000, n_samples_test=400,
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


def plot_variance_vs_residual(x, xlabel, predicted_variance: list, predicted_residual: list, unique_contributions,
                              x_is_log=False, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 6))

    if x_is_log:
        w = 0.05
        width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
        positions_variance = 10 ** (np.log10(x) - w / 2.)
        positions_residual = 10 ** (np.log10(x) + w / 2.)
    else:
        w = (x[-1] if isinstance(x[0], (int, float)) else len(x)) / (len(x) * 5)
        width = lambda _, w: w
        positions_variance = (x if isinstance(x[0], (int, float)) else np.arange(len(x))) - w / 2.
        positions_residual = (x if isinstance(x[0], (int, float)) else np.arange(len(x))) + w / 2.

        # Plot variance partitioning
    ax.boxplot(predicted_variance, positions=positions_variance, widths=width(positions_variance, w), patch_artist=True,
               boxprops=dict(facecolor="C0"), label="variance partitioning")
    # Plot residuals
    ax.boxplot(predicted_residual, positions=positions_residual, widths=width(positions_residual, w), patch_artist=True,
               boxprops=dict(facecolor="C1"), label="residual method")

    ax.set_title("Variance partitioning vs residual method")
    ax.set_subtitle(xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("predicted contribution")
    ax.set_ylim([-0.1, 1.1])
    if isinstance(x[0], (int, float)):
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xlim([10 ** (np.log10(x[0]) - w * 2), 10 ** (np.log10(x[-1]) + w * 2)]) if x_is_log else ax.set_xlim(
            [x[0] - w * 2, x[-1] + w * 2])
    else:
        ax.set_xlim([-0.5, len(x) - 0.5])
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x, rotation=45, ha='right')

    if x_is_log:
        ax.set_xscale("log")

    # draw center line
    ax.axhline(y=unique_contributions[0], color='k', linestyle='--', label='true contribution of $X_0$')

    # Add legend
    ax.legend(loc='upper right')

    # Add text field with variable information
    variable_info = f"unique_contributions: {unique_contributions}\n" + '\n'.join(
        ['{}={!r}'.format(k, v) for k, v in kwargs.items()])
    fig.text(1.1, 0.5, variable_info, ha='center', va='center', fontsize=10)
    plt.show()
