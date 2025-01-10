import numpy as np
from matplotlib import pyplot as plt


def normalize_to_unit_interval(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return scaled_array


def plot_boxplots(predicted_residual, predicted_variance, title, x, x_is_log, xlabel, ylabel, ylim):
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
    medianprops = dict(color='black')
    _ = ax.boxplot(predicted_variance, positions=positions_variance, widths=width(positions_variance, w),
                   patch_artist=True,
                   boxprops=dict(facecolor="C0"), medianprops=medianprops, label="variance partitioning")
    # Plot residuals
    _ = ax.boxplot(predicted_residual, positions=positions_residual, widths=width(positions_residual, w),
                   patch_artist=True,
                   boxprops=dict(facecolor="C1"), medianprops=medianprops, label="residual method")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(ylim)

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

    return fig, ax


def plot_predicted_contributions_box(x, xlabel, predicted_variance: list, predicted_residual: list,
                                     unique_contributions,
                                     x_is_log=False, **kwargs):
    title = "Box plots of predicted contributions displayed in range from 0 to 1"
    ylabel = "predicted contribution"
    ylims = [0, 1]

    fig, ax = plot_boxplots(predicted_residual, predicted_variance, title, x, x_is_log, xlabel, ylabel, ylims)

    # draw center line
    ax.axhline(y=unique_contributions[0], color='k', linestyle='--', label='true contribution of $X_0$')

    # Add legend
    ax.legend(loc='upper right')

    # Add text field with variable information
    variable_info = f"unique_contributions: {unique_contributions}\n" + '\n'.join(
        ['{}={!r}'.format(k, v) for k, v in kwargs.items()])
    fig.text(1.1, 0.5, variable_info, ha='center', va='center', fontsize=10)
    plt.show()


def plot_prediction_error(x, xlabel, predicted_variance: list, predicted_residual: list, unique_contributions,
                          x_is_log=False, **kwargs):
    title = "Deviation from true contribution"
    ylabel = "predicted contribution - true contribution"

    # Calculate the 95th percentile for the whiskers
    whisker_percentile = 95
    variance_whiskers = [np.percentile(np.abs(variance), whisker_percentile) for variance in predicted_variance]
    residual_whiskers = [np.percentile(np.abs(residual), whisker_percentile) for residual in predicted_residual]

    # set y-axis limits to the largest absolute whisker
    max_abs_variance = max(variance_whiskers)
    max_abs_residual = max(residual_whiskers)
    max_total = max(max_abs_variance, max_abs_residual)
    ylim = [-max_total - 0.1, max_total + 0.1]

    # transform data to reflect error from true contribution
    true_contribution = unique_contributions[0]
    predicted_variance = list(np.array(predicted_variance) - true_contribution)
    predicted_residual = list(np.array(predicted_residual) - true_contribution)

    fig, ax = plot_boxplots(predicted_residual, predicted_variance, title, x, x_is_log, xlabel, ylabel, ylim)

    # draw center line
    ax.axhline(y=0, color='k', linestyle='--', label='true contribution of $X_0$')

    # Add legend
    ax.legend(loc='upper right')

    # Add text field with variable information
    variable_info = f"unique_contributions: {unique_contributions}\n" + '\n'.join(
        ['{}={!r}'.format(k, v) for k, v in kwargs.items()])
    fig.text(1.1, 0.5, variable_info, ha='left', va='center', fontsize=10)
    plt.show()


def plot_prediction_scatter(x, xlabel, predicted_variance: list, predicted_residual: list,
                            unique_contributions, scale_to_unit_range=True, **kwargs):
    """
    create scatter plots of predicted variance vs predicted residual to show correlation
    """

    # remove nans and infs
    predicted_variance, predicted_residual = np.nan_to_num(predicted_variance), np.nan_to_num(predicted_residual)

    # center data around true contribution
    true_contribution = unique_contributions[0]
    predicted_variance = list(np.array(predicted_variance) - true_contribution)
    predicted_residual = list(np.array(predicted_residual) - true_contribution)

    # scale data to [-1, 1]
    if scale_to_unit_range:
        predicted_variance = [normalize_to_unit_interval(variance) for variance in predicted_variance]
        predicted_residual = [normalize_to_unit_interval(residual) for residual in predicted_residual]

    # Calculate the number of rows and columns needed
    n_plots = len(predicted_variance)
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False)

    for i, (variance, residual) in enumerate(zip(predicted_variance, predicted_residual)):
        ax[i // ncols, i % ncols].scatter(variance, residual, alpha=0.5)

        title = f"predicted contributions for {x[i]} {xlabel}"
        ax[i // ncols, i % ncols].set_title(f"{x[i]} {xlabel}")
        ax[i // ncols, i % ncols].set_xlabel("variance partitioning predicted contribution")
        ax[i // ncols, i % ncols].set_ylabel("residual method predicted contribution")

        # add text box that displays the correlation coefficient
        corr = np.corrcoef(variance, residual)[0, 1]
        ax[i // ncols, i % ncols].text(0.05, 0.95, rf"$\rho$: {corr:.2f}",
                                       transform=ax[i // ncols, i % ncols].transAxes,
                                       fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        xlims = [np.min(variance) * 1.1 if np.min(variance) < 0 else np.min(variance) * 0.9,
                 np.max(variance) * 1.1 if np.max(variance) > 0 else np.max(variance) * 0.9]
        ylims = [np.min(residual) * 1.1 if np.min(residual) < 0 else np.min(residual) * 0.9,
                 np.max(residual) * 1.1 if np.max(residual) > 0 else np.max(residual) * 0.9]
        ax[i // ncols, i % ncols].set_xlim(xlims)
        ax[i // ncols, i % ncols].set_ylim(ylims)
        # plot x=y
        ax[i // ncols, i % ncols].plot(xlims, ylims, 'k--', label="x=y")
        # add legend
        ax[i // ncols, i % ncols].legend(loc='lower right')

    # remove empty subplots
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(ax.flatten()[i])
    # create additional plot for text containing variable information
    fig.text(1.1, 0.5, '\n'.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()]), ha='left', va='center',
             fontsize=10)