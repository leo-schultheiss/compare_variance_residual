import numpy as np
from matplotlib import pyplot as plt


def plot_variance_vs_residual_box(x, xlabel, predicted_variance: list, predicted_residual: list, unique_contributions,
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
    medianprops = dict(color='black')
    ax.boxplot(predicted_variance, positions=positions_variance, widths=width(positions_variance, w), patch_artist=True,
               boxprops=dict(facecolor="C0"), medianprops=medianprops, label="variance partitioning")
    # Plot residuals
    ax.boxplot(predicted_residual, positions=positions_residual, widths=width(positions_residual, w), patch_artist=True,
               boxprops=dict(facecolor="C1"), medianprops=medianprops, label="residual method")

    ax.set_title("Box plots of predicted contributions in a range from 0 to 1")
    fig.suptitle("Variance partitioning vs residual method")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("predicted contribution")
    ax.set_ylim([0, 1.])
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


def plot_variance_vs_residual_error(x, xlabel, predicted_variance: list, predicted_residual: list, unique_contributions,
                                    x_is_log=False, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 6))

    # transform data to reflect error from true contribution
    true_contribution = unique_contributions[0]
    predicted_variance = list(np.array(predicted_variance) - true_contribution)
    predicted_residual = list(np.array(predicted_residual) - true_contribution)

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
    variance_plot = ax.boxplot(predicted_variance, positions=positions_variance, widths=width(positions_variance, w),
                               patch_artist=True, boxprops=dict(facecolor="C0"), medianprops=medianprops,
                               label="variance partitioning")
    # Plot residuals
    residual_plot = ax.boxplot(predicted_residual, positions=positions_residual, widths=width(positions_residual, w),
                               patch_artist=True, boxprops=dict(facecolor="C1"), medianprops=medianprops,
                               label="residual method")

    ax.set_title("Deviation from true contribution")
    fig.suptitle("Variance partitioning vs residual method")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("predicted contribution - true contribution")

    # set y-axis limits to the largest absolute whisker
    max_abs_variance = max(
        [np.max(np.abs(variance_plot['whiskers'][i].get_ydata())) for i in range(len(variance_plot['whiskers']))])
    max_abs_residual = max(
        [np.max(np.abs(residual_plot['whiskers'][i].get_ydata())) for i in range(len(residual_plot['whiskers']))])
    max_total = max(max_abs_variance, max_abs_residual)
    ax.set_ylim([-max_total - 0.1, max_total + 0.1])

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
    ax.axhline(y=0, color='k', linestyle='--', label='true contribution of $X_0$')

    # Add legend
    ax.legend(loc='upper right')

    # Add text field with variable information
    variable_info = f"unique_contributions: {unique_contributions}\n" + '\n'.join(
        ['{}={!r}'.format(k, v) for k, v in kwargs.items()])
    fig.text(1.1, 0.5, variable_info, ha='center', va='center', fontsize=10)
    plt.show()


def plot_variance_vs_residual_xy_correlation(x, xlabel, predicted_variance: list, predicted_residual: list,
                                             unique_contributions, **kwargs):
    """
    create scatterplots of predicted variance vs predicted residual to show correlation
    """
    def scale_to_minus_one_to_one(array):
        min_val = np.min(array)
        max_val = np.max(array)
        scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
        return scaled_array

    # remove nans and infs
    predicted_variance, predicted_residual = np.nan_to_num(predicted_variance), np.nan_to_num(predicted_residual)

    # create a grid according to the number of samples
    nrows = 3
    ncols = (len(x) + nrows - 1) // nrows  # Ensure enough columns to fit all plots
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False)

    # center data around true contribution
    true_contribution = unique_contributions[0]
    predicted_variance = list(np.array(predicted_variance) - true_contribution)
    predicted_residual = list(np.array(predicted_residual) - true_contribution)

    # scale data to [-1, 1]
    predicted_variance = [scale_to_minus_one_to_one(variance) for variance in predicted_variance]
    predicted_residual = [scale_to_minus_one_to_one(residual) for residual in predicted_residual]

    for i, (variance, residual) in enumerate(zip(predicted_variance, predicted_residual)):
        ax[i // ncols, i % ncols].scatter(variance, residual, alpha=0.5)

        ax[i // ncols, i % ncols].set_title(f"{x[i]} {xlabel}")
        ax[i // ncols, i % ncols].set_xlabel("predicted variance")
        ax[i // ncols, i % ncols].set_ylabel("predicted residual")

        # add text box that displays the correlation coefficient
        corr = np.corrcoef(variance, residual)[0, 1]
        ax[i // ncols, i % ncols].text(0.05, 0.95, rf"$\rho$: {corr:.2f}", transform=ax[i // ncols, i % ncols].transAxes,
                                      fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        lims = [-1.1, 1.1]
        ax[i // ncols, i % ncols].set_xlim(lims)
        ax[i // ncols, i % ncols].set_ylim(lims)
        # plot x=y
        ax[i // ncols, i % ncols].plot(lims, lims, 'k--')
