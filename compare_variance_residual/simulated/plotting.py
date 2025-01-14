import numpy as np
from matplotlib import pyplot as plt


def normalize_to_unit_interval(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return np.nan_to_num(scaled_array)


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
                                     unique_contributions, x_is_log=False, **kwargs):
    title = "Box plots of predicted contributions displayed in range from 0 to 1"
    ylabel = "predicted contribution"
    ylims = [-0.1, 1.1]

    fig, ax = plot_boxplots(predicted_residual, predicted_variance, title, x, x_is_log, xlabel, ylabel, ylims)

    # draw center line
    ax.axhline(y=unique_contributions[0], color='k', linestyle='--', label='true contribution of $X_0$')

    # Add legend
    ax.legend(loc='upper right')

    # Add text field with variable information
    variable_info = f"unique_contributions: {unique_contributions}\n" + '\n'.join(
        ['{}={!r}'.format(k, v) for k, v in kwargs.items()])
    fig.text(1, 0.5, variable_info, ha='left', va='center', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_prediction_error(x, xlabel, predicted_variance: list, predicted_residual: list, unique_contributions,
                          x_is_log=False, **kwargs):
    title = "Deviation from true contribution"
    ylabel = "predicted contribution - true contribution"

    # Calculate the whiskers greatest extent
    predicted_variance = np.array(predicted_variance)
    predicted_residual = np.array(predicted_residual)
    iqr_variance = np.subtract(*np.percentile(predicted_variance, [75, 25]))
    iqr_residual = np.subtract(*np.percentile(predicted_residual, [75, 25]))
    # calculate whiskers
    max_total = np.max([np.percentile(predicted_variance, 75, axis=1) + 1.5 * iqr_variance,
                        np.percentile(predicted_residual, 75, axis=1) + 1.5 * iqr_residual])
    min_total = np.min([np.percentile(predicted_variance, 25, axis=1) - 1.5 * iqr_variance,
                        np.percentile(predicted_residual, 25, axis=1) - 1.5 * iqr_residual])

    # set y-axis limits to the largest absolute whisker
    ylim = [min(0, min_total), max(1, max_total)]

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
    fig.text(1, 0.5, variable_info, ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_prediction_scatter(x, xlabel, predicted_variance: list, predicted_residual: list,
                            unique_contributions, normalize=False, ignore_outliers=False, **kwargs):
    """
    create scatter plots of predicted variance vs predicted residual to show correlation
    """
    # remove 5th percentile and 95th percentile to ignore outliers
    if ignore_outliers:
        predicted_variance = [np.clip(variance, np.percentile(variance, 5), np.percentile(variance, 95)) for variance in
                              predicted_variance]
        predicted_residual = [np.clip(residual, np.percentile(residual, 5), np.percentile(residual, 95)) for residual in
                              predicted_residual]

    # center data around true contribution
    true_contribution = unique_contributions[0]
    predicted_variance = list(np.array(predicted_variance) - true_contribution)
    predicted_residual = list(np.array(predicted_residual) - true_contribution)

    if normalize:
        predicted_variance = [normalize_to_unit_interval(variance) for variance in predicted_variance]
        predicted_residual = [normalize_to_unit_interval(residual) for residual in predicted_residual]

    # Calculate the number of rows and columns needed
    n_plots = len(predicted_variance)
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False, sharex=normalize,
                           sharey=normalize)
    # add title to the figure
    fig.suptitle(f"Scatter plots over {xlabel}: Predicted Variance vs Residual Deviation from True Contribution",
                 fontsize=20, y=1.0)

    for i, (variance, residual) in enumerate(zip(predicted_variance, predicted_residual)):
        ax[i // ncols, i % ncols].scatter(variance, residual, alpha=0.5)
        title = f"{xlabel}: " + (f"{x[i]:02}" if isinstance(x[i], (int, float)) else x[i])
        ax[i // ncols, i % ncols].set_title(title)

        # add text box that displays the correlation coefficient
        corr = np.corrcoef(variance, residual)[0, 1]
        ax[i // ncols, i % ncols].text(0.05, 0.95, rf"$\rho$: {corr:.2f}",
                                       transform=ax[i // ncols, i % ncols].transAxes,
                                       fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        if normalize:
            xlims, ylims = [-1.1, 1.1], [-1.1, 1.1]
        else:
            xlims, ylims = calculate_plot_limits(residual, variance)
        ax[i // ncols, i % ncols].set_xlim(xlims)
        ax[i // ncols, i % ncols].set_ylim(ylims)
        # plot x=y
        ax[i // ncols, i % ncols].plot(xlims, ylims, 'k--', label="x=y")
        # add legend
        ax[i // ncols, i % ncols].legend(loc='lower right')

        # if this will be the last subplot in a column, and i is not in the final row add xticks
        if n_plots < (ncols * nrows) and i // nrows == nrows - 2 and i % ncols == ncols - 1:
            ax[i // ncols, i % ncols].xaxis.set_tick_params(labelbottom=True)

    # add x and y labels
    fig.text(0.5, -0.01, f"variance partitioning predicted - true contribution{', normalized' if normalize else ''}",
             ha='center', fontsize=15)
    fig.text(-0.01, 0.5, f"residual method predicted - true contribution{', normalized' if normalize else ''}",
             rotation='vertical', va='center', fontsize=15)

    # remove empty subplots
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(ax.flatten()[i])
    # create additional plot for text containing variable information
    fig.text(1, 0.5, '\n'.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()]), ha='left', va='center',
             fontsize=10)

    # Adjust layout to increase margins
    plt.tight_layout()
    plt.show()


def calculate_plot_limits(residual, variance):
    variance_lower_perc = np.percentile(variance, 5)
    variance_upper_perc = np.percentile(variance, 95)
    variance_range = variance_upper_perc - variance_lower_perc
    xlims = [
        variance_lower_perc + 0.1 * variance_range if variance_lower_perc < 0 else variance_lower_perc - 0.1 * variance_range,
        variance_upper_perc + 0.1 * variance_range if variance_upper_perc > 0 else variance_upper_perc - 0.1 * variance_range]
    residual_lower_perc = np.percentile(residual, 5)
    residual_upper_perc = np.percentile(residual, 95)
    residual_range = residual_upper_perc - residual_lower_perc
    ylims = [
        residual_lower_perc + 0.1 * residual_range if residual_lower_perc < 0 else residual_lower_perc - 0.1 * residual_range,
        residual_upper_perc + 0.1 * residual_range if residual_upper_perc > 0 else residual_upper_perc - 0.1 * residual_range]
    return xlims, ylims


def plot_experiment(variable_values, variable_name, predicted_variance, predicted_residual,
                    unique_contributions, x_is_log=False, **kwargs):
    plot_predicted_contributions_box(variable_values, variable_name, predicted_variance, predicted_residual,
                                     unique_contributions, x_is_log=x_is_log, **kwargs)
    plot_prediction_error(variable_values, variable_name, predicted_variance, predicted_residual,
                          unique_contributions, x_is_log=x_is_log, **kwargs)
    plot_prediction_scatter(variable_values, variable_name, predicted_variance, predicted_residual,
                            unique_contributions, **kwargs)
