import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

PLOT_HEIGHT = 6

plt.style.use('nord-light-talk')


def normalize_to_unit_interval(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return np.nan_to_num(scaled_array)


def format_variable_values(variable_values):
    # turn floats into two point precision
    if isinstance(variable_values, float):
        return f"{variable_values:.2f}"
    if isinstance(variable_values, list) and isinstance(variable_values[0], float):
        return [f"{v:.2f}" for v in variable_values]
    if isinstance(variable_values[0], (np.ndarray, list)) and isinstance(variable_values[0][0], float):
        return [",".join([f"{v:.2f}" for v in l]) for l in variable_values]
    return variable_values


def create_text(**kwargs):
    return '\n'.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()])


def calculate_plot_limits(residual, variance):
    variance_lower_perc, variance_upper_perc = np.percentile(variance, [5, 95])
    variance_range = variance_upper_perc - variance_lower_perc
    xlims = [
        variance_lower_perc - 0.1 * variance_range,
        variance_upper_perc + 0.1 * variance_range
    ]
    xlims = [min(xlims[0], -0.1 * variance_range), max(xlims[1], 0.1 * variance_range)]

    residual_lower_perc, residual_upper_perc = np.percentile(residual, [5, 95])
    residual_range = residual_upper_perc - residual_lower_perc
    ylims = [
        residual_lower_perc + 0.1 * residual_range if residual_lower_perc < 0 else residual_lower_perc - 0.1 * residual_range,
        residual_upper_perc + 0.1 * residual_range if residual_upper_perc > 0 else residual_upper_perc - 0.1 * residual_range]
    ylims = [min(ylims[0], - 0.1 * residual_range), max(ylims[1], 0.1 * residual_range)]
    return xlims, ylims


def widths_and_posisions(ax, results, x, x_is_log):
    if x_is_log:
        w = 1 / (1.5 * len(results))
        width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
        positions = lambda i: [10 ** (np.log10(pos) + i * w - (len(results) - 1) * w / 2) for pos in x]
    else:
        if isinstance(x[0], (int, float)):
            min_x, max_x = ax.get_xlim()
            w = (max_x - min_x) / (3 * len(results))
            positions = lambda i: [pos + i * w - (len(results) - 1) * w / 2 for pos in x]
        else:
            w = 1 / 8
            positions = lambda i: [pos + i * w - (len(results) - 1) * w / 2 for pos in range(len(results[0]))]
        width = lambda _, w: w
    return positions, w, width


def set_x_axis(ax, results, w, x, x_is_log):
    if isinstance(x[0], (int, float)):
        ax.set_xticks(x)
        ax.set_xticklabels(format_variable_values(x))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_xlim([10 ** (np.log10(x[0]) - w * (1 + len(results) / 2)),
                     10 ** (np.log10(x[-1]) + w * (1 + len(results) / 2))]) if x_is_log else ax.set_xlim(
            [x[0] - w * (1 + len(results) / 2), x[-1] + w * (1 + len(results) / 2)])
    else:
        ax.set_xlim([-0.5, len(x) - 0.5])
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(format_variable_values(x))


def plot_predicted_variances_box(xlabel, x, results, names,
                                 x_is_log=False, save_dir=None, **kwargs):
    title = "Predicted Variance"
    ylabel = "Predicted Variance"
    ylim = [-0.05, 0.5]

    figure_width = 2 * len(x)
    fig, ax = plt.subplots(figsize=(figure_width, PLOT_HEIGHT))
    positions, w, width = widths_and_posisions(ax, results, x, x_is_log)

    # Plot variance partitioning
    medianprops = dict(color='black')
    for i, (result, name) in enumerate(zip(results, names)):
        ax.boxplot(result, positions=positions(i), widths=width(positions(i), w), patch_artist=True,
                   boxprops=dict(facecolor=f"C{i}"), medianprops=medianprops, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(ylim)

    set_x_axis(ax, results, w, x, x_is_log)

    if x_is_log:
        ax.set_xscale("log")

    # draw center line
    if xlabel == "Unique Variance Explained":
        # only draw horizontal line for 1/len(x)
        for i in range(len(x)):
            true_variance = x[i][1]
            # determine x coordinates
            x_min, x_max = ax.get_xlim()
            plot_width = x_max - x_min
            plot_sector = plot_width / len(x)
            xs = [
                x_min + i * plot_sector,
                x_min + (i + 1) * plot_sector
            ]
            ax.plot(xs, [true_variance, true_variance], color='k', linestyle='--')
        # add label for true variancde
        ax.plot([0, 0], [0, 0], color='k', linestyle='--', label=f"true contribution")
    else:
        ax.axhline(y=kwargs["scalars"][1], color='k', linestyle='--', label='true variance')

    # Add text field with variable information
    variable_info = create_text(**kwargs)
    fig.text(1, 0.5, variable_info, ha='left', va='center', fontsize=10)

    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{xlabel}_predicted_variances.png"))

    plt.show()


def plot_mse(variable_name, variable_values, results: list, names: list, scalars,
             x_is_log=False, save_dir=None, **kwargs):
    figure_width = 2 * len(variable_values)
    fig, ax = plt.subplots(figsize=(figure_width, PLOT_HEIGHT))

    # calculate and plot mse for each variable
    if not variable_name == "Unique Variance Explained":
        results_mse = [[np.mean((np.array(predicted_var) - scalars[1]) ** 2) for predicted_var in result] for result in
                       results]
    else:
        true_variances = [row[1] for row in variable_values]
        results_mse = [[np.mean((np.array(predicted) - true) ** 2) for predicted, true in
                        zip(predicted_var, true_variances)] for predicted_var in results]

    _, w, _ = widths_and_posisions(ax, results_mse, variable_values, x_is_log)
    positions = variable_values if isinstance(variable_values[0], (int, float)) else np.arange(
        len(variable_values))

    for i, (result, name) in enumerate(zip(results_mse, names)):
        ax.plot(positions, result, label=name, color=f"C{i}")

    # plot y=0
    ax.axhline(y=0, color='k', linestyle='-')

    ax.set_title("Mean Squared Error")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("MSE")
    max_total = max([np.max(result_mse) for result_mse in results_mse])

    if max_total > 1:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.LogFormatter())
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1])  # Set explicit bottom limit and retain current top limit
    else:
        ylims = ax.get_ylim()
        ax.set_ylim([-0.1 * ylims[1], ylims[1]])  # Only adjust for linear scale

    set_x_axis(ax, variable_values, w, variable_values, x_is_log)
    if x_is_log:
        ax.set_xscale("log")

    # create additional plot for text containing variable information
    fig.text(1, 0.5, create_text(**kwargs), ha='left', va='center',
             fontsize=10)

    ax.legend().set_loc('upper left')
    plt.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{variable_name}_mse.png"))
    plt.show()


# def plot_prediction_scatter(xlabel, x, results, names,
#                             normalize=False, ignore_outliers=False, save_dir=None, **kwargs):
#     """
#     create scatter plots of predicted variance vs predicted residual to show correlation
#     """
#     # remove 5th percentile and 95th percentile to ignore outliers
#     if ignore_outliers:
#         predicted_variance = [np.clip(variance, np.percentile(variance, 5), np.percentile(variance, 95)) for variance in
#                               predicted_variance]
#         predicted_residual = [np.clip(residual, np.percentile(residual, 5), np.percentile(residual, 95)) for residual in
#                               predicted_residual]
#
#     # center data around true variance
#     if xlabel == "Unique Variance Explained":
#         true_variances = [row[1] for row in x]
#         predicted_variance = [np.array(variance) - true_variance for variance, true_variance in zip(predicted_variance, true_variances)]
#         predicted_residual = [np.array(residual) - true_variance for residual, true_variance in zip(predicted_residual, true_variances)]
#         true_variance = max(true_variances)
#     else:
#         true_variance = kwargs["scalars"][1]
#         predicted_variance = list(np.array(predicted_variance) - true_variance)
#         predicted_residual = list(np.array(predicted_residual) - true_variance)
#
#     if normalize:
#         predicted_variance = [normalize_to_unit_interval(variance) for variance in predicted_variance]
#         predicted_residual = [normalize_to_unit_interval(residual) for residual in predicted_residual]
#
#     # Calculate the number of rows and columns needed
#     n_plots = len(predicted_variance)
#     # ncols = int(np.ceil(np.sqrt(n_plots)))
#     # nrows = int(np.ceil(n_plots / ncols))
#     nrows = 1
#     ncols = n_plots
#     fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5), squeeze=False, sharex=True,
#                            sharey=True)
#     # add title to the figure
#     fig.suptitle(f"Error",
#                  fontsize=20, y=1.0)
#
#     for i, (variance, residual) in enumerate(zip(predicted_variance, predicted_residual)):
#         ax[i // ncols, i % ncols].scatter(variance, residual, alpha=0.5)
#         title = f"{xlabel}: {format_variable_values(x)[i]}"
#         ax[i // ncols, i % ncols].set_title(title)
#
#         ax[i // ncols, i % ncols].set_xlabel("predicted - true variance")
#         if i % ncols == 0:
#             ax[i // ncols, i % ncols].set_ylabel("predicted - true variance")
#
#         # add text box that displays the correlation coefficient
#         corr = np.corrcoef(variance, residual)[0, 1]
#         ax[i // ncols, i % ncols].text(0.05, 0.95, rf"$\rho$: {corr:.2f}",
#                                        transform=ax[i // ncols, i % ncols].transAxes,
#                                        fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
#
#         xlims = [-1.1 * true_variance, 1.1 * true_variance]
#         ylims = [-1.1 * true_variance, 1.1 * true_variance]
#
#         ax[i // ncols, i % ncols].set_xlim(xlims)
#         ax[i // ncols, i % ncols].set_ylim(ylims)
#         # plot x=y
#         ax[i // ncols, i % ncols].plot([-10000, 10000], [-10000, 10000], 'k--', label="x=y")
#         # plot X=0, and y=0
#         ax[i // ncols, i % ncols].axvline(x=0, color='k', linestyle='-')
#         ax[i // ncols, i % ncols].axhline(y=0, color='k', linestyle='-')
#         # add legend
#         ax[i // ncols, i % ncols].legend(loc='lower right')
#
#         # if this will be the last subplot in a column, and i is not in the final row add xticks
#         if n_plots < (ncols * nrows) and i // nrows == nrows - 2 and i % ncols == ncols - 1:
#             ax[i // ncols, i % ncols].xaxis.set_tick_params(labelbottom=True)
#
#     # add x and y labels
#     fig.text(0.5, 0, f"Variance Partitioning", ha='center', va='bottom', fontsize=13)
#     fig.text(0, 0.5, f"Residual Method", rotation='vertical', va='center', ha='left', fontsize=13)
#
#     # remove empty subplots
#     for i in range(n_plots, nrows * ncols):
#         fig.delaxes(ax.flatten()[i])
#     # create additional plot for text containing variable information
#     fig.text(1, 0.5, create_text(**kwargs), ha='left', va='center',
#              fontsize=10)
#
#     # Adjust layout to increase margins
#     plt.tight_layout()
#
#     if save_dir is not None:
#         plt.savefig(os.path.join(save_dir, f"{xlabel}_predicted_variances_scatter.png"), pad_inches=1)
#     plt.show()


def plot_experiment(variable_name, variable_values, results, names,
                    x_is_log=False, save_dir=None, **kwargs):
    plot_predicted_variances_box(variable_name, variable_values, results, names,
                                 x_is_log=x_is_log, save_dir=save_dir, **kwargs)
    plot_mse(variable_name, variable_values, results, names,
             x_is_log=x_is_log, save_dir=save_dir, **kwargs)
    # plot_prediction_scatter(variable_name, variable_values, results, names,
    #                         save_dir=save_dir, **kwargs)


def plot_variance_partitioning_results(scalars, scores):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.barplot(
        data=scores,
        palette=["C0", "C0", "C0", "C0", "C1", "C1"],  # Define colors: one for joint/X₁/X₂, one for unique/shared
        ax=ax
    )
    sns.despine(fig)
    theoretical_scores = [sum(scalars), scalars[0] + scalars[1], scalars[0] + scalars[2], scalars[0], scalars[1],
                          scalars[2]]

    # Add lines indicating the maximum possible height for each bar
    for idx, column in enumerate(scores.columns):  # iterate over rows in the DataFrame
        xmin = idx / len(scores.columns)  # Calculate xmin for each bar
        xmax = (idx + 1) / len(scores.columns)  # Calculate xmax for each bar
        plt.axhline(theoretical_scores[idx], linestyle='--', alpha=0.7,
                    xmin=xmin, xmax=xmax, label=fr'Theoretically Explained Variance' if idx == 0 else "")

        plt.text(idx, scores[column].mean() / 2,
                 f"{scores[column].mean():.2f}", ha='center', va='bottom')

    plt.xticks(range(len(scores.columns)), scores.columns, rotation=45)

    plt.ylim(0, 1.05)

    # Ensure the legend is displayed properly
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel(r"Variance Explained (avg. $R^2$ across targets)")
    plt.xlabel(r"Ridge Regression / Variance Partitioning")
    plt.title("Variance Partitioning")
    return fig, ax


def plot_residual_method_results(scalars, d_list, scores):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.barplot(
        data=scores,
        ax=ax,
        palette=["C0", "C0", "C0", "C0", "C1", "C1"]
    )

    sns.despine()

    # Add lines indicating the maximum possible height for each bar
    theoretical_scores = [
        scalars[0] + scalars[1],
        scalars[0] + scalars[2],
        d_list[0] / (d_list[0] + d_list[1]),
        d_list[0] / (d_list[0] + d_list[2]),
        scalars[1],
        scalars[2],
    ]

    for idx, column in enumerate(scores.columns):  # iterate over rows in the DataFrame
        # Ensure index compatibility
        if column not in scores:
            print(f"Column '{column}' is not present in DataFrame. Available columns are: {list(scores.columns)}")
            continue

        xmin = idx / len(scores.columns)  # Calculate xmin for each bar
        xmax = (idx + 1) / len(scores.columns)  # Calculate xmax for each bar
        plt.axhline(theoretical_scores[idx], linestyle='--', alpha=0.7, xmin=xmin, xmax=xmax,
                    label='Theoretically Explained Variance' if idx == 0 else "")

        # Use the mean of the column for positioning text
        mean_column_value = scores[column].mean()
        plt.text(idx, mean_column_value / 2,
                 f"{mean_column_value:.2f}", ha='center', va='center')

    # make xtickmarks diagonal
    plt.xticks(range(len(scores.columns)), scores.columns, rotation=45)

    plt.ylim(0, 1)

    # Ensure the legend is displayed properly
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel(r"Variance Explained (avg. $R^2$ across targets)")
    plt.xlabel("Ridge Regression / Residual Method")
    plt.title("Residual Method")
    return fig, ax


def plot_variance_vs_residual_joint(scalars, error_data):
    fig = plt.figure(figsize=(20, 10))

    range_min = min(error_data["VP Error"].min(), error_data["Residual Error"].min())
    range_max = max(error_data["VP Error"].max(), error_data["Residual Error"].max())
    # add 5% padding
    range_min -= 0.1 * abs(range_min)
    range_max += 0.1 * abs(range_max)

    sns.jointplot(
        data=error_data,
        x="VP Error",  # Variance Partitioning Error
        y="Residual Error",  # Residual Method Error
        hue="Feature Space",
        alpha=0.8,
        kind="scatter",
        xlim=(range_min, range_max),
        ylim=(range_min, range_max),
    )

    # make x and y tick labels black
    plt.tick_params(axis="both", which="major", labelcolor="black")

    # draw 0 lines
    plt.axhline(0, color="black", linestyle="-")
    plt.axvline(0, color="black", linestyle="-")

    # plot x=y line
    plt.plot([-1, 1], [-1, 1], linestyle="--", label="y=x")

    plt.legend()
    plt.suptitle(r"Estimated Unique Contribution per Target ($R^2$)", y=1.05)
    plt.title(rf"$a_\mathbf{{A}}={scalars[0]:.2f}, a_\mathbf{{B}}={scalars[1]:.2f}, a_\mathbf{{C}}={scalars[2]:.2f}$",
              y=1.2)
    plt.xlabel("Variance Partitioning Error")
    plt.ylabel("Residual Method Error")
    plt.show()


if __name__ == '__main__':
    test = [[1.333, 2.4444444], [1 / 3, 2 / 3]]
    print(format_variable_values(test))

    test = [["string1", "string2"], ["string3", "string4"]]
    print(format_variable_values(test))

    test = [1 / 3, 2 / 3]
    print(format_variable_values(test))
