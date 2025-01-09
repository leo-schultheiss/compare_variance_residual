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

    ax.set_title(xlabel)
    fig.suptitle("Variance partitioning vs residual method")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("predicted contribution")
    ax.set_ylim([0 , 1.])
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
