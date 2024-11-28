import numpy as np
from matplotlib import pyplot as plt

import os

import robustness_test.common_utils.training_utils


def correlation_stat_analysis(correlation, savefig=False, alternate_title=None):
    # print statistics
    print("number of voxels", len(correlation))
    print("mean", np.mean(correlation))
    print("std", np.std(correlation))
    print("min", np.min(correlation))
    print("max", np.max(correlation))
    print("median", np.median(correlation))
    # plot histogram with inferno cmap
    # This is the colormap I'd like to use.
    cm = plt.colormaps['inferno']
    # Plot histogram.
    n, bins, patches = plt.hist(correlation, 50, color='green')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    if alternate_title:
        plt.suptitle(alternate_title)
    else:
        if feature != "semantic":
            plt.suptitle(f"{feature} {modality} {low_level_feature}")
        else:
            plt.suptitle(f"{feature} {modality}")
    plt.xlabel("Correlation")
    plt.ylabel("Number of Voxels")
    left, right = plt.xlim()
    plt.xlim(-0.27, 0.610)
    plt.ylim(0, 7300)
    # add mean vertical line
    plt.axvline(x=np.mean(correlation), color='r', linestyle='dashed', linewidth=1)
    if savefig:
        plt.savefig(
            f"../plots/correlation_histogram_{language_model}_{feature}_{modality}_sub{subject}_{low_level_feature}_layer{layer}.png")
    plt.show()
    print()


if __name__ == "__main__":
    language_model = "bert"
    feature = "semantic"  # semantic low-level joint
    modality = "reading"
    subject = 1
    layer = 9
    low_level_feature = "letters"

    # Load the data
    path = robustness_test.common_utils.training_utils.get_prediction_path(language_model, feature, modality, subject, layer=layer)
    print("loading", path)
    correlation = np.nan_to_num(np.load(path, allow_pickle=True))
    correlation_stat_analysis(correlation)
