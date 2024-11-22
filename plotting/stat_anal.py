import numpy as np
from matplotlib import pyplot as plt

import os

language_model = "bert"
feature = "semantic"  # semantic low-level joint
modality = "listening"
subject = "01"
layer = 9
low_level_feature = "phonemes"


def correlation_stat_analysis(correlation, savefig=False):
    # print statistics
    print("mean", np.mean(correlation))
    print("median", np.median(correlation))
    print("std", np.std(correlation))
    print("min", np.min(correlation))
    print("max", np.max(correlation))
    print("sum", np.sum(correlation))
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
    # Load the data
    filename = f"{low_level_feature}.npy" if feature == "low-level" else f"layer_{layer}.npy"
    joint_path_addition = f"{low_level_feature}" if feature == "joint" else ""
    path = os.path.join(f"../{language_model}-{feature}-predictions", modality, subject, joint_path_addition, filename)
    print("loading", path)
    correlation = np.nan_to_num(np.load(path, allow_pickle=True))
    correlation_stat_analysis(correlation)
