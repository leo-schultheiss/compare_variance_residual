import numpy as np
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

from robustness_test.common_utils.visualization import plot_model_histogram_comparison, plot_model_overunder_comparison

import os
import voxelwise_tutorials.viz as viz

language_model = "bert"
modality = "listening"
subject = "01"
layer = 9
low_level_feature = "word_length_std"
mapper = os.path.join("../data", f"subject{subject}_mappers.hdf")

feature_1 = "joint" # semantic low-level joint
feature_2 = "semantic"


def file_path(feature):
    file_name = f"{low_level_feature}.npy" if feature == "low-level" else f"layer_{layer}.npy"
    joint_path_addition = f"{low_level_feature}" if feature == "joint" else ""
    return os.path.join(f"../{language_model}-{feature}-predictions", modality, subject, joint_path_addition, file_name)


path_1 = file_path(feature_1)
path_2 = file_path(feature_2)

correlation_1 = np.nan_to_num(np.load(path_1, allow_pickle=True))
correlation_2 = np.nan_to_num(np.load(path_1, allow_pickle=True))

diff = correlation_1 - correlation_2

flatmap_mask = load_hdf5_array(mapper, key='flatmap_mask')
figsize = np.array(flatmap_mask.shape) / 100.
fig = plt.figure(figsize=figsize)
ax = fig.add_axes((0, 0, 1, 1))
ax.axis('off')
viz.plot_flatmap_from_mapper(diff, mapper, ax=ax, with_curvature=False, alpha=1, cmap='inferno')

title = f"{feature_1} - {feature_2}"
fig.suptitle(title)
fig.show()
fig.savefig(f"../plots/{title}")
