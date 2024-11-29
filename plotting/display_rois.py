import os

import numpy as np
import voxelwise_tutorials.viz as viz
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

from robustness_test.common_utils.feature_utils import get_prediction_path

language_model = "bert"
feature = "semantic"
modality = "reading"
subject = 1
layer = 9
low_level_feature = "phonemes"

path = get_prediction_path(language_model, feature, modality, subject, low_level_feature, layer)
correlation = np.nan_to_num(np.load(path, allow_pickle=True))
mapper_path = os.path.join("../data", f"subject{subject:02}_mappers.hdf")

flatmap_mask = load_hdf5_array(mapper_path, key='flatmap_mask')
figsize = np.array(flatmap_mask.shape) / 100.
fig = plt.figure(figsize=figsize)
ax = fig.add_axes((0, 0, 1, 1))
ax.axis('off')
viz.plot_flatmap_from_mapper(correlation, mapper_path, ax=ax, with_curvature=False, alpha=1, cmap='inferno',
                             vmin=np.min(correlation),
                             vmax=np.max(correlation))

layer_info = "" if feature == "low-level" else f" - layer {layer}"
low_level_info = "" if feature == "semantic" else f" - {low_level_feature}"
figure_name = f"{feature} - subject {subject} - {modality}{low_level_info}{layer_info}"

fig.suptitle(path)
fig.savefig(f"../plots/flatmap_{path.replace(os.path.sep, '_')}.png")
fig.show()
