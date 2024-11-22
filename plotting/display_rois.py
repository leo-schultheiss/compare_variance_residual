import numpy as np
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

import os
import voxelwise_tutorials.viz as viz

from robustness_test.common_utils.training_utils import get_prediction_path

language_model = "bert"
feature = "low-level"
modality = "listening"
subject = "01"
layer = 9
low_level_feature = "phonemes"

path = get_prediction_path(language_model, feature, modality, subject, low_level_feature, layer)
correlation = np.nan_to_num(np.load(path, allow_pickle=True))
mapper = os.path.join("../data", f"subject{subject}_mappers.hdf")

flatmap_mask = load_hdf5_array(mapper, key='flatmap_mask')
figsize = np.array(flatmap_mask.shape) / 100.
fig = plt.figure(figsize=figsize)
ax = fig.add_axes((0, 0, 1, 1))
ax.axis('off')
viz.plot_flatmap_from_mapper(correlation, mapper, ax=ax, with_curvature=False, alpha=1, cmap='inferno', vmin=0, vmax=np.max(correlation))

layer_info = "" if feature == "low-level" else f" - layer {layer}"
low_level_info = "" if feature == "semantic" else f" - {low_level_feature}"
figure_name = f"{feature} - subject {subject} - {modality}{low_level_info}{layer_info}"

fig.suptitle(figure_name)
fig.savefig(f"../plots/{figure_name}")
fig.show()
