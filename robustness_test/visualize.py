import numpy as np
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

import os
from robustness_test.common_utils.visualization import flatmap_subject_voxel_data
import voxelwise_tutorials.viz as viz

language_model = "bert"
modality = "reading"
subject = "01"
layer = 9
feature = "low-level" # "low-level" "joint"
low_level_feature = "numletters"


filename = f"{low_level_feature}.npy" if feature == "low-level" else f"layer_{layer}.npy"
joint_path_addition = f"{low_level_feature}" if feature == "joint" else ""
path = os.path.join(f"../{language_model}-{feature}-predictions", modality, subject, joint_path_addition, filename)
print("loading", path)
correlation = np.load(path, allow_pickle=True)
mapper = os.path.join("../data", f"subject{subject}_mappers.hdf")

# flatmap_subject_voxel_data(subject_number=subject, data=correlation)

flatmap_mask = load_hdf5_array(mapper, key='flatmap_mask')
figsize = np.array(flatmap_mask.shape) / 100.
fig = plt.figure(figsize=figsize)
ax = fig.add_axes((0, 0, 1, 1))
ax.axis('off')
viz.plot_flatmap_from_mapper(correlation, mapper, ax=ax)

fig.suptitle(f"{feature} - Subject {subject} - Layer {layer} - {modality} - {low_level_feature}")
fig.show()