import numpy as np
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

import os
import voxelwise_tutorials.viz as viz

language_model = "bert"
modality = "listening"
subject = "01"
layer = 9
low_level_feature = "phonemes"

path = f"../{language_model}-variance-partitioning/{modality}/{subject}/{low_level_feature}/{layer}/"
intersection = np.nan_to_num(np.load(os.path.join(path, "intersection.npy"), allow_pickle=True))
low_minus_semantic = np.nan_to_num(np.load(os.path.join(path, "low_minus_semantic.npy"), allow_pickle=True))
semantic_minus_low = np.nan_to_num(np.load(os.path.join(path, "semantic_minus_low.npy"), allow_pickle=True))

mapper = os.path.join("../data", f"subject{subject}_mappers.hdf")

flatmap_mask = load_hdf5_array(mapper, key='flatmap_mask')
figsize = np.array(flatmap_mask.shape) / 100.
fig = plt.figure(figsize=figsize)
ax = fig.add_axes((0, 0, 1, 1))
ax.axis('off')

for correlation in [intersection, low_minus_semantic, semantic_minus_low]:

    viz.plot_flatmap_from_mapper(correlation, mapper, ax=ax, with_curvature=False, alpha=1, cmap='inferno', vmin=np.min(correlation), vmax=np.max(correlation))

    graphic = "intersection" if correlation is intersection else "low_minus_semantic" if correlation is low_minus_semantic else "semantic_minus_low"

    figure_name = f"{graphic} - subject {subject} - {modality} - {low_level_feature} - layer {layer}"

    fig.suptitle(figure_name)
    fig.savefig(f"../plots/{figure_name}")
    fig.show()
