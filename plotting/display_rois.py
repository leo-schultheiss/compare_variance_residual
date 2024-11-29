import os

import numpy as np
import voxelwise_tutorials.viz as viz
from matplotlib import pyplot as plt
from voxelwise_tutorials.io import load_hdf5_array

from robustness_test.common_utils.feature_utils import get_prediction_path


def display_flatmap(language_model: str, feature: str, modality: str, subject: int, low_level_feature=None, layer=None,
                    title = None, save_fig: bool = False):
    path = get_prediction_path(language_model, feature, modality, subject, low_level_feature, layer)
    correlation = np.nan_to_num(np.load(path, allow_pickle=True))
    mapper_path = os.path.join("../data", f"subject{subject:02}_mappers.hdf")

    flatmap_mask = load_hdf5_array(mapper_path, key='flatmap_mask')
    figsize = np.array(flatmap_mask.shape) / 100.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis('off')
    viz.plot_flatmap_from_mapper(correlation, mapper_path, ax=ax, with_curvature=False, alpha=1, cmap='inferno',
                                 vmin=0,
                                 vmax=np.max(correlation))

    fig.suptitle(title)
    if save_fig:
        fig.savefig(f"../plots/flatmap_{path.replace(os.path.sep, '_')}.png")
    fig.show()


if __name__ == "__main__":
    language_model = "bert"
    feature = "semantic"
    modality = "reading"
    subject = 1
    layer = 9
    low_level_feature = "letters"

    title = f"{feature} stimulus - subject {subject} - {modality} - layer {layer}, alphas = grid search of 10 values from 10^0 to 10^4"
    display_flatmap(language_model, feature, modality, subject, low_level_feature, layer, title, True)
