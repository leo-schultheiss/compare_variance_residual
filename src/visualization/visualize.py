import os

import matplotlib.pyplot as plt
import numpy as np

from common_utils.hdf_utils import map_to_flat

def flatmap_subject_voxel_data(subject_number, data):
    # Get path to data files
    fdir = os.path.abspath('../../data')
    # Map to subject flatmap
    map_file = os.path.join(fdir, f'subject{subject_number}_mappers.hdf')
    flatmap = map_to_flat(data, map_file)
    # Plot flatmap
    fig, ax = plt.subplots()
    _ = ax.imshow(flatmap, cmap='inferno')
    ax.axis('off')
    plt.show()
    return fig


modality = "reading"
subject = "01"


for layer in range(12):
    joint_correlation = np.load(os.path.join("../../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True)
    standard_correlation = np.load(os.path.join("../../bert-joint-predictions", subject, modality, "letters", f"joint_model_prediction_voxelwise_correlation_layer{layer}.npy"), allow_pickle=True)
    residual_correlation = np.load(os.path.join("../../bert-residuals", modality, "numwords", subject, f"layer_{layer}.npy"), allow_pickle=True)

    # calculate average difference between the two
    diff = np.nan_to_num(standard_correlation) - np.nan_to_num(joint_correlation)
    print(f"Average difference between the two models: {np.mean(diff)}")
    print(f"maximum difference between the two models: {np.max(diff)}")
    print(f"minimum difference between the two models: {np.min(np.abs(diff))}")
    print(f"number of voxels with difference equal to 0.0: {np.sum(diff == 0.0)}")
    print(f"mean of the first model: {np.mean(np.nan_to_num(standard_correlation))}")
    print(f"mean of the second model: {np.mean(np.nan_to_num(joint_correlation))}")

    fig = flatmap_subject_voxel_data(subject, joint_correlation)
    os.makedirs(f"../../plots/{modality}/{subject}", exist_ok=True)
    fig.savefig(f"../../plots/{modality}/{subject}/joint_correlation_layer_{layer}.png")