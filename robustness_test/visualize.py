import os

import matplotlib.pyplot as plt
import numpy as np

from common_utils.hdf_utils import map_to_flat

def flatmap_subject_voxel_data(subject_number, data):
    # Get path to data files
    fdir = os.path.abspath('../data')
    # Map to subject flatmap
    map_file = os.path.join(fdir, f'subject{subject_number}_mappers.hdf')
    flatmap = map_to_flat(data, map_file)
    # Plot flatmap
    fig, ax = plt.subplots()
    _ = ax.imshow(flatmap, cmap='inferno')
    ax.axis('off')
    # add legend


    plt.show()
    return fig


modality = "reading"
subject = "01"
layer = 11
low_level_feature = "letters"
delays = 6

os.makedirs(f"../plots/{modality}/{subject}", exist_ok=True)

for delays in range(1, 10):
    low_level_correlation = np.load(os.path.join("../bert-low-level-predictions", modality, subject, low_level_feature, f"low_level_model_prediction_voxelwise_correlation_delays{delays}.npy"), allow_pickle=True)
    flatmap_subject_voxel_data(subject, low_level_correlation)
    low_level_correlation = np.nan_to_num(low_level_correlation)
    print(f"average correlation: {np.mean(low_level_correlation)}")
    print(f"max correlation: {np.max(low_level_correlation)}")


joint_correlation = np.load(os.path.join("../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True)
standard_correlation = np.load(os.path.join("../bert-joint-predictions", subject, modality, low_level_feature, f"joint_model_prediction_voxelwise_correlation_layer{layer}.npy"), allow_pickle=True)
residual_correlation = np.load(os.path.join("../bert-residuals", modality, "numwords", subject, f"layer_{layer}.npy"), allow_pickle=True)
joint_correlation = np.nan_to_num(joint_correlation)
standard_correlation = np.nan_to_num(standard_correlation)
residual_correlation = np.nan_to_num(residual_correlation)


# calculate average difference between the two
# diff = standard_correlation - joint_correlation
# print(f"Average difference between the two models: {np.mean(diff)}")
# print(f"maximum difference between the two models: {np.max(diff)}")
# print(f"minimum difference between the two models: {np.min(np.abs(diff))}")
# print(f"number of voxels with difference equal to 0.0: {np.sum(diff == 0.0)}")
# print(f"mean of the first model: {np.mean(standard_correlation)}")
# print(f"mean of the second model: {np.mean(joint_correlation)}")
#
# fig = flatmap_subject_voxel_data(subject, standard_correlation)
# fig.savefig(f"../plots/{modality}/{subject}/standard_correlation_layer_{layer}.png")
#
# fig = flatmap_subject_voxel_data(subject, joint_correlation)
# fig.savefig(f"../plots/{modality}/{subject}/joint_correlation_layer_{layer}.png")
#
# # fig = flatmap_subject_voxel_data(subject, residual_correlation)
# # fig.savefig(f"../plots/{modality}/{subject}/residual_correlation_layer_{layer}.png")
#
# fig = flatmap_subject_voxel_data(subject, standard_correlation - joint_correlation)
# fig.savefig(f"../plots/{modality}/{subject}/difference_standard_joint_correlation_layer_{layer}.png")