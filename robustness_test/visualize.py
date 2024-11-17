import os

import numpy as np

from robustness_test.common_utils.visualization import flatmap_subject_voxel_data, plot_model_comparison_rois, plot_model_histogram_comparison

modality = "reading"
subject = "01"
layer = 11
low_level_feature = "letters"

standard_correlation = np.load(os.path.join("../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True)
low_level_correlation = np.load(os.path.join("../bert-low-level-predictions", modality, subject, f"{low_level_feature}.npy"), allow_pickle=True)
joint_correlation = np.load(os.path.join("../bert-joint-predictions", modality, subject, low_level_feature, f"layer_{layer}.npy"), allow_pickle=True)
intersection = np.load("../bert-variance-partitioning/intersection.npy", allow_pickle=True)
standard_minus_low_level = np.load("../bert-variance-partitioning/variance_a_minus_b.npy", allow_pickle=True)
low_level_minus_standard = np.load("../bert-variance-partitioning/variance_b_minus_a.npy", allow_pickle=True)

print(f"minimum intersection: {np.sum(np.nan_to_num(intersection) == 0)}")

flatmap_subject_voxel_data(subject, intersection)
flatmap_subject_voxel_data(subject, standard_minus_low_level)
flatmap_subject_voxel_data(subject, low_level_minus_standard)

plot_model_histogram_comparison(standard_correlation, standard_minus_low_level, "semantic", "semantic \ low level")

