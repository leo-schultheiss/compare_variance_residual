import os

import numpy as np

from robustness_test.common_utils.visualization import flatmap_subject_voxel_data, plot_model_comparison_rois, plot_model_histogram_comparison

modality = "listening"
subject = "01"
layer = 9
low_level_feature = "letters"

standard_correlation = np.load(os.path.join("../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True)
low_level_correlation = np.load(os.path.join("../bert-low-level-predictions", modality, subject, f"{low_level_feature}.npy"), allow_pickle=True)
joint_correlation = np.load(os.path.join("../bert-joint-predictions", modality, subject, low_level_feature, f"layer_{layer}.npy"), allow_pickle=True)
intersection = np.load("../bert-variance-partitioning/intersection.npy", allow_pickle=True)
standard_minus_low_level = np.load("../bert-variance-partitioning/variance_a_minus_b.npy", allow_pickle=True)
low_level_minus_standard = np.load("../bert-variance-partitioning/variance_b_minus_a.npy", allow_pickle=True)

# relace nans with zeros
standard_correlation = np.nan_to_num(standard_correlation)
low_level_correlation = np.nan_to_num(low_level_correlation)
joint_correlation = np.nan_to_num(joint_correlation)
intersection = np.nan_to_num(intersection)
standard_minus_low_level = np.nan_to_num(standard_minus_low_level)
low_level_minus_standard = np.nan_to_num(low_level_minus_standard)

plot_model_histogram_comparison(standard_correlation, joint_correlation, "semantic", "joint")
