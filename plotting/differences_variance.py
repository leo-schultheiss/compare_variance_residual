import numpy as np
from robustness_test.common_utils.training_utils import get_prediction_path
from stat_anal import correlation_stat_analysis
import os

language_model = "bert"
modality = "listening"
subject = "01"
layer = 9
low_level_feature = "phonemes"
mapper = os.path.join("../data", f"subject{subject}_mappers.hdf")

# Load the data
semantic = np.load(get_prediction_path(language_model, "semantic", modality, subject, layer=layer), allow_pickle=True)
semantic_minus_low_level = np.load(
    f"../bert-variance-partitioning/{modality}/{subject}/{low_level_feature}/{layer}/intersection.npy",
    allow_pickle=True)
semantic = np.nan_to_num(semantic)
# if any value is negative, set it to 0
semantic = np.maximum(semantic, 0)
semantic_minus_low_level = np.nan_to_num(semantic_minus_low_level)

correlation_stat_analysis(semantic)
correlation_stat_analysis(semantic_minus_low_level)