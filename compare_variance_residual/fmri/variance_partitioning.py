import os

import numpy as np
import pandas as pd

from features import load_feature, load_brain_data
from results import get_result_path
from ridge import run_ridge_pipeline, run_banded_pipeline


def signed_square(r):
    return r ** 2 * np.sign(r)


def variance_partitioning(data_dir, subject, modality, low_level_feature, alphas=np.logspace(-5, 20, 26), cv=5,
                          number_of_delays=4, n_targets_batch=100, n_alphas_batch=3, n_targets_batch_refit=50,
                          n_iter=5, X_semantic=None, X_low_level=None, Y=None, n_samples_train=None):
    path = get_result_path(modality, subject)

    print("Loading data")
    if X_semantic is None:
        X_semantic, n_samples_train = load_feature(data_dir, "english1000")
    if X_low_level is None:
        X_low_level, n_samples_train = load_feature(data_dir, low_level_feature)
    if Y is None:
        Y, n_samples_train = load_brain_data(data_dir, subject, modality)

    print("Running Variance Partitioning")
    low_level_path = os.path.join(path, f"{low_level_feature}_scores.csv")
    if not os.path.exists(low_level_path):
        print("Running low level")
        low_level_scores = run_ridge_pipeline(X_low_level, Y, n_samples_train, alphas, cv,
                                              number_of_delays, n_targets_batch, n_alphas_batch,
                                              n_targets_batch_refit)
        low_level_scores.to_csv(low_level_path, index=False)
    else:
        print("Loading low level")
        low_level_scores = pd.read_csv(low_level_path)

    english1000_path = os.path.join(path, f"english1000_scores.csv")
    if not os.path.exists(english1000_path):
        print("Running english1000")
        english1000_scores = run_ridge_pipeline(X_semantic, Y, n_samples_train, alphas, cv, number_of_delays,
                                                n_targets_batch,
                                                n_alphas_batch, n_targets_batch_refit)
        english1000_scores.to_csv(english1000_path, index=False)
    else:
        print("Loading english1000")
        english1000_scores = pd.read_csv(english1000_path)

    joint_path = os.path.join(path, f"joint_english1000_{low_level_feature}_scores.csv")
    if not os.path.exists(joint_path):
        joint_features = np.concatenate([X_semantic, X_low_level], axis=1)
        n_features_list = [X_semantic.shape[1], X_low_level.shape[1]]
        joint_scores = run_banded_pipeline(joint_features, n_features_list, Y, n_samples_train, alphas, cv,
                                           n_iter, number_of_delays, n_targets_batch, n_alphas_batch,
                                           n_targets_batch_refit)
        joint_scores.to_csv(joint_path, index=False)
    else:
        joint_scores = pd.read_csv(joint_path)

    vp_path = os.path.join(path, f"vp_english1000_{low_level_feature}_scores.csv")
    if not os.path.exists(vp_path):
        # perform vp
        vp_english1000 = pd.DataFrame()
        col = 'correlation_score'
        # get the intersection of the two sets
        intersection = signed_square(english1000_scores[col]) + signed_square(low_level_scores[col]) - signed_square(
            joint_scores[col])
        difference = signed_square(english1000_scores[col]) - intersection
        difference = np.sqrt(difference)
        difference = np.nan_to_num(difference)

        joint_minus_low = signed_square(joint_scores[col]) - signed_square(low_level_scores[col])
        joint_minus_low = np.sqrt(joint_minus_low)
        joint_minus_low = np.nan_to_num(joint_minus_low)

        # vp_english1000[fr'semantic$\cap${low_level_feature}'] = intersection
        vp_english1000[f'semantic\\{low_level_feature}'] = difference
        vp_english1000[f'joint\\{low_level_feature}'] = joint_minus_low
        vp_english1000.to_csv(vp_path, index=False)
    else:
        vp_english1000 = pd.read_csv(vp_path)
    return vp_english1000
