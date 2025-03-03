import os

import numpy as np
import pandas as pd
from himalaya.backend import get_backend
from himalaya.ridge import Ridge

from features import load_brain_data, load_feature
from results import get_result_path
from ridge import run_ridge_pipeline


def residual_method(data_dir, subject, modality, low_level_feature, run_onsets, alphas=np.logspace(-5, 20, 26),
                    number_of_delays=4, n_targets_batch=100, n_alphas_batch=5, n_targets_batch_refit=50,
                    X_semantic=None, X_low_level=None, Y=None, n_samples_train=None):
    backend = get_backend()
    path = get_result_path(modality, subject)

    print("Loading data")
    if X_semantic is None:
        X_semantic, n_samples_train = load_feature(data_dir, "english1000")
    if Y is None:
        Y, n_samples_train = load_brain_data(data_dir, subject, modality)
    if X_low_level is None:
        X_low_level, n_samples_train = load_feature(data_dir, low_level_feature)

    print("Running Residual")
    print("Cross Features")
    cross_model = Ridge(alpha=1, solver_params=dict(n_targets_batch=n_targets_batch))
    cross_model.fit(X_low_level[:n_samples_train], X_semantic[:n_samples_train])
    cross_path = os.path.join(path, f"cross_{low_level_feature}_english1000_scores.csv")
    if not os.path.exists(cross_path):
        r2_scores = cross_model.score(X_low_level[n_samples_train:], X_semantic[n_samples_train:])
        r2_scores = backend.to_numpy(r2_scores)
        cross_model_data = pd.DataFrame(
            {
                'r2_scores': r2_scores
            }
        )
        cross_model_data.to_csv(cross_path, index=False)

    print("Residual Features")
    residual_path = os.path.join(path, f"residual_{low_level_feature}_english1000_scores.csv")
    if not os.path.exists(residual_path):
        semantic_pred = cross_model.predict(X_low_level)
        semantic_pred = backend.to_numpy(semantic_pred)
        X_semantic_residual = X_semantic - semantic_pred

        residual_scores = run_ridge_pipeline(X_semantic_residual, Y, n_samples_train, run_onsets, alphas,
                                             number_of_delays, n_targets_batch, n_alphas_batch, n_targets_batch_refit)
        residual_scores.to_csv(residual_path)
    else:
        residual_scores = pd.read_csv(residual_path)
    return residual_scores
