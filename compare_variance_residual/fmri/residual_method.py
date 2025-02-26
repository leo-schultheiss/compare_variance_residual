import os

import numpy as np
import pandas as pd
from himalaya.backend import get_backend
from himalaya.ridge import Ridge

from fmri.features import load_brain_data, load_feature
from fmri.results import get_result_path
from fmri.ridge import run_ridge_pipeline


def residual_method(data_dir, subject, modality, low_level_feature, alphas=np.logspace(-5, 20, 26), cv=5,
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
    cross_path = os.path.join(path, f"cross_{low_level_feature}_english1000_scores.csv")
    if os.path.exists(cross_path):
        cross_model = Ridge(alpha=1, solver_params=dict(n_targets_batch=n_targets_batch))
        cross_model.fit(X_low_level[:n_samples_train], X_semantic[:n_samples_train])
        coef = cross_model.coef_

        r2_scores = cross_model.score(X_low_level[n_samples_train:], X_semantic[n_samples_train:])
        r2_scores = backend.to_numpy(r2_scores)
        cross_model_data = pd.DataFrame(
            {
                'coef': coef,
                'r2_scores': r2_scores

            }
        )
        cross_model_data.to_csv(cross_path, index=False)
    else:
        cross_model_data = pd.read_csv(cross_path)
        coef = cross_model_data['coef']
        cross_model = Ridge(alpha=1, solver_params=dict(n_targets_batch=n_targets_batch))
        cross_model.coef_ = coef

    residual_path = os.path.join(path, f"residual_{low_level_feature}_english1000_scores.csv")
    if not os.path.exists(residual_path):
        semantic_pred = cross_model.predict(X_low_level)
        semantic_pred = backend.to_numpy(semantic_pred)
        X_semantic_residual = X_semantic - semantic_pred

        residual_scores = run_ridge_pipeline(X_semantic_residual, Y, n_samples_train, alphas, cv, number_of_delays,
                                             n_targets_batch, n_alphas_batch, n_targets_batch_refit)
        residual_scores.to_csv(residual_path)
    else:
        residual_scores = pd.read_csv(residual_path)
    return residual_scores
