import himalaya.scoring
import numpy as np
import pandas as pd
from himalaya.backend import get_backend
from himalaya.ridge import BandedRidgeCV, ColumnTransformerNoStack, RidgeCV
from himalaya.scoring import r2_score
from scipy.stats import t
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.delayer import Delayer


def compute_p_values(correlation_scores, n):
    t_values = correlation_scores * np.sqrt((n - 2) / (1 - correlation_scores ** 2))
    p_values = 2 * t.sf(np.abs(t_values), df=n - 2)
    return p_values


def calculate_scores(Y, prediction, n_samples_train):
    """
    Calculate correlation, r2 and p-value scores
    """
    n_samples_test = len(Y) - n_samples_train
    correlation_score = np.array(
        [np.corrcoef(Y[n_samples_train:, i], prediction[:, i])[0, 1] for i in range(Y.shape[1])])
    p_values = compute_p_values(correlation_score, n_samples_test)
    r2 = r2_score(Y[n_samples_train:], prediction)
    r2 = get_backend().to_numpy(r2)
    return pd.DataFrame({'correlation_score': correlation_score, 'r2_score': r2, 'p_value': p_values})


def run_pipeline(pipeline, X, Y, n_samples_train):
    """
    Run pipeline and calculate scores
    """
    pipeline.fit(X[:n_samples_train], Y[:n_samples_train])
    prediction = pipeline.predict(X[n_samples_train:])
    prediction = get_backend().to_numpy(prediction)
    return calculate_scores(Y, prediction, n_samples_train)


def run_ridge_pipeline(X, Y, n_samples_train, alphas, cv, number_of_delays, n_targets_batch, n_alphas_batch,
                       n_targets_batch_refit, score_func=himalaya.scoring.r2_score):
    """
    Run ridge pipeline using RidgeCV
    """
    delayer = Delayer(delays=range(1, number_of_delays + 1))
    solver_params = dict(n_targets_batch=n_targets_batch, n_alphas_batch=n_alphas_batch,
                         n_targets_batch_refit=n_targets_batch_refit, score_func=score_func)
    ridge_cv = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
    pipeline = make_pipeline(delayer, ridge_cv)
    return run_pipeline(pipeline, X, Y, n_samples_train)


def run_banded_pipeline(Xs, n_features_list, Y, n_samples_train, alphas, cv, n_iter, number_of_delays,
                        n_targets_batch, n_alphas_batch, n_targets_batch_refit, score_func=himalaya.scoring.r2_score):
    """
    Run banded pipeline using BandedRidgeCV
    """
    delayer = Delayer(delays=range(1, number_of_delays + 1))
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])]
    ct = ColumnTransformerNoStack(transformers=[(f'feature_{i}', delayer, s) for i, s in enumerate(slices)])
    solver_params = dict(alphas=alphas, n_iter=n_iter, n_targets_batch=n_targets_batch, n_alphas_batch=n_alphas_batch,
                         n_targets_batch_refit=n_targets_batch_refit, score_func=score_func)
    banded_ridge_cv = BandedRidgeCV(cv=cv, groups="input", solver_params=solver_params)
    pipeline = make_pipeline(ct, banded_ridge_cv)
    return run_pipeline(pipeline, Xs, Y, n_samples_train)
