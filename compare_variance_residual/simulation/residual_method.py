import himalaya.backend
import numpy as np
from himalaya.ridge import RidgeCV, Ridge


def residual_method(Xs, Y, n_samples_train, alphas=np.logspace(-5, 5, 10), cv=10, use_ols=True,
                    score_func=himalaya.scoring.r2_score, n_targets_batch=1000, n_alphas_batch=5, n_targets_batch_refit=500):
    """
    Compute scores using the residual method.

    Parameters
    ----------
    Xs : list of ndarray
        List of feature spaces.
    Y : ndarray
        Target variable.
    n_samples_train : int
        Number of training samples.
    alphas : ndarray, optional
    cv : int, optional
        Number of cross-validation folds.
    use_ols : bool, optional
        Whether to use ordinary least squares or ridge regression in cross-feature prediction.
    score_func : callable, optional
        Scoring function.
    n_targets_batch : int, optional
        Number of targets to process in parallel.
    n_alphas_batch : int, optional
    """
    backend = himalaya.backend.get_backend()
    solver_params = dict(warn=False, score_func=score_func, n_targets_batch=n_targets_batch,
                         n_alphas_batch=n_alphas_batch, n_targets_batch_refit=n_targets_batch_refit)

    # Handle feature modeling
    if use_ols:
        feature_model = Ridge(alpha=1, solver_params=dict(warn=False, n_targets_batch=n_targets_batch))
    else:
        feature_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)

    feature_scores = []
    residual_scores = []

    for i_to in range(len(Xs)):
        i_from = (i_to + 1) % len(Xs)
        # fit model between feature spaces
        feature_model.fit(Xs[i_from][:n_samples_train], Xs[i_to][:n_samples_train])
        feature_score = feature_model.score(Xs[i_from][n_samples_train:], Xs[i_to][n_samples_train:])
        feature_scores.append(feature_score)

        # predict from one feature to the other to extract shared information
        X_train_predict = feature_model.predict(Xs[i_from][:n_samples_train])
        X_test_predict = feature_model.predict(Xs[i_from][n_samples_train:])
        X_train_predict = backend.asarray(X_train_predict)
        X_test_predict = backend.asarray(X_test_predict)

        # Compute residuals
        X_train_residual = Xs[i_to][:n_samples_train] - X_train_predict
        X_test_residual = Xs[i_to][n_samples_train:] - X_test_predict

        # Train residual model
        residual_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        residual_model.fit(X_train_residual, Y[:n_samples_train])
        residual_score = residual_model.score(X_test_residual, Y[n_samples_train:])
        residual_scores.append(residual_score)

    feature_scores = list(map(backend.to_numpy, feature_scores))
    residual_scores = list(map(backend.to_numpy, residual_scores))

    return feature_scores[0], feature_scores[1], residual_scores[0], residual_scores[1]
