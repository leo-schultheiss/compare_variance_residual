import logging

import himalaya.backend
import numpy as np
from himalaya.ridge import RidgeCV, Ridge


def residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-4, 4, 9), cv=10, use_ols=False,
                    return_full_variance=False, score_func=himalaya.scoring.r2_score, logger=None):
    """
    Provide a function that trains models to compute residuals and scores their predictive
    power using Ridge regression or Ordinary Least Squares (OLS). It also provides an option
    to compute full variance scores if required. The function evaluates models by computing
    residuals and applies second-stage Ridge regression to predict new outputs. Results are returned
    as numpy arrays for residual score, feature score, and optionally, full score.

    Parameters
    ----------
    Xs_train : list or array-like
        A list of training input data. The function assumes the first element contains
        the target features, and the second element contains the feature predictors.
    Xs_test : list or array-like
        A list of testing input data. The structure is analogous to Xs_train.
    Y_train : array-like
        Training target variables.
    Y_test : array-like
        Testing target variables.
    alphas : array-like, optional
        Array of Ridge regularization parameters to explore. Default is np.logspace(-10, 10, 41).
    cv : int, optional
        Number of cross-validation folds to use for RidgeCV. Default is 10.
    use_ols : bool, optional
        If True, OLS regression is used to compute residuals. If False, Ridge regression is used.
        Default is False.
    return_full_variance : bool, optional
        If True, computes the full variance score using original inputs. Default is False.
    score_func : callable, optional
        Scoring function used for evaluating models. Default is himalaya.scoring.r2_score


    Returns
    -------
    tuple
        full_scores[0] : float
            Variance explained by the first full feature set on the test data.
        full_scores[1] : float
            Variance explained by the second full feature set on the test data.
        feature_scores[0] : float
            Score of the first feature predictor's model using the target features.
        feature_scores[1] : float
            Score of the second feature predictor's model using the target features.
        residual_scores[0] : float
            Variance explained by the residual model of the first feature set on the target data.
        residual_scores[1] : float
            Variance explained by the residual model of the second feature set on the target data.
    

    Raises
    ------
    None

    Notes
    -----
    This function assumes compatibility with the `himalaya` library and its backend system.
    The parameter `use_r2` determines the scoring metric to be used across all computations.
    The method works in two stages: first creating residuals with one model, then predicting
    output with another model applied to residuals. Alphas are provided as a hyperparameter
    range for Ridge regression, and the best alpha is selected automatically through
    cross-validation.

    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

    backend = himalaya.backend.get_backend()
    solver_params = dict(warn=False, score_func=score_func, n_targets_batch=1000)

    full_scores = []

    # compute on full feature sets for comparison
    if return_full_variance:
        for i in range(len(Xs_train)):
            full_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
            full_model.fit(Xs_train[i], Y_train)
            full_score = full_model.score(Xs_test[i], Y_test)
            full_scores.append(full_score)

    # Handle feature modeling
    if use_ols:
        Xs_train = list(map(backend.to_numpy, Xs_train))
        Xs_test = list(map(backend.to_numpy, Xs_test))
        feature_model = Ridge(alpha=1, solver_params=solver_params)
    else:
        feature_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)

    feature_scores = []
    residual_scores = []

    for i in range(len(Xs_train)):
        i_from = (i + 1) % len(Xs_train)
        # fit model between feature spaces
        feature_model.fit(Xs_train[i_from], Xs_train[i])
        train_predict = feature_model.predict(Xs_train[i_from])
        test_predict = feature_model.predict(Xs_test[i_from])
        train_predict = backend.asarray(train_predict)
        test_predict = backend.asarray(test_predict)

        feature_score = feature_model.score(Xs_test[i], test_predict)
        feature_scores.append(feature_score)

        # Compute residuals
        train_residual = Xs_train[i] - train_predict
        test_residual = Xs_test[i] - test_predict

        # Train residual model
        residual_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        residual_model.fit(train_residual, Y_train)
        residual_score = residual_model.score(test_residual, Y_test)
        residual_scores.append(residual_score)

    full_scores = list(map(backend.to_numpy, full_scores))
    feature_scores = list(map(backend.to_numpy, feature_scores))
    residual_scores = list(map(backend.to_numpy, residual_scores))

    return full_scores[0], full_scores[1], feature_scores[0], feature_scores[1], residual_scores[0], \
    residual_scores[1]
