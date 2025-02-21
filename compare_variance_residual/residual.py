import himalaya.backend
import numpy as np
from himalaya.ridge import RidgeCV, Ridge


def residual_method(Xs, Y, n_samples_train, alphas=np.logspace(-5, 5, 10), cv=5, use_ols=True,
                    score_func=himalaya.scoring.r2_score, n_targets_batch=100, n_alphas_batch=5):
    """
    Compute performance scores for models using residual-based feature extraction.

    This function calculates the full model scores, feature modeling scores,
    and residual-based scores for datasets represented by multiple feature spaces.
    Residual modeling allows extracting unique contributions from individual
    feature spaces by iteratively predicting and removing shared information.

    Parameters:
        Xs (list of ndarray): A list of feature matrices. Each ndarray represents
            a feature space with dimensions (n_samples, n_features).
        Y (ndarray): Target matrix of shape (n_samples, n_targets).
        n_samples_train (int): Number of samples to use for training.
        alphas (Optional[ndarray]): Array of regularization parameter values
            to explore for Ridge regression. Defaults to np.logspace(-5, 5, 10).
        cv (Optional[int]): Number of cross-validation folds. Defaults to 5.
        use_ols (Optional[bool]): If True, use Ordinary Least Squares (OLS)
            regression instead of RidgeCV for feature modeling. Defaults to False.
        score_func (Optional[callable]): Function to compute a performance score
            between predictions and true targets. Defaults to himalaya.scoring.r2_score.

    Returns:
        tuple: Contains the following six scores in order:
            - Full score for the first feature space
            - Full score for the second feature space
            - Feature modeling score for the first feature space
            - Feature modeling score for the second feature space
            - Residual score for the first feature space
            - Residual score for the second feature space
    """
    backend = himalaya.backend.get_backend()
    solver_params = dict(warn=False, score_func=score_func, n_targets_batch=n_targets_batch,
                         n_alphas_batch=n_alphas_batch)

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
