import himalaya.backend
import numpy as np
from himalaya.ridge import RidgeCV
from sklearn.linear_model import LinearRegression


def residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-4, 4, 9), cv=50, use_ols=False,
                    return_full_variance=False, use_r2=True):
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
    use_r2 : bool, optional
        If True, the r2_score metric is used for scoring. If False, the correlation_score metric
        is used. Default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - residual_score : ndarray
            Scores of the predictive power of residuals.
        - feature_score : ndarray
            Scores of the predictive power of features in the model.
        - full_score : ndarray
            Full variance scores of the model (optional, returned if return_full_variance is True).

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
    backend = himalaya.backend.get_backend()
    score_func = himalaya.scoring.r2_score if use_r2 else himalaya.scoring.correlation_score
    solver_params = dict(warn=False, score_func=score_func, n_targets_batch=1000)

    def train_model(model, train_X, train_y, test_X, test_y):
        """Helper function to fit a model and return predictions and score."""
        model.fit(train_X, train_y)
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)
        score = model.score(test_X, test_y)
        train_pred = backend.asarray(train_pred)
        test_pred = backend.asarray(test_pred)
        return train_pred, test_pred, score

    # Handle feature modeling
    if use_ols:
        features_train, targets_train, features_test, target_test = map(backend.to_numpy,
                                                                        [Xs_train[1], Xs_train[0], Xs_test[1],
                                                                         Xs_test[0]])
        feature_model = LinearRegression()
    else:
        features_train, targets_train, features_test, target_test = Xs_train[1], Xs_train[0], Xs_test[1], Xs_test[0]
        feature_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)

    train_predict, test_predict, feature_score = train_model(
        feature_model, features_train, targets_train, features_test, target_test
    )

    if not use_ols:
        from matplotlib import pyplot as plt
        plt.title("Best alphas for residual model")
        plt.hist(backend.to_numpy(feature_model.best_alphas_))
        plt.show()
    else:
        from matplotlib import pyplot as plt
        plt.title("Coefficients for residual model")
        plt.hist(feature_model.coef_)
        plt.show()

    # Compute residuals
    train_residual = Xs_train[0] - train_predict
    test_residual = Xs_test[0] - test_predict

    # Train residual model
    residual_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
    residual_model.fit(train_residual, Y_train)
    residual_score = residual_model.score(test_residual, Y_test)

    # Optionally compute full variance score
    full_score = None
    if return_full_variance:
        residual_model.fit(Xs_train[0], Y_train)
        full_score = residual_model.score(Xs_test[0], Y_test)

    # Convert results to NumPy
    residual_score, feature_score, full_score = map(
        lambda x: backend.to_numpy(x) if x is not None else None,
        [residual_score, feature_score, full_score]
    )

    return residual_score, feature_score, full_score
