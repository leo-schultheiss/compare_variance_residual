import himalaya.backend
import numpy as np
from himalaya.ridge import RidgeCV
from sklearn.linear_model import LinearRegression


def residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-10, 10, 41), cv=10, use_ols=False,
                    use_r2=True):
    """
    Function to perform the residual regression method.

    The function leverages a multi-stage regression approach to predict targets and assess
    the modeled variance in the residuals of the initial predictions using Ridge regression
    or Ordinary Least Squares (OLS). The method first calculates residuals from initial
    regressions and then trains a second-stage model to predict the residuals. Cross-validated
    Ridge regression or OLS is configurable for the first-stage regression. The function
    computes and returns the mean score of the secondary model on the test data.

    Parameters:
        Xs_train: list
            List of input training datasets.
        Xs_test: list
            List of input test datasets.
        Y_train: array-like
            Target variables for the training dataset.
        Y_test: array-like
            Target variables for the testing dataset.
        alphas: array-like, optional
            Sequence of regularization strength values utilized in Ridge regression.
        cv: int, optional
            Number of cross-validation folds used for Ridge regression.
        use_ols: bool, optional
            Flag indicating whether Ordinary Least Squares should be used instead of Ridge regression.
        use_r2: bool, optional
            Flag specifying whether R-squared scoring is applied.

    Returns:
        float
            Mean score of the residual model on the test data.
    """
    backend = himalaya.backend.get_backend()
    score_func = himalaya.scoring.r2_score if use_r2 else himalaya.scoring.correlation_score

    # train model for creating residuals
    if use_ols:
        model = LinearRegression()
        feature, target = backend.to_numpy(Xs_train[1]), backend.to_numpy(Xs_train[0])
        model.fit(feature, target)
        train_predict = model.predict(backend.to_numpy(Xs_train[1]))
        test_predict = model.predict(backend.to_numpy(Xs_test[1]))
    else:
        solver_params = dict(warn=False, score_func=score_func)
        model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        model.fit(Xs_train[1], Xs_train[0])
        train_predict = model.predict(Xs_train[1])
        test_predict = model.predict(Xs_test[1])

    train_predict, test_predict = backend.asarray(train_predict), backend.asarray(test_predict)

    train_residual = Xs_train[0] - train_predict
    test_residual = Xs_test[0] - test_predict

    solver_params = dict(warn=False, score_func=score_func)
    model_residual = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
    model_residual.fit(train_residual, Y_train)

    score = model_residual.score(test_residual, Y_test)
    score = backend.to_numpy(score)
    mean = np.mean(score)
    return mean
