import himalaya.backend
import numpy as np
from himalaya.ridge import RidgeCV
from sklearn.linear_model import LinearRegression


def residual_method(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-10, 10, 41), cv=10,use_ols=False, ignore_negative_r2=False):
    backend = himalaya.backend.get_backend()

    # train model for creating residuals
    if use_ols:
        model = LinearRegression()
        feature, target = backend.to_numpy(Xs_train[1]), backend.to_numpy(Xs_train[0])
        model.fit(feature, target)
        train_predict = model.predict(backend.to_numpy(Xs_train[1]))
        test_predict = model.predict(backend.to_numpy(Xs_test[1]))
    else:
        solver_params = dict(warn=False)
        model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        model.fit(Xs_train[1], Xs_train[0])
        train_predict = model.predict(Xs_train[1])
        test_predict = model.predict(Xs_test[1])

    train_predict, test_predict = backend.asarray(train_predict), backend.asarray(test_predict)

    train_residual = Xs_train[0] - train_predict
    test_residual = Xs_test[0] - test_predict

    solver_params = dict(warn=False)
    model_residual = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
    model_residual.fit(train_residual, Y_train)

    score = model_residual.score(test_residual, Y_test)
    score = backend.to_numpy(score)

    if ignore_negative_r2:
        score = score[score >= 0]

    mean = np.mean(score)
    return mean