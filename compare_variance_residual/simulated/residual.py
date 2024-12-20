import himalaya.backend
import numpy as np
from himalaya.kernel_ridge import KernelRidgeCV
from sklearn.linear_model import LinearRegression


def residual_method(Xs_train, Xs_test, Y_train, Y_test, use_ols=False):
    backend = himalaya.backend.get_backend()

    # train model for creating residuals
    if use_ols:
        model = LinearRegression()
        feature, target = backend.to_numpy(Xs_train[1]), backend.to_numpy(Xs_train[0])
        model.fit(feature, target)
        train_predict = model.predict(backend.to_numpy(Xs_train[1]))
        test_predict = model.predict(backend.to_numpy(Xs_test[1]))
    else:
        model = KernelRidgeCV(alphas=np.logspace(-10, 10, 41), kernel="linear", solver="eigenvalues", warn=False)
        model.fit(Xs_train[1], Xs_train[0])
        train_predict = model.predict(Xs_train[1])
        test_predict = model.predict(Xs_test[1])

    train_predict, test_predict = backend.asarray(train_predict), backend.asarray(test_predict)

    train_residual = Xs_train[0] - train_predict
    test_residual = Xs_test[0] - test_predict

    model_residual = KernelRidgeCV(alphas=np.logspace(-10, 10, 41), kernel="linear", warn=False)
    model_residual.fit(train_residual, Y_train)

    score = model_residual.score(test_residual, Y_test)
    score = backend.to_numpy(score)
    mean = np.mean(score)
    return mean