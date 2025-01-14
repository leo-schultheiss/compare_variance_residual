import numpy as np
from himalaya.kernel_ridge import Kernelizer, ColumnKernelizer, MultipleKernelRidgeCV
from himalaya.ridge import RidgeCV, GroupRidgeCV
from sklearn.pipeline import make_pipeline


def variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, use_direct_result=False, ignore_negative_r2=False):
    """
    Perform variance partitioning on two feature spaces

    returns: proportional unique variance explained by feature space 0
    """

    # train joint model
    solver_params = dict(n_iter=10, alphas=np.logspace(-10, 10, 41), progress_bar=False)
    model = GroupRidgeCV(groups="input", solver_params=solver_params)
    model.fit(Xs_train, Y_train)
    joint_score = model.score(Xs_test, Y_test)

    # train single model(s)
    solver_params = dict()
    if use_direct_result:
        model = RidgeCV(alphas=np.logspace(-10, 10, 41), solver_params=solver_params)
        model.fit(Xs_train[1], Y_train)
        score = model.score(Xs_test[1], Y_test)

        # calculate unique variance explained by feature space 0 only using the joint model and feature space 1
        X0_unique = joint_score - score
    else:
        # train both models
        model_0 = RidgeCV(alphas=np.logspace(-10, 10, 41), solver_params=solver_params)
        model_0.fit(Xs_train[0], Y_train)
        score_0 = model_0.score(Xs_test[0], Y_test)

        model_1 = RidgeCV(alphas=np.logspace(-10, 10, 41), solver_params=solver_params)
        model_1.fit(Xs_train[1], Y_train)
        score_1 = model_1.score(Xs_test[1], Y_test)

        # calculate unique variance explained by feature space 0
        shared = joint_score - score_0 - score_1
        X0_unique = score_0 - shared

    if ignore_negative_r2:
        X0_unique = X0_unique[X0_unique >= 0]

    mean = np.mean(X0_unique)
    return float(mean)
