import numpy as np
from himalaya.ridge import RidgeCV, GroupRidgeCV


def variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-10, 10, 41), cv=10,
                          direct_variance_partitioning=False, ignore_negative_r2=False):
    """
    Perform variance partitioning on two feature spaces

    Parameters
    ----------
    Xs_train : list of np.ndarray
        List of feature spaces for training
    Xs_test : list of np.ndarray
        List of feature spaces for testing
    Y_train : np.ndarray
        Target for training
    Y_test : np.ndarray
        Target for testing
    alphas : np.ndarray of float, default=np.logspace(-10, 10, 41)
        List of alphas for Ridge regression
    cv : int, default=10
        Number of cross-validation folds
    direct_variance_partitioning : bool, default=False
        Whether to use direct result from feature space 1 (joint model - feature space 1)
        or to use both feature spaces (feature space 0 - shared)
    ignore_negative_r2: bool, default=False
        Whether to ignore negative R2

    Returns
    ----------
    returns : float
        proportional unique variance explained by feature space 0
    """

    # train joint model
    solver_params = dict(n_iter=10, alphas=alphas, progress_bar=False, warn=False)
    model = GroupRidgeCV(groups="input", solver_params=solver_params)
    model.fit(Xs_train, Y_train)
    joint_score = model.score(Xs_test, Y_test)

    # train single model(s)
    solver_params = dict(warn=False)
    if direct_variance_partitioning:
        model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        model.fit(Xs_train[1], Y_train)
        score = model.score(Xs_test[1], Y_test)

        # calculate unique variance explained by feature space 0 only using the joint model and feature space 1
        X0_unique = joint_score - score
    else:
        # train both models
        model_0 = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        model_0.fit(Xs_train[0], Y_train)
        score_0 = model_0.score(Xs_test[0], Y_test)

        model_1 = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
        model_1.fit(Xs_train[1], Y_train)
        score_1 = model_1.score(Xs_test[1], Y_test)

        # calculate unique variance explained by feature space 0
        shared = (score_0 + score_1) - joint_score
        X0_unique = score_0 - shared

    if ignore_negative_r2:
        X0_unique = X0_unique[X0_unique >= 0]

    mean = np.mean(X0_unique)
    return float(mean)
