import logging

import himalaya.scoring
import numpy as np
from himalaya.ridge import RidgeCV, GroupRidgeCV


def variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-4, 4, 9), cv=50,
                          use_r2=True, logger=None) -> tuple:
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
    use_r2 : bool, default=True
        Determines the score metric: if True, use r2 as the score metric; if False, use correlation.
    logger : logging.logger, default=None
        Logger, if none a new one gets instantiated

    Returns
    ----------
    score_0 : float
        Total variance explained by the first feature space.
    score_1 : float
        Total variance explained by the second feature space.
    joint_score : float
        Total variance explained by both feature spaces.
    x0_unique : float
        Unique variance explained by the first feature space.
    x1_unique : float
        Unique variance explained by the second feature space.
    shared : float
        Shared variance between the two feature spaces.
    """
    # Set default logger if not provided
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    from himalaya.backend import get_backend
    backend = get_backend()

    score_func = himalaya.scoring.r2_score if use_r2 else himalaya.scoring.correlation_score

    solver_params = dict(n_iter=10, alphas=alphas, progress_bar=False, warn=False, score_func=score_func,
                         n_targets_batch=1000)
    # train joint model
    joint_model = GroupRidgeCV(groups="input", solver_params=solver_params)
    joint_model.fit(Xs_train, Y_train)
    joint_score = joint_model.score(Xs_test, Y_test)
    logger.debug(fr"$X_0\cup X_1$ best $\alpha$s:")
    logger.debug(f"{joint_model.best_alphas_}")

    solver_params = dict(warn=False, score_func=score_func, n_targets_batch=1000)
    # train single model(s)
    model_0 = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
    model_0.fit(Xs_train[0], Y_train)
    score_0 = model_0.score(Xs_test[0], Y_test)
    logger.debug(fr"$X_0$ best $\alpha$s:")
    logger.debug(f"{model_0.best_alphas_}")

    model_1 = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)
    model_1.fit(Xs_train[1], Y_train)
    score_1 = model_1.score(Xs_test[1], Y_test)
    logger.debug(fr"$X_1$ best $\alpha$s:")
    logger.debug(f"{model_1.best_alphas_}")

    # calculate unique variance explained by feature space 0
    shared = (score_0 + score_1) - joint_score
    x0_unique = score_0 - shared
    x1_unique = score_1 - shared

    # convert back to numpy
    score_0 = backend.to_numpy(score_0)
    score_1 = backend.to_numpy(score_1)
    joint_score = backend.to_numpy(joint_score)
    shared = backend.to_numpy(shared)
    x0_unique = backend.to_numpy(x0_unique)
    x1_unique = backend.to_numpy(x1_unique)

    return score_0, score_1, joint_score, shared, x0_unique, x1_unique
