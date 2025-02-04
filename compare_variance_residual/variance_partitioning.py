import logging

import himalaya.scoring
import numpy as np
from himalaya.ridge import RidgeCV, GroupRidgeCV, Ridge

def variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, alphas=np.logspace(-4, 4, 9), cv=50,
                          score_func=himalaya.scoring.r2_score, use_ols=False, logger=None) -> tuple:
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
    score_func : callable, default=himalaya.scoring.r2_score
        Scoring function used to evaluate the model predictions. Common options include R²
        or other regression metrics compatible with the `himalaya` library.
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

    solver_params = dict(n_iter=10, alphas=alphas, progress_bar=False, warn=False, score_func=score_func,
                         n_targets_batch=1000)
    # train joint model
    joint_model = GroupRidgeCV(groups="input", solver_params=solver_params)

    joint_model.fit(Xs_train, Y_train)
    Y_pred = joint_model.predict(Xs_test)
    joint_score = score_func(Y_test, Y_pred)
    import matplotlib.pyplot as plt
    # plt.hist(backend.to_numpy(joint_model.best_alphas_))
    # plt.xlabel("alpha")
    # plt.ylabel("count")
    # plt.title(fr"$X_0\cup X_1$ best $\alpha$s")
    # plt.show()
    # plt.hist(backend.to_numpy(joint_model.coef_))
    # plt.xlabel("coefficient")
    # plt.ylabel("count")
    # plt.title(fr"$X_0\cup X_1$ coefficients")
    # plt.show()

    # train single models
    if use_ols:
        solver_params = dict(warn=False, n_targets_batch=1000)
        single_model = Ridge(alpha=0.0, solver_params=solver_params)
    else:
        solver_params = dict(warn=False, score_func=score_func,
                             n_targets_batch=1000)
        single_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)

    scores = []
    for i in range(2):
        single_model.fit(Xs_train[i], Y_train)
        Y_pred = single_model.predict(Xs_test[i])
        score = score_func(Y_test, Y_pred)
        scores.append(score)

        # if not use_ols:
        #     plt.hist(backend.to_numpy(single_model.best_alphas_))
        #     plt.xlabel("alpha")
        #     plt.ylabel("count")
        #     plt.title(fr"$X_{i}$ best $\alpha$s")
        #     plt.show()
        # plt.hist(backend.to_numpy(single_model.coef_))
        # plt.xlabel("coefficient")
        # plt.ylabel("count")
        # plt.title(fr"$X_{i}$ coefficients")
        # plt.show()

    score_0, score_1 = scores

    # calculate unique and shared variance
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
