import logging

import himalaya.scoring
import numpy as np
from himalaya.ridge import RidgeCV, GroupRidgeCV, Ridge


def variance_partitioning(Xs, Y, n_samples_train, alphas=np.logspace(-5, 5, 10), cv=5,
                          score_func=himalaya.scoring.r2_score, use_ols=False, n_iter=10, n_targets_batch=100,
                          n_targets_batch_refit=100, n_alphas_batch=None) -> tuple:
    """
        Calculate the shared and unique variance contributions of inputs to an output.

        This function performs variance partitioning to evaluate the unique and shared variance
        contributions of input variables (Xs) to the output (Y). It uses group ridge regression
        and optionally ordinary least squares (OLS) regression to compare the contributions.

        Parameters:
        Xs: list
            A list of input datasets, where each dataset corresponds to a group of features. They
            should be provided as `Xs_train` for training and `Xs_test` for testing in the form
            of partitions.
        Y: ndarray
            The output dataset to which variance contributions are being computed.
        n_samples_train: int
            The number of samples to be used for training in each input dataset and the output.
        alphas: array-like, optional
            Regularization parameter/values for RidgeCV. The default value is a logarithmic
            scale from 10^-4 to 10^4.
        cv: int, optional
            Number of cross-validation folds for RidgeCV (default is 5).
        score_func: callable, optional
            The scoring function to evaluate predictions (default is himalaya.scoring.r2_score).
        use_ols: bool, optional
            Whether to use ordinary least squares instead of group ridge regression (default is False).
        n_iter: int, optional
            Number of iterations for the joint model (default is 10).
        n_targets_batch: int, optional
            Number of targets to process in parallel (default is 50).
        n_targets_batch_refit: int, optional
            Number of targets to process in parallel during refitting (default is 50).
        n_alphas_batch: int, optional
            Number of alphas to process in parallel (default is 1).

        Returns:
        tuple
            A tuple containing six ndarray values:
            - score_0: Score of the first input group.
            - score_1: Score of the second input group.
            - joint_score: Overall joint score considering all input groups.
            - shared: Shared variance contribution between the two input groups.
            - x0_unique: Unique variance contribution of the first input group.
            - x1_unique: Unique variance contribution of the second input group.
    """
    from himalaya.backend import get_backend
    backend = get_backend()

    joint_solver_params = dict(n_iter=n_iter, alphas=alphas, progress_bar=False, warn=False, score_func=score_func,
                               n_targets_batch=n_targets_batch, n_targets_batch_refit=n_targets_batch_refit,
                               n_alphas_batch=n_alphas_batch)
    # train joint model
    joint_model = GroupRidgeCV(groups="input", solver_params=joint_solver_params)

    joint_model.fit([x[:n_samples_train] for x in Xs], Y[:n_samples_train])
    joint_score = joint_model.score([x[n_samples_train:] for x in Xs], Y[n_samples_train:])

    # train single models
    if use_ols:
        solver_params = dict(warn=False, n_targets_batch=n_targets_batch)
        single_model = Ridge(alpha=1.0, solver_params=solver_params)
    else:
        solver_params = dict(warn=False, score_func=score_func, n_targets_batch=n_targets_batch,
                             n_targets_batch_refit=n_targets_batch_refit, n_alphas_batch=n_alphas_batch)
        single_model = RidgeCV(alphas=alphas, cv=cv, solver_params=solver_params)

    scores = []
    for i in range(2):
        single_model.fit(Xs[i][:n_samples_train], Y[:n_samples_train])
        score = single_model.score(Xs[i][n_samples_train:], Y[n_samples_train:])
        scores.append(score)

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
