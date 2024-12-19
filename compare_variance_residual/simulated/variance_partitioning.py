import numpy as np
from himalaya.kernel_ridge import KernelRidgeCV, Kernelizer, ColumnKernelizer, MultipleKernelRidgeCV
from sklearn.pipeline import make_pipeline


def variance_partitioning(Xs_train, Xs_test, Y_train, Y_test, use_refinement=False):
    """
    Perform variance partitioning on two feature spaces

    returns: unique variance explained by each feature space
    """
    # train single models
    solver_params = dict()

    single_predictions = []
    for X_train, X_test in zip(Xs_train, Xs_test):
        model = KernelRidgeCV(alphas=np.logspace(-10, 10, 41), kernel="linear", solver_params=solver_params)
        model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
        single_predictions.append(score)

    # train joint model
    X_train = np.hstack([X for X in Xs_train])
    X_test = np.hstack([X for X in Xs_test])

    # Find the start and end of each feature space X in Xs
    start_and_end = np.concatenate([[0], np.cumsum([X.shape[1] for X in Xs_train])])
    slices = [
        slice(start, end)
        for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]

    # Create a different ``Kernelizer`` for each feature space.
    kernelizers = [("space %d" % ii, Kernelizer(), slice_)
                   for ii, slice_ in enumerate(slices)]
    column_kernelizer = ColumnKernelizer(kernelizers)

    # random search
    solver_params = dict(n_iter=5, alphas=np.logspace(-10, 10, 41), progress_bar=False)
    model_random = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search", solver_params=solver_params,
                                         random_state=42)
    pipe = make_pipeline(column_kernelizer, model_random)
    # Fit the model on all targets
    pipe.fit(X_train, Y_train)

    if use_refinement:
        # refine model using gradient descent
        deltas = pipe[-1].deltas_
        solver_params = dict(max_iter=10, hyper_gradient_method="direct", max_iter_inner_hyper=10,
                             initial_deltas=deltas, progress_bar=False)
        model = MultipleKernelRidgeCV(kernels="precomputed", solver="hyper_gradient", solver_params=solver_params,
                                      random_state=42)
        pipe = make_pipeline(column_kernelizer, model)
        pipe.fit(X_train, Y_train)
        joint_score = pipe.score(X_test, Y_test)
    else:
        joint_score = pipe.score(X_test, Y_test)

    # calculate unique variance explained by each feature space
    X0_unique = joint_score - single_predictions[1]

    return float(X0_unique.mean())
