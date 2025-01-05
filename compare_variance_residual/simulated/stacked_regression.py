import numpy as np
from stacking_fmri import stacking_fmri
from sklearn.datasets import make_regression

def stacking_regression(Xs_train, Xs_test, Y_train, Y_test, ignore_negative_r2=False):
    (
        r2s,
        stacked_r2s,
        r2s_weighted,
        r2s_train,
        stacked_train_r2s,
        S,
    ) = stacking_fmri(
        Y_train,
        Y_test,
        Xs_train,
        Xs_test,
        method="cross_val_ridge",
        # score_f=np.mean_squared_error,
    )

    print("R2s:\n", r2s)
    print("Stacked R2s:\n", stacked_r2s)
    print("R2s weighted:\n", r2s_weighted)

    return np.mean(S, axis=0)

if __name__ == "__main__":
    from compare_variance_residual.simulated.simulation import generate_dataset

    n_features_list = [10, 10]
    n_targets = 10
    n_samples_train = 10
    n_samples_test = 10
    noise_level = 0.0
    unique_contributions = [0.5, 0.5]
    random_distribution = "normal"
    n_runs = 1
    predicted = []
    for run in range(n_runs):
        (Xs_train, Xs_test, Y_train, Y_test) = generate_dataset(
            n_features_list=n_features_list, n_targets=n_targets,
            n_samples_train=n_samples_train, n_samples_test=n_samples_test,
            noise=noise_level, unique_contributions=unique_contributions,
            random_distribution=random_distribution, random_state=run + 42)
        predicted_unique = stacking_regression(Xs_train, Xs_test, Y_train, Y_test)
        predicted.append(predicted_unique)
    print("\nPredicted unique contributions:")
    print(np.mean(predicted, axis=0))
    print("\nTrue unique contributions:")
    unique_contributions = np.array(unique_contributions)
    unique_contributions = unique_contributions / np.sum(unique_contributions)
    print(unique_contributions)