import os

import h5py
import himalaya.scoring
import numpy as np
import pandas as pd
import voxelwise_tutorials.viz as viz
from himalaya.backend import set_backend
from himalaya.ridge import ColumnTransformerNoStack, BandedRidgeCV, Ridge
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import load_hdf5_array

data_dir = "../../data"
backend = set_backend('torch_cuda', on_error='warn')

alphas = np.logspace(-5, 5, 10)
n_iter = 25
n_targets_batch = 100
n_alphas_batch = 5
n_targets_batch_refit = 100
n_splits = 15
chunk_len = 40
num_chunks = 10
cv = 5
score_func = himalaya.scoring.r2_score
trim = 5  # remove 5 TRs from the end of each story
number_of_delays = 4


def get_result_path(modality, subject):
    path = os.path.join("results", modality, f"subject{subject:02}")
    os.makedirs(path, exist_ok=True)
    return path


def signed_square(r):
    return r ** 2 * np.sign(r)


def run_pipeline(features_train, features_val, n_features_list, target_train, targets_val):
    delayer = Delayer(delays=range(1, number_of_delays + 1))

    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [
        slice(start, end)
        for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]
    ct = ColumnTransformerNoStack(transformers=[(f'feature_{i}', delayer, s) for i, s in enumerate(slices)])
    print(ct)

    solver_params = dict(
        alphas=alphas,
        n_iter=n_iter,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
        score_func=score_func
    )
    banded_ridge_cv = BandedRidgeCV(cv=cv, groups="input", solver_params=solver_params)

    pipeline = make_pipeline(
        ct,
        banded_ridge_cv
    )
    pipeline.fit(features_train, target_train)

    correlation_scores = []
    r2_scores = []
    for target_val in targets_val:
        prediction = pipeline.predict(features_val)
        prediction = backend.to_numpy(prediction)

        # calculate correlation for each target
        correlation_score = np.array(
            [np.corrcoef(target_val[:, i], prediction[:, i])[0, 1] for i in range(target_val.shape[1])])
        print(correlation_score.shape)
        correlation_scores.append(correlation_score)

        r2 = himalaya.scoring.r2_score(target_val, prediction)
        r2 = backend.to_numpy(r2)
        r2_scores.append(r2)

    scores = pd.DataFrame(
        {
            'correlation_score_0': correlation_scores[0],
            'correlation_score_1': correlation_scores[1],
            'r2_score_0': r2_scores[0],
            'r2_score_1': r2_scores[1]
        }
    )
    return scores


if __name__ == "__main__":
    features_train = h5py.File(os.path.join(data_dir, 'features', 'features_trn_NEW.hdf'), 'r')
    features_val = h5py.File(os.path.join(data_dir, 'features', 'features_val_NEW.hdf'), 'r')

    for subject in range(1, 10):
        for modality in ["reading", "listening"]:
            Y_train_filename = os.path.join(data_dir, 'responses', f'subject{subject:02}_{modality}_fmri_data_trn.hdf')
            Y_test_filename = os.path.join(data_dir, 'responses', f'subject{subject:02}_{modality}_fmri_data_val.hdf')
            Y_train = load_hdf5_array(Y_train_filename)
            Ys_test = load_hdf5_array(Y_test_filename)
            Y_train = np.vstack([zscore(Y_train[story][:-trim]) for story in Y_train.keys()])
            Ys_test = [np.vstack([zscore(Ys_test[story][i][:-trim]) for story in Ys_test.keys()]) for i in range(2)]
            Y_train, Ys_test = np.nan_to_num(Y_train), np.nan_to_num(Ys_test)

            Y_train = Y_train.astype(np.float32)
            Ys_test = [Y_test.astype(np.float32) for Y_test in Ys_test]

            for low_level_feature in ['letters', 'numletters', 'numphonemes', 'numwords', 'pauses', 'phonemes',
                                      'word_length_std']:
                print(f"Processing subject {subject}, modality {modality}, feature {low_level_feature}")
                path = get_result_path(modality, subject)

                low_level_path = os.path.join(path, f"{low_level_feature}_scores.csv")
                english1000_path = os.path.join(path, f"english1000_scores.csv")
                joint_path = os.path.join(path, f"joint_english1000_{low_level_feature}_scores.csv")
                cross_path = os.path.join(path, f"cross_{low_level_feature}_english1000_scores.csv")
                vp_path = os.path.join(path, f"vp_english1000_{low_level_feature}_scores.csv")
                residual_path = os.path.join(path, f"residual_english1000_{low_level_feature}_scores.csv")

                if os.path.exists(low_level_path) and os.path.exists(english1000_path) and os.path.exists(
                        joint_path) and os.path.exists(residual_path):
                    print(f"Already processed subject {subject}, modality {modality}, feature {low_level_feature}")
                    continue
                else:
                    # print each file that is missing
                    for path in [low_level_path, english1000_path, joint_path, residual_path]:
                        if not os.path.exists(path):
                            print(f"Missing {path}")

                print("Loading data")
                semantic_train = np.vstack(
                    [zscore(features_train[story]['english1000']) for story in features_train.keys()])
                semantic_val = np.vstack([zscore(features_val[story]['english1000']) for story in features_val.keys()])
                low_level_train = np.vstack(
                    [zscore(features_train[story][low_level_feature]) for story in features_train.keys()])
                low_level_val = np.vstack(
                    [zscore(features_val[story][low_level_feature]) for story in features_val.keys()])
                low_level_train, low_level_val = np.nan_to_num(low_level_train), np.nan_to_num(low_level_val)

                low_level_train = low_level_train.astype(np.float32)
                low_level_val = low_level_val.astype(np.float32)
                semantic_train = semantic_train.astype(np.float32)
                semantic_val = semantic_val.astype(np.float32)
                print("Done loading data")

                # Variance Partitioning
                print("Running Variance Partitioning")
                low_level_scores = run_pipeline(low_level_train, low_level_val, [low_level_train.shape[1]], Y_train,
                                                Ys_test)
                low_level_scores.to_csv(low_level_path, index=False)

                english1000_scores = run_pipeline(semantic_train, semantic_val, [semantic_train.shape[1]], Y_train,
                                                  Ys_test)
                english1000_scores.to_csv(english1000_path, index=False)

                joint_features_train = np.concatenate([semantic_train, low_level_train], axis=1)
                joint_features_val = np.concatenate([semantic_val, low_level_val], axis=1)
                joint_scores = run_pipeline(joint_features_train, joint_features_val,
                                            [semantic_train.shape[1], low_level_train.shape[1]], Y_train, Ys_test)
                joint_scores.to_csv(joint_path, index=False)

                # perform vp
                vp_english1000 = pd.DataFrame()
                # iterate all columns containining "correlation"
                for corr in [col for col in joint_scores.columns if "correlation" in col]:
                    # get the column name without the suffix
                    col = corr.split("_")[0]
                    # get the intersection of the two sets
                    intersection = signed_square(english1000_scores[col]) + signed_square(low_level_scores[col]) - signed_square(joint_scores[col])
                    difference = signed_square(english1000_scores[col]) - signed_square(intersection)
                    vp_english1000[col] = difference
                vp_english1000.to_csv(vp_path, index=False)

                # perform residual
                print("Running Residual")
                cross_model = Ridge(alpha=1, solver_params=dict(n_targets_batch=n_targets_batch))
                cross_model.fit(low_level_train, semantic_train)
                r2_scores = cross_model.score(low_level_val, semantic_val)
                r2_scores = backend.to_numpy(r2_scores)
                r2_scores = pd.DataFrame(r2_scores, columns=['r2_cross'])
                r2_scores.to_csv(cross_path, index=False)

                semantic_pred_train = cross_model.predict(low_level_train)
                semantic_pred_val = cross_model.predict(low_level_val)
                semantic_pred_train = backend.to_numpy(semantic_pred_train)
                semantic_pred_val = backend.to_numpy(semantic_pred_val)
                semantic_train_residual_train = semantic_train - semantic_pred_train
                semantic_val_residual_val = semantic_val - semantic_pred_val

                residual_scores = run_pipeline(semantic_train_residual_train, semantic_val_residual_val,
                                               [semantic_train_residual_train.shape[1]], Y_train, Ys_test)
                residual_scores.to_csv(residual_path)
