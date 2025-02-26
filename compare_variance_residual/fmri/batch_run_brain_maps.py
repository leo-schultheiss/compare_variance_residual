import os

import h5py
import himalaya.scoring
import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from himalaya.ridge import ColumnTransformerNoStack, BandedRidgeCV, Ridge
from scipy.stats import zscore
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import load_hdf5_array

from fmri.features import load_brain_data, load_feature
from fmri.ridge import run_ridge_pipeline, run_banded_pipeline

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
number_of_delays = 4


def get_result_path(modality, subject):
    path = os.path.join("results", modality, f"subject{subject:02}")
    os.makedirs(path, exist_ok=True)
    return path


def signed_square(r):
    return r ** 2 * np.sign(r)


if __name__ == "__main__":
    features_train = h5py.File(os.path.join(data_dir, 'features', 'features_trn_NEW.hdf'), 'r')
    features_val = h5py.File(os.path.join(data_dir, 'features', 'features_val_NEW.hdf'), 'r')

    for subject in range(1, 10):
        for modality in ["reading", "listening"]:
            Y_train, Y_test = load_brain_data(data_dir, subject, modality)

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
                semantic_train, semantic_val = load_feature(data_dir, "english1000")
                low_level_train, low_level_val = load_feature(data_dir, low_level_feature)
                print("Done loading data")

                # Variance Partitioning
                print("Running Variance Partitioning")
                low_level_scores = run_ridge_pipeline(low_level_train, low_level_val, Y_train, Y_test, alphas, cv,
                                                      number_of_delays, n_targets_batch, n_alphas_batch,
                                                      n_targets_batch_refit)
                low_level_scores.to_csv(low_level_path, index=False)

                english1000_scores = run_ridge_pipeline(semantic_train, semantic_val, Y_train,
                                                        Y_test, alphas, cv, number_of_delays, n_targets_batch,
                                                        n_alphas_batch, n_targets_batch_refit)
                english1000_scores.to_csv(english1000_path, index=False)

                joint_features_train = np.concatenate([semantic_train, low_level_train], axis=1)
                joint_features_val = np.concatenate([semantic_val, low_level_val], axis=1)
                joint_scores = run_banded_pipeline(joint_features_train, joint_features_val,
                                                   [semantic_train.shape[1], low_level_train.shape[1]], Y_train, Y_test,
                                                   alphas, cv, n_iter, number_of_delays, n_targets_batch,
                                                   n_alphas_batch, n_targets_batch_refit)
                joint_scores.to_csv(joint_path, index=False)

                # perform vp
                vp_english1000 = pd.DataFrame()
                # iterate all columns containining "correlation"
                for corr in [col for col in joint_scores.columns if "correlation" in col]:
                    # get the column name without the suffix
                    col = corr.split("_")
                    # get the intersection of the two sets
                    intersection = signed_square(english1000_scores[col]) + signed_square(
                        low_level_scores[col]) - signed_square(joint_scores[col])
                    difference = signed_square(english1000_scores[col]) - signed_square(intersection)

                    vp_english1000[fr'semantic$\cup${low_level_feature}'] = intersection
                    vp_english1000[f'semantic\\{low_level_feature}'] = difference
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

                residual_scores = run_ridge_pipeline(semantic_train_residual_train, semantic_val_residual_val, Y_train,
                                                     Y_test, alphas, cv, number_of_delays, n_targets_batch,
                                                     n_alphas_batch, n_targets_batch_refit)
                residual_scores.to_csv(residual_path)
