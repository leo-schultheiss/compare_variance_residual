import os

import h5py
import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from himalaya.ridge import Ridge

from fmri.features import load_brain_data, load_feature
from fmri.residual_method import residual_method
from fmri.variance_partitioning import variance_partitioning

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

def signed_square(r):
    return r ** 2 * np.sign(r)


if __name__ == "__main__":
    X_semantic, n_samples_train = load_feature(data_dir, "english1000")

    for subject in range(1, 10):
        for modality in ["reading", "listening"]:
            Y, n_samples_train = load_brain_data(data_dir, subject, modality)

            for low_level_feature in ['letters', 'phonemes']:
                print(f"Running {subject} {modality} {low_level_feature}")
                
                X_low_level, n_samples_train = load_feature(data_dir, low_level_feature)

                variance_partitioning(data_dir, subject, modality, low_level_feature, alphas, cv, number_of_delays,
                                      n_targets_batch, n_alphas_batch, n_targets_batch_refit, n_iter,
                                      X_semantic=X_semantic, X_low_level=X_low_level, Y=Y, n_samples_train=n_samples_train)
                residual_method(data_dir, subject, modality, low_level_feature, alphas, cv, number_of_delays,
                                n_targets_batch, n_alphas_batch, n_targets_batch_refit,
                                X_semantic=X_semantic, X_low_level=X_low_level, Y=Y, n_samples_train=n_samples_train)