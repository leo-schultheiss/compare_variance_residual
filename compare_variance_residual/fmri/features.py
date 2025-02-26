import os

import h5py
import numpy as np
from scipy.stats import zscore

from voxelwise_tutorials.io import load_hdf5_array


def load_brain_data(data_dir, subject, modality, trim=5):
    Y_train_filename = os.path.join(data_dir, 'responses', f'subject{subject:02}_{modality}_fmri_data_trn.hdf')
    Y_test_filename = os.path.join(data_dir, 'responses', f'subject{subject:02}_{modality}_fmri_data_val.hdf')
    Y_train = load_hdf5_array(Y_train_filename)
    Y_test = load_hdf5_array(Y_test_filename)

    Y_train = np.vstack([zscore(Y_train[story][:-trim]) for story in Y_train.keys()])
    n_samples_train = Y_train.shape[0]
    Y_test = [np.vstack([zscore(Y_test[story][i][:-trim]) for story in Y_test.keys()]) for i in range(2)]
    # take average of the two repeats
    Y_test = np.mean(Y_test, axis=0)

    Y = np.vstack([Y_train, Y_test])
    Y = np.nan_to_num(Y)
    Y = zscore(Y)
    Y = Y.astype(np.float32)
    return Y, n_samples_train


def load_feature(data_dir, feature_name):
    Xs_train = h5py.File(os.path.join(data_dir, 'features', 'features_trn_NEW.hdf'), 'r')
    Xs_val = h5py.File(os.path.join(data_dir, 'features', 'features_val_NEW.hdf'), 'r')

    X_train = np.vstack([zscore(Xs_train[story][feature_name]) for story in Xs_train.keys()])
    n_samples_train = X_train.shape[0]
    X_val = np.vstack([zscore(Xs_val[story][feature_name]) for story in Xs_val.keys()])

    X = np.vstack([X_train, X_val])
    X = np.nan_to_num(X)
    X = zscore(X)
    X = X.astype(np.float32)
    return X, n_samples_train