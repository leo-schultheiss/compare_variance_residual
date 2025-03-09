import os

import h5py
import numpy as np
from more_itertools import first
from scipy.stats import zscore
from voxelwise_tutorials.io import load_hdf5_array
from voxelwise_tutorials.utils import explainable_variance


def load_brain_data(data_dir, subject, modality, trim=10):
    Y_train_filename = os.path.join(data_dir, 'responses', f'subject{subject:02}_{modality}_fmri_data_trn.hdf')
    Y_test_filename = os.path.join(data_dir, 'responses', f'subject{subject:02}_{modality}_fmri_data_val.hdf')
    Y_train_hdf = load_hdf5_array(Y_train_filename)
    Y_test_hdf = load_hdf5_array(Y_test_filename)

    run_onsets = [0]

    Y_train = None
    for story in Y_train_hdf.keys():
        story_data = Y_train_hdf[story]

        # some stories are encapsulated in another array
        if story_data.shape[0] == 1:
            print("Encapsulated story")
            story_data = story_data[0]

        story_data = story_data[trim:-trim]

        story_data = story_data.astype(np.float32)
        story_data = np.nan_to_num(story_data)
        story_data = zscore(story_data)

        if Y_train is None:
            Y_train = story_data
        else:
            Y_train = np.vstack([Y_train, story_data])

        run_onsets.append(run_onsets[-1] + story_data.shape[0])
    run_onsets = run_onsets[:-1]

    n_samples_train = Y_train.shape[0]

    Y_test = []
    eval_story = first(Y_test_hdf.keys())
    for i in range(2):
        story_data = Y_test_hdf[eval_story][i][trim:-trim]
        story_data = story_data.astype(np.float32)
        story_data = np.nan_to_num(story_data)
        story_data = zscore(story_data)
        Y_test.append(story_data)

    # calculate explainable variance
    ev = explainable_variance(np.array(Y_test))

    Y_test = np.mean(Y_test, axis=0)

    Y = np.vstack([Y_train, Y_test])
    Y = np.nan_to_num(Y)
    return Y, n_samples_train, run_onsets, ev


def load_feature(data_dir, feature_name, trim=10):
    if feature_name not in ['powspec', 'moten']:
        Xs_train = h5py.File(os.path.join(data_dir, 'features', 'features_trn_NEW.hdf'), 'r')
        Xs_val = h5py.File(os.path.join(data_dir, 'features', 'features_val_NEW.hdf'), 'r')
        X_train = np.vstack([zscore(Xs_train[story][feature_name][trim:-trim]) for story in Xs_train.keys()])
        X_val = np.vstack([zscore(Xs_val[story][feature_name]) for story in Xs_val.keys()])
    elif feature_name == 'powspec':
        X_train = h5py.File(os.path.join(data_dir, 'features', 'features_matrix.hdf'), 'r')['powspec_train']
        X_val = h5py.File(os.path.join(data_dir, 'features', 'features_matrix.hdf'), 'r')['powspec_test']
    else:
        raise(ValueError(f"Feature {feature_name} not found/implemented"))

    n_samples_train = X_train.shape[0]
    X = np.vstack([X_train, X_val])
    X = X.astype(np.float32)
    X = np.nan_to_num(X)
    return X, n_samples_train
