import os

import h5py
import numpy as np
from ridge_utils.ridge import bootstrap_ridge

from hdf_utils import load_data
from npp import zscore


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  ## negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


def load_subject_fmri(data_dir, subject, modality):
    """Load fMRI data for a subject, z-scored across stories"""
    fname_tr5 = os.path.join(data_dir, 'subject{:02d}_{}_fmri_data_trn.hdf'.format(subject, modality))
    trndata5 = load_data(fname_tr5)

    fname_te5 = os.path.join(data_dir, 'subject{:02d}_{}_fmri_data_val.hdf'.format(subject, modality))
    tstdata5 = load_data(fname_te5)

    trim = 5
    zRresp = np.vstack([zscore(trndata5[story][5 + trim:-trim - 5]) for story in trndata5.keys()])
    zPresp = np.vstack([zscore(tstdata5[story][1][5 + trim:-trim - 5]) for story in tstdata5.keys()])

    return zRresp, zPresp


def load_low_level_textual_features(data_dir):
    """
    These files contain low-level textual and speech features
    """
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File(os.path.join(data_dir, 'features_trn_NEW.hdf'), 'r')
    base_features_val = h5py.File(os.path.join(data_dir, 'features_val_NEW.hdf'), 'r')
    return base_features_train, base_features_val


def load_z_low_level_feature(data_dir, low_level_feature, trim=5):
    """
    Load low-level textual features and z-score them across stories
    :param data_dir Directory containing fMRI data
    :return z_score_train, z_score_val
    """
    low_level_train, low_level_val = load_low_level_textual_features(data_dir)
    z_score_train = np.vstack([zscore(low_level_train[story][low_level_feature][5 + trim:-trim]) for story in low_level_train.keys()])
    z_score_val = np.vstack([zscore(low_level_val[story][low_level_feature][5 + trim:-trim]) for story in low_level_val.keys()])
    return z_score_train, z_score_val


def run_regression_and_predict(Rstim, Pstim, data_dir, subject, modality):
    """
    Train a joint model for two feature spaces
    :param Rstim Training stimuli with TR time points and N features. Each feature should be Z-scored across time
    :param Pstim Test stimuli with TP time points and M features. Each feature should be Z-scored across time
    :param data_dir Directory containing fMRI data
    :param subject number from 1 to 9
    :param modality Type of modality of the data, reading or listening
    :return voxelwise_correlations – Predictions of the joint model per layer
    """
    # Run regression
    nboots = 1  # Number of cross-validation runs.
    chunklen = 40  # Length of chunks to break data into.
    nchunks = 20  # Number of chunks to use in the cross-validated training.
    # Training responses with TR time points and M different responses
    zRresp, zPresp = load_subject_fmri(data_dir, subject, modality)
    # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    alphas = np.logspace(1, 3, 10)
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(Rstim), zRresp,
                                                         np.nan_to_num(Pstim), zPresp,
                                                         alphas, nboots, chunklen, nchunks,
                                                         singcutoff=1e-10, single_alpha=True)
    prediction = np.dot(np.nan_to_num(Pstim), wt)
    voxelwise_correlations = np.zeros((zPresp.shape[1],))  # create zero-filled array to hold correlations
    for voxel_index in range(zPresp.shape[1]):
        voxelwise_correlations[voxel_index] = np.corrcoef(zPresp[:, voxel_index], prediction[:, voxel_index])[0, 1]
    return voxelwise_correlations
