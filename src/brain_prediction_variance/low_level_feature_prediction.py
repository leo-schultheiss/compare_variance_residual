import os.path
import threading

import h5py
import numpy as np
import logging

from numpy import number
from ridge_utils.dsutils import make_word_ds, make_phoneme_ds, make_semantic_model
from ridge_utils.ridge import bootstrap_ridge

from common_utils.SemanticModel import SemanticModel, logger
from common_utils.hdf_utils import load_subject_fmri
from common_utils.npp import zscore
from common_utils.stimulus_utils import load_grids_for_stories, load_generic_trfiles
from common_utils.util import make_delayed, create_delayed_low_level_feature

logging.basicConfig(level=logging.DEBUG)


def prediction_joint_model(Rstim, Pstim, data_dir, subject, modality):
    """
    Train a joint model for two feature spaces
    :param Rstim Training stimuli with TR time points and N features. Each feature should be Z-scored across time
    :param Pstim Test stimuli with TP time points and M features. Each feature should be Z-scored across time
    :param data_dir Directory containing fMRI data
    :param subject Subject number from 1 to 9
    :param modality Modality of the data, reading or listening
    :return joint_model_predictions – Predictions of the joint model per layer
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


def train_low_level_model(data_dir, subject_num, modality, low_level_feature, output_dir, numer_of_delays=6):
    # Delay stimuli to account for hemodynamic lag
    delays = range(1, numer_of_delays + 1)
    z_base_feature_train, z_base_feature_val = create_delayed_low_level_feature(data_dir, delays, low_level_feature)

    subject = f'0{subject_num}'
    voxelwise_correlations = prediction_joint_model(z_base_feature_train, z_base_feature_val, data_dir, subject,
                                                    modality)
    # save voxelwise correlations and predictions
    main_dir = os.path.join(output_dir, modality, subject, low_level_feature)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    np.save(os.path.join(str(main_dir), f"low_level_model_prediction_voxelwise_correlation_delays{numer_of_delays}"),
            voxelwise_correlations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="data")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, required=True)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--low_level_feature",
                        help="Low level feature to use. Possible options include:\n"
                             "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    parser.add_argument("output_dir", help="Output directory", type=str)
    args = parser.parse_args()
    print(args)

    threads = []
    for i in range(1, 8):
        thread = threading.Thread(
            train_low_level_model(args.data_dir, args.subject_num, args.modality, args.low_level_feature,
                                  args.output_dir, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
