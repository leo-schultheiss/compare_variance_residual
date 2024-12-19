import os.path

import numpy as np
from ridge_utils.util import make_delayed
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer

from compare_variance_residual.common_utils.feature_utils import load_z_low_level_feature, load_subject_fmri, \
    get_prediction_path
from compare_variance_residual.common_utils.bootstrap_ridge import bootstrap_ridge


def train_low_level_model(data_dir: str, subject_num: int, modality: str, low_level_feature: str, number_of_delays=4):
    """
    Train a model to predict fMRI responses from low level features
    :param data_dir: str, path to data directory
    :param subject_num: int, subject number
    :param modality: str, choose modality (reading or listening)
    :param low_level_feature: str, low level feature to use
    :param number_of_delays: int, number of delays to use
    """
    Rstim, Pstim = load_z_low_level_feature(data_dir, low_level_feature)
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)

    # fit bootstrapped ridge regression model
    corrs, coef, alphas = bootstrap_ridge(Rstim, Rresp, Pstim, Presp)

    # save voxelwise correlations and predictions
    output_file = get_prediction_path(language_model=None, feature="low-level", modality=modality, subject=subject_num,
                                      low_level_feature=low_level_feature)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_file, corrs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train low level feature prediction model")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="../data")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--low_level_feature", help="Low level feature to use. Possible options include:\n"
                                                    "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    args = parser.parse_args()
    print(args)

    from himalaya import backend
    import logging

    backend.set_backend('torch', on_error='warn')
    logging.basicConfig(level=logging.DEBUG)

    train_low_level_model(args.data_dir, args.subject_num, args.modality, args.low_level_feature)
    print("All done!")
