import argparse
import os

import numpy as np
from himalaya.ridge import ColumnTransformerNoStack
from ridge_utils.util import make_delayed
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer

from compare_variance_residual.common_utils.feature_utils import load_downsampled_context_representations, \
    load_subject_fmri, \
    get_prediction_path
from compare_variance_residual.common_utils.ridge import bootstrap_ridge

trim = 5


def predict_brain_activity(data_dir: str, feature_filename: str, language_model: str, layer: int, subject_num: int,
                           modality: str, number_of_delays=4):
    """
    Predict brain activity using semantic representations of words
    :param data_dir: data directory
    :param feature_filename: feature name
    :param language_model: language model
    :param layer: layer of natural language model to use for semantic representation of words
    :param subject_num: subject number
    :param modality: modality 'reading' or 'listening'
    :param number_of_delays: number of delays to account for hemodynamic response
    """
    # Load data
    Rstim, Pstim = load_downsampled_context_representations(data_dir, feature_filename, layer)
    print("Rstim.shape: ", Rstim.shape)
    print("Pstim.shape: ", Pstim.shape)
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)
    print("Rresp.shape: ", Rresp.shape)
    print("Presp.shape: ", Presp.shape)

    # Delay stimuli
    delays = range(1, number_of_delays + 1)

    # fit bootstrapped ridge regression model
    ct = ColumnTransformerNoStack([("semantic", Delayer(delays), slice(0, Rstim.shape[1] - 1))])
    wt, corrs, alphas, all_corrs, ind = bootstrap_ridge(Rstim, Rresp, Pstim, Presp, ct)

    # save results
    output_file = get_prediction_path(language_model, "semantic", modality, subject_num, layer=layer)
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(output_file, corrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain activity")
    parser.add_argument("--data_dir", help="Choose data directory", type=str, default="../data")
    parser.add_argument("--feature_filename", help="Choose feature", type=str, default="../bert_base20.npy")
    parser.add_argument("--language_model", help="Choose language model", type=str, default="bert")
    parser.add_argument("--layer", help="Layer of natural language model to use for semantic representation of words",
                        type=int, default=9)
    parser.add_argument("--subject_num", help="Choose subject", type=int, default=1)
    parser.add_argument("--modality", help="Choose modality", type=str, default="reading")
    args = parser.parse_args()
    print(args)

    from himalaya.backend import set_backend
    import logging

    backend = set_backend("torch", on_error="warn")
    logging.basicConfig(level=logging.DEBUG)

    predict_brain_activity(data_dir=args.data_dir, feature_filename=args.feature_filename,
                           language_model=args.language_model, layer=args.layer, subject_num=args.subject_num,
                           modality=args.modality)
