import argparse
import os

import numpy as np

from compare_variance_residual.fmri_comparison.common_utils.feature_utils import load_downsampled_context_representations, \
    load_subject_fmri, \
    get_prediction_path
from compare_variance_residual.fmri_comparison.common_utils.bootstrap_ridge import bootstrap_ridge

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
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)

    # fit bootstrapped ridge regression model
    corrs, coef, alphas = bootstrap_ridge(Rstim, Rresp, Pstim, Presp, nboots=2)

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
    print("Done")
