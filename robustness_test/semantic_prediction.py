import argparse
import os

import numpy as np
from himalaya.ridge import RidgeCV
from ridge_utils.util import make_delayed

from robustness_test.common_utils.training_utils import load_downsampled_context_representations, load_subject_fmri, \
    get_prediction_path

trim = 5


def predict_brain_activity(data_dir: str, feature_filename: str, language_model: str, layer: int, subject_num: int,
                           modality: str):
    """
    Predict brain activity using semantic representations of words
    :param data_dir: data directory
    :param feature_filename: feature name
    :param layer: layer of natural language model to use for semantic representation of words
    :param subject_num: subject number
    :param modality: modality 'reading' or 'listening'
    """
    # Load data
    training_stim, predicion_stim = load_downsampled_context_representations(data_dir, feature_filename, layer)
    zRresp, zPresp = load_subject_fmri(data_dir, subject_num, modality)

    # Delay stimuli
    numer_of_delays = 4
    delays = range(1, numer_of_delays + 1)
    print("FIR model delays: ", delays)

    delayed_Rstim = make_delayed(np.array(training_stim), delays)
    delayed_Pstim = make_delayed(np.array(predicion_stim), delays)

    print("Rstim.shape: ", delayed_Rstim.shape)
    print("Rresp.shape: ", zRresp.shape)
    print("Pstim.shape: ", delayed_Pstim.shape)
    print("Presp.shape: ", zPresp.shape)

    # Run regression
    model = RidgeCV(alphas=np.logspace(1, 3, 10))
    model.fit(delayed_Rstim, zRresp)
    voxelwise_correlations = model.score(delayed_Pstim, zPresp)

    # save results
    output_file = get_prediction_path(language_model, "semantic", modality, subject_num, layer=layer)
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(output_file, voxelwise_correlations)


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

    predict_brain_activity(data_dir=args.data_dir, feature_filename=args.feature_filename,
                           language_model=args.language_model, layer=args.layer, subject_num=args.subject_num,
                           modality=args.modality)
