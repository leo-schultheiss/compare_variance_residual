import numpy as np
import argparse

import os
from himalaya.ridge import RidgeCV
from ridge_utils.util import make_delayed

from robustness_test.common_utils.training_utils import load_context_representations_interpolated, load_subject_fmri

trim = 5


def predict_brain_activity(data_dir: str, feature_filename: str, layer: int, subject_num: int, modality: str,
                           output_directory: str):
    """
    Predict brain activity using semantic representations of words
    :param data_dir: data directory
    :param feature_filename: feature name
    :param layer: layer of natural language model to use for semantic representation of words
    :param subject_num: subject number
    :param modality: modality 'reading' or 'listening'
    :param output_directory: output directory
    """
    predicion_stim, training_stim = load_context_representations_interpolated(data_dir, feature_filename, layer)

    # Delay stimuli
    numer_of_delays = 4
    delays = range(1, numer_of_delays + 1)
    print("FIR model delays: ", delays)

    delayed_Rstim = make_delayed(np.array(training_stim), delays)
    delayed_Pstim = make_delayed(np.array(predicion_stim), delays)

    zRresp, zPresp = load_subject_fmri(data_dir, subject_num, modality)

    print("Rstim.shape: ", delayed_Rstim.shape)
    print("Rresp.shape: ", zRresp.shape)
    print("Pstim.shape: ", delayed_Pstim.shape)
    print("Presp.shape: ", zPresp.shape)

    # Run regression
    model = RidgeCV()
    model.fit(delayed_Rstim, zRresp)
    voxcorrs = model.score(delayed_Pstim, zPresp)
    raise NotImplementedError("This function is not implemented yet")

    subject = f'0{subject_num}'
    main_dir = os.path.join(output_directory, modality, subject)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    np.save(os.path.join(str(main_dir), "layer_" + str(layer)), voxcorrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain activity")
    parser.add_argument("--data_dir", help="Choose data directory", type=str, default="../data")
    parser.add_argument("--feature_filename", help="Choose feature", type=str, default="../bert_base20.npy")
    parser.add_argument("--layer", help="Layer of natural language model to use for semantic representation of words",
                        type=int, default=9)
    parser.add_argument("--subject_num", help="Choose subject", type=int, default=1)
    parser.add_argument("--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--dirname", help="Choose Directory", type=str, default="../bert-semantic-predictions")
    args = parser.parse_args()
    print(args)

    predict_brain_activity(args.data_dir, args.feature_filename, args.layer, args.subject_num, args.modality,
                           args.dirname)
