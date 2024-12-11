import os.path
import numpy as np
from himalaya.ridge import ColumnTransformerNoStack
from voxelwise_tutorials.delayer import Delayer

from compare_variance_residual.common_utils.feature_utils import load_subject_fmri, \
    load_downsampled_context_representations, \
    get_prediction_path, load_z_low_level_feature
from compare_variance_residual.common_utils.ridge import bootstrap_ridge


def predict_joint_model(data_dir, feature_filename, language_model, subject_num, modality, layer, textual_features,
                        number_of_delays=4):
    # load features
    Pstim, Rstim, transformers = prepare_features(data_dir, feature_filename, layer, textual_features, number_of_delays)
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)
    print("Rresp.shape: ", Rresp.shape)
    print("Presp.shape: ", Presp.shape)

    # fit bootstrapped ridge regression model
    ct = ColumnTransformerNoStack(transformers=transformers)
    corrs, alphas = bootstrap_ridge(Rstim, Rresp, Pstim, Presp, ct)

    # save voxelwise correlations
    output_file = get_prediction_path(language_model, "joint", modality, subject_num, textual_features, layer)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_file, corrs)


def prepare_features(data_dir, feature_filename, layer, textual_features, number_of_delays):
    delays = range(1, number_of_delays + 1)
    all_Rstim, all_Pstim = None, None
    transformers = []
    begin_ind = 0
    # join input features (context representations and low-level textual features)
    for feature in textual_features.split(","):
        if feature == "semantic":
            Rstim, Pstim = load_downsampled_context_representations(data_dir, feature_filename, layer)
        elif feature in ['letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std']:
            Rstim, Pstim = load_z_low_level_feature(data_dir, feature)
        else:
            raise ValueError(f"Textual feature {feature} not found in the dataset")
        print("Rstim.shape: ", Rstim.shape)
        print("Pstim.shape: ", Pstim.shape)
        if all_Rstim is None or all_Pstim is None:
            all_Rstim, all_Pstim = Rstim, Pstim
        else:
            all_Rstim = np.hstack((all_Rstim, Rstim))
            all_Pstim = np.hstack((all_Pstim, Pstim))
        print("all_Rstim.shape: ", all_Rstim.shape)
        print("all_Pstim.shape: ", all_Pstim.shape)
        transformers.append((feature, Delayer(delays), slice(begin_ind, begin_ind + Rstim.shape[1] - 1)))
        begin_ind += Rstim.shape[1]
    return all_Pstim, all_Rstim, transformers


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict fMRI data using joint model")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="../data")
    parser.add_argument("-c", "--feature_filename",
                        help="File with context representations from LM for each story", type=str,
                        default="../bert_base20.npy")
    parser.add_argument("--language_model", help="Language model, where the features are extracted from", type=str,
                        default="bert")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("-l", "--layer", help="layer of the language model to use as input", type=int, default=9)
    parser.add_argument("--textual_features",
                        help="Comma separated, textual feature to use as input. Possible options include:\n"
                             "semantic, letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="semantic,letters")
    args = parser.parse_args()
    print(args)

    from himalaya import backend
    import logging

    backend.set_backend('torch', on_error='warn')
    logging.basicConfig(level=logging.DEBUG)

    logging.basicConfig(level=logging.DEBUG)
    predict_joint_model(args.data_dir, args.feature_filename, args.language_model, args.subject_num, args.modality,
                        args.layer, args.textual_features)
    print("All done!")
