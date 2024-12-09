import os.path

import numpy as np
from himalaya.ridge import ColumnTransformerNoStack
from ridge_utils.util import make_delayed
from sklearn.preprocessing import StandardScaler

from compare_variance_residual.common_utils.feature_utils import load_subject_fmri, \
    load_downsampled_context_representations, \
    get_prediction_path, load_z_low_level_feature
from compare_variance_residual.common_utils.ridge import bootstrap_ridge


def predict_joint_model(data_dir, feature_filename, language_model, subject_num, modality, layer, textual_features,
                        number_of_delays=4):
    textual_features = textual_features.split(",")

    # load features
    Pstim, Rstim, transformers = prepare_features(data_dir, feature_filename, layer, textual_features)
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)
    print("Rresp.shape: ", Rresp.shape)
    # create column transformer listing all column indices for each feature group
    ct = ColumnTransformerNoStack(transformers=transformers)

    # Delay stimuli to account for hemodynamic lag
    delays = range(1, number_of_delays + 1)
    Rstim = make_delayed(Rstim, delays)
    Pstim = make_delayed(Pstim, delays)
    print("Rstim.shape: ", Rstim.shape)
    print("Pstim.shape: ", Pstim.shape)

    # fit bootstrapped ridge regression model
    n_boots = 20  # Number of cross-validation runs.
    chunklen = 40  # Length of chunks to break data into.
    n_chunks = 20  # Number of chunks to use in the cross-validated training.
    alphas = np.logspace(0, 4, 10)
    wt, corrs, alphas, all_corrs, ind = bootstrap_ridge(Rstim, Rresp, Pstim, Presp, alphas, n_boots, chunklen, n_chunks,
                                                        ct, use_corr=True, single_alpha=True)

    # save voxelwise correlations
    output_file = get_prediction_path(language_model, "joint", modality, subject_num, textual_features, layer)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_file, corrs)


def prepare_features(data_dir, feature_filename, layer, textual_features):
    Rstim, Pstim = None, None
    transformers = []
    begin_ind = 0
    # join input features (context representations and low-level textual features)
    for feature in textual_features:
        if feature == "semantic":
            training_stim, prediction_stim = load_downsampled_context_representations(data_dir, feature_filename, layer)
        elif feature in ['letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std']:
            training_stim, prediction_stim = load_z_low_level_feature(data_dir, feature)
        else:
            raise ValueError(f"Textual feature {feature} not found in the dataset")
        print("training_stim.shape: ", training_stim.shape)
        print("prediction_stim.shape: ", prediction_stim.shape)
        if Rstim is None:
            Rstim, Pstim = training_stim, prediction_stim
        else:
            Rstim = np.hstack((Rstim, training_stim))
            Pstim = np.hstack((Pstim, prediction_stim))
        print("Rstim.shape: ", Rstim.shape)
        print("Pstim.shape: ", Pstim.shape)
        transformers.append((feature, StandardScaler(), slice(begin_ind, begin_ind + training_stim.shape[1] - 1)))
        begin_ind += training_stim.shape[1]
    return Pstim, Rstim, transformers


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
