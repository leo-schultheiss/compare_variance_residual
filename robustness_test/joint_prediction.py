import os.path
import h5py
import numpy as np
import logging
from ridge_utils.util import make_delayed

from common_utils.npp import zscore
from common_utils.training_utils import load_subject_fmri, \
    load_low_level_textual_features, load_downsampled_context_representations

logging.basicConfig(level=logging.DEBUG)


def predict_joint_model(data_dir, feature_filename, subject_num, modality, layer, low_level_features, output_dir):
    prediction_stim, training_stim = load_downsampled_context_representations(data_dir, feature_filename, layer)

    Rstim = training_stim
    Pstim = prediction_stim
    # create temporary array of shape 3737 x 0
    # Rstim = np.zeros((training_stim.shape[0], 0))
    # Pstim = np.zeros((prediction_stim.shape[0], 0))
    print("Rstim shape before join: ", Rstim.shape)
    print("Pstim shape before join: ", Pstim.shape)
    # join input features (context representations and low-level textual features)
    low_level_train, low_level_val = load_low_level_textual_features(data_dir)
    for low_level_feature in low_level_features.split(","):
        if low_level_feature not in low_level_train['story_01'].keys():
            raise ValueError(f"Low level feature {low_level_feature} not found in the dataset")
        z_base_feature_train = (
            np.vstack([zscore(low_level_train[story][low_level_feature][5 + 5:-5]) for story in low_level_train.keys()]))
        z_base_feature_val = (
            np.vstack([zscore(low_level_val[story][low_level_feature][5 + 5:-5]) for story in low_level_val.keys()]))
        print("z_base_feature_train shape: ", z_base_feature_train.shape)
        print("z_base_feature_val shape: ", z_base_feature_val.shape)
        Rstim = np.hstack((Rstim, z_base_feature_train))
        Pstim = np.hstack((Pstim, z_base_feature_val))
    print("Rstim shape after join: ", Rstim.shape)
    print("Pstim shape after join: ", Pstim.shape)

    # Delay stimuli to account for hemodynamic lag
    numer_of_delays = 4
    delays = range(1, numer_of_delays + 1)
    Rstim = make_delayed(np.array(Rstim), delays)
    Pstim = make_delayed(np.array(Pstim), delays)

    subject = f'0{subject_num}'
    voxelxise_correlations = prediction_joint_model(Rstim, Pstim, data_dir, subject, modality)
    # save voxelwise correlations and predictions
    main_dir = os.path.join(output_dir, modality, subject, low_level_features)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    np.save(os.path.join(str(main_dir), f"layer_{layer}"),
            voxelxise_correlations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict fMRI data using joint model")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="../data")
    parser.add_argument("-c", "--feature_filename",
                        help="File with context representations from LM for each story", type=str, default="../bert_base20.npy")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="listening")
    parser.add_argument("-l", "--layer", help="layer of the language model to use as input", type=int, default=9)
    parser.add_argument("--low_level_features",
                        help="Low level feature to use. Comma separation possible. Possible options include:\n"
                             "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters,phonemes")
    parser.add_argument("--output_dir", help="Output directory", type=str, default="../bert-joint-predictions")
    args = parser.parse_args()
    print(args)

    predict_joint_model(args.data_dir, args.feature_filename, args.subject_num, args.modality, args.layer,
                        args.low_level_features, args.output_dir)
