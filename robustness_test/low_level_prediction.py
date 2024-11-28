import os.path

import numpy as np
from himalaya.ridge import RidgeCV
from ridge_utils.util import make_delayed

from common_utils.training_utils import load_z_low_level_feature, load_subject_fmri
from robustness_test.common_utils.training_utils import get_prediction_path


def train_low_level_model(data_dir, subject_num, modality, low_level_feature, numer_of_delays=4):
    Rresp, Presp = load_subject_fmri(data_dir, subject_num, modality)
    Rstim, Pstim = load_z_low_level_feature(data_dir, low_level_feature)
    print(f"Rstim shape: {Rstim.shape}\nPstim shape: {Pstim.shape}")

    # delay stimuli to account for hemodynamic lag
    delays = range(1, numer_of_delays + 1)
    delayed_Rstim = make_delayed(np.array(Rstim), delays)
    delayed_Pstim = make_delayed(np.array(Pstim), delays)
    print(f"delayed_Rstim shape: {delayed_Rstim.shape}\ndelayed_Pstim shape: {delayed_Pstim.shape}")

    # train model
    model = RidgeCV(alphas=np.logspace(0, 3, 10), cv=5)
    model.fit(Rstim, Rresp)
    print(model.alpha)
    voxelwise_correlations = model.score(Pstim, Presp)

    # save voxelwise correlations and predictions
    output_file = get_prediction_path(language_model=None, feature="low-level", modality=modality, subject=subject_num,
                                      low_level_feature=low_level_feature)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_file, voxelwise_correlations)


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

    train_low_level_model(args.data_dir, args.subject_num, args.modality, args.low_level_feature)
