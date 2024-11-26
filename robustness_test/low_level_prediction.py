import os.path

import numpy as np

from common_utils.training_utils import load_z_low_level_feature, run_regression_and_predict, make_delayed


def train_low_level_model(data_dir, subject_num, modality, low_level_feature, output_dir, numer_of_delays=4):
    Rstim, Pstim = load_z_low_level_feature(data_dir, low_level_feature)
    print(f"Rstim shape: {Rstim.shape}\nPstim shape: {Pstim.shape}")

    # delay stimuli to account for hemodynamic lag
    delays = range(1, numer_of_delays + 1)
    delayed_Rstim = make_delayed(np.array(Rstim), delays)
    delayed_Pstim = make_delayed(np.array(Pstim), delays)

    print(f"delayed_Rstim shape: {delayed_Rstim.shape}\ndelayed_Pstim shape: {delayed_Pstim.shape}")
    voxelwise_correlations = run_regression_and_predict(delayed_Rstim, delayed_Pstim, data_dir, subject_num,
                                                        modality)

    # save voxelwise correlations and predictions
    main_dir = os.path.join(output_dir, modality, f'0{subject_num}')
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    np.save(os.path.join(str(main_dir), f"{low_level_feature}"),
            voxelwise_correlations)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train low level feature prediction model")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="../data")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, default=1)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--low_level_feature", help="Low level feature to use. Possible options include:\n"
                                                    "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    parser.add_argument("--output_dir", help="Output directory", type=str, default="../bert-low-level-predictions")
    args = parser.parse_args()
    print(args)

    # train_low_level_model(args.data_dir, args.subject_num, args.modality, args.low_level_feature, args.output_dir)

    processes = []
    import multiprocessing

    low_level_features = ["letters", "numletters", "numphonemes", "numwords", "phonemes", "word_length_std"]

    for subject_num in range(1, 3):
        for low_level_feature in low_level_features:
            train_low_level_model(args.data_dir, subject_num, args.modality, low_level_feature, args.output_dir)

    print("All processes finished")
