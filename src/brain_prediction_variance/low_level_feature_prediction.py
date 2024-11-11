import os.path

import numpy as np

from common_utils.training_utils import create_delayed_low_level_feature, run_regression_and_predict


def train_low_level_model(data_dir, subject_num, modality, low_level_feature, output_dir, numer_of_delays=6):
    # Delay stimuli to account for hemodynamic lag
    delays = range(1, numer_of_delays + 1)
    z_base_feature_train, z_base_feature_val = create_delayed_low_level_feature(data_dir, delays, low_level_feature)

    subject = f'0{subject_num}'
    voxelwise_correlations = run_regression_and_predict(z_base_feature_train, z_base_feature_val, data_dir, subject,
                                                        modality)
    # save voxelwise correlations and predictions
    main_dir = os.path.join(output_dir, modality, subject, low_level_feature)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    np.save(os.path.join(str(main_dir), f"low_level_model_prediction_voxelwise_correlation_delays{numer_of_delays}"),
            voxelwise_correlations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="data")
    parser.add_argument("-s", "--subject_num", help="Subject number", type=int, required=True)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--low_level_feature", help="Low level feature to use. Possible options include:\n"
                                                    "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    parser.add_argument("output_dir", help="Output directory", type=str)
    args = parser.parse_args()
    print(args)

    # use processes instead of threads
    processes = []
    import multiprocessing

    for i in range(1, 15):
        p = multiprocessing.Process(target=train_low_level_model, args=(
            args.data_dir, args.subject_num, args.modality, args.low_level_feature, args.output_dir, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # compare results
    for i in range(1, 15):
        main_dir = os.path.join(args.output_dir, args.modality, f'0{args.subject_num}', args.low_level_feature)
        voxelwise_correlations = np.load(
            os.path.join(str(main_dir), f"low_level_model_prediction_voxelwise_correlation_delays{i}.npy"))
        print(
            f"Voxelwise correlation for {args.low_level_feature} with {i} delays: {np.nan_to_num(voxelwise_correlations).mean()}")
