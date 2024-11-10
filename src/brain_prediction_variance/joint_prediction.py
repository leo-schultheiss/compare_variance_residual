import os.path

import h5py
import numpy as np
import logging
from ridge_utils.dsutils import make_word_ds, make_phoneme_ds, make_semantic_model
from ridge_utils.ridge import bootstrap_ridge

from common_utils.SemanticModel import SemanticModel, logger
from common_utils.hdf_utils import load_subject_fmri
from common_utils.npp import zscore
from common_utils.stimulus_utils import load_grids_for_stories, load_generic_trfiles
from common_utils.util import make_delayed

logging.basicConfig(level=logging.DEBUG)
data_dir = "data"


def signed_squared_correlation(rho):
    return rho ** 2 * np.sign(rho)


def union(v):
    """
    joint model correlations
    """
    pass


def intersection(rho_1, rho_2, rho_1_union_rho_2):
    """
    Approximates shared variance between two models
    :param rho_1 encoding performance for feature space 1
    :param rho_2 encoding performance for feature space 2
    :param rho_1_union_rho_2 encoding performance for joint model
    """
    import math
    return math.sqrt(signed_squared_correlation(rho_1) + signed_squared_correlation(rho_2) - signed_squared_correlation(
        rho_1_union_rho_2))


def load_low_level_textual_features():
    """
    These files contain low-level textual and speech features
    """
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File(os.path.join(data_dir, 'features_trn_NEW.hdf'), 'r')
    base_features_val = h5py.File(os.path.join(data_dir, 'features_val_NEW.hdf'), 'r')
    return base_features_train, base_features_val


def prediction_joint_model(Rstim, Pstim, data_dir, subject, modality, num_layers=12):
    """
    Train a joint model for two feature spaces
    :param Rstim Training stimuli with TR time points and N features. Each feature should be Z-scored across time
    :param Pstim Test stimuli with TP time points and M features. Each feature should be Z-scored across time
    :param data_dir Directory containing fMRI data
    :param subject Subject number from 1 to 9
    :param modality Modality of the data, reading or listening
    :return joint_model_predictions – Predictions of the joint model per layer
    """
    # Run regression
    nboots = 1  # Number of cross-validation runs.
    chunklen = 40  # Length of chunks to break data into.
    nchunks = 20  # Number of chunks to use in the cross-validated training.
    correlations_per_layer = []
    for layer_number in np.arange(num_layers):
        # Training responses with TR time points and M different responses
        zRresp, zPresp = load_subject_fmri(data_dir, subject, modality)
        # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
        alphas = np.logspace(1, 3, 10)
        wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(Rstim[layer_number]), zRresp,
                                                             np.nan_to_num(Pstim[layer_number]), zPresp,
                                                             alphas, nboots, chunklen, nchunks,
                                                             singcutoff=1e-10, single_alpha=True)
        prediction = np.dot(np.nan_to_num(Pstim), wt)
        voxelwise_correlations = np.zeros((zPresp.shape[1],))  # create zero-filled array to hold correlations
        for voxel_index in range(zPresp.shape[1]):
            voxelwise_correlations[voxel_index] = np.corrcoef(zPresp[:, voxel_index], prediction[:, voxel_index])[0, 1]
        correlations_per_layer.append(voxelwise_correlations)
    return correlations_per_layer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="data")
    parser.add_argument("-c", "--context_representations",
                        help="File with context representations from LM for each story", type=str, required=True)
    parser.add_argument("-s", "--subjectNum", help="Subject number", type=int, required=True)
    parser.add_argument("--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--layers", help="Number of layers", type=int, required=True)
    parser.add_argument("--low_level_feature",
                        help="Low level feature to use. Possible options include:\n"
                             "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    parser.add_argument("output_dir", help="Output directory", type=str)
    args = parser.parse_args()

    context_representations = np.load(args.context_representations, allow_pickle=True)

    training_story_names = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                            'life', 'myfirstdaywiththeyankees', 'naked',
                            'odetostepfather', 'souls', 'undertheinfluence']
    testing_story_names = ['wheretheressmoke']
    all_story_names = training_story_names + testing_story_names

    grids = load_grids_for_stories(all_story_names)

    # Load TRfiles
    trfiles = load_generic_trfiles(all_story_names, root="stimuli/trfiles")

    # Make word and phoneme datasequences
    word_data_sequences = make_word_ds(grids, trfiles)  # dictionary of {storyname : word DataSequence}
    eng1000 = SemanticModel.load(os.path.join(args.data_dir, "english1000sm.hf5"))

    semantic_sequence_representations = dict()  # dictionary to hold projected stimuli {story name : projected DataSequence}
    for i in np.arange(len(all_story_names)):
        print(all_story_names[i])
        semantic_sequence_representations[all_story_names[i]] = []
        for layer in np.arange(args.layers):
            temp = make_semantic_model(word_data_sequences[all_story_names[i]], [eng1000], [985])
            temp.data = np.nan_to_num(context_representations.item()[all_story_names[i]][layer])
            semantic_sequence_representations[all_story_names[i]].append(temp)

    # Downsample stimuli, since fMRI data covers multiple words in one TR
    interpolation_type = "lanczos"  # filter type used for averaging across representations
    window = 3  # number of lobes in Lanczos filter
    # num_layers = 12
    down_sampled_semantic_sequences = dict()  # dictionary to hold down-sampled stimuli
    for story in all_story_names:
        down_sampled_semantic_sequences[story] = []
        for layer in np.arange(args.layers):
            temp = semantic_sequence_representations[story][layer].chunksums(interpolation_type, window=window)
            down_sampled_semantic_sequences[story].append(temp)

    trim = 5
    training_stim = {}
    prediction_stim = {}
    for layer in np.arange(args.layers):
        training_stim[layer] = []
        training_stim[layer].append(
            np.vstack(
                [zscore(down_sampled_semantic_sequences[story][layer][5 + trim:-trim]) for story in
                 training_story_names]))

    for layer in np.arange(args.layers):
        prediction_stim[layer] = []
        prediction_stim[layer].append(
            np.vstack(
                [zscore(down_sampled_semantic_sequences[story][layer][5 + trim:-trim]) for story in
                 testing_story_names]))
    story_lengths = [len(down_sampled_semantic_sequences[story][0][5 + trim:-trim]) for story in training_story_names]
    print(story_lengths)

    # Delay stimuli to account for hemodynamic lag
    numer_of_delays = 6
    delays = range(1, numer_of_delays + 1)

    print("FIR model delays: ", delays)
    print(np.array(training_stim[0]).shape)
    delayed_Rstim = []
    for layer in np.arange(args.layers):
        delayed_Rstim.append(make_delayed(np.array(training_stim[layer])[0], delays))

    delayed_Pstim = []
    for layer in np.arange(args.layers):
        delayed_Pstim.append(make_delayed(np.array(prediction_stim[layer])[0], delays))

    # Print the sizes of these matrices
    print("delayed_Rstim shape: ", delayed_Rstim[0].shape)
    print("delayed_Pstim shape: ", delayed_Pstim[0].shape)

    # join input features (context representations and low-level textual features)
    base_features_train, base_features_val = load_low_level_textual_features()

    trim = 5
    np.random.seed(9)
    z_base_feature_train = np.vstack(
        [zscore(base_features_train[story][args.low_level_feature][5 + trim:-trim]) for story in
         base_features_train.keys()])
    z_base_feature_val = np.vstack(
        [zscore(base_features_val[story][args.low_level_feature][5 + trim:-trim]) for story in
         base_features_val.keys()])
    print("base features train shape: ", np.shape(z_base_feature_train))
    print("base features val shape: ", np.shape(z_base_feature_val))

    # join input features (context representations and low-level textual features)
    Rstim = [np.hstack((delayed_Rstim[layer_number], z_base_feature_train)) for layer_number in np.arange(args.layers)]
    Pstim = [np.hstack([delayed_Pstim[layer_number], z_base_feature_val]) for layer_number in np.arange(args.layers)]

    subject = f'0{args.subjectNum}'

    voxelxise_correlations = prediction_joint_model(Rstim, Pstim, args.data_dir, subject, args.modality,
                                                    args.layers)

    # save voxelwise correlations and predictions
    main_dir = os.path.join(args.output_dir, args.modality, subject, args.low_level_feature)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    for layer in range(len(voxelxise_correlations)):
        np.save(os.path.join(str(main_dir), f"joint_model_prediction_voxelwise_correlation_layer{layer}"),
                voxelxise_correlations[layer])
