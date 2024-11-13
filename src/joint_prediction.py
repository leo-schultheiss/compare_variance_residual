import os.path

import h5py
import numpy as np
import logging
from ridge_utils.dsutils import make_word_ds, make_semantic_model
from ridge_utils.ridge import bootstrap_ridge

from common_utils.SemanticModel import SemanticModel
from common_utils.npp import zscore
from common_utils.stimulus_utils import load_grids_for_stories, load_generic_trfiles
from common_utils.training_utils import make_delayed, load_subject_fmri, \
    load_low_level_textual_features

logging.basicConfig(level=logging.DEBUG)


def load_low_level_textual_features(data_dir):
    """
    These files contain low-level textual and speech features
    """
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File(os.path.join(data_dir, 'features_trn_NEW.hdf'), 'r')
    base_features_val = h5py.File(os.path.join(data_dir, 'features_val_NEW.hdf'), 'r')
    return base_features_train, base_features_val


def prediction_joint_model(Rstim, Pstim, data_dir, subject, modality):
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
    # Training responses with TR time points and M different responses
    zRresp, zPresp = load_subject_fmri(data_dir, subject, modality)
    # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    alphas = np.logspace(1, 3, 10)
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(Rstim), zRresp,
                                                         np.nan_to_num(Pstim), zPresp,
                                                         alphas, nboots, chunklen, nchunks,
                                                         singcutoff=1e-10, single_alpha=True)
    prediction = np.dot(np.nan_to_num(Pstim), wt)
    voxelwise_correlations = np.zeros((zPresp.shape[1],))  # create zero-filled array to hold correlations
    for voxel_index in range(zPresp.shape[1]):
        voxelwise_correlations[voxel_index] = np.corrcoef(zPresp[:, voxel_index], prediction[:, voxel_index])[0, 1]
    return voxelwise_correlations


def predict_joint_model(data_dir, context_representations, subject_num, modality, layer, low_level_feature, output_dir):
    context_representations = np.load(context_representations, allow_pickle=True)
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
    eng1000 = SemanticModel.load(os.path.join(data_dir, "english1000sm.hf5"))

    semantic_sequence_representations = dict()  # dictionary to hold projected stimuli {story name : projected DataSequence}
    for i in np.arange(len(all_story_names)):
        semantic_sequence_representations[all_story_names[i]] = []
        temp = make_semantic_model(word_data_sequences[all_story_names[i]], [eng1000], [985])
        temp.data = np.nan_to_num(context_representations.item()[all_story_names[i]][layer])
        semantic_sequence_representations[all_story_names[i]] = temp

    # Downsample stimuli, since fMRI data covers multiple words in one TR
    interpolation_type = "lanczos"  # filter type used for averaging across representations
    window = 3  # number of lobes in Lanczos filter
    down_sampled_semantic_sequences = dict()  # dictionary to hold down-sampled stimuli
    for story in all_story_names:
        down_sampled_semantic_sequences[story] = semantic_sequence_representations[story].chunksums(interpolation_type,
                                                                                                    window=window)
    trim = 5
    training_stim = np.vstack(
        [zscore(down_sampled_semantic_sequences[story][5 + trim:-trim]) for story in
         training_story_names])

    prediction_stim = np.vstack(
        [zscore(down_sampled_semantic_sequences[story][5 + trim:-trim]) for story in
         testing_story_names])

    print("Training stimuli shape: ", training_stim.shape)
    print("Prediction stimuli shape: ", prediction_stim.shape)

    # join input features (context representations and low-level textual features)
    low_level_train, low_level_val = load_low_level_textual_features(data_dir)
    z_score_train = np.vstack(
        [zscore(low_level_train[story][low_level_feature][5 + 5:-5]) for story in low_level_train.keys()])
    z_score_val = np.vstack(
        [zscore(low_level_val[story][low_level_feature][5 + 5:-5]) for story in low_level_val.keys()])
    z_base_feature_train, z_base_feature_val = z_score_train, z_score_val
    print("z_base_feature_train shape: ", z_base_feature_train.shape)
    print("z_base_feature_val shape: ", z_base_feature_val.shape)
    Rstim = np.hstack((training_stim, z_base_feature_train))
    Pstim = np.hstack((prediction_stim, z_base_feature_val))

    # Delay stimuli to account for hemodynamic lag
    numer_of_delays = 6
    delays = range(1, numer_of_delays + 1)
    print("FIR model delays: ", delays)
    print("Rstim shape: ", np.array(Rstim).shape)
    print("Pstim shape: ", np.array(Pstim).shape)

    delayed_Rstim = make_delayed(np.array(Rstim), delays)
    delayed_Pstim = make_delayed(np.array(Pstim), delays)
    # Print the sizes of these matrices
    print("delayed_Rstim shape: ", delayed_Rstim.shape)
    print("delayed_Pstim shape: ", delayed_Pstim.shape)
    subject = f'0{subject_num}'

    voxelxise_correlations = prediction_joint_model(Rstim, Pstim, data_dir, subject, modality)
    # save voxelwise correlations and predictions
    main_dir = os.path.join(output_dir, modality, subject, low_level_feature)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    np.save(os.path.join(str(main_dir), f"joint_model_prediction_voxelwise_correlation_layer{layer}"),
            voxelxise_correlations)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument("-d", "--data_dir", help="Directory containing data", type=str, default="data")
    parser.add_argument("-c", "--context_representations",
                        help="File with context representations from LM for each story", type=str, required=True)
    parser.add_argument("-s", "--subjectNum", help="Subject number", type=int, required=True)
    parser.add_argument("-m", "--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("-l", "--layer", help="layer of the language model to use as input", type=int, default=9)
    parser.add_argument("--low_level_feature",
                        help="Low level feature to use. Possible options include:\n"
                             "letters, numletters, numphonemes, numwords, phonemes, word_length_std",
                        type=str, default="letters")
    parser.add_argument("output_dir", help="Output directory", type=str)
    args = parser.parse_args()
    print(args)

    predict_joint_model(args.data_dir, args.context_representations, args.subjectNum, args.modality, args.layer,
                        args.low_level_feature, args.output_dir)
