import math
import os.path

import h5py
import numpy as np
from ridge_utils.dsutils import make_word_ds, make_phoneme_ds, make_semantic_model
from ridge_utils.ridge import bootstrap_ridge
from torch.ao.quantization import default_embedding_fake_quant

from brain_prediction_standard.predict_brain_activity import data_dir
from common_utils.SemanticModel import SemanticModel
from common_utils.hdf_utils import load_subject_fmri
from common_utils.npp import zscore
from common_utils.stimulus_utils import load_grids_for_stories, load_generic_trfiles
from common_utils.util import make_delayed


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
    return math.sqrt(signed_squared_correlation(rho_1) + signed_squared_correlation(rho_2) - signed_squared_correlation(
        rho_1_union_rho_2))

def load_low_level_textual_features():
    """
    These files contain low-level textual and speech features
    """
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File(os.path.join(data_dir, 'features_trn_NEW.hdf'), 'r+')
    base_features_val = h5py.File(os.path.join(data_dir,'features_val_NEW.hdf'), 'r+')
    return base_features_train, base_features_val


def train_joint_model(Rstim, Pstim, data_dir, subject, modality, num_layers=12):
    """
    Train a joint model for two feature spaces
    :param Rstim – Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    :param Pstim – Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    :param data_dir – Directory containing fMRI data
    :param subject – Subject number
    :param modality – Modality of the data

    :return joint_model_predictions – Predictions of the joint model per layer
    """
    # Run regression
    nboots = 1  # Number of cross-validation runs.
    chunklen = 40  #
    nchunks = 20
    correlations_per_layer = []
    for layer_number in np.arange(num_layers):
        # Training responses with TR time points and M different responses
        zRresp, zPresp = load_subject_fmri(data_dir, subject, modality)
        alphas = np.logspace(1, 3,
                             10)  # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
        wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(Rstim[layer_number]), zRresp,
                                                             np.nan_to_num(Pstim[layer_number]), zPresp,
                                                             alphas, nboots, chunklen, nchunks,
                                                             singcutoff=1e-10, single_alpha=True)
        pred = np.dot(np.nan_to_num(delayed_Pstim[layer_number]), wt)

        # np.save(os.path.join(main_dir+'/'+save_dir, "test_"+str(eachlayer)),zPresp)
        # np.save(os.path.join(main_dir+'/'+save_dir, "pred_"+str(eachlayer)),pred)
        voxcorrs = np.zeros((zPresp.shape[1],))  # create zero-filled array to hold correlations
        for vi in range(zPresp.shape[1]):
            voxcorrs[vi] = np.corrcoef(zPresp[:, vi], pred[:, vi])[0, 1]
        print(voxcorrs)
        correlations_per_layer.append(voxcorrs)
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
    parser.add_argument('--model_1', type=str, help='numpy file containing model prediction correlation data')
    parser.add_argument('--model_2', type=str, help='numpy file containing model prediction correlation data')
    args = parser.parse_args()

    subject = f'0{args.subjectNum}'
    context_representations = np.load(args.context_representations, allow_pickle=True)
    # print(context_representations.item().keys())

    training_story_names = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                            'life', 'myfirstdaywiththeyankees', 'naked',
                            'odetostepfather', 'souls', 'undertheinfluence']
    testing_story_names = ['wheretheressmoke']
    all_story_names = training_story_names + testing_story_names

    grids = load_grids_for_stories(all_story_names)

    # Load TRfiles
    trfiles = load_generic_trfiles(all_story_names, root="stimuli/trfiles")

    # Make word and phoneme datasequences
    wordseqs = make_word_ds(grids, trfiles)  # dictionary of {storyname : word DataSequence}
    phonseqs = make_phoneme_ds(grids, trfiles)  # dictionary of {storyname : phoneme DataSequence}
    eng1000 = SemanticModel.load(os.path.join(args.data_dir, "english1000sm.hf5"))

    semantic_sequence_representations = dict()  # dictionary to hold projected stimuli {story name : projected DataSequence}
    for i in np.arange(len(all_story_names)):
        print(all_story_names[i])
        semantic_sequence_representations[all_story_names[i]] = []
        for layer in np.arange(args.layers):
            temp = make_semantic_model(wordseqs[all_story_names[i]], [eng1000], [985])
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
    print("delRstim shape: ", delayed_Rstim[0].shape)
    print("delPstim shape: ", delayed_Pstim[0].shape)

    joint_model_predictions = train_joint_model(delayed_Rstim, delayed_Pstim, args.data_dir, subject, args.modality, args.layers)

    main_dir = os.path.join(args.data_dir, args.modality, subject)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # if model_1 and model_2 args are given, load them
    if args.model_1 and args.model_2:
        model_1 = np.load(args.model_1, allow_pickle=True)
        model_2 = np.load(args.model_2, allow_pickle=True)
    else:
        raise NotImplementedError("Must provide model_1 and model_2 args")
