import os

import git
import h5py
import numpy as np
from ridge_utils.dsutils import make_word_ds, make_semantic_model
from ridge_utils.ridge import bootstrap_ridge

from robustness_test.common_utils.SemanticModel import SemanticModel
from robustness_test.common_utils.hdf_utils import load_data
from robustness_test.common_utils.npp import zscore
from robustness_test.common_utils.stimulus_utils import load_grids_for_stories, load_generic_trfiles


def load_subject_fmri(data_dir: str, subject: int, modality: str):
    """Load fMRI data for a subject, z-scored across stories"""
    fname_tr5 = os.path.join(data_dir, f'subject{subject:02}_{modality}_fmri_data_trn.hdf')
    trndata5 = load_data(fname_tr5)

    fname_te5 = os.path.join(data_dir, f'subject{subject:02}_{modality}_fmri_data_val.hdf')
    tstdata5 = load_data(fname_te5)

    trim = 5
    zRresp = np.vstack([zscore(trndata5[story][5 + trim:-trim - 5]) for story in trndata5.keys()])
    zPresp = np.vstack([zscore(tstdata5[story][1][5 + trim:-trim - 5]) for story in tstdata5.keys()])

    zRresp = np.nan_to_num(zRresp)
    zPresp = np.nan_to_num(zPresp)

    return zRresp, zPresp


def load_low_level_textual_features(data_dir):
    """
    These files contain low-level textual and speech features
    """
    # 'letters', 'numletters', 'numphonemes', 'numwords', 'phonemes', 'word_length_std'
    base_features_train = h5py.File(os.path.join(data_dir, 'features_trn_NEW.hdf'), 'r')
    base_features_val = h5py.File(os.path.join(data_dir, 'features_val_NEW.hdf'), 'r')
    return base_features_train, base_features_val


def load_z_low_level_feature(data_dir, low_level_feature, trim=5):
    """
    Load low-level textual features and z-score them across stories
    :param data_dir Directory containing fMRI data
    :return z_score_train, z_score_val
    """
    low_level_train, low_level_val = load_low_level_textual_features(data_dir)
    z_score_train = np.vstack(
        [zscore(low_level_train[story][low_level_feature][5 + trim:-trim]) for story in low_level_train.keys()])
    z_score_val = np.vstack(
        [zscore(low_level_val[story][low_level_feature][5 + trim:-trim]) for story in low_level_val.keys()])
    return z_score_train, z_score_val


def run_regression_and_predict(Rstim, Pstim, data_dir, subject, modality):
    """
    Train a joint model for two feature spaces
    :param Rstim Training stimuli with TR time points and N features. Each feature should be Z-scored across time
    :param Pstim Test stimuli with TP time points and M features. Each feature should be Z-scored across time
    :param data_dir Directory containing fMRI data
    :param subject number from 1 to 9
    :param modality Type of modality of the data, reading or listening
    :return voxelwise_correlations – Predictions of the joint model per layer
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


def get_prediction_path(language_model: str, feature: str, modality: str, subject: int, low_level_feature=None, layer=None):
    def get_git_root():
        git_repo = git.Repo(".", search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root

    if type(subject) == int:
        subject = f"{subject:02}"

    path_base = os.path.join(get_git_root(), "predictions", language_model, feature, modality, subject)

    filename = f"{low_level_feature}.npy" if feature == "low-level" else f"layer_{layer}.npy"
    joint_path_addition = f"{low_level_feature}" if feature == "joint" else ""

    path = os.path.join(str(path_base), joint_path_addition, filename)
    return path


def load_context_representations_interpolated(data_dir: str, feature_file: str, layer: int, save_file=None):
    """
    Load context representations from a file and downsample them to match the TRs
    :param data_dir: directory where data is stored
    :param feature_file: name of the file containing the context representations
    :param layer: layer of the model to use
    :param save_file: name of the file to save the downsampled context representations
    :return: downsampled context representations
    """
    stimul_features = np.load(feature_file, allow_pickle=True)
    # print(stimul_features.item().keys())
    training_story_names = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                            'life', 'myfirstdaywiththeyankees', 'naked',
                            'odetostepfather', 'souls', 'undertheinfluence']
    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    prediction_story_names = ['wheretheressmoke']
    all_story_names = training_story_names + prediction_story_names
    grids = load_grids_for_stories(all_story_names, root="../stimuli/grids")
    # Load TRfiles
    trfiles = load_generic_trfiles(all_story_names, root="../stimuli/trfiles")
    # Make word and phoneme datasequences
    wordseqs = make_word_ds(grids, trfiles)  # dictionary of {storyname : word DataSequence}
    eng1000 = SemanticModel.load(os.path.join(data_dir, "english1000sm.hf5"))
    semanticseqs = dict()  # dictionary to hold projected stimuli {story name : projected DataSequence}
    for story in all_story_names:
        semanticseqs[story] = make_semantic_model(wordseqs[story], [eng1000], [985])
    story_filenames = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                       'life', 'myfirstdaywiththeyankees', 'naked',
                       'odetostepfather', 'souls', 'undertheinfluence', 'wheretheressmoke']
    semanticseqs = dict()
    for i in np.arange(len(all_story_names)):
        semanticseqs[all_story_names[i]] = []
        temp = make_semantic_model(wordseqs[all_story_names[i]], [eng1000], [985])
        temp.data = np.nan_to_num(stimul_features.item()[story_filenames[i]][layer])
        semanticseqs[all_story_names[i]] = temp
    # Downsample stimuli
    interptype = "lanczos"  # filter type
    window = 3  # number of lobes in Lanczos filter
    downsampled_semanticseqs = dict()  # dictionary to hold downsampled stimuli
    for story in all_story_names:
        downsampled_semanticseqs[story] = semanticseqs[story].chunksums(interptype, window=window)

    if save_file:
        np.save(save_file, downsampled_semanticseqs)

    trim = 5
    training_stim = np.vstack(
        [zscore(downsampled_semanticseqs[story][5 + trim:-trim]) for story in training_story_names])
    predicion_stim = np.vstack(
        [zscore(downsampled_semanticseqs[story][5 + trim:-trim]) for story in prediction_story_names])
    return predicion_stim, training_stim
