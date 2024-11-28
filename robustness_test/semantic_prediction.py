import numpy as np
import argparse

import os
from himalaya.ridge import GroupRidgeCV

from robustness_test.common_utils.training_utils import load_context_representations_interpolated

trim = 5


def predict_brain_activity(data_dir, subject_num, featurename, modality, dirname, layer):
    predicion_stim, training_stim = load_context_representations_interpolated(data_dir, featurename, layer)
    # story_lengths = [len(downsampled_semanticseqs[story][0][5 + trim:-trim]) for story in training_story_names]
    # print(story_lengths)

    # Delay stimuli
    numer_of_delays = 4
    delays = range(1, numer_of_delays + 1)

    # print("FIR model delays: ", delays)
    # print(np.array(training_stim).shape)

    delayed_Rstim = make_delayed(np.array(training_stim), delays)
    delayed_Pstim = make_delayed(np.array(predicion_stim), delays)

    # print("delRstim shape: ", delayed_Rstim.shape)
    # print("delPstim shape: ", delayed_Pstim.shape)
    subject = f'0{subject_num}'
    main_dir = os.path.join(dirname, modality, subject)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    # Run regression
    model = GroupRidgeCV(alphas=np.logspace(0, 4, 10), groups=story_lengths, cv=5, n_jobs=1)
    voxcorrs = None
    # voxcorrs = run_regression_and_predict(delayed_Rstim, delayed_Pstim, data_dir, subject_num, modality)
    raise NotImplementedError("This function is not implemented yet")

    np.save(os.path.join(str(main_dir), "layer_" + str(layer)), voxcorrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain activity")
    parser.add_argument("--data_dir", help="Choose data directory", type=str, default="../data")
    parser.add_argument("--subject_num", help="Choose subject", type=int, default=1)
    parser.add_argument("--featurename", help="Choose feature", type=str, default="../bert_base20.npy")
    parser.add_argument("--modality", help="Choose modality", type=str, default="reading")
    parser.add_argument("--layer", help="Layer of natural language model to use for semantic representation of words",
                        type=int, default=9)
    parser.add_argument("--dirname", help="Choose Directory", type=str, default="../bert-semantic-predictions")
    args = parser.parse_args()
    print(args)

    # predict_brain_activity(args.data_dir, args.subject_num, args.featurename, args.modality, args.dirname, args.layer)

    import multiprocessing

    processes = []

    for layer in range(12):
        print(f"layer {layer}")
        for modality in ['reading', 'listening']:
            predict_brain_activity(args.data_dir, args.subject_num, args.featurename, modality, args.dirname, layer)
    print("All done")
