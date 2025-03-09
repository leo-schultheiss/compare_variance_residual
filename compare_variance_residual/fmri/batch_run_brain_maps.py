import os

import numpy as np
from himalaya.backend import set_backend

from features import load_brain_data, load_feature
from residual_method import residual_method
from results import get_result_path
from variance_partitioning import variance_partitioning

data_dir = "../../data"
backend = set_backend('torch_cuda', on_error='warn')

if __name__ == "__main__":
    X_semantic, n_samples_train = load_feature(data_dir, "english1000")

    for subject in range(1, 10):
        for low_level_feature, modality in zip(['moten', 'powspec'], ["reading", "listening"]):
            print(f"Running {subject} {modality} {low_level_feature}")
            Y, _, ev = load_brain_data(data_dir, subject, modality)
            np.savetxt(os.path.join(get_result_path(modality, subject), "ev.csv"), ev, delimiter=",", header="ev")
            X_low_level, n_samples_train = load_feature(data_dir, low_level_feature)
            variance_partitioning(data_dir, subject, modality, low_level_feature, X_semantic=X_semantic,
                                  X_low_level=X_low_level, Y=Y, n_samples_train=n_samples_train)
            residual_method(data_dir, subject, modality, low_level_feature, X_semantic=X_semantic,
                            X_low_level=X_low_level, Y=Y, n_samples_train=n_samples_train)
