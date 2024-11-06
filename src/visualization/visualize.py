import os

import matplotlib.pyplot as plt
import numpy as np

from common_utils.hdf_utils import map_to_flat


modality = "reading"
subject = "01"
layer = 11

correlation_data = np.load(os.path.join("../../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True)


def plot_subject_voxel_data(subject_number, data):
    # Get path to data files
    fdir = os.path.abspath('../../data')
    # Map to subject flatmap
    map_file = os.path.join(fdir, f'subject{subject_number}_mappers.hdf')
    flatmap = map_to_flat(data, map_file)
    # Plot flatmap
    fig, ax = plt.subplots()
    _ = ax.imshow(flatmap, cmap='inferno')
    ax.axis('off')
    plt.show()


plot_subject_voxel_data(subject, correlation_data)