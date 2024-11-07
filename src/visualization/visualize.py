import os

import matplotlib.pyplot as plt
import numpy as np

from common_utils.hdf_utils import map_to_flat

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


modality = "reading"
subject = "01"
layer = 11

# 1D arrays of correlation data
correlation_data =  [
    np.load(os.path.join("../../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True),
    np.load(os.path.join("../../bert-joint-predictions", subject, modality, "letters",f"joint_model_prediction_voxelwise_correlation_layer{layer}.npy"), allow_pickle=True),
]

# calculate average difference between the two
diff = np.nan_to_num(correlation_data[0]) - np.nan_to_num(correlation_data[1])
print(f"Average difference between the two models: {np.mean(diff)}")
print(f"maximum difference between the two models: {np.max(diff)}")


for data in correlation_data:
    plot_subject_voxel_data(subject, data)