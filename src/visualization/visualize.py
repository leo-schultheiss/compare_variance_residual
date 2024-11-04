import os

import matplotlib.pyplot as plt
import numpy as np

from common_utils.hdf_utils import map_to_flat

# Get path to data files
fdir = os.path.abspath('../../data')

modality = "reading"
subject = "01"
layer = 11

correlation_data = np.load(os.path.join("../../bert-predictions", modality, subject, f"layer_{layer}.npy"), allow_pickle=True)

# Map to subject flatmap
map_file = os.path.join(fdir, 'subject{}_mappers.hdf'.format(subject))
flatmap = map_to_flat(correlation_data, map_file)

# Plot flatmap
fig, ax = plt.subplots()
_ = ax.imshow(flatmap, cmap='inferno')
ax.axis('off')
plt.show()