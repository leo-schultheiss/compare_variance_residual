# Comparing Variance Partitioning and the Residual Method for Interpreting Brain Recordings

This repository contains code for the Bachelor's Thesis "Comparing Variance Partitioning and the Residual Method for Interpreting Brain Recordings".

## Data and Features

The fMRI data and its related features should be at the repository root with the following structure:
```
data/
├─ features/
│  ├─ features_trn_NEW.hdf
│  ├─ ...
├─ mappers/
│  ├─ subject01_mappers.hdf
│  ├─ ...
├─ responses/
│  ├─ subject01_listening_fmri_data_trn.hdf
│  ├─ ...
```

The mappers, subject responses and feature can be found at https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri

## Quickstart 

After downloading said files above, head over to the [quickstart notebook](compare_variance_residual/quickstart.ipynb) for a short intro into how to use the code contained in this package.

## TODO 

- separate notebooks from python files 
  - the python files contain the implementations for variance partitioning, the residual method, simulated data 
  - the notebooks contain analyses of both methods and their properties under different circumstances
- move the data folder mentioned above to somewhere else (see [voxelwise_tutorials](https://github.com/gallantlab/voxelwise_tutorials) repo