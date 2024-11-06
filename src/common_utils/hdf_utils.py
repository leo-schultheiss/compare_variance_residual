"""
Utility function: loading data from hdf5 files and loading mapper files to display data on the
cortical surface.

"""
import os

import h5py
import numpy as np
import scipy.sparse

from common_utils.npp import zscore


def load_data(fname, key=None):
    """Function to load data from an hdf file.

    Parameters
    ----------
    fname: string
        hdf5 file name
    key: string
        key name to load. If not provided, all keys will be loaded.

    Returns
    -------
    data : dictionary
        dictionary of arrays

    """
    data = dict()
    with h5py.File(fname) as hf:
        if key is None:
            for k in hf.keys():
                print("{} will be loaded".format(k))
                data[k] = hf[k][()]
        else:
            data[key] = hf[key][()]
    return data


def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file

    Parameters
    ----------
    fname: string
        file name containing array to be loaded
    varname: string
        name of variable to be loaded

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.

    By Mark Lescroart

    """
    with h5py.File(fname) as hf:
        data = (hf['%s_data'%varname], hf['%s_indices'%varname], hf['%s_indptr'%varname])
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf['%s_shape'%varname])
    return sparsemat


def map_to_flat(voxels, mapper_file):
    """Generate flatmap image for an individual subject from voxel array

    This function maps a list of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    voxels: array
        n x 1 array of voxel values to be mapped
    mapper_file: string
        file containing mapping arrays

    Returns
    -------
    image : array
        flatmap image, (n x 1024)

    By Mark Lescroart

    """
    pixmap = load_sparse_array(mapper_file, 'voxel_to_flatmap')
    with h5py.File(mapper_file, mode='r') as hf:
        pixmask = hf['flatmap_mask'][()]
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(pixmask.shape)).astype(voxels.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
    img[pixmask] = mimg
    return img.T[::-1]

def load_subject_fmri(data_dir, subject, modality):
    """Load fMRI data for a subject, z-scored across stories"""
    fname_tr5 = os.path.join(data_dir, 'subject{}_{}_fmri_data_trn.hdf'.format(subject, modality))
    trndata5 = load_data(fname_tr5)
    print(trndata5.keys())

    fname_te5 = os.path.join(data_dir, 'subject{}_{}_fmri_data_val.hdf'.format(subject, modality))
    tstdata5 = load_data(fname_te5)
    print(tstdata5.keys())

    trim = 5
    zRresp = np.vstack([zscore(trndata5[story][5 + trim:-trim - 5]) for story in trndata5.keys()])
    zPresp = np.vstack([zscore(tstdata5[story][1][5 + trim:-trim - 5]) for story in tstdata5.keys()])

    return zRresp, zPresp
