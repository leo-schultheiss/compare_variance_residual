import os


def get_data_home(dataset="moth_hour", data_home=None) -> str:
    # adapted from https://github.com/gallantlab/voxelwise_tutorials/blob/main/voxelwise_tutorials/io.py
    """Return the path of the data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times. By default the data dir is set to a folder named
    'compare_variance_residual' in the user home folder. Alternatively, it can be set
    by the 'COMPARE_VARIANCE_RESIDUAL' environment variable or programmatically
    by giving an explicit folder path. The '~' symbol is expanded to the user
    home folder. If the folder does not already exist, it is automatically
    created.

    Parameters
    ----------
    dataset : str | None
        Optional name of a particular dataset subdirectory, to append to the
        data_home path.
    data_home : str | None
        Optional path to data dir, to use instead of the default one.

    Returns
    -------
    data_home : str
        The path to the data directory.
    """
    if data_home is None:
        data_home = os.environ.get(
            'COMPARE_VARIANCE_RESIDUAL',
            os.path.join('~', 'compare_variance_residual'))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    if dataset is not None:
        data_home = os.path.join(data_home, dataset)

    return data_home

def get_results_home(result_home=None) -> str:
    return get_data_home("results", result_home)

if __name__ == "__main__":
    print(get_data_home())
    print(get_results_home())