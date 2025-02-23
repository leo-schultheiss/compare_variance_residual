import numpy as np
import itertools as itools
from sklearn.model_selection import BaseCrossValidator


class TemporalChunkSplitter(BaseCrossValidator):
    """
    TemporalChunkSplitter is a class that generates train-test splits for time-series data.
    It is designed to split data into chunks of a fixed length, and then randomly select a fixed number of chunks
    to be used as the test set. The remaining chunks are used as the training set. This process is repeated for a
    fixed number of iterations.

    Parameters
    ----------
    num_splits : int
        Number of splits to generate.
    chunk_len : int
        Length of each chunk.
    num_chunks : int
        Number of chunks to use as the test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_indexes : list
        List of indexes for the training set.
    tune_indexes : list
        List of indexes for the test set.
    """

    # rng = np.random.RandomState(seed)
    # all_indexes = range(num_examples)
    # index_chunks = list(zip(*[iter(all_indexes)] * chunk_len))
    # splits_list = []
    # for _ in range(num_splits):
    #     rng.shuffle(index_chunks)
    #     tune_indexes_ = list(itools.chain(*index_chunks[:num_chunks]))
    #     train_indexes_ = list(set(all_indexes) - set(tune_indexes_))
    #     splits_list.append((train_indexes_, tune_indexes_))
    #
    # valinds = [splits[1] for splits in splits_list]
    #
    # correlation_matrices = []
    # for bi in counter(range(nboots), countevery=1, total=nboots):
    #     train_indexes_, tune_indexes_ = splits_list[bi]
    #
    #     # Select data
    #     stim_train_ = stim_train[train_indexes_, :]
    #     stim_test_ = stim_train[tune_indexes_, :]
    #     resp_train_ = resp_train[train_indexes_, :]
    #     resp_test_ = resp_train[tune_indexes_, :]
    #

    def __init__(self, num_splits, chunk_len, num_chunks, seed=42):
        self.num_splits = num_splits
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.num_splits

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.seed)
        num_examples = len(X)
        all_indexes = range(num_examples)
        index_chunks = list(zip(*[iter(all_indexes)] * self.chunk_len))

        # creates bootstrapped, random (with replacement) index slices
        for _ in range(self.num_splits):
            rng.shuffle(index_chunks)
            tune_indexes = list(itools.chain(*index_chunks[:self.num_chunks]))
            train_indexes = list(set(all_indexes) - set(tune_indexes))
            yield train_indexes, tune_indexes