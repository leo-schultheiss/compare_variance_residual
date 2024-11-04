import math

import numpy as np


def signed_squared_correlation(rho):
    return rho**2 * np.sign(rho)


def union(v):
    """
    joint model correlations
    """
    pass


def intersection(rho_1, rho_2, rho_1_union_rho_2):
    """
    Approximates shared variance between two models
    :param rho_1 encoding performance for feature space 1
    :param rho_2 encoding performance for feature space 2
    :param rho_1_union_rho_2 encoding performance for joint model
    """
    return math.sqrt(signed_squared_correlation(rho_1) + signed_squared_correlation(rho_2) - signed_squared_correlation(rho_1_union_rho_2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument('model_1', type=str, help='numpy file containing model prediction correlation data')
    parser.add_argument('model_2', type=str, help='numpy file containing model prediction correlation data')

    args = parser.parse_args()

    model_1 = np.load(args.model_1, allow_pickle=True)
    model_2 = np.load(args.model_2, allow_pickle=True)