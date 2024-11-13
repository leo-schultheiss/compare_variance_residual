import os

import numpy as np


def ssc(data: np.array):
    """
    Calculate the signed squared correlation of a matrix
    :param data: np.array
    :return: np.array
    """
    return (data ** 2) * np.sign(data)


def variance_partitioning(model_a, model_b, joint_model, output_dir):
    # load numpy correlation data
    model_a = np.load(model_a, allow_pickle=True)
    model_b = np.load(model_b, allow_pickle=True)
    joint_model = np.load(joint_model, allow_pickle=True)

    # remove nan values
    model_a = np.nan_to_num(model_a)
    model_b = np.nan_to_num(model_b)
    joint_model = np.nan_to_num(joint_model)

    # estimate the explained variance of each model using signed squared correlation
    squared_intersection = ssc(model_a) + ssc(model_b) - ssc(joint_model)
    squared_variance_a_minus_b = ssc(model_a) - squared_intersection
    squared_variance_b_minus_a = ssc(model_b) - squared_intersection

    # take roots of the squared values
    intersection = np.sqrt(squared_intersection)
    variance_a_minus_b = np.sqrt(squared_variance_a_minus_b)
    variance_b_minus_a = np.sqrt(squared_variance_b_minus_a)

    # save the results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "intersection.npy"), intersection)
    np.save(os.path.join(output_dir, "variance_a_minus_b.npy"), variance_a_minus_b)
    np.save(os.path.join(output_dir, "variance_b_minus_a.npy"), variance_b_minus_a)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Perform variance partitioning on two models")
    parser.add_argument("model_a", help="Model correlation data (Feature space A)", type=str)
    parser.add_argument("model_b", help="Model correlation data (Feature space B)", type=str)
    parser.add_argument("joint_model", help="Joint model correlation data (Feature space AUB)", type=str)
    parser.add_argument("output_dir", help="Output directory", type=str)
    args = parser.parse_args()
    print(args)

    variance_partitioning(args.model_a, args.model_b, args.joint_model, args.output_dir)
