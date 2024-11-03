import numpy as np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CheXpert NN argparser")
    parser.add_argument('model_1', type=str, help='numpy file containing model prediction correlation data')
    parser.add_argument('model_2', type=str, help='numpy file containing model prediction correlation data')

    args = parser.parse_args()

    model_1 = np.load(args.model_1, allow_pickle=True)
    model_2 = np.load(args.model_2, allow_pickle=True)