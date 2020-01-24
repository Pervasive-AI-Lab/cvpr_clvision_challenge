import numpy as np
import pickle as pkl
import os
import logging
from PIL import Image

from core50.dataset import CORE50


def main(args):

    # Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50('core50/data/', scenario=args.scenario)

    # Get the fixed test set
    test_x, test_y = dataset.get_test_set()

    # loop over the training incremental batches
    for i, train_batch in enumerate(dataset):

        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y = train_batch

        print("----------- batch {0} -------------".format(i))
        print("train shape: {0}, test_shape: {0}"
              .format(train_x.shape, train_y.shape))


    #TODO(final evaluation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="nicv2_79",
        choices=['ni', 'nc', 'nic', 'nicv2_79','nicv2_196', 'nicv2_391')
    parser.add_argument('--data_folder', type=str, default='data',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('-da', '--dataset', type=str, default='core50',
        help='Name of the dataset (default: core50).')



    args = parser.parse_args()

    main(args)
