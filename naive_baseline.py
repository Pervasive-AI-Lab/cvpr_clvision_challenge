#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-02-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

Getting Started example for the CVPR 2020 CLVision Challenge. It will
load the data and create the submission file for you in
cvpr_clvision_challenge/submissions/

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from core50.dataset import CORE50
import torch
import torch.nn.functional as F
from utils import models
from utils.train_test import train_net, get_accuracy, test_multitask


def main(args):
    print("Scenario chosen: ", args.scenario)
    # Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50(root='core50/data/', scenario=args.scenario,
                     preload=args.preload_data)

    # Get the fixed test set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()
    print("Recovering test set...")
    full_testset = dataset.get_full_test_set()

    # model
    if args.classifier == 'ResNet18':
        classifier = models.ResNet18(args).to(args.device)
    elif args.classifier == 'MLP':
        classifier = models.MLP(args).to(args.device)

    opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # loop over the training incremental batches
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        print("----------- batch {0} -------------".format(i))
        print("train shape: {0}, test_shape: {0}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, t,
            args.epochs
        )
        stats = test_multitask(
            classifier, full_valdidset, args.batch_size
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="ni",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
                        choices=['ResNet18', 'MLP'])
    parser.add_argument('-hs', '--hidden_size', type=int, default=64,
                        help='Number of channels in each convolution layer of '
                             'the VGG network or hidden size of an MLP. If '
                             'None, kept to default')

    # CL
    parser.add_argument('--replay', type=bool, default=True,
                        help='enable replay')
    parser.add_argument('--mem_size', type=int, default=600,
                        help='number of saved samples per class')
    parser.add_argument('--replay_size', type=int, default=60,
                        help='number of replays per batch')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers to use for data-loading '
                             '(default: 1).')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    # convert from per class to total memory
    args.mem_size = args.mem_size * args.n_classes

    main(args)
