#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Massimo Caccia, Pau Rodriguez,        #
# Lorenzo Pellegrini. All rights reserved.                                     #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-02-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

Getting Started example for the CVPR 2020 CLVision Challenge. It will load the
data and create the submission file for you in the
cvpr_clvision_challenge/submissions directory.

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import time
import copy
from core50.dataset import CORE50
import torch
import numpy as np
from utils.train_test import train_net, test_multitask, preprocess_imgs
import torchvision.models as models
from utils.common import create_code_snapshot


def main(args):

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root='core50/data/', scenario=args.scenario,
                     preload=args.preload_data)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    if args.classifier == 'ResNet18':
        classifier = models.resnet18(pretrained=True)
#         classifier.fc = torch.nn.Dropout(0.4)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
    
    if args.classifier == 'ResNet34':
        classifier = models.resnet34(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
        
    if args.classifier == 'SqueezeNet1.1':
        classifier = models.squeezenet1_1(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
        
    if args.dropout:
        print(f"Using Dropout of {args.dropout}")
        dropout_layer_fc = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(512, 128),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(64, args.n_classes)
        )
        classifier.fc = dropout_layer_fc
        

    opt = torch.optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=0.0001)
#     opt = torch.optim.Adam(classifier.parameters(), lr=args.lr)
#     init_lr = args.lr
#     lr = 2 * init_lr
    criterion = torch.nn.CrossEntropyLoss()

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = None

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch
        
        # trying opt here for decay
#         lr = lr*0.5
#         opt = torch.optim.SGD(classifier.parameters(), lr=lr)

        # adding eventual replay patterns to the current batch
        idxs_cur = np.random.choice(
            train_x.shape[0], args.replay_examples, replace=False
        )

        if i == 0:
            ext_mem = [train_x[idxs_cur], train_y[idxs_cur]]
        else:
            ext_mem = [
                np.concatenate((train_x[idxs_cur], ext_mem[0])),
                np.concatenate((train_y[idxs_cur], ext_mem[1]))]

        train_x = np.concatenate((train_x, ext_mem[0]))
        train_y = np.concatenate((train_y, ext_mem[1]))

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        # train the classifier on the current batch/task
        _, _, stats = train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, t,
            args.epochs, preproc=preprocess_imgs
        )
        if args.scenario == "multi-task-nc":
            heads.append(copy.deepcopy(classifier.fc))

        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test on the validation set
        stats, _ = test_multitask(
            classifier, full_valdidset, args.batch_size,
            preproc=preprocess_imgs, multi_heads=heads,
            last_layer_name="fc", verbose=False
        )

        valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(
        classifier, full_testset, args.batch_size, preproc=preprocess_imgs,
        multi_heads=heads, last_layer_name="fc", verbose=False
    )

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
                        choices=['ResNet18', "ResNet34", "SqueezeNet1.1"])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    
    parser.add_argument('-dp', '--dropout', type=float, default=0.0, 
                        help='dropout as a propotion between 0 and 1')

    # Continual Learning
    parser.add_argument('--replay_examples', type=int, default=0,
                        help='data examples to keep in memory for each batch '
                             'for replay.')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    main(args)
