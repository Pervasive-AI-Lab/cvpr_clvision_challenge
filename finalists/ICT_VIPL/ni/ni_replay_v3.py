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
from core50.dataset_npl import CORE50
import torch
import numpy as np
from utils.train_test_npl import train_net, test_multitask
import torchvision.models as models
from utils.common import create_code_snapshot, Cache

def main(args):

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # vars to update over time
    vars = Cache(args.cache_dir)
    vars.valid_acc = []
    vars.ext_mem_sz = []
    vars.ram_usage = []
    vars.heads = []
    vars.batch = 0
    vars.ext_mem = None

    # load cache
    if args.cache_dir:
        vars.load()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root=args.data_dir, scenario=args.scenario,
                     start_batch=vars.batch, cumul=args.cumul)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    if args.cls == 'ResNet18':
        classifier = models.resnet18(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
    if args.cls == 'ResNet34':
        classifier = models.resnet34(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
    if args.cls == 'ResNet50':
        classifier = models.resnet50(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)
    if args.cls == 'ResNet101':
        classifier = models.resnet101(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)
    if args.cls == 'WideResNet50':
        classifier = models.wide_resnet50_2(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)
    if args.cls == 'ResNext50':
        classifier = models.resnext50_32x4d(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)

    criterion = torch.nn.CrossEntropyLoss()

    if hasattr(vars, 'state_dict'):
      classifier.load_state_dict(vars.state_dict)

    # loop over the training incremental batches (x, y, t)
    for train_batch in dataset:
        train_x, train_y, t = train_batch

        # adding eventual replay patterns to the current batch
        if args.replay_strategy == 'equal_interval':
            idxs_cur = np.arange(1, train_x.shape[0], train_x.shape[0] / args.replay_examples, dtype=np.int64)
        else:
            idxs_cur = np.random.choice(train_x.shape[0], args.replay_examples, replace=False)

        if vars.batch == 0:
            vars.ext_mem = [train_x[idxs_cur], train_y[idxs_cur]]
        else:
            vars.ext_mem = [
                np.concatenate((train_x[idxs_cur], vars.ext_mem[0])),
                np.concatenate((train_y[idxs_cur], vars.ext_mem[1]))]

        train_x = np.concatenate((train_x, vars.ext_mem[0]))
        train_y = np.concatenate((train_y, vars.ext_mem[1]))

        print("----------- batch {0} -------------".format(vars.batch))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)
        if args.lr_step > 0:
          lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
        else:
          lr_scheduler = None

        # train the classifier on the current batch/task
        _, _, stats = train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, t,
            args.epochs, args.aug, lr_scheduler=lr_scheduler
        )
        if args.scenario == "multi-task-nc":
            vars.heads.append(copy.deepcopy(classifier.fc))

        # collect statistics
        vars.ext_mem_sz += stats['disk']
        vars.ram_usage += stats['ram']

        # test on the validation set
        stats, _ = test_multitask(
            classifier, full_valdidset, args.batch_size,
            multi_heads=vars.heads, verbose=False
        )

        vars.valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

        vars.batch += 1
        if args.cache_dir:
            vars.state_dict = classifier.state_dict()
            vars.save()

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(vars.valid_acc), elapsed, np.average(vars.ram_usage),
            np.max(vars.ram_usage), np.average(vars.ext_mem_sz), np.max(vars.ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(
        classifier, full_testset, args.batch_size,
        multi_heads=vars.heads, verbose=False
    )

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="ni",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--data_dir', type=str, default="/ram/")

    # Model
    parser.add_argument('--cls', '--classifier', type=str, default='WideResNet50',
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'WideResNet50', 'ResNext50'])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs')

    # Continual Learning
    parser.add_argument('--replay_examples', type=int, default=0,
                        help='data examples to keep in memory for each batch '
                             'for replay.')
    parser.add_argument('--replay_strategy', type=str, default='random',
                        help='the stratedy to choose the data examples to keep in memory for each batch for replay',
                        choices=['random', 'equal_interval'])

    # Misc
    parser.add_argument('--sub_dir', type=str, default="ni",
                        help='directory of the submission file for this exp.')

    parser.add_argument('--cache_dir', type=str, default='',
                        help='directory of the cache file.')

    # More
    parser.add_argument('--trainable_backbone', type=str, default='all',
                        choices=['all', 'first', 'none'])

    parser.add_argument('--cumul', default=False, action='store_true',
                        help='If True the cumulative scenario is assumed, the incremental scenario otherwise')

    parser.add_argument('--lr_step', type=int, default=0,
                        help='learning rate step')

    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='learning rate gamma')

    parser.add_argument('--aug', type=str, default=None,
                        choices=['l1', 'l2', 'l3'])

    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    main(args)
