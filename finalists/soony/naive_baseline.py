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

import os
import time
import copy
import torch
import argparse
import numpy as np
import torchvision.models as models

from core50.dataset import CORE50
from utils.common import create_code_snapshot
from utils.train_test import test_multitask



#=================================================
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

from utils.train_test import train_multitask_nc, train_latent_replay, train_latent_replay2

    
class ReplayNet3(torch.nn.Module):
    def __init__(self, features1, features2, features3, fc):
        super(ReplayNet3, self).__init__()
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.fc = fc
    def forward(self, x, use_latent=False):
        if use_latent:
            y = self.fc(x)
        else:
            y = self.get_latent(x)
            y = self.fc(y)
        return y
    def get_latent(self, x):
        x1 = self.features1(x)
        x1 = torch.flatten(x1, 1)
        x2 = self.features2(x)
        x2 = torch.flatten(x2, 1)
        x3 = self.features3(x)
        x3 = torch.flatten(x3, 1)
        y = torch.cat((x1, x2, x3), dim=1)
        return y
    
    
        
def fc_build(in_feature, out_feature):
    net = nn.Sequential(
        nn.Linear(in_feature, 4096),
        #nn.ReLU(True),
        nn.ELU(),
        nn.AlphaDropout(p=0.2),
        nn.Linear(4096, 4096),
        nn.ELU(),
        nn.AlphaDropout(p=0.2),
        nn.Linear(4096, out_feature))
    return net

#=================================================






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
    

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = None
    heads_valid = []
    
    prev_acc = 0
    replay_mem = {}
    
    backbone1 = models.resnext101_32x8d(pretrained=True)
    backbone2 = models.resnext50_32x4d(pretrained=True)
    backbone3 = models.resnet152(pretrained=True)
    backbone4 = models.resnet101(pretrained=True)
    backbone5 = models.resnet50(pretrained=True)
    backbone6 = models.resnet34(pretrained=True)
    backbone7 = models.resnet18(pretrained=True)
    backbone8 = models.densenet161(pretrained=True)
    features1 = torch.nn.Sequential(*list(backbone1.children())[:-1])
    features2 = torch.nn.Sequential(*list(backbone2.children())[:-1])
    features3 = torch.nn.Sequential(*list(backbone3.children())[:-1])
    features4 = torch.nn.Sequential(*list(backbone4.children())[:-1])
    features5 = torch.nn.Sequential(*list(backbone5.children())[:-1])
    features6 = torch.nn.Sequential(*list(backbone6.children())[:-1])
    features7 = torch.nn.Sequential(*list(backbone7.children())[:-1])
    features8 = torch.nn.Sequential(*list(backbone8.children())[:-1],nn.ReLU(),nn.AdaptiveAvgPool2d(1))
    
    criterion = torch.nn.CrossEntropyLoss()
    tf_train = transforms.Compose([
               transforms.ToPILImage(),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    tf_valid = transforms.Compose([
               transforms.ToPILImage(),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # model
    if args.scenario == "multi-task-nc":
        args.epochs = 3
        args.batch_size = 100
        args.lr = 0.03
        
        fc = fc_build(2048+2048+512, args.n_classes)
        classifier = ReplayNet3(features2, features5, features6, fc)
        
    if args.scenario == "ni":
        args.epochs = 1
        args.batch_size = 900
        args.lr = 0.0016
        fc = fc_build(2048+2048+2208, args.n_classes)
        classifier = ReplayNet3(features1, features2, features8, fc)
        
    if args.scenario == "nic":
        args.epochs = 1
        args.batch_size = 900
        args.lr = 0.078
        fc = fc_build(2048+2048+2208, args.n_classes)
        classifier = ReplayNet3(features1, features2, features8, fc)

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch
        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)
        
        # train the classifier on the current batch/task 
        if args.scenario == "multi-task-nc":
            opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)
            _, _, stats = train_multitask_nc(args, opt, classifier, criterion, train_x, train_y, t, tf_train, heads, full_valdidset, tf_valid)
            heads.append(copy.deepcopy(classifier.fc))
        if args.scenario == "ni":
            opt = torch.optim.Adam(classifier.fc.parameters(), lr=args.lr/(i+1)/2)

            if i < 7:
                _, _, stats = train_latent_replay2(args, opt, classifier, criterion, train_x, train_y, replay_mem, t, tf_train, full_valdidset, tf_valid, prev_acc, run=True)
            else:
                args.epochs = 100
                _, _, stats = train_latent_replay2(args, opt, classifier, criterion, train_x, train_y, replay_mem, t, tf_train, full_valdidset, tf_valid, prev_acc, run=True)
        if args.scenario == "nic":
            opt = torch.optim.Adam(classifier.fc.parameters(), lr=args.lr/(i+1))
            if i < 390:
                _, _, stats = train_latent_replay2(args, opt, classifier, criterion, train_x, train_y, replay_mem, t, tf_train, full_valdidset, tf_valid, prev_acc, run=True)
            else:
                args.epochs = 100
                _, _, stats = train_latent_replay2(args, opt, classifier, criterion, train_x, train_y, replay_mem, t, tf_train, full_valdidset, tf_valid, prev_acc, run=True)
            

        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test on the validation set
        stats, _ = test_multitask(args, classifier, full_valdidset, tf_valid, multi_heads=heads, verbose=False)
        valid_acc += stats['acc']
        prev_acc = stats['acc'][0]
        
        '''
        if heads != []:
            with torch.no_grad():
                classifier.fc.weight.fill_(0)
                classifier.fc.bias.fill_(0)
        '''
        
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.scenario # + '-{:.5f}'.format(stats['acc'][0])
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
    stats, preds = test_multitask(args, classifier, full_testset, tf_valid, multi_heads=heads, verbose=False)

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
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18')
    parser.add_argument('--cuda_index', type=int, default=0, help='cuda index')
    parser.add_argument('--num_worker', type=int, default=32, help='number of worker threads')

    # Optimization
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=600, help='batch_size')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--early_stopping', type=float, default=0.9999, help='early stopping')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')

    # Continual Learning
    parser.add_argument('--ewc', type=bool, default=False)
    parser.add_argument('--latent_replay', type=bool, default=False)
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
    args.device = 'cuda' if args.cuda else 'cpu'
    torch.cuda.set_device(args.cuda_index)

    main(args)
