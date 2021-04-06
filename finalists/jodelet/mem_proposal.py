#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Submission for CVPR 2020 CLVision Challenge
# Copyright (c) 2020. Jodelet Quentin, Vincent Gripon, and Tsuyoshi Murata. All rights reserved.
# Copyrights licensed under the CC-BY-NC 4.0 License.
# See the accompanying LICENSE file for terms. 

# Based on the naive_baseline.py by Vincenzo Lomonaco, Massimo Caccia, 
# Pau Rodriguez, Lorenzo Pellegrini (Under the CC BY 4.0 License)
# From https://github.com/vlomonaco/cvpr_clvision_challenge

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
import torchvision.models as models
from utils.common import create_code_snapshot

from mem_train import ReservoirNet, train_net, test_multitask

def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.

        Args:
        classifierReturns:
            tensor: pre-processed batch.

    """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def main(args):

    torch.manual_seed(0)

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic tracks"
    # and assuming the core50 location in ./core50/data/
    dataset = CORE50(root='core50/data/', scenario=args.scenario,
                     preload=args.preload_data)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # Model by Facebook AI (https://arxiv.org/abs/1905.00546)
    # https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    # This model is released under the CC-BY-NC 4.0 license. See LICENSE file in the Facebook AI repository for additional details. 
    featureModel = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    featureModel.fc = torch.nn.Identity()
    featureModel.eval()

    featureModel = featureModel.to(args.device)

    # Creation of the classification layer
    memorySize = 128
    learningRate = 0.005
    epochs = 1
    classifier = ReservoirNet(lambda : torch.nn.Linear(2048,args.n_classes), args.n_classes*memorySize, (2048,), (1,))
    classifier = classifier.to(args.device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learningRate)
    

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []

    
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        _, _, stats = train_net(featureModel, classifier, optimizer, epochs, args.batchSize, args.batchSize_backbone, args.device, train_x, train_y, t, preproc=preprocess_imgs)
        if args.scenario == "multi-task-nc":
            heads.append(copy.deepcopy(classifier))
            classifier = ReservoirNet(lambda : torch.nn.Linear(2048,args.n_classes), args.n_classes*memorySize, (2048,), (1,))
            classifier = classifier.to(args.device)
            optimizer = torch.optim.SGD(classifier.parameters(), lr=learningRate)


        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']
        
        stats, _ = test_multitask(featureModel, classifier, args.batchSize_backbone, args.device, full_valdidset, preproc=preprocess_imgs, multi_heads=heads, verbose=False)

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
    start = time.time()
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(featureModel, classifier, args.batchSize_backbone, args.device, full_testset, preproc=preprocess_imgs, multi_heads=heads, verbose=False)

    
    elapsed = (time.time() - start) / 60
    print("Inference Time: {}m".format(elapsed))
    
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

    # Optimization
    parser.add_argument('--batchSize', type=int, default=32,
                        help='batchSize')
    
    parser.add_argument('--batchSize_backbone', type=int, default=64,
                        help='batchSize_backbone')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    main(args)
