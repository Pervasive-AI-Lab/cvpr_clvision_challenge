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
from utils.train_test import train_net, test_multitask
import torchvision.models as models
from utils.common import create_code_snapshot
from skimage import color
import kornia
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from efficientnet_pytorch import EfficientNet

def brightness_tweak(img, in_type="RGB", out_type="RGB"):
    if in_type == "RGB":
        lab_img = color.rgb2lab(img)
    elif in_type == "LAB":
        lab_img = img

    std_lightness = np.std(lab_img[:, :, 0])

    h, w, _ = lab_img.shape
    brightness_mask = np.ones((h, w))

    k = (np.random.rand() - 0.5) * 3
    scale = k * std_lightness
    brightness_mask *= scale

    brightness_mask += lab_img[:, :, 0]
    brightness_mask[brightness_mask < 0] = 0
    brightness_mask[brightness_mask > 100] = 100
    lab_img[:, :, 0] = brightness_mask

    if out_type == "RGB":
        return color.lab2rgb(lab_img).astype("float32")
    elif out_type == "LAB":
        return lab_img


def main(args):

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()


    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(scenario=args.scenario,
                     preload=args.preload_data, root='core50/data/')

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    classifier = EfficientNet.from_pretrained('efficientnet-b7')
    #classifier.fc = torch.nn.Linear(2560, args.n_classes)

    #classifier = models.resnet18(pretrained=True)
    classifier.fc = torch.nn.Linear(classifier._fc.in_features, args.n_classes)

    torch.nn.init.kaiming_uniform(classifier.fc.weight)
    #torch.nn.init.kaiming_uniform(classifier.fc[2].weight)

    opt = torch.optim.SGD(classifier.parameters(), lr=args.lr,weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = None
    heritage_x = None
    heritage_y = None

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        print(train_x.shape[0], args.replay_examples)


        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor).cuda()

        #item = train_x[0,:]
        #print(torch.unique(item))
        #print(item.squeeze().numpy().shape)
        #cv2.imwrite('x.jpg', item.squeeze().numpy())
        #print(item.permute(2,0,1).shape)
        #item = jitter(item.permute(2,0,1).reshape(1,3,128,128)/255)
        #print(item.shape)
        #cv2.imwrite('y.jpg', item.squeeze().permute(1,2,0).numpy()*255)
        #print(torch.unique(item))
        #exit()


        train_x = train_x / 255
        print(train_x.shape)

        train_x[:, :, :, 0] = ((train_x[:, :, :, 0] - 0.485) / 0.229)
        train_x[:, :, :, 1] = ((train_x[:, :, :, 1] - 0.456) / 0.224)
        train_x[:, :, :, 2] = ((train_x[:, :, :, 2] - 0.406) / 0.225)
        train_x = train_x.permute((0, 3, 1, 2))



        randlimit = 2000

        idxs_cur = np.random.choice(
            train_x.shape[0], randlimit, replace=False
        )

        if i == 0:

            ext_mem = [train_x[idxs_cur], train_y[idxs_cur]]
        else:
            ext_mem = [
                torch.cat((train_x[idxs_cur],heritage_x, ext_mem[0])),
                torch.cat((train_y[idxs_cur],heritage_y, ext_mem[1]))]


        original_count = train_x.shape[0]



        train_x = torch.cat((train_x, ext_mem[0]),0)
        train_y = torch.cat((train_y, ext_mem[1]),0)

        originality = torch.zeros(train_y.shape)
        originality[0:original_count] = 1


        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        # train the classifier on the current batch/task
        _, _, stats, all_elites_x, all_elites_y = train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, t,
            args.epochs, preproc=None, originality=originality
        )

        heritage_x = all_elites_x.cpu()
        heritage_y = all_elites_y

        if args.scenario == "multi-task-nc":
            heads.append(copy.deepcopy(classifier.fc))

        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test on the validation set
        stats, _ = test_multitask(
            classifier, full_valdidset, args.batch_size,
            preproc=None, multi_heads=heads, verbose=False
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
        classifier, full_testset, args.batch_size,
        multi_heads=heads, verbose=False
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
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
                        choices=['ResNet18'])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs')

    # Continual Learning
    parser.add_argument('--replay_examples', type=int, default=2000,
                        help='data examples to keep in memory for each batch '
                             'for replay.')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="ni",
                        help='directory of the submission file for this exp.')

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    main(args)
