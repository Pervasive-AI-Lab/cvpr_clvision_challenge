#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Massimo Caccia, Pau Rodriguez,        #
# Lorenzo Pellegrini. All rights reserved.                                     #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 8-11-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""
General useful functions for machine learning with Pytorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from sklearn.utils import class_weight
import os
import numpy as np
import torch
import time
import torch.nn.functional as F
from .common import check_ext_mem, check_ram_usage, aug_data

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class EpochData(Dataset):
    def __init__(self, x, y, aug):
        super().__init__()
        self.x = x
        self.y = y
        self.aug = aug
        # transform function

        if self.aug:
            self.transform = transforms.Compose([
                aug_data(self.aug),
                transforms.ToPILImage(),
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])

    def __getitem__(self, index):
        img = Image.open(self.x[index])
        # img = img.resize((224, 224))
        img = self.transform(np.uint8(img))
        return img, self.y[index]

    def __len__(self):
        return self.x.shape[0]


def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, params_dir, lwf_params, focal_params, aug,
              use_cuda=True,
              mask=None,
              post_scaling=False):
    """
    Train a Pytorch model from pre-loaded tensors.

        Args:
            optimizer (object): the pytorch optimizer.
            model (object): the pytorch model to train.
            criterion (func): loss function.
            mb_size (int): mini-batch size.
            x (tensor): train data.
            y (tensor): train labels.
            t (int): task label.
            train_ep (int): number of training epochs.
            preproc (func): test iterations.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
        Returns:
            ave_loss (float): average loss across the train set.
            acc (float): average accuracy over training.
            stats (dict): dictionary of several stats collected.
    """

    cur_ep = 0
    cur_train_t = t
    stats = {"ram": [], "disk": []}

    # if aug:
    #     x = aug_data(np.uint8(x), aug)

    # if preproc:
    #     x = preproc(x)

    # (train_x, train_y, train_w), it_x_ep = pad_data(
    #     [x, y, w], mb_size
    # )

    # shuffle_in_unison(
    #     [train_x, train_y, train_w], 0, in_place=True
    # )

    model = maybe_cuda(model, use_cuda=use_cuda)

    if lwf_params[0]:
        import copy
        old_model = copy.deepcopy(model)
        old_model.train(False)

    acc = None
    ave_loss = 0

    dataset = EpochData(x, y, aug)
    loader = DataLoader(dataset, batch_size=mb_size, shuffle=True, num_workers=4)

    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem(os.path.join('cl_ext_mem', params_dir)))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, total_cnt, ave_loss = 0, 0, 0

        start_time = time.time()
        for it, [x_mb, y_mb] in enumerate(loader):

            optimizer.zero_grad()

            x_mb = maybe_cuda(x_mb)
            y_mb = maybe_cuda(y_mb)

            logits = model(x_mb)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)

            # focal loss
            if focal_params[0]:
                loss *= ((1 - torch.softmax(logits, dim=1)[range(len(y_mb)), y_mb]) ** focal_params[2])
                loss *= focal_params[1]
                loss = loss.mean()

            ave_loss += loss.item()

            # lwf loss
            lwf_loss = torch.tensor(0.0)
            if lwf_params[0]:
                lwf_loss = -(F.softmax(old_model(x_mb) / lwf_params[2], dim=1) * F.log_softmax(logits / lwf_params[2],
                                                                                               dim=1)).sum(
                    dim=1).mean()
                loss += lwf_loss * lwf_params[1]

            loss.backward()
            optimizer.step()

            total_cnt += y_mb.size(0)
            acc = correct_cnt.item() * 1. / total_cnt
            ave_loss /= total_cnt

            if it % 100 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'lwf loss {:.6f}, '
                    'running train acc: {:.3f}, '
                    'time: {:.1f}s'
                        .format(it, ave_loss, lwf_loss.item(), acc, time.time() - start_time)
                )
                start_time = time.time()

        cur_ep += 1

    return ave_loss, acc, stats


def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def test_multitask(
        model, test_set, mb_size, priors, use_cuda=True, multi_heads=[], verbose=True,
        post_scaling=False):
    """
    Test a model considering that the test set is composed of multiple tests
    one for each task.

        Args:
            model (nn.Module): the pytorch model to test.
            test_set (list): list of (x,y,t) test tuples.
            mb_size (int): mini-batch size.
            preproc (func): image preprocess function.
            use_cuda (bool): if we want to use gpu or cpu.
            multi_heads (list): ordered list of "heads" to be used for each
                                task.
        Returns:
            stats (float): collected stasts of the test including average and
                           per class accuracies.
    """

    model.eval()

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    priors = torch.from_numpy(priors).type(torch.FloatTensor)
    priors = maybe_cuda(priors, use_cuda=use_cuda)

    for (x, y), t in test_set:

        # if preproc:
        #     x = preproc(x)
        dataset = EpochData(x, y, None)
        loader = DataLoader(dataset, batch_size=mb_size, num_workers=4)

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            if verbose:
                print("Using head: ", t)
            with torch.no_grad():
                model.fc.weight.copy_(multi_heads[t].weight)
                model.fc.bias.copy_(multi_heads[t].bias)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        # test_x = torch.from_numpy(x).type(torch.FloatTensor)
        # test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            for x_mb, y_mb in loader:

                x_mb = maybe_cuda(x_mb)
                y_mb = maybe_cuda(y_mb)

                logits = model(x_mb)

                probs = torch.softmax(logits, dim=1)

                if post_scaling:
                    probs /= priors

                _, pred_label = torch.max(probs, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

                # print(pred_label)
                # print(y_mb)
            acc = correct_cnt.item() / len(dataset)

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    # reset the head for the next batch
    if multi_heads:
        if verbose:
            print("classifier reset...")
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.bias.fill_(0)

    return stats, preds
