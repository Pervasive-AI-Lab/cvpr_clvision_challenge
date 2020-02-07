#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
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

import numpy as np
import torch
from torch.autograd import Variable
from .common import pad_data, shuffle_in_unison


def get_accuracy(model, criterion, batch_size, test_x, test_y, test_it,
                 use_cuda=False, mask=None):
    """
    Test accuracy given a model and the test data.

        Args:
            model (nn.Module): the pytorch model to test.
            criterion (func): loss function.
            batch_size (int): mini-batch size.
            test_x (tensor): test data.
            test_y (tensor): test labels.
            test_it (int): test iterations.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
        Returns:
            ave_loss (float): average loss across the test set.
            acc (float): average accuracy.
            accs (list): average accuracy for class.
    """

    correct_cnt, ave_loss = 0, 0
    model = maybe_cuda(model, use_cuda=use_cuda)

    num_class = torch.max(test_y) + 1
    hits_per_class = [0] * num_class
    pattern_per_class = [0] * num_class

    for i in range(test_it):
        # indexing
        start = i * batch_size
        end = (i + 1) * batch_size

        x = Variable(
            maybe_cuda(test_x[start:end], use_cuda=use_cuda), volatile=True
        )
        y = Variable(
            maybe_cuda(test_y[start:end], use_cuda=use_cuda), volatile=True
        )

        logits = model(x)

        if mask is not None:
            # we put an high negative number so that after softmax that prob
            # will be zero and not contribute to the loss
            idx = (torch.FloatTensor(mask).cuda() == 0).nonzero()
            idx = idx.view(idx.size(0))
            logits[:, idx] = -10e10

        loss = criterion(logits, y)
        _, pred_label = torch.max(logits.data, 1)
        correct_cnt += (pred_label == y.data).sum()
        ave_loss += loss.data[0]

        for label in y.data:
            pattern_per_class[int(label)] += 1

        for i, pred in enumerate(pred_label):
            if pred == y.data[i]:
                hits_per_class[int(pred)] += 1

    accs = np.asarray(hits_per_class) / \
           np.asarray(pattern_per_class).astype(float)

    acc = correct_cnt * 1.0 / test_y.size(0)

    ave_loss /= test_y.size(0)

    return ave_loss, acc, accs


def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, mask=None,
              record_stats=False):
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
            record_stats (bool): if we want to save collect sparsity stats.
        Returns:
            ave_loss (float): average loss across the train set.
            acc (float): average accuracy over training.
            stats (dict): dictionary of several stats collected.
    """

    cur_ep = 0
    cur_train_t = t
    stats = {'perc_on_avg': [], 'perc_on_std': [], 'on_idxs': []}

    if preproc:
        x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    for ep in range(train_ep):

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
            logits = model(x_mb)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 100 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

        cur_ep += 1

        if record_stats:
            perc_on_avg = np.mean(model.active_perc_list)
            perc_on_std = np.std(model.active_perc_list)

            print(
                "Average active units: {}%, std {}%: "
                .format(perc_on_avg, perc_on_std)
            )

            stats['perc_on_avg'].append(perc_on_avg)
            stats['perc_on_std'].append(perc_on_std)

    return ave_loss, acc, stats


def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.

        Args:
            img_batch (tensor): batch of images.
            scale (bool): if we want to scale the images between 0 an 1.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            norm (bool): if we want to normalize them.
        Returns:
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
        model, test_set, mb_size, preproc=None, use_cuda=True, multi_heads=[],
        mask=False):
    """
    Test a model considering that the test set is composed of multiple tests
    one for each task.

        Args:
            model (nn.Module): the pytorch model to test.
            test_set (list): list of (x,y,t) test tuples.
            mb_size (int): mini-batch size.
            preproc (func): image preprocess function.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
            multi_heads (list): ordered list of "heads" to be used for each
                                task.
        Returns:
            stats (float): collected stasts of the test including average and
                           per class accuracies.
    """

    model.eval()

    acc_x_task = []
    stats = {'accs': []}

    for (x, y), t in test_set:

        # reduce test set 20 substampling
        # idx = range(0, y.shape[0], 20)
        # x = np.take(x, idx, axis=0)
        # y = np.take(y, idx, axis=0)

        if preproc:
            x = preproc(x)

        (test_x, test_y), it_x_ep = pad_data(
            [x, y], mb_size
        )

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            print("Using head: ", t)
            model.classifier = multi_heads[t]

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        test_y = torch.from_numpy(test_y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            minclass = np.min(y)
            maxclass = np.max(y)

            for it in range(it_x_ep):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)
                logits = model(x_mb)

                if mask:
                    totclass = logits.size(1)
                    mymask = np.asarray([0] * totclass)
                    mymask[minclass:maxclass+1] = 1
                    mymask = maybe_cuda(torch.from_numpy(mymask),
                                        use_cuda=use_cuda)
                    # print(mymask)
                    logits = mymask * logits

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()

            acc = correct_cnt.item() / test_y.shape[0]

        print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    print("------------------------------------------")
    print("Avg. acc:", np.mean(acc_x_task))
    print("------------------------------------------")

    # reset the head for the next batch
    if multi_heads:
        print("classifier reset...")
        model.reset_classifier()

    return stats
