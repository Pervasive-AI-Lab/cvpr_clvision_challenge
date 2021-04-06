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

import numpy as np
import torch
from torch.autograd import Variable
from .common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage
import kornia
import torchvision
def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, mask=None, originality=None):
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

    (train_x, train_y, originality), it_x_ep = pad_data(
        [x, y, originality], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y, originality], 0, in_place=True
    )

    jitter = kornia.augmentation.ColorJitter(
        brightness=np.random.rand() * 0.25,
        contrast=np.random.rand() * 0.25,
        saturation=np.random.rand() * 0.5,
        hue=np.random.rand() * 4)


    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0
    all_elites_x = None
    all_elites_y = None
    itemwise_criterion = torch.nn.CrossEntropyLoss(reduce=False)
    elites_x = []
    elites_y = []

    inheritage_count = 1000#train_x.shape[0]//10
    all_item_losses = torch.zeros((train_x.shape[0],))

    originality_in_this_epoch = torch.LongTensor((train_x.shape[0])*2).random_(0, 2)

    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()

            x_mb = train_x[start:end].cuda()
            y_mb = train_y[start:end]
            originality_mb = originality[start:end]
            originality_in_this_epoch_mb = originality_in_this_epoch[start:end]

            for i in range(x_mb.shape[0]):

                if originality_mb[i] == 1 and originality_in_this_epoch_mb[i] == 1:

                    x_mb_remapped  = x_mb[i,:]#torch.Size([3, 128, 128])

                    x_mb_remapped[0,:, :] = x_mb_remapped[0,:, :]*0.229 + 0.485
                    x_mb_remapped[1,:, :] = x_mb_remapped[1,:, :]*0.224 + 0.456
                    x_mb_remapped[2,:, :] = x_mb_remapped[2,:, :]*0.225 + 0.406

                    #x_mb_remapped = jitter(x_mb_remapped.view(1,3,128,128))

                    flipper = kornia.augmentation.RandomHorizontalFlip(p=0.5)
                    x_mb_remapped = flipper(x_mb_remapped)

                    x_mb_remapped = kornia.augmentation.functional.apply_rotation(x_mb_remapped,kornia.augmentation.random_generator.random_rotation_generator(1, 180))

                    x_mb_remapped = kornia.augmentation.functional.apply_erase_rectangles(x_mb_remapped, {
                        "widths": torch.rand(1) * (128 // 4) + (128 // 8),
                        "heights": torch.rand(1) * (128 // 4) + (128 // 8),
                        "xs": torch.rand(1) * 128,
                        "ys": torch.rand(1) * 128,
                        "values": torch.ones(1) * torch.mean(x_mb_remapped[0, ...]).view(1).cpu()})

                    x_mb_remapped = x_mb_remapped.squeeze()

                    x_mb_remapped[0, :, :] = ((x_mb_remapped[0, :, :] - 0.485) / 0.229)
                    x_mb_remapped[1, :, :] = ((x_mb_remapped[1, :, :] - 0.456) / 0.224)
                    x_mb_remapped[2, :, :] = ((x_mb_remapped[2, :, :] - 0.406) / 0.225)

                    x_mb[i,:] = x_mb_remapped


            if it != 0:
                x_mb = torch.cat((x_mb,elites_x),0)
                y_mb = torch.cat((y_mb,elites_y),0)

            logits = model(x_mb)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(x_mb)
                itemwise_loss = itemwise_criterion(logits, y_mb)
                optimizer.zero_grad()

                _, elite_ids = torch.topk(itemwise_loss, mb_size)
                elite_ids = elite_ids[0:mb_size//8]

            model.train()


            elites_x = x_mb[elite_ids,:]
            elites_y = y_mb[elite_ids]



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

        originality_in_this_epoch_mb = 1 - originality_in_this_epoch_mb

        if ep == 2:
            with torch.no_grad():
                for it in range(it_x_ep):
                    start = it * mb_size
                    end = (it + 1) * mb_size

                    optimizer.zero_grad()

                    x_mb = train_x[start:end].cuda()
                    y_mb = train_y[start:end]

                    logits = model(x_mb)

                    _, pred_label = torch.max(logits, 1)
                    correct_cnt += (pred_label == y_mb).sum()

                    itemwise_loss = criterion(logits, y_mb)
                    all_item_losses[start:end] = itemwise_loss

    _, elite_ids = torch.topk(all_item_losses, inheritage_count+200)

    all_elites_x = train_x[elite_ids[200:inheritage_count+200],:]
    all_elites_y = train_y[elite_ids[200:inheritage_count+200]]

    return ave_loss, acc, stats, all_elites_x, all_elites_y




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
        model, test_set, mb_size, preproc=None, use_cuda=True, multi_heads=[], verbose=True):
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

    for (x, y), t in test_set:

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            if verbose:
                print("Using head: ", t)
            with torch.no_grad():
                model.fc.weight.copy_(multi_heads[t].weight)
                model.fc.bias.copy_(multi_heads[t].bias)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        test_x = test_x / 255
        test_x[:, :, :, 0] = ((test_x[:, :, :, 0] - 0.485) / 0.229)
        test_x[:, :, :, 1] = ((test_x[:, :, :, 1] - 0.456) / 0.224)
        test_x[:, :, :, 2] = ((test_x[:, :, :, 2] - 0.406) / 0.225)
        test_x = test_x.permute((0, 3, 1, 2))



        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            iters = test_y.size(0) // mb_size + 1
            for it in range(iters):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

                # print(pred_label)
                # print(y_mb)
            acc = correct_cnt.item() / test_y.shape[0]

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
