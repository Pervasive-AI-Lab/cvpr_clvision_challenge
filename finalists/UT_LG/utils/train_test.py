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
from .common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage
from .utils import resize_tensor
from torch.utils import data



class aug_dataset(data.Dataset):
    def __init__(self, x, y, augmentations=None):
        self.x = x
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.augmentations = augmentations  # save the augmentations

    def __len__(self):
        return self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        # no need to convert to tensor, because image is converted to tensor already by the pipeline
        augmented = self.augmentations(image=self.x[idx])

        return augmented['image'], self.y[idx]

def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, resize=False):
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
    training_set = data.TensorDataset(train_x, train_y)
    params = {'batch_size': mb_size,
              'shuffle': True,
              'num_workers': 4}
    dataloader = data.DataLoader(training_set, **params)

    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it, batch in enumerate(dataloader):
            batch_x, batch_y = batch

            optimizer.zero_grad()

            if resize:
                batch_x = resize_tensor(batch_x, 224)


            x_mb = maybe_cuda(batch_x, use_cuda=use_cuda)
            y_mb = maybe_cuda(batch_y, use_cuda=use_cuda)
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

    return ave_loss, acc, stats

def train_net_aug(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, aug, use_cuda=True, verbose=False):

    cur_ep = 0
    acc = None
    ave_loss = 0
    stats = {"ram": [], "disk": []}
    model = maybe_cuda(model, use_cuda=use_cuda)

    train_ds = aug_dataset(x, y, augmentations=aug)

    # Initilize the dataloader
    trainloader = data.DataLoader(train_ds, batch_size=mb_size, shuffle=True, num_workers=4)
    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()
        if verbose:
            print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it, batch in enumerate(trainloader):

            batch_x, batch_y = batch

            optimizer.zero_grad()

            x_mb = maybe_cuda(batch_x, use_cuda=use_cuda)
            y_mb = maybe_cuda(batch_y, use_cuda=use_cuda)
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

            if it % 100 == 0 and verbose:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

        cur_ep += 1

    return ave_loss, acc, stats


def train_net_aug_indep(optimizer, model_list, criterion, mb_size, x, y, t,
                  train_ep, valset, val_batch, train_aug, test_aug, use_cuda=True, verbose=False):
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
    acc = None
    ave_loss = 0
    stats = {"ram": [], "disk": []}
    model = maybe_cuda(model_list[t], use_cuda=use_cuda)
    train_ds = aug_dataset(x, y, augmentations=train_aug)

    # Initilize the dataloader
    trainloader = data.DataLoader(train_ds, batch_size=mb_size, shuffle=True, num_workers=4)
    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        model.active_perc_list = []
        model.train()
        if verbose:
            print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it, batch in enumerate(trainloader):

            batch_x, batch_y = batch

            optimizer.zero_grad()

            x_mb = maybe_cuda(batch_x, use_cuda=use_cuda)
            y_mb = maybe_cuda(batch_y, use_cuda=use_cuda)
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

            if it % 100 == 0 and verbose:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

        cur_ep += 1
        if verbose:
            test_multi_models_aug(model_list, valset, val_batch, test_aug)
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
        model, test_set, mb_size, preproc=None, use_cuda=True, multi_heads=[], verbose=True, resize=False):
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

        if preproc:
            x = preproc(x)

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

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            iters = test_y.size(0) // mb_size + 1
            for it in range(iters):

                start = it * mb_size
                end = (it + 1) * mb_size
                batch_x = test_x[start:end]
                if resize:
                    batch_x = resize_tensor(batch_x, 224)
                batch_y = test_y[start:end]

                x_mb = maybe_cuda(batch_x, use_cuda=use_cuda)
                y_mb = maybe_cuda(batch_y, use_cuda=use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

            acc = correct_cnt.item() / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))


    # reset the head for the next batch
    if multi_heads:
        if verbose:
            print("classifier reset...")
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.bias.fill_(0)

    return stats, preds


def test_multitask_aug(
        model, test_set, mb_size, test_aug, use_cuda=True, verbose=True):
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


        model = maybe_cuda(model, use_cuda=use_cuda)

        test_ds = aug_dataset(x, y, augmentations=test_aug)

        # Initilize the dataloader
        testloader = data.DataLoader(test_ds, batch_size=mb_size, shuffle=False, num_workers=4)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():


            for it, batch in enumerate(testloader):
                batch_x, batch_y = batch
                x_mb = maybe_cuda(batch_x, use_cuda=use_cuda)
                y_mb = maybe_cuda(batch_y, use_cuda=use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

            acc = correct_cnt.item() / y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    stats['acc'].append(np.mean(acc_x_task))

    return stats, preds

def test_multi_models(
        model_list, test_set, mb_size, preproc=None, use_cuda=True, verbose=True):
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



    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:
        if preproc:
            x = preproc(x)

        model = model_list.get(t, model_list[0])
        model.eval()
        if verbose:
            print("Using head: ", t)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

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

    return stats, preds

def test_multi_models_aug(
        model_list, test_set, mb_size, test_aug, use_cuda=True, verbose=True):
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



    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:


        model = model_list.get(t, model_list[0])
        model.eval()
        if verbose:
            print("Using head: ", t)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_ds = aug_dataset(x, y, augmentations=test_aug)
        # Initilize the dataloader
        testloader = data.DataLoader(test_ds, batch_size=mb_size, shuffle=False, num_workers=4)
        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            for it, batch in enumerate(testloader):
                batch_x, batch_y = batch

                x_mb = maybe_cuda(batch_x, use_cuda=use_cuda)
                y_mb = maybe_cuda(batch_y, use_cuda=use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

                # print(pred_label)
                # print(y_mb)
            acc = correct_cnt.item() / y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)


    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    return stats, preds

def test_multi_models_small(
        model_list, match_list, test_set, mb_size, preproc=None, use_cuda=True, verbose=True):
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



    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    for (x, y), t in test_set:
        if preproc:
            x = preproc(x)


        model = model_list.get(t, model_list[0])
        dict = match_list.get(t, match_list[0])
        model.eval()
        if verbose:
            print("Using head: ", t)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

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
                pred_label = pred_label.data.cpu().numpy()
                pred_label_new = np.copy(pred_label)
                for k, v in dict.items():
                    pred_label_new[pred_label == v] = k
                y_mb = y_mb.data.cpu().numpy()
                correct_cnt += (pred_label_new == y_mb).sum()
                preds += list(pred_label_new)

            acc = correct_cnt / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)
        model.cpu()

    stats['acc'].append(np.mean(acc_x_task))

    if verbose:
        print("------------------------------------------")
        print("Avg. acc:", stats['acc'])
        print("------------------------------------------")

    return stats, preds