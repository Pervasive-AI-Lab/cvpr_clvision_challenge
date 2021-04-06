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
from models.resnet_drl import ResNet_DRL, ResNeSt_DRL


def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, mask=None,pad=True,
              ER_type=None, ext_mem={}, mem_size=0, drl_lmb=0., out_dim=50, 
              scenario='ni', one_hot=False):
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

    if pad:
        (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
        )
    else:
        rd = x.shape[0]%mb_size
        train_x = x[:-rd]
        train_y = y[:-rd]
        it_x_ep = train_x.shape[0] // mb_size

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    if one_hot:
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    else:
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if ER_type and t > 0:
        
        num_mem = t 
        print('mem keys',ext_mem.keys())
        if ER_type == 'BER':
            per_task_size = [mb_size]*num_mem
            
        elif ER_type == 'ER':
            per_task_size = [(mb_size)//num_mem]*num_mem


    model.active_perc_list = []
    model.train()
    for ep in range(train_ep):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()  

            x_mb, y_mb = train_x[start:end], train_y[start:end]

            if ER_type and t>0:   

                mem_x, mem_y = [], []
                
                for i in range(t):
                    mem = ext_mem[i]
                    ids = np.random.choice(len(mem[0]), size=per_task_size[i])
                    mem_x.append(mem[0][ids])
                    mem_y.append(mem[1][ids])


                mem_x = np.concatenate(mem_x)
                mem_y = np.concatenate(mem_y)
                mem_x = torch.from_numpy(mem_x).type(torch.FloatTensor)
                if one_hot:
                    mem_y = torch.from_numpy(mem_y).type(torch.FloatTensor)
                else:
                    mem_y = torch.from_numpy(mem_y).type(torch.LongTensor)

                x_mb, y_mb = torch.cat([x_mb,mem_x]), torch.cat([y_mb,mem_y])


            x_mb = maybe_cuda(x_mb, use_cuda=use_cuda)
            y_mb = maybe_cuda(y_mb, use_cuda=use_cuda)

            logits = model(x_mb)

            if isinstance(model,ResNet_DRL) or isinstance(model,ResNeSt_DRL):
                loss = criterion(logits,y_mb,model.H,drl_lmb,out_dim,use_cuda)
            else:
                loss = criterion(logits, y_mb)
                
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, pred_label = torch.max(logits, 1)
            if one_hot:
                _, y_mb = torch.max(y_mb, 1)
                
            correct_cnt += (pred_label == y_mb).sum()

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
        model, test_set, mb_size, preproc=None, use_cuda=True, multi_heads=[], verbose=True,one_hot=False):
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
                if one_hot:
                    model.fc[0].weight.copy_(multi_heads[t][0].weight)
                    model.fc[0].bias.copy_(multi_heads[t][0].bias)
                else:
                    model.fc.weight.copy_(multi_heads[t].weight)
                    model.fc.bias.copy_(multi_heads[t].bias)

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt = 0

        with torch.no_grad():

            iters = int(np.ceil(test_y.size(0) / mb_size))
            for it in range(iters):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)
                
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                if len(y_mb.shape) > 1:
                    _, y_mb = torch.max(logits, 1)
                    
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
            if one_hot:
                model.fc[0].weight.fill_(0)
                model.fc[0].bias.fill_(0)
            else:
                model.fc.weight.fill_(0)
                model.fc.bias.fill_(0)

    return stats, preds
