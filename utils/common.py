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
General useful functions for machine learning prototyping based on numpy.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np


def shuffle_in_unison(dataset, seed, in_place=False):
    """
    Shuffle two (or more) list in unison. It's important to shuffle the images
    and the labels maintaining their correspondence.

        Args:
            dataset (dict): list of shuffle with the same order.
            seed (int): set of fixed Cifar parameters.
            in_place (bool): if we want to shuffle the same data or we want
                             to return a new shuffled dataset.
        Returns:
            list: train and test sets composed of images and labels, if in_place
                  is set to False.
    """

    np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

        Args:
            x (tensor): logits on which to apply the softmax function.
        Returns:
            tensor: softmax vector of batched softmax vectors.
    """

    f = x - np.max(x)
    return np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
    # If you do not care about stability use line above:
    # return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def count_lines(fpath):
    """
    Count line in file.

        Args:
            fpath (str): file path.
        Returns:
            int: number of lines in the file.
    """

    num_imgs = 0
    with open(fpath, 'r') as f:
        for line in f:
            if '/' in line:
                num_imgs += 1
    return num_imgs


def pad_data(dataset, mb_size):
    """
    Padding all the matrices contained in dataset to suit the mini-batch
    size. We assume they have the same shape.

        Args:
            dataset (str): sets to pad to reach a multile of mb_size.
            mb_size (int): mini-batch size.
        Returns:
            list: padded data sets
            int: number of iterations needed to cover the entire training set
                 with mb_size mini-batches.
    """

    num_set = len(dataset)
    x = dataset[0]
    # computing test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # padding data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        for i, data in enumerate(dataset):
            dataset[i] = np.concatenate((data[:n_to_add], data))
    if num_set == 1:
        dataset = dataset[0]

    return dataset, it


def compute_one_hot(train_y, class_count):
    """
    Compute one-hot from labels.

        Args:
            train_y (list): list of int labels.
            class_count (int): total number of classes.
        Returns:
            tensor: one-hot encoding of the input tensor.

    """

    target_y = np.zeros((train_y.shape[0], class_count), dtype=np.float32)
    target_y[np.arange(train_y.shape[0]), train_y.astype(np.int8)] = 1

    return target_y


def imagenet_batch_preproc(img_batch, rgb_swap=True, channel_first=True,
                           avg_sub=True):
    """
    Pre-process batch of PIL img for Imagenet pre-trained models with caffe.
    It may be need adjustements depending on the pre-trained model
    since it is training dependent.

        Args:
            img_batch (tensor): batch of images.
            rgb_swap (bool): if we want to swap the channels order.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            avg_sub (bool): if we want to subtract the average pixel value
                            for each channel.
        Returns:
            tensor: pre-processed batch.
    """

    # we assume img is a 3-channel image loaded with PIL
    # so img has dim (w, h, c)

    if rgb_swap:
        # Swap RGB to BRG
        img_batch = img_batch[:, :, :, ::-1]

    if avg_sub:
        # Subtract channel average
        img_batch[:, :, :, 0] -= 104
        img_batch[:, :, :, 1] -= 117
        img_batch[:, :, :, 2] -= 123

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def remove_some_labels(dataset, labels_set, scale_labels=False):
    """
    This method simply remove patterns with labels contained in
    the labels_set.

        Args:
            dataset (list): training set composed of data and labels.
            labels_set (list): set of labels to remove.
            scale_labels (bool): if we want to change the actual label number
                                 to start from zero or not.
        Returns:
            list: reduced set of data and labels.
    """

    data, labels = dataset
    for label in labels_set:
        # Using fun below copies data
        mask = np.where(labels == label)[0]
        labels = np.delete(labels, mask)
        data = np.delete(data, mask, axis=0)

    if scale_labels:
        # scale labels if they do not start from zero
        min = np.min(labels)
        labels = (labels - min)

    return [data, labels]


def change_some_labels(dataset, labels_set, change_set):
    """
    This method simply change labels contained in
    the labels_set.

        Args:
            dataset (list): training set composed of data and labels.
            labels_set (list): labels to change.
            change_set (list): corrisponding changed labels.
        Returns:
            list: changed set of data and labels.
    """

    data, labels = dataset
    for label, change in zip(labels_set, change_set):
        mask = np.where(labels == label)[0]
        labels = np.put(labels, mask, change)

    return data, labels


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and sparse labels.
        Args:
            predictions (tensor): batched predictions.
            labels (list): list on integers labels.
        Returns:
            float: error rate.
    """

    # return the accuracy
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])



