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
General useful functions for machine learning prototyping based on numpy.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import psutil
import shutil


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


def check_ext_mem(ext_mem_dir):
    """
    Compute recursively the memory occupation on disk of ::ext_mem_dir::
    directory.

        Args:
            ext_mem_dir (str): path to the directory.
        Returns:
            ext_mem (float): Occupation size in Megabytes
    """

    ext_mem = sum(
        os.path.getsize(
            os.path.join(dirpath, filename)) for
                dirpath, dirnames, filenames in os.walk(ext_mem_dir)
                    for filename in filenames
    ) / (1024 * 1024)

    return ext_mem


def check_ram_usage():
    """
    Compute the RAM usage of the current process.

        Returns:
            mem (float): Memory occupation in Megabytes
    """

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)

    return mem


def create_code_snapshot(code_dir, dst_dir):
    """
    Copy the code that generated the exps as a backup.

        Args:
            code_dir (str): root dir of the project
            dst_dir (str): where to put the code files
    """

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for dirpath, dirnames, filenames in os.walk(code_dir):
        for filename in filenames:
            if ".py" in filename and ".pyc" not in filename:
                try:
                    shutil.copy(os.path.join(dirpath, filename), dst_dir)
                except shutil.SameFileError:
                    pass


