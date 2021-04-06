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

Basic data loader for the CVPR2020 CLVision Challenge.

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import pickle as pkl
import logging
from hashlib import md5
import numpy as np
from PIL import Image


class CORE50(object):
    """ CORe50 CLVision Challenge Data Loader.

    Args:
        root (string): Root directory of the dataset where ``core50_128x128``,
            ``paths.pkl``, ``LUP.pkl``, ``labels.pkl``, ``core50_imgs.npz``
            live. For example ``~/data/core50``.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``multi-task-nc``, ``nic``.
        train (bool, optional): If True, creates the dataset from the training
            set, otherwise creates from test set.
        cumul (bool, optional): If True the cumulative scenario is assumed, the
            incremental scenario otherwise. Practically speaking ``cumul=True``
            means that for batch=i also batch=0,...i-1 will be added to the
            available training data.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        start_batch (int, optional): One of the training incremental batches
            from 0 to max-batch - 1. Remember that for the ``ni``,
            ``multi-task-nc`` and ``nic`` we have respectively 8, 9 and 391
            incremental batches. If ``train=False`` this parameter will be
            ignored.
    """

    new2old_names = {'ni': 'ni', 'multi-task-nc': 'nc', 'nic': 'nicv2_391'}
    nbatch = {
        'ni': 8,
        'nc': 9,
        'nicv2_391': 391
    }

    classnames = ['adapter1', ' adapter2', ' adapter3', ' adapter4', ' adapter5', ' mobile_phone1', ' mobile_phone2',
                  ' mobile_phone3', ' mobile_phone4', ' mobile_phone5', ' scissors1', ' scissors2', ' scissors3',
                  ' scissors4', ' scissors5', ' light_bulb1', ' light_bulb2', ' light_bulb3', ' light_bulb4',
                  ' light_bulb5', ' can1', ' can2', ' can3', ' can4', ' can5', ' glasses1', ' glasses2', ' glasses3',
                  ' glasses4', ' glasses5', ' ball1', ' ball2', ' ball3', ' ball4', ' ball5', ' marker1', ' marker2',
                  ' marker3', ' marker4', ' marker5', ' cup1', ' cup2', ' cup3', ' cup4', ' cup5', ' remote_control1',
                  ' remote_control2', ' remote_control3', ' remote_control4', ' remote_control5']

    def __init__(self, root='', scenario='ni', cumul=False,
                 run=0, start_batch=0, task_sep=False):
        """" Initialize Object """

        self.root = os.path.expanduser(root)
        self.scenario = self.new2old_names[scenario]
        self.cumul = cumul
        self.run = run
        self.batch = start_batch
        self.task_sep = task_sep

        print("Loading paths...")
        with open(os.path.join(root, 'paths.pkl'), 'rb') as f:
            self.paths = pkl.load(f)

        print("Loading LUP...")
        with open(os.path.join(root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        print("Loading labels...")
        with open(os.path.join(root, 'labels.pkl'), 'rb') as f:
            self.labels = pkl.load(f)

        # to be changed
        self.tasks_id = []
        self.labs_for_task = []

        if self.scenario == 'nc':
            self.task_sep = True
        else:
            self.task_sep = False

        print("preparing CL benchmark...")
        for i in range(self.nbatch[self.scenario]):
            if self.task_sep:
                self.tasks_id.append(i)
                self.labs_for_task.append(
                    list(set(self.labels[self.scenario][run][i]))
                )
            else:
                self.tasks_id.append(0)

    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """

        scen = self.scenario
        run = self.run
        batch = self.batch

        if self.batch == self.nbatch[scen]:
            raise StopIteration

        # Getting the right indexis
        if self.cumul:
            train_idx_list = []
            for i in range(self.batch + 1):
                train_idx_list += self.LUP[scen][run][i]
        else:
            train_idx_list = self.LUP[scen][run][batch]

        # loading data
        print("Loading data...")
        # Getting the actual paths
        train_paths = []
        for idx in train_idx_list:
            train_paths.append(os.path.join(self.root, self.paths[idx]))
        # loading imgs
        # ribo
        # train_x = self.get_batch_from_paths(train_paths).astype(np.float32)
        train_x = np.asarray(train_paths, dtype='S128')

        # In either case we have already loaded the y
        if self.cumul:
            train_y = []
            for i in range(self.batch + 1):
                train_y += self.labels[scen][run][i]
        else:
            train_y = self.labels[scen][run][batch]

        train_y = np.asarray(train_y, dtype=np.int64)

        # Update state for next iter
        self.batch += 1

        return train_x, train_y, self.tasks_id[self.batch - 1]

    next = __next__  # python2.x compatibility.

    def get_full_valid_set(self, reduced=True):
        """
        Return the test set (the same for each inc. batch).
        """

        scen = self.scenario
        run = self.run
        valid_idx_list = self.LUP[scen][run][-1]
        valid_y = self.labels[scen][run][-1]

        if self.scenario == 'nc':

            valid_set = []
            i_valid_paths = {i: [] for i in range(self.nbatch[self.scenario])}
            idx_x_task = {i: [] for i in range(self.nbatch[self.scenario])}
            y_x_task = {i: [] for i in range(self.nbatch[self.scenario])}

            for idx, y in zip(valid_idx_list, valid_y):
                for i in range(self.nbatch[self.scenario]):
                    if y in self.labs_for_task[i]:
                        idx_x_task[i].append(idx)
                        y_x_task[i].append(y)
                        i_valid_paths[i].append(
                            os.path.join(self.root, self.paths[idx]))

            for i in range(self.nbatch[self.scenario]):
                # ribo
                # i_valid_x = self.get_batch_from_paths(i_valid_paths[i]) \
                #     .astype(np.float32)
                i_valid_x = i_valid_paths[i]
                i_valid_y = np.asarray(y_x_task[i], dtype=np.float32)

                if reduced:
                    # reduce valid set by 20
                    idx = range(0, i_valid_y.shape[0], 20)
                    i_valid_x = np.take(i_valid_x, idx, axis=0)
                    i_valid_y = np.take(i_valid_y, idx, axis=0)

                valid_set.append([(i_valid_x, i_valid_y), i])


        else:
            # test paths
            valid_paths = []
            for idx in valid_idx_list:
                valid_paths.append(os.path.join(self.root, self.paths[idx]))

            # test imgs
            # ribo
            # valid_x = self.get_batch_from_paths(valid_paths) \
            #     .astype(np.float32)
            valid_x = np.asarray(valid_paths, dtype='S128')
            valid_y = np.asarray(valid_y, dtype=np.int64)

            if reduced:
                # reduce valid set by 20
                idx = range(0, valid_y.shape[0], 20)
                valid_x = np.take(valid_x, idx, axis=0)
                valid_y = np.take(valid_y, idx, axis=0)

            valid_set = [[(valid_x, valid_y), self.tasks_id[self.batch - 1]]]

        return valid_set

    def get_full_test_set(self):
        """
        Return the full test set (no labels)
        """

        filelist_path = self.root + "test_filelist.txt"
        filelist_tlabeled_path = self.root + "test_filelist_tlabeled.txt"
        test_img_dir = self.root + "core50_challenge_test"

        if self.scenario == 'nc':
            test_paths = {i: [] for i in range(self.nbatch[self.scenario])}
            with open(filelist_tlabeled_path, "r") as rf:
                for line in rf:
                    path, task_label = line.split()
                    test_paths[int(task_label.strip())].append(
                        os.path.join(test_img_dir, path.strip()))

            full_test = []
            for i in range(self.nbatch[self.scenario]):
                # ribo
                # test_x = self.get_batch_from_paths(test_paths[i]) \
                #     .astype(np.float32)
                test_x = np.asarray(test_paths[i], dtype='S128')
                full_test.append(
                    [(test_x, np.asarray([-1] * len(test_paths[i]))), i])

        else:
            test_paths = []
            with open(filelist_path, "r") as rf:
                for line in rf:
                    test_paths.append(os.path.join(test_img_dir, line.strip()))
            # ribo
            # test_x = self.get_batch_from_paths(test_paths).astype(np.float32)
            test_x = np.asarray(test_paths, dtype='S%d' % len(test_paths[0]))

            full_test = [[(test_x, np.asarray([-1] * len(test_paths))), 0]]

        return full_test

    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False):
        """ Given a number of abs. paths it returns the numpy array
        of all the images. """

        # Getting root logger
        # log = logging.getLogger('mylogger')

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = paths.shape[0]
        # hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        # log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        # file_path = None

        # if compress:
        #     file_path = snap_dir + hexdigest + ".npz"
        #     if os.path.exists(file_path) and not on_the_fly:
        #         loaded = True
        #         with open(file_path, 'rb') as f:
        #             npzfile = np.load(f)
        #             x, y = npzfile['x']
        # else:
        #     x_file_path = snap_dir + hexdigest + "_x.bin"
        #     if os.path.exists(x_file_path) and not on_the_fly:
        #         loaded = True
        #         with open(x_file_path, 'rb') as f:
        #             x = np.fromfile(f, dtype=np.uint8) \
        #                 .reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')

                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            # if not on_the_fly:
            #     # Then we save x
            #     if compress:
            #         with open(file_path, 'wb') as g:
            #             np.savez_compressed(g, x=x)
            #     else:
            #         x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x


if __name__ == "__main__":

    # Create the dataset object for example with the "multi-task-nc"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50(root='data/', scenario="multi-task-nc")

    # Get the fixed valid set
    print("Recovering validation set...")
    full_valdiset = dataset.get_full_valid_set()
    # Get the fixed test set
    print("Recovering test set...")
    full_testset = dataset.get_full_test_set()

    # loop over the training incremental batches
    for i, (x, y, t) in enumerate(dataset):
        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y: {0}"
              .format(x.shape, y.shape))

        # use the data
        pass
