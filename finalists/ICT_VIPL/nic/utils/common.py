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
import pickle as pkl
import time

import imgaug as ia
from imgaug import augmenters as iaa


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


def aug_data(aug_type):
    # print('Augmenting')

    if aug_type == 'l1':
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
        ])
    elif aug_type == 'l2':
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order
    elif aug_type == 'l3':
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        seq = iaa.Sequential(
            [
                #
                # Apply the following augmenters to most images.
                #
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images

                # crop some of the images by 0-10% of their height/width
                sometimes(iaa.Crop(percent=(0, 0.1))),

                # Apply affine transformations to some of the images
                # - scale to 80-120% of image height/width (each axis independently)
                # - translate by -20 to +20 relative to height/width (per axis)
                # - rotate by -45 to +45 degrees
                # - shear by -16 to +16 degrees
                # - order: use nearest neighbour or bilinear interpolation (fast)
                # - mode: use any available mode to fill newly created pixels
                #         see API or scikit-image for which modes are available
                # - cval: if the mode is constant, then use a random brightness
                #         for the newly created pixels (e.g. sometimes black,
                #         sometimes white)
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                #
                # Execute 0 to 5 of the following (less important) augmenters per
                # image. Don't execute all of them, as that would often be way too
                # strong.
                #
                iaa.SomeOf((0, 5),
                           [
                               # Convert some images into their superpixel representation,
                               # sample between 20 and 200 superpixels per image, but do
                               # not replace all superpixels with their average, only
                               # some of them (p_replace).
                               sometimes(
                                   iaa.Superpixels(
                                       p_replace=(0, 1.0),
                                       n_segments=(20, 200)
                                   )
                               ),

                               # Blur each image with varying strength using
                               # gaussian blur (sigma between 0 and 3.0),
                               # average/uniform blur (kernel size between 2x2 and 7x7)
                               # median blur (kernel size between 3x3 and 11x11).
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),

                               # Sharpen each image, overlay the result with the original
                               # image using an alpha between 0 (no sharpening) and 1
                               # (full sharpening effect).
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                               # Same as sharpen, but for an embossing effect.
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                               # Search in some images either for all edges or for
                               # directed edges. These edges are then marked in a black
                               # and white image and overlayed with the original image
                               # using an alpha of 0 to 0.7.
                               sometimes(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0, 0.7)),
                                   iaa.DirectedEdgeDetect(
                                       alpha=(0, 0.7), direction=(0.0, 1.0)
                                   ),
                               ])),

                               # Add gaussian noise to some images.
                               # In 50% of these cases, the noise is randomly sampled per
                               # channel and pixel.
                               # In the other 50% of all cases it is sampled once per
                               # pixel (i.e. brightness change).
                               iaa.AdditiveGaussianNoise(
                                   loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                               ),

                               # Either drop randomly 1 to 10% of all pixels (i.e. set
                               # them to black) or drop them on an image with 2-5% percent
                               # of the original size, leading to large dropped
                               # rectangles.
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout(
                                       (0.03, 0.15), size_percent=(0.02, 0.05),
                                       per_channel=0.2
                                   ),
                               ]),

                               # Invert each image's channel with 5% probability.
                               # This sets each pixel value v to 255-v.
                               iaa.Invert(0.05, per_channel=True),  # invert color channels

                               # Add a value of -10 to 10 to each pixel.
                               iaa.Add((-10, 10), per_channel=0.5),

                               # Change brightness of images (50-150% of original value).
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),

                               # Improve or worsen the contrast of images.
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                               # Convert each image to grayscale and then overlay the
                               # result with the original with random alpha. I.e. remove
                               # colors with varying strengths.
                               iaa.Grayscale(alpha=(0.0, 1.0)),

                               # In some images move pixels locally around (with random
                               # strengths).
                               sometimes(
                                   iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                               ),

                               # In some images distort local areas with varying strength.
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                           ],
                           # do all of the above augmentations in random order
                           random_order=True
                           )
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    else:
        raise Exception()

    return seq.augment_image


def aug_data_ori(images, aug_type):
    print('Augmenting')

    if aug_type == 'l1':
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
        ])
    elif aug_type == 'l2':
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order
    elif aug_type == 'l3':
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        seq = iaa.Sequential(
            [
                #
                # Apply the following augmenters to most images.
                #
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images

                # crop some of the images by 0-10% of their height/width
                sometimes(iaa.Crop(percent=(0, 0.1))),

                # Apply affine transformations to some of the images
                # - scale to 80-120% of image height/width (each axis independently)
                # - translate by -20 to +20 relative to height/width (per axis)
                # - rotate by -45 to +45 degrees
                # - shear by -16 to +16 degrees
                # - order: use nearest neighbour or bilinear interpolation (fast)
                # - mode: use any available mode to fill newly created pixels
                #         see API or scikit-image for which modes are available
                # - cval: if the mode is constant, then use a random brightness
                #         for the newly created pixels (e.g. sometimes black,
                #         sometimes white)
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                #
                # Execute 0 to 5 of the following (less important) augmenters per
                # image. Don't execute all of them, as that would often be way too
                # strong.
                #
                iaa.SomeOf((0, 5),
                           [
                               # Convert some images into their superpixel representation,
                               # sample between 20 and 200 superpixels per image, but do
                               # not replace all superpixels with their average, only
                               # some of them (p_replace).
                               sometimes(
                                   iaa.Superpixels(
                                       p_replace=(0, 1.0),
                                       n_segments=(20, 200)
                                   )
                               ),

                               # Blur each image with varying strength using
                               # gaussian blur (sigma between 0 and 3.0),
                               # average/uniform blur (kernel size between 2x2 and 7x7)
                               # median blur (kernel size between 3x3 and 11x11).
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),

                               # Sharpen each image, overlay the result with the original
                               # image using an alpha between 0 (no sharpening) and 1
                               # (full sharpening effect).
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                               # Same as sharpen, but for an embossing effect.
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                               # Search in some images either for all edges or for
                               # directed edges. These edges are then marked in a black
                               # and white image and overlayed with the original image
                               # using an alpha of 0 to 0.7.
                               sometimes(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0, 0.7)),
                                   iaa.DirectedEdgeDetect(
                                       alpha=(0, 0.7), direction=(0.0, 1.0)
                                   ),
                               ])),

                               # Add gaussian noise to some images.
                               # In 50% of these cases, the noise is randomly sampled per
                               # channel and pixel.
                               # In the other 50% of all cases it is sampled once per
                               # pixel (i.e. brightness change).
                               iaa.AdditiveGaussianNoise(
                                   loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                               ),

                               # Either drop randomly 1 to 10% of all pixels (i.e. set
                               # them to black) or drop them on an image with 2-5% percent
                               # of the original size, leading to large dropped
                               # rectangles.
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout(
                                       (0.03, 0.15), size_percent=(0.02, 0.05),
                                       per_channel=0.2
                                   ),
                               ]),

                               # Invert each image's channel with 5% probability.
                               # This sets each pixel value v to 255-v.
                               iaa.Invert(0.05, per_channel=True),  # invert color channels

                               # Add a value of -10 to 10 to each pixel.
                               iaa.Add((-10, 10), per_channel=0.5),

                               # Change brightness of images (50-150% of original value).
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),

                               # Improve or worsen the contrast of images.
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                               # Convert each image to grayscale and then overlay the
                               # result with the original with random alpha. I.e. remove
                               # colors with varying strengths.
                               iaa.Grayscale(alpha=(0.0, 1.0)),

                               # In some images move pixels locally around (with random
                               # strengths).
                               sometimes(
                                   iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                               ),

                               # In some images distort local areas with varying strength.
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                           ],
                           # do all of the above augmentations in random order
                           random_order=True
                           )
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    else:
        raise Exception()

    images = seq(images=images)
    return images


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


class Cache:
    def __init__(self, cache_dir):
        self._cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def load(self):
        if not os.path.exists(os.path.join(self._cache_dir, 'latest_model.txt')):
            print('No existing cache...train from scratch')
            return
        print('Cache found')
        with open(os.path.join(self._cache_dir, 'latest_model.txt'), 'r') as fp:
            latest_model = fp.read().strip()
        with open(os.path.join(self._cache_dir, latest_model), 'rb') as fp:
            self.state_dict = pkl.load(fp)
            print('Model %s loaded.' % latest_model)
        with open(os.path.join(self._cache_dir, 'extra_vars.pkl'), 'rb') as fp:
            extra_vars = pkl.load(fp)
        for key, value in extra_vars.items():
            if hasattr(self, key):
                setattr(self, key, value[:extra_vars['next_batch_idx']] if isinstance(value, list) else value)
                print('Variable %s loaded.' % key)
        assert int(os.path.splitext(latest_model)[0].partition('-')[-1]) == extra_vars['next_batch_idx']

    def save(self, verbal=False, keep_latest_only=True):
        model_name = '%s-%d.pkl' % (time.strftime('%Y%m%d%H%M%S', time.localtime()), len(self.tasks_done))
        if keep_latest_only and os.path.exists(os.path.join(self._cache_dir, 'latest_model.txt')):
            with open(os.path.join(self._cache_dir, 'latest_model.txt'), 'r') as fp:
                old_model_name = fp.readline().strip()
                os.remove(os.path.join(self._cache_dir, old_model_name))

        with open(os.path.join(self._cache_dir, 'latest_model.txt'), 'w') as fp:
            fp.write(model_name)
        with open(os.path.join(self._cache_dir, model_name), 'wb') as fp:
            pkl.dump(self.state_dict, fp)
            if verbal:
                print('Model %s saved.' % model_name)
        extra_vars = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                extra_vars[key] = getattr(self, key)
                if verbal:
                    print('Variable %s saved.' % key)
        with open(os.path.join(self._cache_dir, 'extra_vars.pkl'), 'wb') as fp:
            pkl.dump(extra_vars, fp)
