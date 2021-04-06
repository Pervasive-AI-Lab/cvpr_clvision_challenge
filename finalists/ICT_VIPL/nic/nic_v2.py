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

import shutil
from PIL import Image
import signal
import random
import argparse
import os
import time
import copy
from core50.dataset_npl import CORE50
import torch
import numpy as np
from utils.train_test_v2 import train_net, test_multitask
import torchvision.models as models
from utils.common import create_code_snapshot, Cache


def save_accs(val_accs, sub_dir):
    with open(os.path.join(sub_dir, 'val_accs.txt'), 'w') as fout:
        for val_acc in val_accs:
            fout.write('%f' % val_acc + os.linesep)


def main(args):
    # print args recap
    print(args, end="\n\n")

    # params dir
    params_dir = 'run_%d-' % args.run if args.run > 0 else ''
    params_dir += ('%d-' % args.batch_size if not args.batch_size == 64 else '')
    params_dir += '%s_epochs_%d-wd_%f-lr_%.4f' % (args.classifier, args.epochs, args.weight_decay, args.lr)
    params_dir += '_adam' if args.optimizer == 'adam' else ''
    if args.joint_training:
        params_dir += '_joint_training'
    else:
        params_dir += ('-seed_%d' % args.random_seed)
        params_dir += ('' if args.aug == 'none' else '-%s' % args.aug)
        params_dir += (
            ('-replay_%s_%d' % (args.select_criterion,
                                args.replay_examples)) if args.replay_examples > 0 else '')
        params_dir += ('-dynamic' if args.dynamic_exem else '')
        params_dir += ('-lwf_' + str(args.lwf_T) + '_' + str(args.lwf_weight)) if args.lwf else ''
        params_dir += (('-focal_' + str(args.focal_T) + '_' + str(args.focal_weight)) if args.focal else '')
        params_dir += '_post_scaling' if not args.no_post_scaling else ''
    print('FOLDER: %s' % params_dir)

    # do not remove this line
    start = time.time()

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # deterministic
    if not args.no_det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root=args.data_dir, scenario=args.scenario, cumul=args.joint_training, run=args.run)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()

    # model
    if args.classifier == 'ResNet18':
        classifier = models.resnet18(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)  # change the fc layer match CLVision
    elif args.classifier == 'ResNet34':
        classifier = models.resnet34(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)  # change the fc layer match CLVision
    elif args.classifier == 'ResNet50':
        classifier = models.resnet50(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)  # change the fc layer match CLVision
    elif args.classifier == 'ResNet101':
        classifier = models.resnet101(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)  # change the fc layer match CLVision
    elif args.classifier == 'MobileNet':
        classifier = models.mobilenet_v2(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)  # change the fc layer match CLVision
    elif args.classifier == 'ResNeXt101':
        classifier = models.resnext101_32x8d(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)  # change the fc layer match CLVision
    elif args.classifier == 'WideResNet50':
        classifier = models.wide_resnet50_2(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, args.n_classes)
    else:
        raise Exception('Invalid classifier name')

    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.focal:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # vars to update over time

    if args.cache_dir == '':
        cache_dir = ''
    else:
        cache_dir = args.cache_dir if args.submit_ver else os.path.join(args.cache_dir, params_dir)

    vars = Cache(cache_dir)
    vars.valid_acc = []
    vars.ext_mem_sz = []
    vars.ram_usage = []
    vars.heads = []
    vars.tasks_done = []
    vars.next_batch_idx = 0
    ext_mem = None

    # load cache
    if cache_dir:
        vars.load()

    if hasattr(vars, 'state_dict'):
        classifier.load_state_dict(vars.state_dict)

    val_accs = vars.valid_acc[:vars.next_batch_idx]

    # load exemplars
    if not args.joint_training and args.replay_examples > 0:

        if not args.save_exemplars:
            assert vars.next_batch_idx == 0
        else:
            # exemplars dir
            exem_dir = os.path.join('cl_ext_mem', params_dir)
            if not os.path.exists(exem_dir):
                os.makedirs(exem_dir)

        if vars.next_batch_idx > 0:
            Xs, Ys = [], []
            for filename in os.listdir(exem_dir):
                filepath = os.path.join(exem_dir, filename)
                Xs.append(np.array(Image.open(filepath)))
                Ys.append(float(os.path.splitext(filename)[0].split('_')[-1]))
            ext_mem = [np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)]

    # submissions dir
    if args.submit_ver:
        sub_dir = os.path.join('submissions', args.sub_dir)
    else:
        sub_dir = os.path.join('submissions', args.sub_dir, params_dir)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # seen classes
    seen_classes = set()

    # ave losses
    ave_losses = []

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if i == args.to_batch_idx:
            print('Early stop at batch %d' % (i + 1))
            break

        # skip batches when continuing training
        if i < vars.next_batch_idx:
            print('Skipping batch %d' % i)
            continue

        train_x, train_y, t = train_batch  # t is the task indicato
        # update seen classes
        seen_classes.update(set(train_y.astype(int)))

        num_new_classes = len(set(train_y))
        num_new_samples = len(train_x)

        # adding eventual replay patterns to the current batch
        if not args.joint_training and ext_mem is not None and args.replay_examples > 0:
            train_x = np.concatenate((train_x, ext_mem[0]))
            train_y = np.concatenate((train_y, ext_mem[1]))

        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        if t in vars.tasks_done:
            # continue
            pass

        # train the classifier on the current batch/task
        ave_loss, _, stats = train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, t,
            args.epochs, params_dir, (args.lwf, args.lwf_weight, args.lwf_T) if i > 0 else (False, 0, args.lwf_T),
            (args.focal, args.focal_weight, args.focal_T),
            aug=args.aug if args.aug is not 'none' else None
        )

        sample_nums = np.array([np.sum(train_y == class_idx) for class_idx in range(len(dataset.classnames))])
        priors = sample_nums / np.sum(sample_nums)
        priors[priors == 0.] = np.inf

        if args.scenario == "multi-task-nc":
            vars.heads.append(copy.deepcopy(classifier.fc))

        # collect statistics
        vars.ext_mem_sz += stats['disk']
        vars.ram_usage += stats['ram']

        # test on the validation set
        stats, pred_labels = test_multitask(
            classifier, full_valdidset, args.batch_size, priors,
            multi_heads=vars.heads, verbose=False,
            post_scaling=not args.no_post_scaling
        )

        vars.valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")
        if isinstance(stats['acc'], list):
            val_accs.extend(stats['acc'])
        else:
            val_accs.append(stats['acc'])

        # save the validation accuracies
        save_accs(val_accs, sub_dir)

        # update next batch_idx
        vars.next_batch_idx = i + 1

        vars.tasks_done.append(t)
        if cache_dir:
            vars.state_dict = classifier.state_dict()
            vars.save()

        # select and store exemplars
        num_exemplars_now = args.replay_examples
        if args.dynamic_exem:
            if len(ave_losses) > 20:
                abs_gain = int(np.clip(np.ceil(np.abs(ave_loss - np.mean(ave_losses)) / np.std(ave_losses)),
                                       a_max=3, a_min=0))
                if np.mean(ave_losses) > ave_loss:
                    num_exemplars_now -= abs_gain
                elif np.mean(ave_losses) < ave_loss:
                    num_exemplars_now += abs_gain
        ave_losses.append(ave_loss)

        if not args.joint_training and args.replay_examples > 0:
            if args.select_criterion == 'random':
                idxs_cur = np.random.choice(
                    num_new_samples, num_exemplars_now * num_new_classes, replace=False
                )
            elif args.select_criterion == 'first':
                if i == 0:
                    idxs_cur = np.array(
                        np.concatenate([np.array(range(10)) * 300 + i for i in range(num_exemplars_now)]))
                else:
                    idxs_cur = np.array(range(num_exemplars_now))
            elif args.select_criterion == 'linspace':
                idxs_cur = np.linspace(0, num_new_samples, endpoint=False, dtype=int,
                                       num=num_exemplars_now * num_new_classes)

            # store exemplars
            exemplars = [train_x[idxs_cur], train_y[idxs_cur]]
            if args.save_exemplars:
                for img_i in range(len(idxs_cur)):
                    img = exemplars[0][img_i]
                    label = int(exemplars[1][img_i])
                    shutil.copy(img, os.path.join(exem_dir, '%d_%d_%d.jpg' % (i, img_i, label)))

            # concatenate with the previous ones
            if i == 0:
                ext_mem = exemplars
            else:
                ext_mem = [
                    np.concatenate((exemplars[0], ext_mem[0])),
                    np.concatenate((exemplars[1], ext_mem[1]))]

        signal.signal(signal.SIGINT, s)

    # Generate submission.zip
    # directory with the code snapshot to generate the results

    # save the validation accuracies
    save_accs(val_accs, sub_dir)

    # final analysis
    if not args.submit_ver:

        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        full_no_reduced_validset = dataset.get_full_valid_set(reduced=False)
        stats, pred_labels = test_multitask(
            classifier, full_no_reduced_validset, args.batch_size,
            priors=np.ones([len(CORE50.classnames)]), multi_heads=vars.heads, verbose=False,
            post_scaling=not args.no_post_scaling
        )

        with open(sub_dir + "/val_preds.txt", "w") as wf:
            for pred in pred_labels:
                wf.write(str(pred) + "\n")

        conf_mat = confusion_matrix(full_no_reduced_validset[0][0][1].astype(np.int64), pred_labels)
        plt.figure(figsize=(10, 9))
        sns.heatmap(conf_mat, vmax=45, square=True, xticklabels=CORE50.classnames,
                    yticklabels=CORE50.classnames)
        plt.savefig(os.path.join(sub_dir, 'final_conf_mat.pdf'))
        plt.close()

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    metadata_file = os.path.join(sub_dir, 'metadata.txt')
    if not os.path.exists(metadata_file):
        elapsed = (time.time() - start) / 60
        print("Training Time: {}m".format(elapsed))
        with open(sub_dir + "/metadata.txt", "w") as wf:
            for obj in [
                np.average(vars.valid_acc), elapsed, np.average(vars.ram_usage),
                np.max(vars.ram_usage), np.average(vars.ext_mem_sz), np.max(vars.ext_mem_sz)
            ]:
                wf.write(str(obj) + "\n")
    else:
        print('Metadata_file exists: %s' % metadata_file)

    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(
        classifier, full_testset, args.batch_size, priors=np.ones(len(CORE50.classnames)),
        multi_heads=vars.heads, verbose=False
    )

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="nic",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--data_dir', type=str, default="core50/data/")

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet50')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')

    # Continual Learning
    parser.add_argument('--random_seed', type=int, default=1993)
    parser.add_argument('--select_criterion', type=str, default='random')
    parser.add_argument('--replay_examples', type=int, default=5,
                        help='data examples to keep in memory for each batch '
                             'for replay.')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--joint_training', action='store_true')
    parser.add_argument('--aug', type=str, default='l3', help='l1|l2|l3')
    parser.add_argument('--no_post_scaling', action='store_true', help='post_scaling')
    parser.add_argument('--dynamic_exem', action='store_true', help='exem number based on loss')

    # learning without forgetting
    parser.add_argument('--lwf', action='store_true')
    parser.add_argument('--lwf_T', type=float, default=2.)
    parser.add_argument('--lwf_weight', type=float, default=1.)

    # focal loss
    parser.add_argument('--focal', action='store_true', help='focal loss')
    parser.add_argument('--focal_T', type=float, default=2.)
    parser.add_argument('--focal_weight', type=float, default=1.)

    # debug
    parser.add_argument('--run', type=int, default=0, help='which order to run')
    parser.add_argument('--to_batch_idx', type=int, default=-1)
    parser.add_argument('--no_det', action='store_true',
                        help='whether to use deterministic mode to improve reproducibility')
    parser.add_argument('--save_exemplars', action='store_true',
                        help='save exemplars in the storage mainly for debugging')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="nic",
                        help='directory of the submission file for this exp.')
    parser.add_argument('--submit_ver', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="",
                        help='directory of the cache file.')

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    start_time = time.time()
    main(args)
    print('Total running time: %.2f secs' % (time.time() - start_time))
