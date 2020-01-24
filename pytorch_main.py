import numpy as np
import pickle as pkl
import os
import logging
from PIL import Image
import torch
import torch.nn.functional as F
from pdb import set_trace

from core50.dataset import CORE50
from utils import models


def main(args):

    ## Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    ## and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50('core50/data/', scenario=args.scenario)

    ## Get the fixed test set
    #test_x, test_y = dataset.get_test_set()

    ## CLASSIFIER
    if args.classifier == 'ResNet18':
        classifier = models.ResNet18(args).to(args.device)
    elif args.classifier == 'MLP':
        classifier = models.MLP(args).to(args.device)

    if args.replay:
        from utils.buffer import Buffer
        buffer = Buffer(args)

    opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)

    ## loop over the training incremental batches
    for i, current_batch in enumerate(dataset):

        ## WARNING train_batch is NOT a mini-batch, but one incremental batch!
        ## You can later train with SGD indexing train_x and train_y properly.
        current_x, current_y = current_batch

        ## convert to pytorch and resize
        current_x = torch.tensor(current_x).view([-1, *args.input_size])
        current_y = torch.tensor(current_y, dtype=torch.long)

        print("----------- batch {0} -------------".format(i))
        print("train shape: {0}".format(current_x.shape))

        ## create input data
        if args.replay and i>0:
            replay_data = buffer.sample(args.replay_size)
            replay_x, replay_y = replay_data[0], replay_data[1]
            input_x, input_y = torch.cat([current_x, replay_x]), torch.cat([current_y, replay_y])
        else:
            input_x, input_y = current_x, current_y

        ## predict
        logits = classifier(input_x[:args.batch_size])
        loss = F.cross_entropy(logits, input_y[:args.batch_size])

        ## train
        opt.zero_grad()
        loss.backward()
        opt.step()

        if args.replay:
            buffer.add_reservoir(current_x, current_y, None, i)

        if i % args.print_every == 0:
            logits = classifier(input_x)
            pred = logits.argmax(dim=1, keepdim=True)
            train_acc = pred.eq(input_y.view_as(pred)).sum().item() / pred.size(0)
            print('training error: {:.2f} \t training accuracy {:2f}'.format(loss, train_acc))

            if i % args.eval_every == 0:
            #TODO(not sure what we do yet here)
                pass

    #TODO(final evaluation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="nicv2_79",
        choices=['ni', 'nc', 'nic', 'nicv2_79','nicv2_196', 'nicv2_391'])
    parser.add_argument('--data_folder', type=str, default='data',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('-da', '--dataset', type=str, default='core50',
        help='Name of the dataset (default: core50).')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
        choices=['ResNet18', 'MLP'])
    parser.add_argument('-hs', '--hidden_size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        'or hidden size of an MLP. If None, kept to default')

    # CL
    parser.add_argument('--replay', type=bool, default=True,
        help='enable replay')
    parser.add_argument('--mem_size', type=int, default=600,
        help='number of saved samples per class')
    parser.add_argument('--replay_size', type=int, default=60,
        help='number of replays per batch')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
        help='batch_size')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)

    args = parser.parse_args()

    # Bonus args

    if args.dataset=='core50':
        args.input_size = [128, 128, 3]
        args.input_size = [3, 128, 128]
        args.n_classes = 50
    else:
        Exception('Not implemented yet!')

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    args.mem_size = args.mem_size*args.n_classes #convert from per class to total memory

    main(args)
