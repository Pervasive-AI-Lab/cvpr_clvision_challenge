import numpy as np
import pickle as pkl
import os
import logging
from PIL import Image
import torch
from pdb import set_trace

from core50.dataset import CORE50


def main(args):

    ## Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    ## and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50('core50/data/', scenario="nicv2_79")

    '''
    ## Get the fixed test set
    test_x, test_y = dataset.get_test_set()

    ## CLASSIFIER
    #TODO(load some models)
    if args.classifier == 'ResNet18':
        classifier = ResNet18(args.n_classes, nf=20, input_size=args.input_size)
    elif args.classifier == 'MLP':
        classifier = classifier(args).to(args.device)
    '''

    if args.replay:
        from utils.buffer import Buffer
        buffer = Buffer(args)

    #opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)

    ## loop over the training incremental batches
    for i, train_batch in enumerate(dataset):

        ## WARNING train_batch is NOT a mini-batch, but one incremental batch!
        ## You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y = train_batch

        print("----------- batch {0} -------------".format(i))
        print("train shape: {0}, test_shape: {0}"
              .format(train_x.shape, train_y.shape))

        '''
        if args.replay and i>0:
            old_x, old_y = buffer.sample()
            train_x, train_y = torch.cat([train_x, old_x]), torch.cat([train_y, old_y])

        logits = model(train_x)
        loss = F.cross_entropy(logits, target)
        pred = logits.argmax(dim=1, keepdim=True)

        if args.replay:
            buffer.add_reservoir(train_x, train_y, None, i)

        if i % args.print_every == 0:
            train_acc = pred.eq(target.view_as(pred)).sum().item() / pred.size(0)
            print('training error: {:.2f} \t training accuracy {:2f}'
                  .format(loss, train_acc))

            if i % args.eval_every == 0:
            #TODO(not sure what we do yet here)
                pass

        '''
    #TODO(final evaluation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--mode', type=str, default='train',
        choices=['train', 'test'])
    parser.add_argument('--data_folder', type=str, default='data',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('-da', '--dataset', type=str, default='core50',
        help='Name of the dataset (default: core50).')

    # Model
    parser.add_argument('-hs', '--hidden_size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        'or hidden size of an MLP. If None, kept to default')

    # CL
    parser.add_argument('--replay', type=bool, default=True,
        help='enable replay')
    parser.add_argument('--mem_size', type=int, default=600,
        help='number of saved samples per class')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num_epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--patience', type=int, default=5,
        help='Number of epochs without a valid loss decrease we can wait')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    # Bonus args

    if args.dataset=='core50':
        args.input_size = [128, 128, 3]
        args.n_classes = 100
    else:
        Exception('Not implemented yet!')

    args.cuda = torch.cuda.is_available()

    args.mem_size = args.mem_size*args.n_classes #convert from per class to total memory

    main(args)
