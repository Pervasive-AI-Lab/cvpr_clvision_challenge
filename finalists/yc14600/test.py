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

import argparse
import os
import time
import csv
import copy
import gc
from core50.dataset import CORE50
import torch
import numpy as np
from utils.train_test import train_net, test_multitask
from core50.dataset import preprocess_imgs
import torchvision.models as models
from models.resnet_drl import resnet18, resnet50
from models.drloss import cl_dr_loss_softmax,cl_dr_loss_sigmoid
from utils.common import create_code_snapshot



def shuffle_data(*X):
    
    N = X[0].shape[0]
    idx = np.arange(N)
    
    np.random.shuffle(idx)  
    rt = [x[idx] for x in X]
    return rt

def one_hot_encoder(label,H=None,out=None):
    
    N = label.shape[0]
    if H is None:
        H = int(np.max(label)+1)
    label = label.astype(np.int32)
    
    if out is None:
        Y = np.zeros((N,H),dtype=np.float32)
    else:
        Y = out * 0
        
    Y[range(N),label] = 1

    
    return Y

def main(args):

    # print args recap
    print(args, end="\n\n")
    # directory with the code snapshot to generate the results
    sub_dir = os.path.join('submissions', args.sub_dir, '-'.join(time.ctime().replace(':','').split(' '))) 
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    with open(os.path.join(sub_dir,'configures.txt'),'w') as f:
        f.write(str(args))
    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
    dataset = CORE50(root=args.data_dir, scenario=args.scenario,
                     preload=args.preload_data,train_size=args.train_size)

    # Get the validation set
    print("Recovering validation set...")
    full_valdidset = dataset.get_full_valid_set()
    print('val len',len(full_valdidset),len(full_valdidset[0][0][0]),full_valdidset[0][1])
    for i,vt in enumerate(full_valdidset):
        val_x = preprocess_imgs(vt[0][0])
        val_y = vt[0][1]
        full_valdidset[i] = [(val_x,val_y),vt[1]]
    print('val len after',len(full_valdidset),len(full_valdidset[0][0][0]),full_valdidset[0][1])

    # model
    one_hot, incl_fc = False, True
    if args.drl_lmb > 0:
        if args.sigmoid and args.scenario != 'ni':
            criterion = cl_dr_loss_sigmoid
            one_hot = True

        else:
            criterion = cl_dr_loss_softmax
    else:
        if args.sigmoid and args.scenario != 'ni':
            criterion = torch.nn.BCELoss()
            one_hot = True

        else:
            criterion = torch.nn.CrossEntropyLoss()

    if args.classifier == 'ResNet18':
        classifier = resnet18(pretrained=True,incl_fc=incl_fc) if args.drl_lmb>0 else models.resnet18(pretrained=True)         
        if one_hot:
            classifier.fc = torch.nn.Sequential(torch.nn.Linear(512, args.n_classes),torch.nn.Sigmoid())
        else:
            classifier.fc = torch.nn.Linear(512, args.n_classes)
    elif args.classifier == 'ResNet50':
        classifier = resnet50(pretrained=True,incl_fc=incl_fc) if args.drl_lmb>0 else models.resnet50(pretrained=True)
        if one_hot:
            classifier.fc = torch.nn.Sequential(torch.nn.Linear(2048, args.n_classes),torch.nn.Sigmoid())
        else:
            classifier.fc = torch.nn.Linear(2048, args.n_classes)

    elif args.classifier == 'ResNetst':
        if args.drl_lmb > 0:
            from models.resnet_drl import resnest50
        else:
            from ResNeSt.resnest.torch import resnest50
        #torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        classifier = resnest50(pretrained=True)#torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        if one_hot:
            classifier.fc = torch.nn.Sequential(torch.nn.Linear(2048, args.n_classes),torch.nn.Sigmoid())
        else:
            classifier.fc = torch.nn.Linear(2048, args.n_classes)

############################################### fix layers ###########################################
    fix_layer = []
    for l in range(1,4):
        if str(l) in args.fix:
            fix_layer.append('layer'+str(l))
            if l == 1:
                fix_layer+=['conv1','bn1']
    print('fix layers: {}'.format(fix_layer))
    for name, value in classifier.named_parameters():
        name = name.split('.')
        if name[0] in fix_layer:
            value.requires_grad = False


############################################# config optimizer #########################################
    if args.opt == 'sgd':
        opt = torch.optim.SGD(classifier.parameters(), lr=args.lr,momentum=args.mmt,nesterov=args.nestro)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(classifier.parameters(),lr=args.lr,betas=(args.beta1,args.beta2))
    elif args.opt == 'adamw':
        opt = torch.optim.AdamW(classifier.parameters(),lr=args.lr,betas=(args.beta1,args.beta2))
    elif args.opt == 'adagrad':
        opt = torch.optim.Adagrad(classifier.parameters(),lr=args.lr,lr_decay=args.lr_decay)
    elif args.opt == 'rmsprop':
        opt = torch.optim.RMSprop(classifier.parameters(),lr=args.lr,alpha=args.alpha,momentum=args.mmt)
    else:
        raise NotImplementedError('Please add implementation for optimizer {} first.'.format(args.opt))
    
 

    #criterion = cl_dr_loss if args.drl_lmb > 0 else torch.nn.CrossEntropyLoss()

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []
    ext_mem = {} 
    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch
        #train_x, train_y = train_x[:200],train_y[:200]
        #if i > 2:
        #    break
        train_x = preprocess_imgs(train_x)
        if one_hot:
            train_y = one_hot_encoder(train_y,H=args.n_classes)

        # adding eventual replay patterns to the current batch
        if args.scenario!='offline':
            if args.fix_budget:
                if not ext_mem:
                    total_mem = args.replay_examples
                else:
                    args.replay_examples = total_mem//(len(ext_mem)+1)

                          
            idxs_cur = np.random.choice(
            train_x.shape[0], min(args.replay_examples,train_x.shape[0]), replace=False
            )
            if args.replay_examples > 0:
                
                mem_x = train_x[idxs_cur]
                mem_y = train_y[idxs_cur]
                ext_mem.update({i:[mem_x, mem_y]})
           
                if i > 0 and not args.ER_type:
                    
                    train_x = np.concatenate([train_x]+ [ext_mem[k][0] for k in range(i)])
                    train_y = np.concatenate([train_y]+ [ext_mem[k][1] for k in range(i)])
                
            
        if args.final:
            if args.scenario != 'multi-task-nc' and i == len(dataset.tasks_id)-1:
                print('add final')
                val_x,val_y = full_valdidset[0][0][0],full_valdidset[0][0][1]

                train_x = np.concatenate([train_x,val_x])
                train_y = np.concatenate([train_y,val_y])   
                if one_hot:
                    val_y = one_hot_encoder(val_y,H=args.n_classes)             

            if args.scenario == 'multi-task-nc':
                print('add final')
                val_x,val_y = full_valdidset[i][0][0],full_valdidset[i][0][1] 
                train_x = np.concatenate([train_x,val_x])
                train_y = np.concatenate([train_y,val_y])   
                if one_hot:
                    val_y = one_hot_encoder(val_y,H=args.n_classes)                
                        
        
        print("----------- batch {0} -------------".format(i))
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        print("Task Label: ", t)

        # train the classifier on the current batch/task
        pad = False if args.scenario=='offline' else True
        _, _, stats = train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, i,
            args.epochs, preproc=None, use_cuda=args.cuda, pad=pad,
            ER_type=args.ER_type, ext_mem=ext_mem, mem_size=args.replay_examples,
            drl_lmb=args.drl_lmb, out_dim=args.n_classes,scenario=args.scenario,
            one_hot=one_hot
        )
        if args.scenario == "multi-task-nc" and not args.singlehead:
            heads.append(copy.deepcopy(classifier.fc))

        # collect statistics
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test on the validation set
        stats, _ = test_multitask(
            classifier, full_valdidset, args.batch_size,
            preproc=None, multi_heads=heads, verbose=False, one_hot=one_hot
        )

        valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

        
        if args.fix_budget and i > 0:
            print('fix budget update',args.replay_examples)
            for i, mem in ext_mem.items():
                ext_mem[i] = [mem[0][:args.replay_examples],mem[1][:args.replay_examples]]
        
        gc.collect()

    print("Avg. acc for all batches:")
    print(np.mean(valid_acc))

    # Generate submission.zip

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    with open(sub_dir+'valid_accuracy.csv','w') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(valid_acc)

    # test_preds.txt: with a list of labels separated by "\n"
    
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(
        classifier, full_testset, args.batch_size, preproc=preprocess_imgs,
        multi_heads=heads, verbose=True,one_hot=one_hot
    )
    print("------------------------------------------")
    print("Test acc: {}".format(stats['acc']))
    print("------------------------------------------")

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic','offline'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')
    parser.add_argument('--train_size', type=int,default=-1,
                        help='training size for offline tests,-1 use all data')
    parser.add_argument('--singlehead', default=False, action='store_true')
    parser.add_argument('--final',default=False,action='store_true',
                        help='add valid set to training set in final training')
    parser.add_argument('--sigmoid', default=False, action='store_true',
                        help='use sigmoid corss entropy except ni task')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet50',
                        choices=['ResNet18','ResNet50','ResNetst'])
    parser.add_argument('-noc', '--no_cuda', default=False,action='store_true',
                        help='use cuda or not')
    parser.add_argument('--fix', type=str, default='',
                       help='specify which layer to be fixed with pre-trained parameters.')


    # Optimization
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--opt', type=str, default='sgd',choices=['sgd','adam','adamw','adagrad','rmsprop'],
                        help='type of optimizer')
    parser.add_argument('--mmt', type=float, default=0.,
                        help='momemtum of optimizer sgd or rmsprop')
    parser.add_argument('--nestro',  default=False,action='store_true',
                        help='nesterov of optimizer sgd')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 of optimizer adam*')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 of optimizer adam')
    parser.add_argument('--lr_decay', type=float, default=0.,
                        help='lr_decay of optimizer adagrad')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='alpha of optimizer rmsprop')


    # Continual Learning
    parser.add_argument('--replay_examples', type=int, default=0,
                        help='data examples to keep in memory for each batch '
                             'for replay.')
    parser.add_argument('--ER_type', type=str, default=None,choices=['ER','BER'],
                        help='use BER or ER')

    parser.add_argument('--fix_budget', default=False, action='store_true',
                        help='fix memory budget or not')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')
    parser.add_argument('--data_dir',type=str,default='core50/data/',help='folder of keeping core50 data')

    # DRL config
    parser.add_argument('--drl_lmb',type=float,default=0.,help='lambda of DRL')

    
    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = not args.no_cuda#torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
