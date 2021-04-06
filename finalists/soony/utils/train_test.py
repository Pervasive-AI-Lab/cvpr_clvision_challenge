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




#==========================================================
import copy
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

class clvc_dataset(Dataset):
    def __init__(self, image, label, transforms=None):
        self.image = image
        self.label = label
        self.transform = transforms
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        image = self.image[index,:,:,:].astype('uint8')
        label = torch.tensor(self.label[index].astype('int64'))
        if self.transform is not None:
            image = self.transform(image)
        #print(image.shape)
        return image, label

class clvc_latentset(Dataset):
    def __init__(self, args, model, image, label, replay_mem, transforms=None):
        self.labels = label
        self.replay_mem = replay_mem
        data = clvc_dataset(image, label, transforms)
        load = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            for batch_idx, (data_x,data_y) in enumerate(load):
                data_x = data_x.to(args.device)
                latent = model.get_latent(data_x)
                latent = latent.detach().cpu()
                if not 'latent' in self.replay_mem:
                    self.replay_mem['latent'] = latent
                    self.replay_mem['labels'] = data_y
                else:
                    self.replay_mem['latent'] = torch.cat([self.replay_mem['latent'],latent],dim=0)
                    self.replay_mem['labels'] = torch.cat([self.replay_mem['labels'],data_y],dim=0)
    def __len__(self):
        return len(self.replay_mem['labels'])
    def __getitem__(self, index):
        x = self.replay_mem['latent'][index]
        y = self.replay_mem['labels'][index]
        return x, y

    
def train_multitask_nc(args, optimizer, model, criterion, train_x, train_y, task_id, tf_train, heads, full_valdidset, tf_valid, prev_acc=0):
    stats = {"ram": [], "disk": []}
    
    acc = None
    ave_loss = 0
    val_acc_cp = []
    val_acc_max = prev_acc
    if prev_acc > 0:
        torch.save(model.state_dict(),args.scenario+'.pt')
    
    patience = args.patience
    early_stopping = False
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, min_lr=0.000001)
    
    # Training Loop
    for ep in range(args.epochs):
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        
        print("training ep: ", ep)
        train_data = clvc_dataset(train_x, train_y, transforms=tf_train)
        train_load = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
        model.to(args.device)
        model.train()
        
        correct_cnt, ave_loss, sample_cnt = 0, 0, 0
        
        for batch_idx, (image,label) in enumerate(train_load):
            image, label = image.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            logits = model(image)
            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == label).sum()
            sample_cnt += label.size(0)
            loss = criterion(logits, label)
            ave_loss += loss.item()
            loss.backward()
            optimizer.step()
            acc = correct_cnt.item() / sample_cnt
            ave_loss /= sample_cnt

            if batch_idx % 100 == 0:
                print('==>>> it: {}, avg. loss: {:.6f}, running train acc: {:.3f}'
                        .format(batch_idx, ave_loss, acc) )
                
        model_valid = copy.deepcopy(model)
        heads_valid = heads.copy()
        heads_valid.append(copy.deepcopy(model_valid.fc))
        torch.save(model.state_dict(),args.scenario+'_task_{}.pt'.format(task_id))
        
        val_stats, _ = test_multitask(args, model_valid, full_valdidset, tf_valid, multi_heads=heads_valid, verbose=False, criterion=criterion)
        val_loss = np.mean(val_stats['loss'])
        val_acc = np.mean(val_stats['acc'])
        val_acc_cp.append(val_acc)
        print('Validation loss / acc: {:.7f} / {:.4f}'.format(val_loss, val_acc))
                
        scheduler.step(val_acc)
        
        if val_acc_max < val_acc:
            print('model save at ep',ep)
            torch.save(model.state_dict(),args.scenario+'.pt')
            val_acc_max = val_acc
        
        if ep > patience:
            for i in range(len(val_acc_cp)-patience):
                if val_acc_cp[-1] < val_acc_cp[i]:
                    early_stopping=True
                    
                    break
        
        if val_acc > args.early_stopping or early_stopping:
            print('Early stopping at ep ', ep)
            break
                    
    print('load saved model')
    model.load_state_dict(torch.load(args.scenario+'.pt'))
    torch.save(model.state_dict(),args.scenario+'_task_{}.pt'.format(task_id))
                
    return ave_loss, acc, stats

def train_latent_replay(args, optimizer, model, criterion, train_x, train_y, replay_mem, task_id, tf_train, full_valdidset, tf_valid, prev_acc=0):
    stats = {"ram": [], "disk": []}
    
    acc = None
    ave_loss = 0
    val_acc_cp = []
    val_acc_max = prev_acc
    if prev_acc > 0:
        torch.save(model.state_dict(),args.scenario+'.pt')
    
    patience = args.patience
    early_stopping = False
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, min_lr=0.000001)
    
    # Training Loop
    for ep in range(args.epochs):
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        
        train_data = clvc_latentset(args, model, train_x, train_y, replay_mem, transforms=tf_train)
        train_load = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
        
        print("training ep: ", ep)
        model.to(args.device)
        model.train()
        
        correct_cnt, ave_loss, sample_cnt = 0, 0, 0
        
        for batch_idx, (image,label) in enumerate(train_load):
            image, label = image.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            logits = model(image, use_latent=True)
            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == label).sum()
            sample_cnt += label.size(0)
            loss = criterion(logits, label)
            ave_loss += loss.item()
            loss.backward()
            optimizer.step()
            acc = correct_cnt.item() / sample_cnt
            ave_loss /= sample_cnt

            if batch_idx % 100 == 0:
                print('==>>> it: {}, avg. loss: {:.7f}, running train acc: {:.3f}'
                        .format(batch_idx, ave_loss, acc) )
                
        val_stats, _ = test_multitask(args, model, full_valdidset, tf_valid, verbose=False, criterion=criterion)
        val_loss = np.mean(val_stats['loss'])
        val_acc = np.mean(val_stats['acc'])
        val_acc_cp.append(val_acc)
        print('Validation loss / acc: {:.7f} / {:.4f}'.format(val_loss, val_acc))
                
        scheduler.step(val_acc)
        
        
        
        if val_acc_max < val_acc:
            print('model save at ep',ep)
            torch.save(model.state_dict(),args.scenario+'.pt')
            val_acc_max = val_acc
            
        
        if ep > patience:
            for i in range(len(val_acc_cp)-patience):
                if val_acc_cp[-1] < val_acc_cp[i]:
                    early_stopping=True
                    break
        
        if val_acc > args.early_stopping or early_stopping:
            print('Early stopping at ep ', ep)
            break
                    
    print('load saved model')
    model.load_state_dict(torch.load(args.scenario+'.pt'))
            
    return ave_loss, acc, stats



def train_latent_replay2(args, optimizer, model, criterion, train_x, train_y, replay_mem, task_id, tf_train, full_valdidset, tf_valid, prev_acc=0, run=False):
    stats = {"ram": [], "disk": []}
    
    acc = None
    ave_loss = 0
    val_acc_cp = []
    val_acc_max = prev_acc
    if prev_acc > 0:
        torch.save(model.state_dict(),args.scenario+'.pt')
    
    train_data = clvc_latentset(args, model, train_x, train_y, replay_mem, transforms=tf_train)
    train_load = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    
    if run:
        patience = args.patience
        early_stopping = False

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, min_lr=0.000001)

        # Training Loop
        for ep in range(args.epochs):
            stats['disk'].append(check_ext_mem("cl_ext_mem"))
            stats['ram'].append(check_ram_usage())

            print("training ep: ", ep)
            model.to(args.device)
            model.train()

            correct_cnt, ave_loss, sample_cnt = 0, 0, 0

            for batch_idx, (image,label) in enumerate(train_load):
                image, label = image.to(args.device), label.to(args.device)
                optimizer.zero_grad()
                logits = model(image, use_latent=True)
                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == label).sum()
                sample_cnt += label.size(0)
                loss = criterion(logits, label)
                ave_loss += loss.item()
                loss.backward()
                optimizer.step()
                acc = correct_cnt.item() / sample_cnt
                ave_loss /= sample_cnt

                if batch_idx % 100 == 0:
                    print('==>>> it: {}, avg. loss: {:.7f}, running train acc: {:.3f}'
                            .format(batch_idx, ave_loss, acc) )

            val_stats, _ = test_multitask(args, model, full_valdidset, tf_valid, verbose=False, criterion=criterion)
            val_loss = np.mean(val_stats['loss'])
            val_acc = np.mean(val_stats['acc'])
            val_acc_cp.append(val_acc)
            print('Validation loss / acc: {:.7f} / {:.4f}'.format(val_loss, val_acc))

            scheduler.step(val_acc)

            if val_acc_max < val_acc:
                print('model save at ep',ep)
                torch.save(model.state_dict(),args.scenario+'.pt')
                val_acc_max = val_acc

            if ep > patience:
                for i in range(len(val_acc_cp)-patience):
                    if val_acc_cp[-1] < val_acc_cp[i]:
                        early_stopping=True
                        break

            if val_acc > args.early_stopping or early_stopping:
                print('Early stopping at ep ', ep)
                break

        print('load saved model')
        model.load_state_dict(torch.load(args.scenario+'.pt'))
            
    return ave_loss, acc, stats

#==========================================================





def test_multitask(args, model, test_set, tf_valid, multi_heads=[], verbose=True, criterion=None):

    acc_x_task = []
    stats = {'accs': [], 'acc': [], 'loss':[]}
    preds = []
    
    model.to(args.device)
    model.eval()

    for (x, y), t in test_set:
        data = clvc_dataset(x, y, transforms=tf_valid)
        load = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        correct_cnt, ave_loss, sample_cnt = 0, 0, 0

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            model.load_state_dict(torch.load(args.scenario+'_task_{}.pt'.format(t)))
        

        with torch.no_grad():
            for batch_idx, (image,label) in enumerate(load):
                image, label = image.to(args.device), label.to(args.device)
                logits = model(image)
                if criterion is not None:
                    loss = criterion(logits, label)
                    ave_loss += loss.item()
                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == label).sum()
                preds += list(pred_label.data.cpu().numpy())
                sample_cnt += label.size(0)
            acc = correct_cnt.item() / sample_cnt
            if criterion is not None:
                ave_loss /= sample_cnt

        if verbose:
            print('TEST Loss/Acc. Task {} ==>>> loss: {:.7f}, acc: {:.3f}'.format(t, ave_loss, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)
        if criterion is not None:
            stats['loss'].append(ave_loss)

    stats['acc'].append(np.mean(acc_x_task))


    return stats, preds






