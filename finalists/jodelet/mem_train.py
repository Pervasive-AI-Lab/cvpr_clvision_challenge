#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Submission for CVPR 2020 CLVision Challenge
# Copyright (c) 2020. Jodelet Quentin, Vincent Gripon, and Tsuyoshi Murata. All rights reserved.
# Copyrights licensed under the CC-BY-NC 4.0 License.
# See the accompanying LICENSE file for terms. 

# Based on the utils.train_test.py by Vincenzo Lomonaco, Massimo Caccia, 
# Pau Rodriguez, Lorenzo Pellegrini (Under the CC BY 4.0 License)
# From https://github.com/vlomonaco/cvpr_clvision_challenge

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
from utils.common import check_ext_mem, check_ram_usage


class ReservoirNet(torch.nn.Module) :
    def __init__(self, modelClass, memorySize, memoryDataDim, memoryTargetDim, sameDeviceMemory=True) :
        super(ReservoirNet, self).__init__()
        self.net = modelClass()
        self.memory = MemoryReservoir(memorySize, memoryDataDim, memoryTargetDim)
        self.sameDeviceMemory = sameDeviceMemory

    def forward(self, x) :
        return self.net(x)
    
    def addToMemory(self, tasks, data, targets) :
        self.memory.add(tasks, data, targets)
    
    def sampleFromMemory(self, size) :
        return self.memory.sample(size)


class MemoryReservoir(torch.nn.Module) :
    def __init__(self, memorySize, dataDim, targetDim=[1,]) : 
        super(MemoryReservoir, self).__init__()
        self.register_buffer("memoryData", torch.empty([memorySize, *dataDim], dtype=torch.float))
        self.register_buffer("memoryTarget", torch.empty([memorySize, *targetDim], dtype=torch.long))
        self.register_buffer("observed", torch.zeros([1], dtype=torch.long))
    
    def add(self, tasks, inputs, targets) :
        for i in range(len(targets)) :
            if self.observed < self.memoryTarget.size(0)  :
                self.memoryData[self.observed] = inputs[i]
                self.memoryTarget[self.observed] = targets[i]
            else :
                pos = torch.randint(self.observed.item(), (1,)).item()
                if pos < self.memoryTarget.size(0) : 
                    self.memoryData[pos] = inputs[i]
                    self.memoryTarget[pos] = targets[i]
            
            self.observed += 1
    
    def sample(self, size) :
        datas = torch.FloatTensor().to(self.memoryData.device)
        targets = torch.LongTensor().to(self.memoryData.device)

        if self.observed.item() > 0 :
            datas = torch.empty([size, *self.memoryData.size()[1:]], dtype=torch.float, device=self.memoryData.device)
            targets = torch.empty([size], dtype=torch.long, device=self.memoryData.device)

            lenght = min([self.memoryTarget.size(0), self.observed.item()])
            randID = torch.randint(lenght, (size,))
            for i in range(len(randID)) :
                datas[i] = self.memoryData[randID[i]].unsqueeze(0)
                targets[i] = self.memoryTarget[randID[i]].unsqueeze(0)
        
        return datas, targets


def train_net(featureModel, classifier, optimizer, epochs, batchSize, batchSize_backbone, device, x, y, t, preproc=None):

    cur_ep = 0
    cur_train_t = t
    stats = {"ram": [], "disk": []}

    if preproc:
        x = preproc(x)

    train_x = torch.from_numpy(x).type(torch.FloatTensor)
    train_y = torch.from_numpy(y).type(torch.LongTensor)

    acc = None

    classifier.train()

    for ep in range(epochs):

        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())

        correct_cnt, avg_loss = 0, 0
        
        order = torch.randperm(train_y.size(0))
        ### Change start here ###
        #iters_backbone = order.size(0) // batchSize_backbone + 1
        iters_backbone = int(np.ceil(order.size(0)/batchSize_backbone))
        ### Change end here ###
        for it in range(iters_backbone):
            start_backbone = it * batchSize_backbone
            end_backbone = (it + 1) * batchSize_backbone

            x_backbone = train_x[order[start_backbone:end_backbone]].to(device)
            y_backbone = train_y[order[start_backbone:end_backbone]].to(device)

            with torch.no_grad():
                features = featureModel(x_backbone)   

        
            iters = features.size(0) // (batchSize//2)

            for it in range(iters):

                start = it * (batchSize//2)
                end = (it + 1) * (batchSize//2)

                x_memo, y_memo = classifier.sampleFromMemory(batchSize//2)

                x_mb = torch.cat((features[start:end], x_memo))
                y_mb = torch.cat((y_backbone[start:end], y_memo))

                optimizer.zero_grad()
                logits = classifier(x_mb)


                pred_label = torch.argmax(logits, dim=1)
                correct_cnt += (pred_label == y_mb).sum()

                loss = torch.nn.functional.cross_entropy(logits, y_mb)
                avg_loss += loss.item()

                loss.backward()
                optimizer.step()

                classifier.addToMemory(y_backbone[start:end], features[start:end], y_backbone[start:end])

                acc = correct_cnt.item() / \
                    ((it + 1) * y_mb.size(0))
                avg_loss /= ((it + 1) * y_mb.size(0))
    

        cur_ep += 1

    return acc, avg_loss, stats


def test_multitask(featureModel, classifier, batchSize, device, test_set, preproc=None, multi_heads=[], verbose=True):

    acc_x_task = []
    stats = {'accs': [], 'acc': []}
    preds = []

    classifier.eval()

    for (x, y), t in test_set:       

        if preproc:
            x = preproc(x)

        if multi_heads != [] and len(multi_heads) > t:
            if verbose:
                print("Using head: ", t)
            classifier = multi_heads[t]

        acc = None

        test_x = torch.from_numpy(x).type(torch.FloatTensor)
        test_y = torch.from_numpy(y).type(torch.LongTensor)

        correct_cnt = 0
        total = 0

        with torch.no_grad():

            ### Change start here ###
            #iters = test_y.size(0) // batchSize + 1
            iters = int(np.ceil(test_y.size(0)/batchSize))
            ### Change end here ###
            for it in range(iters):

                start = it * batchSize
                end = (it + 1) * batchSize
                total += end - start

                x_mb = test_x[start:end].to(device)
                y_mb = test_y[start:end].to(device)

                pred_label = torch.argmax(classifier(featureModel(x_mb)), dim=1)
                
                correct_cnt += (pred_label == y_mb).sum()
                preds += list(pred_label.data.cpu().numpy())

            acc = correct_cnt.item() / test_y.shape[0]

        if verbose:
            print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)
    stats['acc'].append(np.mean(acc_x_task))

    return stats, preds

