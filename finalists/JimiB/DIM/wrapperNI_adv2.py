import os
import time
import copy
import six
import sys

import numpy as np

#import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
#import torch.nn as nn
#import torch

import matplotlib.pyplot as plt

### tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch

from networks.DIM_model_Int import *

from networks.train_nets import *
from pre_proc.loader import data_split,data_split_Tr_CV,LoadDataset,data_org
from pre_proc.transf import Transform 
from networks.WSched import GradualWarmupScheduler

class NI_wrap():
    def __init__(self,dataset,val_data,device,path,load=True,replay=True):
        '''
        Args:
        TO DO: complete Args
        eventully add a flag for saving CLDIM weights
        '''
        self.load = load
        self.replay = replay
        self.stats = {"ram": [], "disk": []}
        self.dataset = dataset
        self.val_data = val_data
        ################# transformation for training
        self.tr = transforms.Compose([
    
            transforms.ToPILImage(),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.6),
                transforms.ColorJitter(contrast=0.4),
                transforms.ColorJitter(saturation=0.4),
                ]),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation(180, resample=3, expand=False, center=None, fill=0),
                transforms.RandomAffine(30, translate=(.1,.1), scale=(0.95,1.05), shear=5, resample=False, fillcolor=0)
            ]),

            transforms.ToTensor(),
            #Cutout(4,20,p=0.6),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
        ################## transformation for validation and test set
        self.trT = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
        #Transform(affine=0.5, train=True,cutout_ratio=0.6,ssr_ratio=0.6,flip = 0.6)
        self.device = device
        self.path = path 
        
    def train(self):
        acc_time = []
        data_test = self.val_data[0][0][0]
        labels_test = self.val_data[0][0][1]
        for i, train_batch in enumerate(self.dataset):
            
            writerDIM = SummaryWriter('runs/experiment_DIM'+str(i))
            data,labels, t = train_batch

            index_tr,index_cv,coreset = data_split(data.shape[0],777)
            
            # adding eventual replay patterns to the current batch
            if i == 0:
                ext_mem = [data[coreset], labels[coreset]]
#                dataC = np.concatenate((data[index_tr], data[index_cv]),axis=0)
#                labC = np.concatenate((labels[index_tr],labels[index_cv]),axis=0)
 
                dataC = data[index_tr]#np.concatenate((data[index_tr]),axis=0)
                labC = labels[index_tr]#np.concatenate((labels[index_tr],axis=0)
            else:
                dataP = ext_mem[0]
                labP = ext_mem[1]
 
                ext_mem = [
                    np.concatenate((data[coreset], ext_mem[0])),
                    np.concatenate((labels[coreset], ext_mem[1]))]
                if self.replay:
                    #dataC = np.concatenate((data[index_tr], data[index_cv],dataP),axis=0)
                    #labC = np.concatenate((labels[index_tr],labels[index_cv],labP),axis=0)
                    dataC = np.concatenate((data[index_tr],dataP),axis=0)
                    labC = np.concatenate((labels[index_tr],labP),axis=0)
                else:
                    dataC = np.concatenate((data[index_tr], data[index_cv]),axis=0)
                    labC = np.concatenate((labels[index_tr],labels[index_cv]),axis=0)
 
            del data,labels,train_batch 
         
        
            print("----------- batch {0} -------------".format(i))
            print("Task Label: ", t)
            trC,cvC = data_split_Tr_CV(dataC.shape[0],777)
            if i==0: 
                train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC)
                val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC)
            else:
                #mod for previous batches MI
                dataR = data_org(dataP,labP)
                train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC,ref=dataR)
                val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC,ref=dataR)

            print('Training set: {0} \nValidation Set {1}'.format(train_set.__len__(),val_set.__len__()))
            batch_size=32
            batch_sizeV = 32
            if trC.shape[0]%batch_size==1:
                batch_size-=1

            if cvC.shape[0]%batch_sizeV==1:
                batch_sizeV-=1

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_set, batch_size=batch_sizeV, shuffle=False)
            dataloaders = {'train':train_loader,'val':valid_loader}

            if i ==0:        
                prior = False
                ep=30
                dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
                dim_model.to(self.device)
                
                writer = SummaryWriter('runs/experiment_C'+str(i))
                lr_new = 0.00001
                epWU = 5
            else:
                epWU = 2
                prior = True
                ep=8
                if i>2:
                    ep=5
                lr_new =0.000005
                

            optimizer = torch.optim.Adam(dim_model.parameters(),lr=lr_new)
            scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1) #there is also MultiStepLR
            
            #scheduler = #GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=epWU, after_scheduler=sched)
            
            tr_dict_enc = {'ep':ep,'writer':writerDIM,'best_loss':1e10,'t_board':True,
                           'gamma':.1,'beta':.5,'Prior_Flag':prior}    
           

            if i==0 and self.load:
                print('Load DIM model weights first step')
                dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T0cset128_cnn.pt'))
            else:
                ############################## Train Encoder########################################
                dim_model,self.stats = trainEnc_MIadv(self.stats,dim_model, optimizer, scheduler,dataloaders,self.device,tr_dict_enc)
                ####################################################################################
                #torch.save(dim_model.state_dict(), self.path + 'weights/weightsDIM_T'+str(i)+'cset128_cnn.pt')

            #### Cross_val Testing

            test_set = LoadDataset(data_test,labels_test,transform=self.trT)
            batch_size = 32
            if data_test.shape[0]%batch_size==1:
                batch_size-=1
            
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            score= []
            dim_model.eval()
            
            for inputs, labels in test_loader:
                torch.cuda.empty_cache()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device) 
                _,_,_,pred=dim_model(inputs)
                
                pred_l = pred.data.cpu().numpy()
                score.append(np.sum(np.argmax(pred_l,axis=1)==labels.data.cpu().numpy())/pred_l.shape[0])
            print('TEST PERFORMANCES:', np.asarray(score).mean())
            acc_time.append(np.asarray(score).mean())
            del test_set,test_loader
        self.dim_model = dim_model

        acc_time = np.asarray(acc_time)
        return self.stats,acc_time
        
    def test(self,test_data,standalone=False):
        
        if standalone:
            self.dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
            self.dim_model.to(self.device)
            
            self.dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T7cset128_cnn.pt'))

        
        test_set = LoadDataset(test_data[0][0][0],transform=self.trT)
        batch_size=32
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        score = None
        self.dim_model.eval()
        for inputs in test_loader:
            torch.cuda.empty_cache()
            inputs = inputs.to(self.device)
            _,_,_,pred = self.dim_model(inputs)
            pred_l = pred.data.cpu().numpy()
            
            if score is None:
                score = np.argmax(pred_l,axis=1)
            else:
                score = np.concatenate((score,np.argmax(pred_l,axis=1)),axis=0)      
        return score


