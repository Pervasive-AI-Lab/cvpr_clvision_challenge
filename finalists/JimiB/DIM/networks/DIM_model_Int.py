import torch.nn as nn
import torch
import numpy
import sys
import torchvision.models as models
from networks.model import *
from networks.mi_networks import *


class DIM_model(nn.Module):
    def __init__(self,batch_s = 32,num_classes =64,feature=False,out_class = 50,model_ty=101):
        super().__init__()
        ###########pytorch pretrained mod
        if model_ty==101:
            print('resnext101_32x8d')
            model_ft = models.resnext101_32x8d(pretrained=True)
        else:
            print('resnext50_32x4d')
            model_ft = models.resnext50_32x4d(pretrained=True)
            
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        self.encoder = nn.Sequential(*list(model_ft.children())[:8])
        self.head =  model_ft.avgpool
        self.head2 = model_ft.fc
        
        #test input output size and channel to use
        fake_in = torch.ones([2,3,128,128])
        out1 = self.encoder(fake_in)
        #print(out1.size())        
        out2 = self.head(out1)
        out2 = torch.flatten(out2, 1)
        inputCL = out2.size(1)
        out2 = self.head2(out2)
        #print(out2.size())
        
        n_inputL = out1.size(1)
        n_inputG = out2.size(1)
        n_units = 2048
        #model = model.to(torch.device('cuda:0'))
        
        #classifier
        self.cl = nn.Linear(inputCL, 50, bias=False)

        # insert in the model mutual information networks
        self.global_MI = Local_MI_Gl_Feat(n_input = n_inputG,n_units = n_units)
        
        self.local_MI = Local_MI_1x1ConvNet(n_inputL,n_units)
        
        self.features_g = n_units
        self.features_l = n_units
        
        self.feature = feature

    def forward(self,x):
        self.batch = x.size(0)
        C_phi = self.encoder(x)
        buff = self.head(C_phi)
        buff = torch.flatten(buff, 1)
        #print(buff.size())
        class_r = self.cl(buff)
        
        E_phi = self.head2(buff)

        if self.feature:
            A_phi = E_phi
        
        E_phi = self.global_MI(E_phi)
        C_phi = self.local_MI(C_phi)
        E_phi = E_phi.view(self.batch,self.features_g,1)
        C_phi = C_phi.view(self.batch,self.features_l,-1)
        if self.feature:
            return E_phi,C_phi,A_phi,class_r
        else:
            return E_phi,C_phi
