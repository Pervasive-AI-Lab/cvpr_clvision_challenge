import torch
import numpy as np


def one_hot_encoder(label,out_dim=None,out=None):
    
    N = label.shape[0]
    
    if out is None:
        Y = torch.zeros((N,out_dim),dtype=torch.float32)
    else:
        Y = out * 0.
    label = label.type(torch.LongTensor)  
    idx = torch.arange(N)
    Y[idx,label] = 1
    return Y





def dr_loss(y,H,lmb,out_dim,use_cuda=True,one_hot=False):
    if not one_hot:
        yids = one_hot_encoder(y,out_dim=out_dim)
        yids = torch.mm(yids, torch.t(yids))
    else:
        yids = torch.mm(y, torch.t(y))
    
    N = y.shape[0]
    mask = torch.eye(N) 
    loss = 0.
    for h in H:
        if len(h.size()) > 2:
            h = torch.reshape(h,[N,-1])

    sim = torch.mm(h,torch.t(h))
    if use_cuda:
        yids = yids.cuda()
        sim = sim.cuda()
        mask = mask.cuda()
    loss += 0.5*torch.mean(sim*(1.-yids)-sim*(mask-yids))

    return loss*lmb


def cl_dr_loss_softmax(y_hat,y,H,lmb,out_dim,use_cuda=True):

    cl = torch.nn.CrossEntropyLoss()(y_hat,y)
    drl = dr_loss(y,H,lmb,out_dim,use_cuda)

    return cl+drl

def cl_dr_loss_sigmoid(y_hat,y,H,lmb,out_dim,use_cuda=True):
    
    cl = torch.nn.BCELoss()(y_hat,y)
    drl = dr_loss(y,H,lmb,out_dim,use_cuda,one_hot=True)

    return cl+drl
    