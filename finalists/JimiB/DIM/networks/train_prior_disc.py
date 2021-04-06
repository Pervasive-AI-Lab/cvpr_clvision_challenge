import numpy as np
import torch
import h5py

####libraries
from networks.gradient_penalty import contrastive_gradient_penalty
from networks.dim_loss import get_positive_expectation,get_negative_expectation

def train_disc(network,X_P,X_Q,optimizerD,metrics,gradient_penalty=1.0):

    '''
    Args:
        network: discriminator
        X_P: prior sample input.
        X_Q: current sample input.
        gradient_penalty: Gradient penalty amount.
        
    Legend:
        E_pos: prior expectation,
        Q_pos: current expectation,
        P_samples: prior samples,
        Q_samples: current samples
    '''

        
    network.zero_grad() 
    nnLin = nn.Sigmoid()
    #Q_samples = nnLin(Q_samples)
    P_samples = network(X_P)
    Q_samples = network(X_Q)

    E_pos = get_positive_expectation(P_samples, measure)
    E_neg = get_negative_expectation(Q_samples, measure)
    difference = E_pos - E_neg

    gp_loss_P = contrastive_gradient_penalty(network, X_P, gradient_penalty)
    gp_loss_Q = contrastive_gradient_penalty(network, X_Q, gradient_penalty)
    gp_loss = 0.5 * (gp_loss_P + gp_loss_Q)

    metrics['E_P[D(x)]'] += P_samples.data.cpu().numpy().mean()
    metrics['E_Q[D(x)]'] += Q_samples.data.cpu().numpy().mean()
    
    loss = -difference + gp_loss
    metrics['Disc_loss'] += loss.data.cpu().numpy()
    
    loss.backward()
    optimizedD.step()
    
    return network,Q_samples
    
class sample_prior():
    def __init__(self,path,device):
         self.path = path
#         self.device = device
#         dset= h5py.File(path+'.hdf5','r') 
#         prior_distrib =  np.asarray(dset['data'])
#         self.prior_distrib = torch.as_tensor(prior_distrib, dtype = torch.float32,device=torch.device('cpu'))
#         del prior_distrib
        
    def sample(self,size):
        #index = np.random.choice(np.arange(self.prior_distrib.size(0)), size, replace=False)        
        sample_prior = np.random.random_sample((size,64))#self.prior_distrib[index,:]
        sample_prior.to(device)
        return sample_prior

def save_prior_dist(encoder,dataloader,device,path=None,save=False):
    '''
    Generate data fro classifier training
    '''
    data =None
    lab = None
    for inputs, labels in dataloader:
        torch.cuda.empty_cache()
        inputs = inputs.to(device)        
        _,_,pred = encoder(inputs)
        pred_l = pred.data.cpu().numpy()
        if data is None:
            data = pred_l
            lab = labels.data.cpu().numpy()
        else:
            data = np.concatenate((data,pred_l),axis=0)
            lab = np.concatenate((lab,labels.data.cpu().numpy()),axis=0)
        del pred
    if save:
        with h5py.File(path,'w') as f:
            dset = f.create_dataset('data',data=data)
            dset = f.create_dataset('labels',data=lab)
    else:
        return data,lab
