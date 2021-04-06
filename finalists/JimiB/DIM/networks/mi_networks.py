import torch
import numpy as np 
import torch.nn as nn
#import sys
#sys.path.append('/media/DATA/jbonato/cvpr_clvision_challenge/DIM/')
from networks.misc import Permute

class Glob_MI(nn.Module):
    """
    Global MI  network architecture
    """
    def __init__(self,n_input,n_units):
        super().__init__()
        
        self.block = nn.Sequential(
                nn.Linear(n_input, n_units, bias=False),
                nn.BatchNorm1d(n_units),
                nn.ReLU(),
                nn.Linear(n_units, n_units),
                nn.BatchNorm1d(n_units),
                nn.ReLU(),
                nn.Linear(n_units,1)
            )
        self.block(kai_weights_init)
    def forward(self,x):
        return self.block(x)




class Local_MI_Gl_Feat(nn.Module):
    """
    mutual Information nn for global feature: linearly embedding of 
    global feature in higher-dim space
    """
    def __init__(self,n_input,n_units,bn=False):
        super().__init__()
    
        self.bn = bn

        assert n_units >= n_input, 'Check MI_Enc_Global'

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
                nn.Linear(n_input, n_units, bias=False),
                nn.BatchNorm1d(n_units),
                nn.ReLU(),
                nn.Linear(n_units, n_units)
            )
        
        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.bool)
        for i in range(n_input):
            eye_mask[i, i] = True

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)


        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        """
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: network output.
        """


        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)

        return h

class Local_MI_1x1ConvNet(nn.Module):
    """Simple custom 1x1 convnet.
    """
    def __init__(self, n_input, n_units,):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.bool)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = True
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """
            Args:
                x: Input tensor.
            Returns:
                torch.Tensor: network output.
        """

        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h


class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim
        return

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x

def kai_weights_init(m):
    if type(m)== nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

def weights_init(m):
    ''' Weight initializer of DCGAN probably we need to play with it
        look at AAE weight discriminator makhzani 2015
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    
class Prior_discriminator(nn.Module):
    '''Classificator of fake/real global features vector. real features sampled from the prior distribution
        fake features are obtained from the surrent encoder architecture
        TODO: initialization of parameters + remeber sigmoid into the output 
        
    '''
    def __init__(self,n_input):
        self.linear_layer = nn.Sequential(
                nn.Linear(n_input, 1000, bias=False),
                nn.BatchNorm1d(1000),
                nn.ReLU(),
                nn.Linear(1000, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(200,1)
        )
        self.linear_layer.apply(weights_init)
    def forward(self,x):
        x = self.linear_layer(x)
        return x