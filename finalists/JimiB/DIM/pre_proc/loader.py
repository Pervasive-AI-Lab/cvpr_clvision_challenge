import numpy
import six
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import numpy as np
import gc


def data_split(n_dataset,seed,debug=False):
    '''This module divide data_batchin:
        90%(training, cv) and 10%(coreset)
        Args:
        -data
        -labels
        -seed: for splitting
        return:
        training, cv and coreset
    '''
    
    train_data_size = 10 if debug else int(n_dataset * 0.9)
    tr = 10 if debug else int(train_data_size * 0.8)
    cv = 10 if debug else int(train_data_size * 0.2)   
    core = 10 if debug else int(n_dataset - train_data_size)
 
    perm = np.random.RandomState(seed).permutation(n_dataset)
    
    return perm[core:core+tr],perm[core+tr:],perm[:core]
                                                  
def data_split_Tr_CV(train_data_size,seed,debug=False):
    '''This module divide data_batchin:
        80%(training, cv) and 20%(coreset)
        Args:
        -data
        -labels
        -seed: for splitting
        return:
        training, cv 
    '''
    tr = 10 if debug else int(train_data_size * 0.8)
    cv = 10 if debug else int(train_data_size * 0.2)   
   
 
    perm = np.random.RandomState(seed).permutation(train_data_size)
    
    return perm[:tr],perm[tr:]

def data_org(data,lab):
    L = lab.max()
    gen_list = []
    for i in range(int(L)+1):
        pt = np.where(lab==i)
        gen_list.append(data[pt[0],:,:,:])
    return gen_list

class LoadDataset(Dataset):
    """Data loader with parallel Batch for MI maxim across batches. If loader is empty is a standard loader"""

    def __init__(self,images,labels=None,transform=None,indices=None,ref=None):
        """
        Args:
            images -> np.arr (samples,H,W,C)
            labels -> np.arr (sample,)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not(indices is None):
            self.index = indices
        else:
            self.index = np.arange(images.shape[0])
            
        self.im = images[self.index].astype(np.uint8)
        if not(labels is None):
            self.lb = labels[self.index]
        else: 
            self.lb = None
        self.transform = transform
        self.ref = ref
        if not(self.ref is None):
            self.standardize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
 
    def __len__(self):
        return len(self.im)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.im[idx]
        #print(image.shape)
        if self.transform:
            image = self.transform(image)
        if self.ref is None:    
            if not(self.lb is None):
                label = self.lb[idx]
                return image,label
            else:
                return image
        else:
            if not(self.lb is None):
                label = self.lb[idx]
                #pt = np.where(self.ref[1]==label)
                #print(len(self.ref),int(label))
                vec_choice = self.ref[int(label)]
                idx_ref = np.random.randint(vec_choice.shape[0], size=1)#np.random.choice(np.arange(),1) 
                
                reference = vec_choice[idx_ref,:,:,:] 
                out = torch.empty((3,128,128,2),dtype=torch.float32)
                out[:,:,:,0]=image
                out[:,:,:,1]=self.standardize(reference[0])
                #print(out.size(),image.size())
                return out,label
            else:
                return image

            
class LoadFeat(Dataset):
    """Data loader with parallel Batch for MI maxim across batches. If loader is empty is a standard loader"""

    def __init__(self,feat,labels=None,transform=None,indices=None,ref=None):
        """
        Args:
            images -> np.arr (samples,H,W,C)
            labels -> np.arr (sample,)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not(indices is None):
            self.index = indices
        else:
            self.index = np.arange(feat.shape[0])
            
        self.im = feat[self.index]
        if not(labels is None):
            self.lb = labels[self.index]
        else: 
            self.lb = None

 
    def __len__(self):
        return len(self.im)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torch.from_numpy(self.im[idx])
 
        if not(self.lb is None):
            label = self.lb[idx]
            return image,label
        else:
            return image

