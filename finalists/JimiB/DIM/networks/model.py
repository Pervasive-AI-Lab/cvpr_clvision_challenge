import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


#__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        feature_C = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # insert secon output for the creation of local and global vector for DIM
        x = self.classifier(x)
        return x,feature_C


def alexnet(pretrained=False, progress=True, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],progress=progress)
        model.load_state_dict(state_dict)
    return model


def init_weights(m):
    if type(m)== nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

class classifier(nn.Module):
    def __init__(self,n_input,n_class):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(n_input, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, n_class)
            )
        self.linear.apply(init_weights)
        
    def forward(self,x):
        out = self.linear(x)
        return out
def lin_block(inp,out):
    return nn.Sequential(nn.Linear(inp,out, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(out))

class _classifier(nn.Module):
    def __init__(self,n_input,n_class,n_neurons):
        super().__init__()
        self.n_input = n_input
        self.n_class = n_class
        self.n_neurons = n_neurons 
        self.linear = self._make_layer()
        self.linear.apply(init_weights)
    def _make_layer(self):
        layer = []
        layer.append(lin_block(self.n_input,self.n_neurons[0]))
        
        for j in range(len(self.n_neurons)-1):
            layer.append(lin_block(self.n_neurons[j],self.n_neurons[j+1]))
            
        layer.append(nn.Linear(self.n_neurons[-1],self.n_class,bias=True))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        return self.linear(x)