import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from core50.dataset import CORE50
from utils.augmentation import *

def setup_classifier(cls_name, n_classes, args):
    if cls_name == 'resnet18':
        classifier = models.resnet18(pretrained=True)
        classifier.fc = torch.nn.Linear(512, n_classes)
    elif cls_name == 'resnext101':
        classifier = models.densenet161(pretrained=True)
        for idx, (param_name, param) in enumerate(classifier.named_parameters()):
            if idx < 73:
                # print(idx, param_name)
                param.requires_grad = False
        classifier.fc = torch.nn.Linear(512, n_classes)
    elif cls_name == 'densenet161':
        classifier = models.densenet161(pretrained=True)
        classifier.fc = torch.nn.Linear(512, n_classes)
    elif cls_name == 'dense_freeze_till3':
        classifier = models.densenet161(pretrained=True)
        for idx, (param_name, param) in enumerate(classifier.named_parameters()):
            if idx < 117:
                param.requires_grad = False
        classifier.fc = torch.nn.Linear(512, n_classes)
    elif cls_name == 'dense_freeze_till4':
        classifier = models.densenet161(pretrained=True)
        for idx, (param_name, param) in enumerate(classifier.named_parameters()):
            if idx < 336:
                param.requires_grad = False
        classifier.fc = torch.nn.Linear(512, n_classes)
    elif cls_name == 'densenet161_complex':
        classifier = models.densenet161(pretrained=True)
        fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, n_classes)
        )
        classifier.fc = fc
    elif cls_name == 'efficientnetb5':
        classifier = EfficientNet.from_pretrained('efficientnet-b5', num_classes=n_classes)

    elif cls_name == 'efficientnetb4':
        classifier = EfficientNet.from_pretrained('efficientnet-b4', num_classes=n_classes)
    else:
        raise Exception('wrong cls')
    return classifier

def setup_opt(optimizer, model, args):
    if optimizer == 'SGD':
        nesterov = True if args.nesterov == 'True' else False
        optim = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                nesterov=nesterov,
                                weight_decay=args.weight_decay)
    elif optimizer == 'AdamW':
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad)
    elif optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    return optim

def setup_dataset(scenario, preload=True):
    dataset = CORE50(root='core50/data/', scenario=scenario,
                          preload=preload)
    full_valdidset = dataset.get_full_valid_set()
    num_tasks = dataset.nbatch[dataset.scenario]
    return dataset, full_valdidset, num_tasks

def setup_aug(aug_type):
    augmentation = {
        'center_224': (aug_crop_224, aug_test_crop_224),
        'center_128': (aug_crop_128, aug_test_crop_128),
        '224': (aug_224, aug_test_224),
        'trainNo128_testcrop224': (aug_128, aug_test_crop_224)

    }
    return augmentation[aug_type]

def cal_memsize(scenario, replay_size):
    if scenario == 'nic':
        return 3000 + 390 * replay_size
    elif scenario == 'ni':
        return replay_size * 8