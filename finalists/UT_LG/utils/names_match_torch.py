import torch
from models.EWC import EWC
from models.Tiny_torch import Tiny
from models.task_mem import task_mem
from models.indep_models import indep_models
from models.indep_models_smallFC import indep_models_smallFC
from utils.setup_elements import aug_crop_128, aug_test_crop_128, aug_crop_224, aug_test_crop_224,aug_224, aug_test_224

optimizers = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam
}

methods = {
    'task_mem': task_mem,
    "EWC": EWC,
    'Tiny': Tiny,
    'multi_models': indep_models,
    'multi_models_smallFC': indep_models_smallFC
}

