import torchgeometry as tgm
import torch

from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import kornia


def warp(x):
    hor_flip = kornia.augmentation.RandomHorizontalFlip()
    ver_flip = kornia.augmentation.RandomVerticalFlip()
    out = hor_flip(x)
    out = ver_flip(out)
    return out