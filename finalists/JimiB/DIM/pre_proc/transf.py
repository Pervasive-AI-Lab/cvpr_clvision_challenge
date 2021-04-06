import numpy as np
import six
import albumentations as A
import numpy as np
import cv2
from skimage.transform import AffineTransform, warp
import numpy as np
import pandas as pd
import gc

# already included into loader
def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.
        Args:
            img_batch (tensor): batch of images.
            scale (bool): if we want to scale the images between 0 an 1.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            norm (bool): if we want to normalize them.
        Returns:
            tensor: pre-processed batch.
    """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def affine_image(img):
    """
    Args:
        img: (h, w) or (1, h, w)
    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
#     if img.ndim == 3:
#         img = img[0]

    # --- scale ---
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 7
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image =np.empty_like(img)
    for i in range(3):
        transformed_image[:,:,i] = warp(img[:,:,i], tform)
    
    return transformed_image



def resize(image, size=(128, 128)):
    return cv2.resize(image, size)



def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image):
    return aug(image=image)['image']


def affine_image(img):
    """
    Args:
        img: (h, w) or (1, h, w)
    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
#     if img.ndim == 3:
#         img = img[0]

    # --- scale ---
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 7
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image =np.empty_like(img)
    for i in range(3):
        transformed_image[:,:,i] = warp(img[:,:,i], tform)
    
    return transformed_image



def resize(image, size=(128, 128)):
    return cv2.resize(image, size)



def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image):
    return aug(image=image)['image']


class Transform:
    def __init__(self, affine=0., train=True,cutout_ratio=0.,ssr_ratio=0.,flip=0.):
        
        self.affine = affine
        self.train = train
        self.cutout_ratio = cutout_ratio
        self.ssr_ratio = ssr_ratio
        self.flip=flip
        self.crop=.5
        self.hue =.8
        #print('wwwwwwwwwww')
    def __call__(self, example):
        if self.train:
            x, y = example
            
            #print(x.shape,y)
        else:
            x = example
            
        x = np.transpose(x, (1,2,0))
        
        #--- Augmentation ---
        if _evaluate_ratio(self.affine):
            x = affine_image(x)
        
        #albumentations...
        
        else:
            if _evaluate_ratio(self.cutout_ratio):
                #print('w')
                x = apply_aug(A.CoarseDropout(max_holes=16, max_height=8, max_width=8, p=1.0), x)

            if _evaluate_ratio(self.ssr_ratio):
                #print('w2')
                x = apply_aug(A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=180,
                    p=1.0), x)
                
            if _evaluate_ratio(self.flip):
                #print('w3')
                x = np.flip(x,axis=0).copy()
                
            if _evaluate_ratio(self.flip):
                #print('w4')
                x = np.flip(x,axis=1).copy()
            if _evaluate_ratio(self.crop):
                x = apply_aug(A.augmentations.transforms.RandomSizedCrop((90,128), 128, 128, w2h_ratio=1.0, interpolation=3, always_apply=False, p=1.0),x)
            if _evaluate_ratio(self.hue):
                x = apply_aug(A.augmentations.transforms.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1),x)
                
        x = np.transpose(x, (2,0,1)) 
        if self.train:
            y = y.astype(np.int64)
            return x, y
        else:
            return x

