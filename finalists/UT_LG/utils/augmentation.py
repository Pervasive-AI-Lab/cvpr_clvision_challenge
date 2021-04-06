import albumentations as A
from albumentations.pytorch import ToTensor

aug_test_crop_128 = A.Compose(
    [
        A.CenterCrop(100, 100),
        A.Resize(128, 128),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensor()
    ], p=1)

aug_crop_128 = A.Compose(
    [
        A.CenterCrop(100, 100),
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
            ],
            p=0.5
        ),
        A.OneOf(
            [
                # apply one of transforms to 50% of images
                A.RandomContrast(0.4),  # apply random contrast
                A.RandomGamma((20, 180)),  # apply random gamma
                A.RandomBrightness(0.4),  # apply random brightness
            ],
            p=0.5
        ),
        A.OneOf(
            [
                # apply one of transforms to 50% images
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(
                    distort_limit=2,
                    shift_limit=0.5
                ),
            ],
            p=0.3
        ),
        A.Resize(128, 128),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),

        ToTensor()
    ],
    p=1
)
aug_test_crop_224 = A.Compose(
    [
        A.CenterCrop(100, 100),
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensor()
    ], p=1)

aug_crop_224 = A.Compose(
    [
        A.CenterCrop(100, 100),
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
            ],
            p=0.5
        ),
        A.OneOf(
            [
                # apply one of transforms to 50% of images
                A.RandomContrast(0.4),  # apply random contrast
                A.RandomGamma((20, 180)),  # apply random gamma
                A.RandomBrightness(0.4),  # apply random brightness
            ],
            p=0.5
        ),
        A.OneOf(
            [
                # apply one of transforms to 50% images
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(
                    distort_limit=2,
                    shift_limit=0.5
                ),
            ],
            p=0.3
        ),
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),

        ToTensor()
    ],
    p=1
)

aug_test_224 = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensor()
    ], p=1)

aug_224 = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),

        ToTensor()
    ],
    p=1
)

aug_128 = A.Compose(
    [
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1),
            ],
            p=0.5
        ),
        A.OneOf(
            [
                # apply one of transforms to 50% of images
                A.RandomContrast(0.4),  # apply random contrast
                A.RandomGamma((20, 180)),  # apply random gamma
                A.RandomBrightness(0.4),  # apply random brightness
            ],
            p=0.5
        ),
        A.OneOf(
            [
                # apply one of transforms to 50% images
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(
                    distort_limit=2,
                    shift_limit=0.5
                ),
            ],
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),

        ToTensor()
    ],
    p=1
)


