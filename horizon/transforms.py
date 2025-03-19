"""
SEA.AI
authors: Kevin Serrano,

Transforms used for training the horizon detection model.
Transforms are implemented using the albumentations library.
Useful links:
- https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/
- https://albumentations.ai/docs/getting_started/transforms_and_targets/
"""

from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch.transforms import ToTensorV2

import utils.albumentations16 as A16
import utils.albumextensions as Ax
from utils.general import LOGGER, colorstr
from utils.horizon import points_to_pitch_theta  # points_to_hough


def log_transforms(transforms: List[BasicTransform], prefix: str) -> None:
    """Log transforms."""
    LOGGER.info(
        f"{prefix} {', '.join(f'{x}'.replace('always_apply=False, ', '') for x in transforms if x.p)}"
    )


def geometric_augment(imgsz: Tuple[int,int]) -> List[BasicTransform]:
    """
    Geometric transforms for augmentation.
    """
    return [
        A.RandomCropFromBorders(p=0.1),
        Ax.ResizeIfNeeded(max_size=max(imgsz)),
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(
            min_height=imgsz[0], min_width=imgsz[1], border_mode=cv2.BORDER_CONSTANT, value=0
        ),
        A.Affine(
            p=0.75,
            scale=(0.8, 1.1),
            rotate=(-20, 20),
            translate_percent=(-0.1, 0.1),
            balanced_scale=True,
            keep_ratio=True,
            fit_output=False,
            mode=cv2.BORDER_CONSTANT,
            cval=0,
        ),
    ]


def horizon_augment_rgb(
    imgsz: Tuple[int,int],
    im_compression_prob: float,
    prefix=colorstr("albumentations rgb:"),
) -> A.Compose:
    """
    Augmentations for RGB images.

    Args:
        imgsz (Tuple[int,int]): image size (height, width)
        im_compression_prob (float): Image compression propability

    """

    T = [
        # weather transforms
        A.OneOf(
            [
                A.RandomRain(p=1),
                A.RandomSunFlare(p=1),
                A.RandomFog(p=1, fog_coef_lower=0.1, fog_coef_upper=0.3),
            ],
            p=0.05,
        ),
        # image transforms
        A.RandomCropFromBorders(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.CLAHE(p=0.05),
        A.Blur(p=0.05),
        A.ImageCompression(quality_lower=75, p=im_compression_prob),
        # geometric transforms
        *geometric_augment(imgsz),
        # torch-related transforms
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(p=1.0),
    ]
    log_transforms(T, prefix)
    return A.Compose(
        T, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )


def horizon_base_rgb(imgsz: Tuple[int,int]) -> A.Compose:
    """
    No augmentation, just resize and normalize.

    Args:
        imgsz  (Tuple[int,int]): image size (height, width)
    """
    return A.Compose(
        [
            # geometric transforms
            Ax.ResizeIfNeeded(max_size=max(imgsz)),
            A.PadIfNeeded(
                min_height=imgsz[0],
                min_width=imgsz[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),  # letterbox
            # torch-related transforms
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def horizon_augment_ir16bit(
    imgsz: Tuple[int,int], im_compression_prob: float, prefix=colorstr("albumentations ir16bit:")
) -> A.Compose:
    """
    Augmentations for 16-bit IR images.

    Args:
        imgsz (Tuple[int,int]): image size (height, width)
        im_compression_prob (float): Image compression propability
    """

    T = [
        # image transforms (16-bit compatible)
        A16.Clip(p=1.0, lower_limit=(0.2, 0.25), upper_limit=(0.4, 0.45)),
        A16.CLAHE(p=0.5, clip_limit=(3, 5), tile_grid_size=(0, 0)),
        A16.NormalizeMinMax(p=1.0),
        A.UnsharpMask(p=0.5, threshold=5),
        A.ToRGB(p=1.0),
        A.ImageCompression(quality_lower=50, p=im_compression_prob),
        # geometric transforms
        *geometric_augment(imgsz),
        # torch-related transforms
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(p=1.0),
    ]
    log_transforms(T, prefix)
    return A.Compose(
        T, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )


def horizon_base_ir16bit(
    imgsz: Tuple[int,int],
    lower_limit: float = 15000 / 65535,
    upper_limit: float = 28000 / 65535,
    clahe: bool = True,
    unsharp_mask: bool = True,
) -> A.Compose:
    """
    No augmentation, just resize and normalize.

    Args:
        imgsz (Tuple[int,int]): image size (height, width)
        lower_limit (float): lower limit for clipping
        upper_limit (float): upper limit for clipping
        clahe (bool): apply CLAHE
        unsharp_mask (bool): apply unsharp mask
    """
    return A.Compose(
        [
            # image transforms (16-bit compatible)
            A16.Clip(
                p=1.0, lower_limit=(lower_limit,) * 2, upper_limit=(upper_limit,) * 2
            ),
            A16.CLAHE(
                p=1.0 if clahe else 0.0, clip_limit=(4, 4), tile_grid_size=(0, 0)
            ),
            A16.NormalizeMinMax(p=1.0),
            A.UnsharpMask(p=1.0 if unsharp_mask else 0.0, threshold=5),
            A.ToRGB(p=1.0),
            # geometric transforms
            Ax.ResizeIfNeeded(max_size=max(imgsz)),
            A.PadIfNeeded(
                min_height=imgsz[0],
                min_width=imgsz[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),  # letterbox
            # torch-related transforms
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def points_to_normalised_pitch_theta(
    points: np.ndarray,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    """Convert points to normalised pitch and theta."""
    # normalize points
    points = np.array(points)
    points[:, 0] /= image_w
    points[:, 1] /= image_h

    # convert to pitch, theta
    x1, y1, x2, y2 = points.flatten()
    pitch, theta = points_to_pitch_theta(x1, y1, x2, y2)
    return np.array([pitch, theta])
