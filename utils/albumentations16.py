import random
from typing import Sequence, Union, Tuple, cast

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import (
    ImageOnlyTransform,
    BaseTransformInitSchema,
)

from albumentations.core.pydantic import (
    NonNegativeFloatRangeType,
    OnePlusFloatRangeType,
    ZeroOneRangeType,
)


def convert_16bit_to_8bit(im, augment=True):
    if len(im.shape) == 3 and im.shape[2] == 3:
        # for some reason, some 16bit images have 3 channels
        im = im[:, :, 0]
    transform = get_16_to_8_transform(augment)
    return transform(image=im)["image"]


def get_16_to_8_transform(augment):
    if augment:
        # meant to be used for training, randomness is important
        return A.Compose(
            [
                # 15000/65535 = 0.228, 28000/65535 = 0.427
                Clip(p=1.0, lower_limit=(0.2, 0.25), upper_limit=(0.4, 0.45)),
                CLAHE(p=0.5, clip_limit=(3, 5), tile_grid_size=(0, 0)),
                NormalizeMinMax(p=1.0),
                A.UnsharpMask(p=0.5, threshold=5),
                A.ToRGB(p=1.0),
            ]
        )
    else:
        llimit = 15000 / 65535
        ulimit = 28000 / 65535
        # meant to be used for validation, deterministic
        return A.Compose(
            [
                Clip(p=1.0, lower_limit=(llimit, llimit), upper_limit=(ulimit, ulimit)),
                CLAHE(p=1.0, clip_limit=(4, 4), tile_grid_size=(0, 0)),
                NormalizeMinMax(p=1.0),
                A.UnsharpMask(p=0.0, threshold=5),
                A.ToRGB(p=1.0),
            ]
        )


class CLAHE(ImageOnlyTransform):
    """
    Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
            If (0, 0), optimal value will be calculated based on image size.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16
    """

    class InitSchema(BaseTransformInitSchema):
        clip_limit: OnePlusFloatRangeType = (1.0, 4.0)
        tile_grid_size: NonNegativeFloatRangeType = (8, 8)

    def __init__(
        self,
        clip_limit: Union[float, Sequence[float]] = 4.0,
        tile_grid_size: Union[float, Sequence[float]] = (8, 8),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.clip_limit = cast(Tuple[float, float], clip_limit)
        self.tile_grid_size = cast(Tuple[int, int], tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        if self.tile_grid_size == (0, 0):
            # compute tile_grid_size based on image size
            tile_grid_size = (round(max(img.shape) / 160),) * 2
        else:
            tile_grid_size = self.tile_grid_size
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        """Returns names of arguments that are used in __init__ method of the transform."""
        return ("clip_limit", "tile_grid_size")


class NormalizeMinMax(ImageOnlyTransform):
    """
    Normalize image to 0-255 range using min-max scaling.

    Targets:
        image

    Image types:
        uint8, uint16
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(p=p, always_apply=always_apply)

    def apply(self, img, **params):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    def get_transform_init_args_names(self):
        """Returns names of arguments that are used in __init__ method of the transform."""
        return ()


class Clip(ImageOnlyTransform):
    """
    Clip image to a certain range.

    Args:
        lower_limit (float or (float, float)): lower limit value for clipping.
            If lower_limit is a single float value, the range will be (0, lower_limit). Default: (0.1, 0.2).
        upper_limit (float or (float, float)): upper limit value for clipping.
            If upper_limit is a single float value, the range will be (upper_limit, 1). Default: (0.8, 0.9).

    Targets:
        image

    Image types:
        uint8, uint16
    """

    class InitSchema(BaseTransformInitSchema):
        lower_limit: ZeroOneRangeType = (0.1, 0.2)
        upper_limit: ZeroOneRangeType = (0.8, 0.9)

    def __init__(
        self,
        lower_limit: Union[float, Sequence[float]] = (0.1, 0.2),
        upper_limit: Union[float, Sequence[float]] = (0.8, 0.9),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.lower_limit = cast(Tuple[float, float], lower_limit)
        self.upper_limit = cast(Tuple[float, float], upper_limit)

    def apply(self, img, lower_limit=0.1, upper_limit=0.9, **params):
        max_val = np.iinfo(img.dtype).max
        a_min = int(lower_limit * max_val)
        a_max = int(upper_limit * max_val)
        return np.clip(img, a_min, a_max)

    def get_params(self):
        return {
            "lower_limit": random.uniform(self.lower_limit[0], self.lower_limit[1]),
            "upper_limit": random.uniform(self.upper_limit[0], self.upper_limit[1]),
        }

    def get_transform_init_args_names(self):
        """Returns names of arguments that are used in __init__ method of the transform."""
        return ("lower_limit", "upper_limit")
