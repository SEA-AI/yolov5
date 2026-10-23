"""Custom augmentations following the Albumentations API."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.geometric import functional as F
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from pydantic import ValidationInfo, field_validator


class SafeRandomCrop(A.RandomCrop):
    """Crop a random part of the input.

    Unlike the base RandomCrop which raises an error, this version automatically adjusts oversized crop dimensions
    to match the input image size while maintaining random positioning.

    Args:
        height: height of the crop.
        width: width of the crop.
        p: probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        # do not throw an error if the crop size exceeds the image size
        # instead, dynamically adjust the crop size to the image size
        height = min(self.height, image_shape[0])
        width = min(self.width, image_shape[1])

        h_start = self.py_random.random()
        w_start = self.py_random.random()
        crop_coords = fcrops.get_crop_coords(image_shape, (height, width), h_start, w_start)
        return {"crop_coords": crop_coords, "pad_params": None}


class ResizeIfNeeded(DualTransform):
    """
    Resize an image if its dimensions exceed specified maximum values.

    Args:
        max_size (int, list of int, optional): maximum size of the longest side of the image after transformation.
            When using a list, max size will be randomly selected from the values in the list.
        max_size_hw (tuple of int, optional): maximum height and width of the image after transformation.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        max_size: Optional[Union[int, List[int]]]
        max_size_hw: Optional[Tuple[int, int]]
        interpolation: int

        @field_validator("max_size", "max_size_hw")
        @classmethod
        def check_positive(cls, v: Any, info: ValidationInfo) -> Any:
            if v is not None and any(x < 1 for x in (v if isinstance(v, (list, tuple)) else [v])):
                raise ValueError(f"{info.field_name} must be bigger or equal to 1.")
            return v

    def __init__(
        self,
        max_size: Optional[Union[int, List[int]]] = None,
        max_size_hw: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1,
    ):
        super().__init__(p=p)
        if not any([max_size, max_size_hw]):
            raise ValueError("At least one of max_size or max_size_hw must be set")
        self.max_size = max_size
        self.max_size_hw = max_size_hw
        self.interpolation = interpolation

    @staticmethod
    def _compute_scale(
        height: int, width: int, max_size: Optional[int], max_size_hw: Optional[Tuple[int, int]]
    ) -> float:
        """Compute the downscale factor (<= 1) that satisfies the maximum constraints."""
        scale = 1.0
        if max_size_hw is not None:
            scale = min(scale, max_size_hw[0] / height, max_size_hw[1] / width)
        if max_size is not None:
            scale = min(scale, max_size / max(height, width))
        return scale

    def apply(self, img: np.ndarray, max_size: Optional[int] = None, **params) -> np.ndarray:
        height, width = img.shape[:2]
        scale = self._compute_scale(height, width, max_size, self.max_size_hw)
        if scale < 1.0:  # only scale down
            return cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=self.interpolation)
        return img

    def apply_to_keypoints(self, keypoints: np.ndarray, max_size: Optional[int] = None, **params) -> np.ndarray:
        height, width = params["shape"][:2]
        scale = self._compute_scale(height, width, max_size, self.max_size_hw)
        return F.keypoints_scale(keypoints, scale, scale)

    def get_params(self) -> dict[str, Optional[int]]:
        max_size = self.max_size
        if isinstance(max_size, (list, tuple)):
            max_size = self.py_random.choice(max_size)
        return {"max_size": max_size}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "max_size_hw", "interpolation")
