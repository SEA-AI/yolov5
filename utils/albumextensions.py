"""Custom augmentations following the Albumentations API."""

import random
from typing import Dict, Optional, Tuple, Union, cast, List

import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as F
from albumentations.core.pydantic import InterpolationType, ScaleIntType
from albumentations.core.transforms_interface import DualTransform, BaseTransformInitSchema

from pydantic import field_validator, ValidationInfo


class MaxSizeHWInitSchema(BaseTransformInitSchema):
    max_size: int | list[int] | None
    max_size_hw: tuple[int, int] | None
    interpolation: InterpolationType

    @field_validator("max_size")
    @classmethod
    def check_max_size(cls, v: ScaleIntType | None, info: ValidationInfo) -> int | list[int] | None:
        if v is None:
            return None
        result = v if isinstance(v, (list, tuple)) else [v]
        for value in result:
            if value < 1:
                raise ValueError(f"{info.field_name} must be bigger or equal to 1.")
        return cast(Union[int, List[int], None], result)

    @field_validator("max_size_hw")
    @classmethod
    def check_max_size_hw(cls, v: tuple[int, int] | None, info: ValidationInfo) -> tuple[int, int] | None:
        if v is None:
            return None
        if not isinstance(v, tuple) or len(v) != 2:
            raise ValueError(f"{info.field_name} must be a tuple of two integers")
        if not all(x >= 1 for x in v):
            raise ValueError(f"All values in {info.field_name} must be bigger or equal to 1")
        return v


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
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    class InitSchema(MaxSizeHWInitSchema):
        pass

    def __init__(
        self,
        max_size: Optional[Union[int, list]] = None,
        max_size_hw: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.max_size = max_size
        self.max_size_hw = max_size_hw
        self.interpolation = interpolation

        if not any([max_size, max_size_hw]):
            raise ValueError("At least one of max_size or max_size_hw must be set")

    def _compute_scale(self, height: int, width: int, max_size: Optional[int], max_size_hw: Optional[Tuple[int, int]]) -> float:
        """Compute the scale factor based on image dimensions and maximum constraints."""
        scale_height = scale_width = 1.0

        if max_size_hw is not None:
            max_height, max_width = max_size_hw
            if height > max_height:
                scale_height = max_height / height
            if width > max_width:
                scale_width = max_width / width

        if max_size is not None and max(height, width) > max_size:
            scale = max_size / max(height, width)
            scale_height = scale_width = min(scale_height, scale_width, scale)

        return min(scale_height, scale_width)

    def apply(
        self,
        img: np.ndarray,
        max_size: Optional[int] = None,
        max_size_hw: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        **params,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        final_scale = self._compute_scale(height, width, max_size, max_size_hw)

        if final_scale < 1.0:  # Only resize if we need to scale down
            new_height = int(height * final_scale)
            new_width = int(width * final_scale)
            return cv2.resize(img, (new_width, new_height), interpolation=interpolation)

        return img

    def apply_to_bbox(self, bbox: np.ndarray, **params) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        max_size: Optional[int] = None,
        max_size_hw: Optional[Tuple[int, int]] = None,
        **params,
    ) -> np.ndarray:
        height = params["rows"]
        width = params["cols"]

        final_scale = self._compute_scale(height, width, max_size, max_size_hw)
        final_scale = min(1.0, final_scale)  # don't scale up

        return F.keypoints_scale(keypoints, final_scale, final_scale)

    def get_params(self) -> Dict[str, Optional[int]]:
        max_size = None
        if self.max_size:
            max_size = self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)

        return {"max_size": max_size, "max_size_hw": self.max_size_hw}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "max_size_hw", "interpolation")
