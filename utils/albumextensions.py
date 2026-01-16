"""Custom augmentations following the Albumentations API."""

import random
from typing import Dict, List, Optional, Tuple, Union, cast, Any, Literal
from typing_extensions import Self

import cv2
import numpy as np
import albumentations as A
from albumentations.augmentations.geometric import functional as F
from albumentations.core.pydantic import InterpolationType
from albumentations.core.types import ScaleIntType
from albumentations.core.utils import to_tuple
from albumentations.augmentations.crops import functional as fcrops
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from pydantic import ValidationInfo, field_validator, model_validator


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

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]

        image_height, image_width = image_shape

        # do not throw an error if the crop size exceeds the image size
        # instead, dynamically adjust the crop size to the image size
        height = min(self.height, image_height)
        width = min(self.width, image_width)

        h_start = random.random()
        w_start = random.random()
        crop_coords = fcrops.get_crop_coords(image_shape, (height, width), h_start, w_start)
        return {"crop_coords": crop_coords}


class ThermalRandomBrightnessContrast(A.RandomBrightnessContrast):
    """Random brightness and contrast adjustment for thermal images."""

    def apply(self, img: np.ndarray, alpha: float, beta: float, **params: Any) -> np.ndarray:
        if not is_grayscale(img):
            return img
        return super().apply(img, alpha, beta, **params)


class ThermalRandomGamma(A.RandomGamma):
    """Random gamma adjustment for thermal images."""

    def apply(self, img: np.ndarray, gamma: float, **params: Any) -> np.ndarray:
        if not is_grayscale(img):
            return img
        return super().apply(img, gamma, **params)


class ThermalHorizontalMotionBlur(A.ImageOnlyTransform):
    """Simulate left-right (horizontal) thermal motion blur by convolving with an exponential decay kernel.

    This transform applies a 1D exponential blur kernel along the horizontal direction (central row),
    parameterized by a "time constant" tau. The kernel simulates blur either to the left, right, or,
    if direction='random', randomly chooses left or right each call (never both or at arbitrary angles).

    Ideally, grayscale images will be blurred; color images are returned unchanged.

    Notes:
        - Blur is always strictly leftward or rightward (horizontal) in the image; never a 2D (angled) motion blur.
        - direction="random" chooses randomly between "left" or "right" on each application.
        - Kernel is non-symmetric and highly directional; not a standard motion blur or box filter.
        - Gaussian noise is applied after convolution, pixelwise and independently.
        - Underlying convolution uses OpenCV's filter2D via Albumentations.
        - Useful for simulating "thermal" or exponential-trail blur typically seen in certain sensing scenarios.

    Targets:
        image

    Image types:
        uint8, float32

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> from utils.albumextensions import ThermalHorizontalMotionBlur
        >>> image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> transform = ThermalHorizontalMotionBlur(tau_range=(2, 10), direction="random", p=1.0)
        >>> result = transform(image=image)
        >>> motion_blurred_image = result["image"]
    """

    class InitSchema(BaseTransformInitSchema):
        tau_range: Union[float, Tuple[float, float]]
        direction: Literal["left", "right", "random"]
        noise_std_range: Union[float, Tuple[float, float]]
        skip_rgb: bool = True

        @model_validator(mode="after")
        def process_blur(self) -> Self:
            self.tau_range = to_tuple(self.tau_range, 1)
            self.noise_std_range = to_tuple(self.noise_std_range, 0)

            if self.tau_range[0] < 1:
                raise ValueError("tau_range must be greater than or equal to 1")

            if self.direction not in {"left", "right", "random"}:
                raise ValueError("direction must be 'left', 'right', or 'random'")

            if self.noise_std_range[0] < 0:
                raise ValueError("noise_std_range must be greater than or equal to 0")

            return self

    def __init__(
        self,
        tau_range: Union[float, Tuple[float, float]] = (1, 20),
        direction: Literal["left", "right", "random"] = "random",
        noise_std_range: Union[float, Tuple[float, float]] = (0.01, 0.02),
        skip_rgb: bool = True,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        """
        Args:
            tau_range: Blur decay parameter, float or (min, max).
            direction: Blur direction: 'left' (rightward), 'right' (leftward), or
                'random' (randomly choose left or right direction per call).
            noise_std_range: Noise std as fraction of max pixel, float or (min, max).
            skip_rgb: If True, skip blurring for RGB images. An rgb image in this context is
                an image with 3 channels which are not all the same.
            p: Probability of applying the transform.
            always_apply: If set, always apply.
        """
        super().__init__(p=p, always_apply=always_apply)
        self.tau_range = cast("Tuple[float, float]", tau_range)
        self.direction = direction
        self.noise_std_range = cast("Tuple[float, float]", noise_std_range)
        self.skip_rgb = skip_rgb

    def apply(
        self, img: np.ndarray, kernel: np.ndarray, anchor: tuple[int, int], noise_std: float, **params: Any
    ) -> np.ndarray:
        # Blur is only applied to grayscale images
        if not is_grayscale(img) and self.skip_rgb:
            return img

        # blur image
        img = cv2.filter2D(img, -1, kernel, anchor=anchor).astype(img.dtype)

        # add gaussian noise
        noise = np.zeros_like(img[..., 0:1], dtype=np.float32)
        cv2.randn(noise, 0.0, noise_std)
        noisy = img.astype(np.float32) + noise * np.iinfo(img.dtype).max

        # clip and return
        return np.clip(noisy, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max).astype(img.dtype)

    def get_params(self) -> dict[str, Any]:
        # Kernel length is proportional to tau
        if self.direction == "random":
            direction = 1 if random.random() < 0.5 else -1
        else:
            direction = 1 if self.direction == "left" else -1

        def exp(tau: float, t: np.ndarray) -> np.ndarray:
            return np.exp(-direction * t / tau)

        tau = random.uniform(self.tau_range[0], self.tau_range[1])
        length = int(np.ceil(2.5 * tau))
        length += 1 if length % 2 == 0 else 0  # Ensure odd

        kernel = np.zeros((1, length))
        kernel[0, :] = exp(tau, np.arange(length))
        kernel /= kernel.sum()

        # OpenCV's filter2D uses the 'anchor' parameter to set the kernel's alignment point.
        # For asymmetric kernels like this motion blur, the anchor should be set to the start (left edge)
        # for right-moving blur or end (right edge) for left-moving blur.
        anchor = (0, 0) if direction > 0 else (length - 1, 0)

        noise_std = random.uniform(self.noise_std_range[0], self.noise_std_range[1])
        return {"kernel": kernel, "anchor": anchor, "noise_std": noise_std}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("tau_range", "direction", "noise_std_range", "skip_rgb")


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
        if any(x < 1 for x in v):
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

    def _compute_scale(
        self, height: int, width: int, max_size: Optional[int], max_size_hw: Optional[Tuple[int, int]]
    ) -> float:
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


def is_grayscale(img: np.ndarray, tol: float = 1e-6) -> bool:
    """Naive check if an image is grayscale."""
    if img.ndim == 2:
        return True
    if img.ndim != 3 or img.shape[2] != 3:
        return False
    # Check all channels same (within tol)
    diff = img.max(axis=2) - img.min(axis=2)
    return diff.max() <= tol
