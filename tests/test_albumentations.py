# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Behavior tests for SEA.AI's albumentations extensions, so library upgrades can't silently change them."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

A = pytest.importorskip("albumentations")

import horizon.transforms as T  # noqa: E402
import utils.albumentations16 as A16  # noqa: E402
import utils.albumextensions as Ax  # noqa: E402
from utils.augmentations import Albumentations, PreAlbumentations  # noqa: E402
from utils.dataloaders import imread_16bit_compatible  # noqa: E402


def coded_image(h, w):
    """Image whose pixel values encode their position, so crops can be located."""
    return (np.arange(h * w, dtype=np.int64).reshape(h, w) % 251).astype(np.uint8)[..., None].repeat(3, 2)


def thermal_image(h=120, w=160, seed=0):
    """Synthetic 16-bit thermal frame within the 15000-28000 band used by SEA.AI IR cameras."""
    return np.random.default_rng(seed).integers(12000, 31000, (h, w), dtype=np.uint16)


# SafeRandomCrop ---------------------------------------------------------------------------------------------------
def test_safe_random_crop_larger_than_image_returns_whole_image():
    im = coded_image(40, 60)
    out = Ax.SafeRandomCrop(height=100, width=100, p=1.0)(image=im)["image"]
    assert np.array_equal(out, im)


def test_safe_random_crop_smaller_than_image_is_a_sub_block():
    im = coded_image(40, 60)
    for _ in range(10):
        out = Ax.SafeRandomCrop(height=16, width=24, p=1.0)(image=im)["image"]
        assert out.shape == (16, 24, 3)
        hits = [
            (y, x)
            for y in range(40 - 16 + 1)
            for x in range(60 - 24 + 1)
            if np.array_equal(im[y : y + 16, x : x + 24], out)
        ]
        assert hits, "crop is not a contiguous block of the input"


def test_safe_random_crop_keeps_yolo_boxes_valid():
    t = A.Compose(
        [Ax.SafeRandomCrop(height=32, width=32, p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True),
    )
    for _ in range(10):
        out = t(image=coded_image(64, 64), bboxes=[[0.5, 0.5, 0.4, 0.4]], class_labels=[1])
        assert out["image"].shape[:2] == (32, 32)
        for b in out["bboxes"]:
            assert all(0.0 <= v <= 1.0 for v in b)


# ResizeIfNeeded ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, shape_in, shape_out",
    [
        ({"max_size_hw": (100, 100)}, (200, 400), (50, 100)),  # width limits: scale 0.25
        ({"max_size_hw": (100, 400)}, (200, 400), (100, 200)),  # height limits: scale 0.5
        ({"max_size": 100}, (200, 400), (50, 100)),  # longest side
        ({"max_size_hw": (100, 100)}, (50, 80), (50, 80)),  # never upscales
    ],
)
def test_resize_if_needed_shapes(kwargs, shape_in, shape_out):
    out = Ax.ResizeIfNeeded(**kwargs)(image=coded_image(*shape_in))["image"]
    assert out.shape[:2] == shape_out


def test_resize_if_needed_max_size_list_picks_one():
    for _ in range(10):
        out = Ax.ResizeIfNeeded(max_size=[50, 100])(image=coded_image(200, 400))["image"]
        assert max(out.shape[:2]) in (50, 100)


def test_resize_if_needed_scales_keypoints():
    t = A.Compose(
        [Ax.ResizeIfNeeded(max_size_hw=(100, 100))],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    out = t(image=coded_image(200, 400), keypoints=[(200, 100), (400, 0)])
    assert np.allclose(np.array(out["keypoints"])[:, :2], [(50, 25), (100, 0)])


def test_resize_if_needed_requires_a_limit():
    with pytest.raises(ValueError):
        Ax.ResizeIfNeeded()


# 16-bit transforms ------------------------------------------------------------------------------------------------
def reference_16_to_8(im, lo=15000 / 65535, hi=28000 / 65535, clip_limit=4):
    """Deterministic 16->8 bit conversion written with plain OpenCV, matching get_16_to_8_transform(augment=False)."""
    im = np.clip(im, int(lo * 65535), int(hi * 65535))
    tile = (round(max(im.shape) / 160),) * 2
    im = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(im)
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)


def test_clip_16bit():
    im = thermal_image()
    out = A16.Clip(p=1.0, lower_limit=(0.25, 0.25), upper_limit=(0.4, 0.4))(image=im)["image"]
    assert out.dtype == np.uint16
    assert np.array_equal(out, np.clip(im, int(0.25 * 65535), int(0.4 * 65535)))


def test_normalize_min_max_16bit_to_uint8():
    out = A16.NormalizeMinMax(p=1.0)(image=thermal_image())["image"]
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_clahe_16bit_auto_tile_grid():
    im = thermal_image(320, 640)
    out = A16.CLAHE(p=1.0, clip_limit=(4, 4), tile_grid_size=(0, 0))(image=im)["image"]
    assert np.array_equal(out, cv2.createCLAHE(clipLimit=4, tileGridSize=(4, 4)).apply(im))


@pytest.mark.parametrize("channels", [1, 3])
def test_convert_16bit_to_8bit_deterministic(channels):
    im = thermal_image()
    src = im if channels == 1 else np.stack([im, im // 2, im // 3], -1)  # 3-ch 16-bit frames use channel 0
    out = A16.convert_16bit_to_8bit(src, augment=False)
    assert out.shape == (120, 160, 3) and out.dtype == np.uint8
    assert np.array_equal(out, reference_16_to_8(im))


def test_convert_16bit_to_8bit_augment():
    for seed in range(5):
        out = A16.convert_16bit_to_8bit(thermal_image(seed=seed), augment=True)
        assert out.shape == (120, 160, 3) and out.dtype == np.uint8


def test_imread_16bit_compatible(tmp_path):
    f = str(tmp_path / "ir.png")
    cv2.imwrite(f, thermal_image())
    out = imread_16bit_compatible(f, augment16=False)
    assert out.shape == (120, 160, 3) and out.dtype == np.uint8


# YOLOv5 dataloader augmentations ----------------------------------------------------------------------------------
def test_albumentations_keeps_labels():
    labels = np.array([[0, 0.5, 0.5, 0.2, 0.2], [1, 0.3, 0.3, 0.1, 0.1]], dtype=np.float32)
    aug = Albumentations(size=64, hyp={"compression": 1.0})
    assert aug.transform is not None
    im, out = aug(coded_image(64, 64), labels.copy())
    assert im.shape == (64, 64, 3) and out.shape == labels.shape
    assert np.allclose(out, labels, atol=1e-6)  # pixel-only transforms leave boxes untouched


def test_pre_albumentations_crops_to_size():
    aug = PreAlbumentations(size=32, hyp={"pre_crop": 1.0})
    assert aug.transform is not None
    for _ in range(10):
        im, out = aug(coded_image(64, 96), np.array([[0, 0.5, 0.5, 0.4, 0.4]], dtype=np.float32))
        assert im.shape == (32, 32, 3)
        assert ((out[:, 1:] >= 0) & (out[:, 1:] <= 1)).all()


# Horizon transforms -----------------------------------------------------------------------------------------------
def test_horizon_base_rgb_letterbox_and_keypoints():
    # 100x320 -> resized to 50x160 (scale 0.5) -> padded to 96x160 with 23 rows on top
    out = T.horizon_base_rgb((96, 160))(image=coded_image(100, 320), keypoints=[(0, 50), (320, 50)])
    assert tuple(out["image"].shape) == (3, 96, 160)
    assert np.allclose(np.array(out["keypoints"])[:, :2], [(0, 48), (160, 48)])


def test_horizon_base_ir16bit():
    out = T.horizon_base_ir16bit((96, 160))(image=thermal_image(), keypoints=[(0, 60), (160, 60)])
    assert tuple(out["image"].shape) == (3, 96, 160)
    assert len(out["keypoints"]) == 2


@pytest.mark.parametrize("ir", [False, True])
def test_horizon_augment_pipelines_run(ir):
    t = T.horizon_augment_ir16bit((96, 160), 0.5) if ir else T.horizon_augment_rgb((96, 160), 0.5)
    for seed in range(5):
        im = thermal_image(seed=seed) if ir else coded_image(120, 160)
        out = t(image=im, keypoints=[(0, 60), (160, 60)])
        assert tuple(out["image"].shape) == (3, 96, 160)
        assert len(out["keypoints"]) == 2  # remove_invisible=False keeps both horizon points
