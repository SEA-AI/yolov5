"""
Shared utilities for ONNX export scripts.

This module provides common functionality used by both export_ahoy.py and export_yolo.py
to minimize code duplication.
"""

from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

import torch

from export import export_onnx, export_onnx_trt7_compatible
from utils.general import LOGGER

if TYPE_CHECKING:
    from models.custom import YOLO, AHOY


def get_weights_path(weights_path: str) -> str | None:
    """Get model weights from local path or W&B registry/run.

    Args:
        weights_path: Local path or W&B artifact in format:
                     - <collection_name>:<version> (for registry)
                     - <entity>/<project>/<artifact_name>:<version> (for run)

    Returns:
        Path to model weights file
    """

    # Check if it's a local file first
    if Path(weights_path).exists():
        return weights_path

    try:
        import wandb
    except ImportError:
        LOGGER.error("Please install wandb to download models from W&B registry")
        return weights_path

    api = wandb.Api()

    # Build candidates: registry (collection:version) first, then run artifact path
    candidates = []
    if ":" in weights_path and "/" not in weights_path.split(":")[0]:
        candidates.append((f"wandb-registry-model/{weights_path}", "registry"))
    candidates.append((weights_path, "run"))

    for artifact_name, kind in candidates:
        LOGGER.info(f"Attempting to download from {kind}: {artifact_name}")
        artifact_path = api.artifact(name=artifact_name).download(root=Path("artifacts", weights_path))
        return str(next(Path(artifact_path).glob("*.pt")))

    for artifact_name, kind in candidates:
        try:
            artifact_path = api.artifact(name=artifact_name).download(root=Path("artifacts", weights_path))
            artifact_path = str(next(Path(artifact_path).glob("*.pt")))
            LOGGER.info(f"Successfully downloaded {kind}: {artifact_name}")
            return artifact_path
        except Exception as e:
            LOGGER.warning(f"Download failed for {artifact_name} ({kind}): {e}")

    return None


def transform_sz(imgsz: int | List[int] | Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert size specifications to (height, width) tuple format.

    Args:
        imgsz: Int, list, or tuple representing dimensions.
            - If int: converted to (imgsz, imgsz)
            - If list/tuple with 1 or 2 elements: converted to (imgsz[0], imgsz[-1])

    Returns:
        Tuple in (height, width) format.

    Raises:
        ValueError: If imgsz is not int, list, or tuple, or if list/tuple has more than 2 elements.
    """
    # Handle scalar case (single integer)
    if isinstance(imgsz, int):
        return imgsz, imgsz
    if not isinstance(imgsz, (list, tuple)) or len(imgsz) not in (1, 2):
        raise ValueError(f"imgsz must be int or a list/tuple of 1 or 2 elements, got {imgsz}")
    return imgsz[0], imgsz[-1]


def export_model_to_onnx(
    model: "YOLO | AHOY",
    imgsz: Tuple[int, int],
    batch_size: int,
    fname: str,
    dynamic: bool = False,
    simplify: bool = False,
    trt7_compatible: bool = False,
):
    """
    Export a model to ONNX format.

    This is a generic export function that handles the common export logic
    for both YOLO and AHOY models.

    Args:
        model: The model to export (must have prepare_for_export and register_io_hooks methods)
        imgsz: Image size as (height, width) tuple
        batch_size: Batch size for the model
        fname: Output filename for the ONNX model
        dynamic: Whether to export with dynamic batch size
        simplify: Whether to simplify the exported model
        trt7_compatible: Whether to export TensorRT 7 compatible model

    Returns:
        Path to the exported ONNX file
    """

    if not fname:
        input_size = f"{imgsz[0]}x{imgsz[1]}"
        base = f"{type(model).__name__.lower()}"
        if isinstance(model.obj_det_weights, list) and len(model.obj_det_weights) > 1:
            base = f"{base}ensemble"
        fname = f"{base}_b{batch_size}_sz{input_size}.onnx"
    LOGGER.info(f"🚀 Exporting model {type(model).__name__} to {fname}...")

    model.prepare_for_export(dynamic=dynamic)
    model.register_io_hooks()  # inp: uint8 -> fp32/fp16 / 255.0, out: fp16 -> fp32

    # Create dummy input
    image = torch.zeros((batch_size, 3, imgsz[0], imgsz[1]), device=model.device).byte()  # B, C, H, W
    # https://github.com/NVIDIA/TensorRT/issues/3026#issuecomment-1570419758
    image = image.float() if trt7_compatible else image
    LOGGER.info(f"🔮 Dummy input...{image.shape}, {image.dtype}")

    model(image)  # need to run once to get the model to JIT compile

    export_func = export_onnx_trt7_compatible if trt7_compatible else export_onnx
    result = export_func(
        model,
        im=image,
        file=Path(fname),
        dynamic=dynamic,
        simplify=simplify,
        opset=12,
    )

    return result
