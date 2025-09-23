"""
Export AHOY to ONNX format.

ONNX is an open standard for machine learning models that enables interoperability 
between different frameworks and platforms.
https://onnx.ai/

The exported ONNX model can be used with various inference engines and accelerators,
including TensorRT for optimized GPU inference.

Example:
    # Using local weights files:
    python export_ahoy.py \
        --det-weights yolov5n.pt \
        --hor-weights yolov11n-obb.pt \
        --imgsz 640 \
        --infsz 320
        --batch-size 2 \
        --fuse \
        --half \
        --fname ahoy.onnx

    # Using W&B artifacts:
    python export_ahoy.py \
        --det-weights YOLOv5n-IR:latest \
        --hor-weights YOLOv5h-IR:latest \
        --imgsz 640 \
        --batch-size 2 \
        --fuse \
        --half \
        --fname ahoy.onnx

NOTE: For TensorRT 7 compatible models, use the --trt7-compatible flag.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from export import export_onnx, export_onnx_trt7_compatible
from models.custom import AHOY
from utils.general import LOGGER


def get_weights_path(weights_path: str) -> str:
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
    
    # Try registry first (format: collection:version)
    if ":" in weights_path and "/" not in weights_path.split(":")[0]:
        try:
            collection, version = weights_path.split(":")
            artifact_name = f"wandb-registry-model/{collection}:{version}"
            LOGGER.info(f"Attempting to download from registry: {artifact_name}")
            
            artifact_path = api.artifact(name=artifact_name).download(
                root=Path("artifacts", weights_path)
            )
            return str(next(Path(artifact_path).glob("*.pt")))
            
        except Exception as e:
            LOGGER.warning(f"Failed to download from registry: {e}")
    
    # Try as direct run artifact (format: entity/project/artifact:version)
    try:
        LOGGER.info(f"Attempting to download as run artifact: {weights_path}")
        artifact_path = api.artifact(name=weights_path).download(
            root=Path("artifacts", weights_path.replace("/", "_").replace(":", "_"))
        )
        return str(next(Path(artifact_path).glob("*.pt")))
        
    except Exception as e:
        LOGGER.error(f"Failed to download from W&B run: {e}")


def _transform_sz(imgsz: int | List[int] | Tuple[int, int]) -> Tuple[int, int]:
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


def main(
    det_weights: str,
    hor_weights: str,
    imgsz: int | Tuple[int, int],
    infsz: int | Tuple[int, int] | None,
    batch_size: int,
    half: bool,
    fuse: bool,
    dynamic: bool = False,
    simplify: bool = False,
    trt7_compatible: bool = False,
    fname: str = "",
):
    """Export the model to TensorRT engine."""
    # Transform image size to (height, width) format
    imgsz = _transform_sz(imgsz)
    infsz = _transform_sz(imgsz) if infsz is None else _transform_sz(infsz)
    det_weights = get_weights_path(det_weights)
    hor_weights = get_weights_path(hor_weights)

    model = AHOY(
        obj_det_weigths=det_weights,
        hor_det_weights=hor_weights,
        fp16=half,
        fuse=fuse,
        imgsz=imgsz,
        infsz=infsz,
    )

    if not fname:
        input_size = f"{imgsz[0]}x{imgsz[1]}"
        fname = f"{type(model).__name__.lower()}_b{batch_size}_sz{input_size}.onnx"
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
    _ = export_func(
        model,
        im=image,
        file=Path(fname),
        dynamic=dynamic,
        simplify=simplify,
        opset=12,
    )


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-dw",
        "--det-weights",
        type=str,
        required=True,
        help="Path to the detection model weights.",
    )
    parser.add_argument(
        "-hw",
        "--hor-weights",
        type=str,
        required=True,
        help="Path to the horizontal model weights.",
    )
    parser.add_argument("-sz", "--imgsz", nargs="+", type=int, default=[640, 640], help="image input shape (h, w)")
    parser.add_argument(
        "--infsz",
        nargs="+",
        type=int,
        default=None,
        help="image shape during inference (h, w). If not specified, uses same size as imgsz",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size.",
    )
    parser.add_argument(
        "-fu",
        "--fuse",
        action="store_true",
        help="Fuse convolution and batchnorm layers.",
    )
    parser.add_argument(
        "-hf",
        "--half",
        action="store_true",
        help="Export half-precision model.",
    )
    parser.add_argument(
        "-si",
        "--simplify",
        action="store_true",
        help="Simplify the exported model.",
    )
    parser.add_argument(
        "-trt7",
        "--trt7-compatible",
        action="store_true",
        help="Export TensorRT 7 compatible model.",
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default="",
        help="Filename for the exported model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(**vars(args))
