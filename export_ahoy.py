"""
Export AHOY or AHOYOBB( with Horizon-OBB) to ONNX format.

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
        --fname ahoy.onnx \
        --obb 


# flatten this
    python export_ahoy.py --det-weights yolov5n.pt --hor-weights yolov11n-obb.pt --imgsz 640 --infsz 320 --batch-size 2 --fuse --half --fname ahoy.onnx --obb


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
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from export import export_onnx, export_onnx_trt7_compatible
from models.custom import AHOY, AHOYOBB
from models.yolo import Detect

logging.basicConfig(level=logging.INFO)


def get_weights_path(weights_path: str) -> str:
    """Get model weights from local path or W&B registry.

    Args:
        weights_path: Local path or W&B artifact in format <collection_name>:<version>

    Returns:
        Path to model weights file
    """
    if Path(weights_path).exists():
        return weights_path

    try:
        import wandb
    except ImportError:
        logging.error("Please install wandb to download models from W&B registry")
        return weights_path

    try:
        REGISTRY = "model"
        collection, version = weights_path.split(":")

        api = wandb.Api()
        artifact_name = f"wandb-registry-{REGISTRY}/{collection}:{version}"
        artifact_path = api.artifact(name=artifact_name).download()
        return next(Path(artifact_path).glob("*.pt"))

    except Exception as e:
        logging.error(f"Failed to download from W&B: {str(e)}")
        raise e
    

def _transform_sz(imgsz: int | List[int] | Tuple[int,int]) -> Tuple[int,int]:
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
    infsz: int | Tuple[int, int],
    batch_size: int,
    half: bool,
    fuse: bool,
    obb: bool,
    trt7_compatible: bool = False,
    fname: str = "",
):
    """Export the model to TensorRT engine."""
    # Transform image size to (height, width) format
    imgsz = _transform_sz(imgsz)
    infsz = _transform_sz(infsz)
    det_weights = get_weights_path(det_weights)
    hor_weights = get_weights_path(hor_weights)

    model_class = AHOYOBB if obb else AHOY
    model = model_class(
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
    print(f"🚀 Exporting model {type(model).__name__} to {fname}...")

    inplace = False  # default
    dynamic = False  # default

    # Update model
    model.eval()
    print("✨ Preparing the model for export...")
    for _, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True
    model.register_io_hooks()  # inp: uint8 -> fp32/fp16 / 255.0, out: fp16 -> fp32

    # Create dummy input
    image = torch.zeros((batch_size, 3, imgsz[0], imgsz[1]), device=model.device).byte() # B, C, H, W
    # https://github.com/NVIDIA/TensorRT/issues/3026#issuecomment-1570419758
    image = image.float() if trt7_compatible else image
    print(f"🔮 Dummy input...{image.shape}, {image.dtype}")

    model(image)  # need to run once to get the model to JIT compile

    export_func = export_onnx_trt7_compatible if trt7_compatible else export_onnx
    f, _ = export_func(
        model,
        im=image,
        file=Path(fname),
        dynamic=dynamic,
        simplify=False,
        opset=12,
    )
    print(f"🎉 Model successfully exported to {f}! 🚀")


def _parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "-sz",
        "--imgsz",
        nargs="+", 
        type=int, 
        default=[640, 640], 
        help="image input shape (h, w)"
    )
    parser.add_argument(
        "--infsz",
        nargs="+", 
        type=int, 
        default=[640, 640], 
        help="image shape during inference (h, w)"
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
        "--obb",
        action="store_true",
        help="Use OBB model for the horizon",
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
