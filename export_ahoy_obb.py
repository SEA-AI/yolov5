"""
Export AHOY with yoloOBB as horizon model to ONNX format. 

ONNX is an open standard for machine learning models that enables interoperability 
between different frameworks and platforms.
https://onnx.ai/

The exported ONNX model can be used with various inference engines and accelerators,
including TensorRT for optimized GPU inference.

Example:
    # Using local weights files:
    python export_ahoy_obb.py \
        --det-weights yolov5n.pt \
        --hor-weights yolov11n-obb.pt \
        --imgsz 640 \
        --infsz 320
        --batch-size 2 \
        --fuse \
        --half \
        --fname ahoy.onnx

# flatten this
    python export_ahoy_obb.py --det-weights yolov5n.pt --hor-weights yolov11n-obb.pt --imgsz 640 --infsz 320 --batch-size 2 --fuse --half --fname ahoy.onnx


    # Using W&B artifacts:
    python export_ahoy_obb.py \
        --det-weights YOLOv5n-IR:latest \
        --hor-weights yolov11n-obb-IR:latest \
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
from export_ahoy import _transform_sz, get_weights_path
from models.custom import AHOYOBB
from models.yolo import Detect

logging.basicConfig(level=logging.INFO)

def main(
    det_weights: str,
    hor_weights: str,
    imgsz: int | Tuple[int, int],
    infsz: int | Tuple[int, int],
    batch_size: int,
    half: bool,
    fuse: bool,
    trt7_compatible: bool = False,
    fname: str = "",
):
    """Export the model to TensorRT engine."""
    # Transform image size to (height, width) format
    imgsz = _transform_sz(imgsz)
    infsz = _transform_sz(infsz)
    det_weights = get_weights_path(det_weights)
    hor_weights = get_weights_path(hor_weights)

    model = AHOYOBB(
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
