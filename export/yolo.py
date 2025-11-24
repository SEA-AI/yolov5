"""
Export YOLO to ONNX format.

ONNX is an open standard for machine learning models that enables interoperability 
between different frameworks and platforms.
https://onnx.ai/

The exported ONNX model can be used with various inference engines and accelerators,
including TensorRT for optimized GPU inference.

Example:
    # Using local weights files:
    python export/yolo.py \
        --det_weights yolov5n.pt \
        --imgsz 640 \
        --infsz 320 \
        --batch-size 2 \
        --fuse \
        --half \
        --fname yolo.onnx

    # Using W&B artifacts:
    python export/yolo.py \
        --det_weights YOLOv5n-IR:latest \
        --imgsz 640 \
        --batch-size 2 \
        --fuse \
        --half \
        --fname yolo.onnx

NOTE: For TensorRT 7 compatible models, use the --trt7-compatible flag.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.export import export_model_to_onnx, get_weights_path, transform_sz
from models.custom import SeaYOLO

def main(
    det_weights: str,
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
    """Export the YOLO model to ONNX format."""
    # Transform image size to (height, width) format
    imgsz = transform_sz(imgsz)
    infsz = transform_sz(imgsz) if infsz is None else transform_sz(infsz)
    det_weights = get_weights_path(det_weights)

    model = SeaYOLO(
        obj_det_weights=det_weights,
        fp16=half,
        fuse=fuse,
        imgsz=imgsz,
        infsz=infsz,
    )

    export_model_to_onnx(
        model=model,
        imgsz=imgsz,
        batch_size=batch_size,
        fname=fname,
        dynamic=dynamic,
        simplify=simplify,
        trt7_compatible=trt7_compatible,
    )


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-dw",
        "--det-weights",
        type=str,
        required=True,
        help="Path to the object detection model weights or W&B artifact (e.g., 'YOLOv5n-IR:latest').",
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
        "-dy",
        "--dynamic",
        action="store_true",
        help="Export with dynamic batch size.",
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

