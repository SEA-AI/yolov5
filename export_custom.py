"""
Export YOLO or AHOY to ONNX. Mode: AHOY if --hor-weights is set, else YOLO.
Weights can be local paths or W&B artifacts (e.g. entity/project/run:v0).

Usage:
  # YOLO single
  python export_custom.py --det-weights YOLOv5n-IR:latest --imgsz 480 640 --half --fuse --fname yolo.onnx
  python export_custom.py --det-weights sea-ai/yolo-train/run_xxx:v0 --imgsz 480 640 --half --fuse

  # YOLO ensemble
  python export_custom.py --det-weights YOLOv5m-IR:latest YOLOv5n-H:latest --imgsz 480 640 --half --fuse

  # AHOY (set --hor-weights)
  python export_custom.py --det-weights YOLOv5n-MIX:latest --hor-weights sea-ai/ultralytics/run_xxx:v0 \\
      --imgsz 1080 3600 --infsz 572 1920 --batch-size 1 --half --fuse

Use --trt7-compatible to export TensorRT 7 compatible model.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from export import export_onnx, export_onnx_trt7_compatible
from models.custom import AHOY, YOLO
from utils.general import LOGGER


def export_model_to_onnx(
    model: YOLO | AHOY,
    imgsz: Tuple[int, int],
    batch_size: int,
    fname: str,
    dynamic: bool = False,
    simplify: bool = False,
    trt7_compatible: bool = False,
):
    """
    Export a model to ONNX format.

    Generic export logic for both YOLO and AHOY models.

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
        base = type(model).__name__.lower()
        if isinstance(model.obj_det_weights, list) and len(model.obj_det_weights) > 1:
            base = f"{base}ensemble"
        fname = f"{base}_b{batch_size}_sz{input_size}.onnx"
    LOGGER.info(f"🚀 Exporting model {type(model).__name__} to {fname}...")

    model.prepare_for_export(dynamic=dynamic)
    model.register_io_hooks()  # inp: uint8 -> fp32/fp16 / 255.0, out: fp16 -> fp32

    image = torch.zeros((batch_size, 3, imgsz[0], imgsz[1]), device=model.device).byte()  # B, C, H, W
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
        opset=17,
    )

    return result


def main(
    det_weights: List[str],
    imgsz: int | Tuple[int, int],
    infsz: int | Tuple[int, int] | None,
    batch_size: int,
    half: bool,
    fuse: bool,
    hor_weights: str | None = None,
    device: str = "cpu",
    dynamic: bool = False,
    simplify: bool = False,
    trt7_compatible: bool = False,
    fname: str = "",
):
    """Export YOLO or AHOY to ONNX. If hor_weights is set, build AHOY; else YOLO (ensemble if len(det_weights) > 1)."""
    if hor_weights is not None:
        model = AHOY(
            obj_det_weights=det_weights,
            hor_det_weights=hor_weights,
            fp16=half,
            fuse=fuse,
            imgsz=imgsz,
            infsz=infsz,
            device=device,
        )
    else:
        model = YOLO(
            weights=det_weights,
            fp16=half,
            fuse=fuse,
            imgsz=imgsz,
            infsz=infsz,
            device=device,
        )

    export_model_to_onnx(
        model=model,
        imgsz=model.imgsz,
        batch_size=batch_size,
        fname=fname,
        dynamic=dynamic,
        simplify=True if len(det_weights) > 1 else simplify,
        trt7_compatible=trt7_compatible,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export YOLO or AHOY to ONNX. Use --hor-weights to export AHOY.",
    )
    parser.add_argument(
        "-dw",
        "--det-weights",
        nargs="+",
        required=True,
        metavar="WEIGHTS",
        help=(
            "Detection weights: path(s) or W&B artifact (e.g. entity/project/run:v0). "
            "One = single model; two+ = ensemble object detection model."
        ),
    )
    parser.add_argument(
        "-hw",
        "--hor-weights",
        type=str,
        default=None,
        help="Horizontal weights path or W&B artifact. If set, export AHOY.",
    )
    parser.add_argument(
        "-sz",
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Input image size (h w).",
    )
    parser.add_argument(
        "--infsz",
        nargs="+",
        type=int,
        default=None,
        help="Model resizes internally to this (h w) for inference. Default: --imgsz (no resize).",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size for export.")
    parser.add_argument("-fu", "--fuse", action="store_true", help="Fuse conv+bn.")
    parser.add_argument("-hf", "--half", action="store_true", help="FP16 export.")
    parser.add_argument("-si", "--simplify", action="store_true", help="Run ONNX simplifier.")
    parser.add_argument("-dy", "--dynamic", action="store_true", help="Dynamic batch axis.")
    parser.add_argument("-trt7", "--trt7-compatible", action="store_true", help="TensorRT 7 compatible ONNX.")
    parser.add_argument("-dv", "--device", type=str, default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu.")
    parser.add_argument("-f", "--fname", type=str, default="", help="Output ONNX filename (default: auto).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(**vars(args))
