#!/usr/bin/env python3
"""
Example usage of OneberryYolo class for dual-model YOLO inference.

This script demonstrates how to use the OneberryYolo class which combines
a medium and secondary YOLOv5 model, with the secondary model taking precedence
when predictions overlap.

Usage:
    python examples/oneberry_yolo_usage.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import from models
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from models.custom import OneberryYolo


def example_oneberry_usage():
    """Example of using OneberryYolo for dual-model inference."""
    
    print("OneberryYolo Usage Example")
    print("=" * 50)
    
    # Example model weights paths (you'll need to provide actual paths)
    medium_weights = "yolov5m.pt"     # Path to medium model
    secondary_weights = "yolov5n.pt"  # Path to secondary model (has priority)
    
    print(f"Medium model: {medium_weights}")
    print(f"Secondary model: {secondary_weights}")
    
    # Initialize OneberryYolo model
    model = OneberryYolo(
        medium_weights=medium_weights,
        secondary_weights=secondary_weights,
        device="cpu",  # or "cuda" if available
        fp16=False,
        fuse=True,
        imgsz=(640, 640),
        iou_threshold=0.5,  # IoU threshold for overlap detection
    )
    
    print(f"Model initialized on device: {model.device}")
    print(f"Model stride: {model.stride}")
    print(f"Model names: {model.names}")
    
    # Create dummy input for testing
    batch_size = 2
    dummy_input = torch.randint(0, 255, (batch_size, 3, 640, 640), dtype=torch.uint8)
    
    print(f"Input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print(f"Output shape: {predictions[0].shape if isinstance(predictions, tuple) else predictions.shape}")
    
    # Example: Register hooks for preprocessing/postprocessing (useful for export)
    model.register_io_hooks()
    print("IO hooks registered for export readiness")
    
    # Cleanup hooks
    model.remove_hooks()
    print("Hooks removed")
    
    print("\nOneberryYolo example completed successfully!")


def export_example():
    """Example of exporting OneberryYolo to ONNX."""
    print("\nONNX Export Example")
    print("=" * 30)
    
    print("To export OneberryYolo to ONNX, use:")
    print("python export/yolo.py \\")
    print("    --det_weights yolov5m.pt \\")
    print("    --secondary_weights yolov5n.pt \\")
    print("    --imgsz 640 640 \\")
    print("    --batch-size 1 \\")
    print("    --iou-threshold 0.5 \\")
    print("    --fuse \\")
    print("    --fname oneberry_yolo.onnx")


if __name__ == "__main__":
    try:
        example_oneberry_usage()
        export_example()
    except FileNotFoundError as e:
        print(f"Model weights not found: {e}")
        print("Please download YOLOv5 weights or adjust the paths in this example.")
        export_example()
    except Exception as e:
        print(f"Error: {e}")
        export_example()