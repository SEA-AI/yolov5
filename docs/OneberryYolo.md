# OneberryYolo: Dual-Model YOLO Inference

OneberryYolo is a custom PyTorch module that combines two YOLOv5 models (medium and secondary) in a single inference pipeline. **When predictions from both models overlap, the secondary model's predictions take precedence and override the medium model's predictions.**

## Features

- **Dual-model inference**: Combines medium and secondary YOLOv5 models
- **Priority-based overlap handling**: Secondary model predictions take precedence over overlapping medium model predictions
- **IoU-based filtering**: Configurable IoU threshold for determining overlaps
- **Export compatibility**: Can be exported to ONNX format using the existing export infrastructure
- **Preprocessing/postprocessing hooks**: Compatible with SeaYOLO's hook system
- **Flexible sizing**: Supports different input and inference image sizes with automatic padding/scaling

## Use Cases

- **Hybrid detection**: Use medium model for broad coverage and secondary model for specific, high-priority detections
- **Performance optimization**: Balance between detection accuracy (medium) and speed (secondary)
- **Priority-based detection**: When certain objects need to be detected with higher priority/confidence by the secondary model

## Usage

### Basic Python Usage

```python
from models.custom import OneberryYolo

# Initialize the model
model = OneberryYolo(
    medium_weights="yolov5m.pt",
    secondary_weights="yolov5n.pt",  # Secondary model has priority over overlaps
    device="cuda",  # or "cpu"
    fp16=True,
    fuse=True,
    imgsz=(640, 640),
    iou_threshold=0.5,  # IoU threshold for overlap detection
)

# Inference
import torch
dummy_input = torch.randint(0, 255, (1, 3, 640, 640), dtype=torch.uint8)
predictions = model(dummy_input)
```

### ONNX Export

#### Single Model (SeaYOLO)
```bash
python export/yolo.py \
    --det_weights yolov5n.pt \
    --imgsz 640 \
    --batch-size 1 \
    --fuse \
    --fname single_yolo.onnx
```

#### Dual Model (OneberryYolo)
```bash
python export/yolo.py \
    --det_weights yolov5m.pt \
    --secondary_weights yolov5n.pt \
    --imgsz 640 \
    --batch-size 1 \
    --iou-threshold 0.5 \
    --fuse \
    --fname oneberry_yolo.onnx
```

## Parameters

### OneberryYolo.__init__()

- `medium_weights` (str): Path to medium model weights file
- `secondary_weights` (str): Path to secondary model weights file (takes priority over overlapping predictions)
- `device` (str|torch.device): Device to run models on (auto-selected if None)
- `fp16` (bool): Use half precision (fp16)
- `fuse` (bool): Fuse conv and batch norm layers
- `imgsz` (Tuple[int, int]): Input image size (height, width)
- `infsz` (Tuple[int, int], optional): Inference size (height, width)
- `iou_threshold` (float): IoU threshold for determining overlapping detections (default: 0.5)

### Export Parameters

- `--det_weights`: Path to medium model weights (required)
- `--secondary_weights`: Path to secondary model weights (enables OneberryYolo mode, has priority over overlapping predictions)
- `--iou-threshold`: IoU threshold for overlap detection (default: 0.5)
- `--imgsz`: Input image size
- `--batch-size`: Batch size for export
- `--fuse`: Fuse conv/bn layers
- `--half`: Export in half precision
- `--fname`: Output filename

## How It Works

1. **Dual Inference**: Both medium and secondary models process the same input
2. **Prediction Extraction**: Valid detections (confidence > 0) are extracted from both models
3. **Overlap Detection**: IoU is calculated between all medium and secondary detections
4. **Priority Filtering**: Medium model detections that overlap with secondary detections (IoU ≥ threshold) are removed
5. **Combination**: Secondary detections (priority) + non-overlapping medium detections are combined
6. **Output**: Combined predictions are returned in the same format as single-model output

**Key Point**: The secondary model always takes precedence - any medium model detection that overlaps with a secondary model detection is discarded.

## IoU Threshold Guidelines

- **0.3-0.4**: More aggressive filtering, fewer medium detections retained
- **0.5**: Balanced overlap detection (recommended default)
- **0.6-0.7**: Conservative filtering, more medium detections retained
- **0.8+**: Very conservative, only highly overlapping detections are filtered

## Architecture

```
Input Image
     │
     ├─► Medium Model ─► Medium Predictions
     │                        │
     └─► Secondary Model ──► Secondary Predictions (PRIORITY)
                              │
                              ▼
                         Overlap Filter
                              │
                              ▼
                      Combined Predictions
                  (Secondary takes precedence)
```

## Examples

See `examples/oneberry_yolo_usage.py` for a complete usage example.

## Compatibility

- Compatible with existing YOLOv5 export infrastructure
- Works with W&B artifacts
- Supports TensorRT 7 compatibility mode
- Compatible with dynamic batch sizes
- Supports model simplification via ONNX-Simplifier