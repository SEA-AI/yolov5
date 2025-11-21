# Update model weight handling and OBB model detection

## What?

Enhanced model weight downloading capabilities in `export_ahoy.py` to support both W&B registry and W&B run artifacts, and added OBB model detection functionality in `utils/torch_utils.py` to automatically identify Oriented Bounding Box models.

## Why?

These changes are necessary to improve the model loading workflow by supporting multiple W&B artifact sources and enabling automatic model type detection for OBB models, which streamlines the export process and reduces manual configuration requirements.

## How?

- **Enhanced weight downloading**: Updated `get_weights_path()` function to handle both registry format (`collection:version`) and run artifact format (`entity/project/artifact:version`) with appropriate error handling and logging
- **OBB model detection**: Added `is_obb_weights()` function that loads checkpoint files and checks if the model is an `OBBModel` type, enabling automatic model type detection
- **Import updates**: Added `OBBModel` import from `ultralytics.nn.tasks` to support the new detection functionality

## Testing

The changes can be tested by:

1. **W&B Registry Download**: 
   ```bash
   python export_ahoy.py --det-weights YOLOv5n-IR:latest --hor-weights YOLOv5h-IR:latest
   ```

2. **W&B Run Artifact Download**:
   ```bash
   python export_ahoy.py --det-weights entity/project/artifact:version --hor-weights entity/project/artifact:version
   ```

3. **OBB Model Detection**: The `is_obb_weights()` function will automatically detect OBB models when loading checkpoints, enabling proper model type handling in the AHOY class.

All functionality includes appropriate error handling and logging for debugging purposes. 