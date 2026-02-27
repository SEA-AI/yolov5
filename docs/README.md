# MSParadiseAI

> [!note]
> Documentation not strictly related to ultralytics, but rather on how it is used in the context of the MSParadiseAI project.

## How was the Latest YOLOv5 BBox Model Trained?

### Class Mapping

The object labels are inconsistent to say the least, to make the situation _better_, we map them to a more reduced set of classes. Here's a snippet on how to do it:

```python
dataset = fo.load_dataset("...")
class_map = read_yaml("../yolov5/data/class_map.yaml")  # Actual path

valid_classes = {
    key: value for key, value in class_map.items() if value != "None"
}

# make a clone of the ground_truth_det field
dataset.clone_sample_field(
    "ground_truth_det",
    "ground_truth_train_det",
)

# apply the class map to the ground_truth_train_det field
dataset = dataset.filter_labels(
    "ground_truth_train_det",
    F("label").is_in(list(valid_classes.keys())),
    only_matches=False,
)
dataset.keep()  # deletes fields that are not in the valid_classes
dataset.save()

dataset = dataset.map_labels("ground_truth_train_det", class_map)
```

### Export dataset

Use `scripts/fo_to_yolo.py` to export a FiftyOne dataset to YOLOv5 format with optional W&B registration.

```bash
python scripts/fo_to_yolo.py \
  --dataset-name "MY_DATASET" \
  --export-dir "/home/sea-ai/Documents" \
  --class-map "data/class_map.yaml" \
  --split-mode split \        # split | train | val
  --split-by trip \           # field name | random (skipped if TRAIN_/VAL_ tags already exist)
  --noise-ratio 0.25 \        # omit to use all samples
  --val-ratio 0.2 \
  --wandb --wandb-entity sea-ai --wandb-org sea-ai-org --wandb-collection "MY_DATASET" \
  --tags-suffix v0
```

- A description is prompted interactively and is **required** to proceed.
- The output folder is `<export-dir>/<dataset-name>_<tags-suffix>/`.
- `dataset.yaml` uses `path: .` (portable — works wherever the folder is moved or downloaded from W&B).
- If `TRAIN_<suffix>` and `VAL_<suffix>` tags already exist on the samples, the existing split is used automatically.
- Omit `--wandb` to skip upload entirely. Omit `--wandb-org` to upload the artifact without linking to the Dataset Registry.

### Combine datasets

Use `scripts/combine_datasets.py` to merge multiple exported datasets into a single self-contained dataset for training. All source datasets must share the same class list.

Each entry in `--datasets` can be a local path or a W&B artifact reference — they can be mixed freely.

```bash
# Local paths
python scripts/combine_datasets.py \
  --datasets "/home/sea-ai/Documents/DATASET_A_v0" "/home/sea-ai/Documents/DATASET_B_v0" \
  --output-dir "/home/sea-ai/Documents/COMBINED_v0" \
  --wandb --wandb-entity sea-ai --wandb-collection "my-combined-dataset"

# W&B artifact refs — no local setup needed, downloaded automatically
python scripts/combine_datasets.py \
  --datasets "sea-ai/dataset-registry/DATASET_A:v0" "sea-ai/dataset-registry/DATASET_B:v0" \
  --wandb --wandb-entity sea-ai --wandb-collection "my-combined-dataset"
```

- A description is prompted interactively and is **required** to proceed.
- Local paths and W&B artifact refs can be mixed freely.
- `--output-dir` and `--download-dir` are optional — both default to a temp directory if not set.
- Images and labels from all source datasets are copied into `images/` and `labels/`, prefixed with `d0_`, `d1_`, etc. to avoid filename collisions.
- `dataset.yaml` uses `path: .` (portable — works wherever the folder is moved or downloaded from W&B).
- The W&B artifact contains the full combined dataset (images + labels + yaml) — self-contained and usable on any machine.
- W&B artifact entries are declared as lineage inputs in the uploaded artifact.
- Omit `--wandb-org` to upload the artifact without linking to the Dataset Registry.
- Pass the output YAML directly to training: `python train.py --data /path/to/combined/dataset.yaml`.

### Training IR Entrypoint

```python
from yolov5 import train

train.run(
    data="path/to/dataset.yaml",  # obtained from export
    hyp="yolov5/data/hyps/hyp.sea-ai-IR.yaml",
    device=0,
    epochs=100,
    batch_size=-1,  # auto-batching
    imgsz=640,
    weights="yolov5n.pt",
    single_cls=False,
    close_mosaic=10,
    single_cls_val=True,
)
```

### Training RGB Entrypoint

```python
from yolov5 import train

train.run(
    data="path/to/dataset.yaml",  # obtained from export
    hyp="yolov5/data/hyps/hyp.sea-ai.yaml",
    device=0,
    epochs=100,
    batch_size=-1,  # auto-batching
    imgsz=1280,
    weights="yolov5n6.pt",
    single_cls=False,
    single_cls_val=True,
    im_compression_prob=0.9,
)
```


## How was the Latest YOLOv5 Horizon Model Trained?

> [!important]
> The `torch.utils.data.Dataset` needs a `fiftyone.Dataset` to be passed to it. It will create a view of the dataset with the samples that have a horizon line annotation. Meaning, during training, all samples WITHOUT a horizon line annotation will be ignored.

> [!note]
> Fiftyone is only needed during the instantiation of the `torch.utils.data.Dataset` object. It will load the list of filepaths and targets into memory.

### Training IR Entrypoint

```python
from yolov5.horizon import train

train.run(
    dataset_name="TRAIN_IR_ALL_2024_09_IMAGE_BB",
    train_tag="TRAIN_by_sequence",
    val_tag="VAL_by_sequence",
    epochs=100,
    batch_size=-1,  # auto-batching
    imgsz=640,
    weights="yolov5n.pt",
)
```

### Training RGB Entrypoint

```python
from yolov5.horizon import train

train.run(
    dataset_name="TRAIN_RGB_ALL_2025_02_IMAGE_BB",  # "TRAIN_IR_ALL_2024_09_IMAGE_BB"
    train_tag="TRAIN_v0",
    val_tag="VAL_v0",
    epochs=100,
    batch_size=-1,  # auto-batching
    imgsz=1280,
    weights="yolov5n6.pt"
)
```
