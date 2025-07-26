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

YOLOv5 uses a defined format for their dataset. Since SEA.AI uses fiftyone for data management, we need to export the dataset in the YOLOv5 format. Luckily, fiftyone has a built-in function to export the dataset in the YOLOv5 format.

Here's a snippet of the code that exports the dataset:
```python
# for IR
fo_splits = [f"TRAIN_by_sequence", f"VAL_by_sequence"]
# for RGB
fo_splits = [f"TRAIN_v0", f"VAL_v0"]

yolo_splits = ["train", "val"]

for fo_split, yolo_split in zip(fo_splits, yolo_splits):
    split: fo.DatasetView = dataset.match_tags(fo_split)
    split.export(
        export_dir="path/to/export/dir",
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth_train_det",
        classes=classes,
        split=yolo_split,
        export_media=True,
    )
```

> [!note]
> For more details on how the tags are defined, check the `fo_to_yolo.py` file in this same directory.

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
