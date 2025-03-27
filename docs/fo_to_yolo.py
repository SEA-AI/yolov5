"""
Prepare a dataset for training a YOLOv5 model.
1. Subsample the dataset to have a ratio of noise samples to annotated samples.
2. Split the dataset into train and validation based on a field.
3. Apply a category map to the dataset.
4. Export the dataset to YOLO format.

Usage:
python fo_to_yolo.py \
  --dataset-name "TRAIN_RL_SPLIT_THERMAL_2024_03" \
  --export-dir "/mnt/datasets/yolo" \
  --class-map "../../../yolov5/data/class_map.yaml" \
  --split-by "sequence" \
  --noise-ratio 0.25 \
  --val-ratio 0.2 \
  --tags-suffix "v2" \
  --fo-tags "tag_one" "tag_two" \
  --debug
"""

import argparse
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
import yaml
from fiftyone import ViewField as F


def read_yaml(yaml_file: str) -> dict:
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def subsample_dataset(dataset: fo.DatasetView, noise_ratio: float) -> fo.DatasetView:
    """
    Subsample the dataset to have a ratio of noise samples to annotated samples.
    """
    annotated = dataset.exists("ground_truth_det.detections", True)
    noise = dataset.exists("ground_truth_det.detections", False)

    n_noise = min(len(noise), int(len(annotated) * noise_ratio))
    return annotated + noise.sort_by("uniqueness", reverse=True).limit(n_noise)


def split_by_field(
    dataset: fo.DatasetView, field: str, val_ratio: float, tags_suffix: str
) -> fo.DatasetView:
    """
    Split and tag the dataset into train and validation based on a field.
    """
    counts = dataset.count_values(field)

    # sort by count
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    # every nth key is a validation group, where n = 1/val_ratio
    val_keys = list(counts.keys())[:: int(1 / val_ratio)]

    train_n = sum(v for k, v in counts.items() if k not in val_keys)
    val_n = sum(v for k, v in counts.items() if k in val_keys)
    total_n = train_n + val_n
    print(f"Ratio train/val split: {train_n/total_n:.2f}:{val_n/total_n:.2f}")

    # untag samples with previous tags
    dataset.untag_samples(f"TRAIN_{tags_suffix}")
    dataset.untag_samples(f"VAL_{tags_suffix}")

    dataset.match(~F(field).is_in(val_keys)).tag_samples(f"TRAIN_{tags_suffix}")
    dataset.match(F(field).is_in(val_keys)).tag_samples(f"VAL_{tags_suffix}")
    return dataset


def apply_category_map(dataset: fo.DatasetView, class_map: dict) -> fo.DatasetView:
    """
    Apply a category map to the dataset.
    """
    not_none_class_map = {
        key: value for key, value in class_map.items() if value != "None"
    }

    dataset = dataset.filter_labels(
        "ground_truth_det",
        F("label").is_in(list(not_none_class_map.keys())),
        only_matches=False,
    )
    dataset.keep()
    dataset.save()

    return dataset.map_labels("ground_truth_det", class_map)


def export_as_yolo_dataset(
    dataset: fo.DatasetView,
    dataset_name: str,
    tags_suffix: str,
    export_dir: str,
    classes: list[str],
    debug: bool,
):
    """
    Export the dataset to YOLO format.
    """
    fo_splits = [f"TRAIN_{tags_suffix}", f"VAL_{tags_suffix}"]
    yolo_splits = ["train", "val"]

    for fo_split, yolo_split in zip(fo_splits, yolo_splits):
        print(f"📤 Exporting {yolo_split} dataset...")
        if debug:
            split: fo.DatasetView = dataset.match_tags(fo_split).take(
                100, seed=51
            )  # Tirar take para exportar tudo
        else:
            split: fo.DatasetView = dataset.match_tags(fo_split)
        split.export(
            export_dir=str(Path(export_dir) / dataset_name),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth_det",
            classes=classes,
            split=yolo_split,
            export_media=True,
        )


def load_dataset(dataset_name: str, fo_tags: list[str] = None) -> fo.DatasetView:
    """
    Load a dataset from FiftyOne.
    """
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' not found in FiftyOne")

    dataset = fo.load_dataset(dataset_name)
    dataset = dataset.match_tags(fo_tags) if fo_tags else dataset

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}'{f' with tags {fo_tags}' if fo_tags else ''} is empty"
        )

    return dataset


def export_dataset(
    dataset_name: str,
    export_dir: str = "/mnt/datasets/yolo",
    split_by: str = "sequence",
    class_map: str = "./yolov5/data/class_map.yaml",
    noise_ratio: float = 0.25,
    val_ratio: float = 0.2,
    tags_suffix: str = "v0",
    fo_tags: list[str] = None,
    debug: bool = False,
):
    """
    Prepare a dataset for training a YOLOv5 model.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to fetch and split.
    export_dir : str, optional
        Directory to export the training data, by default "/mnt/datasets/yolo".
    split_by : str, optional
        DB field to split the training data, by default "trip".
    class_map : str, optional
        Path to the class map, by default "./yolov5/data/class_map.yaml".
    noise_ratio : float, optional
        Ratio of noise samples to add to the training set, by default 0.25.
    val_ratio : float, optional
        Ratio of validation samples to add to the training set, by default 0.2.
    tags_suffix : str, optional
        Suffix to add to the "TRAIN" and "VAL" tags, by default "v0".
    fo_tags : str or list of str (separated by space), by default None
        Tags to filter the dataset in FiftyOne.
    debug : bool, optional
        Debug mode, by default False.
    """
    _ = [fo.delete_dataset(d) for d in fo.list_datasets() if d.startswith("tmp_")]
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    # make sure export_dir is empty
    shutil.rmtree((Path(export_dir) / dataset_name).as_posix(), ignore_errors=True)

    dataset = load_dataset(dataset_name, fo_tags)
    print(
        f"📡 Found {len(dataset)} samples in dataset {dataset_name}"
        + (f" with tags {fo_tags}" if fo_tags else "")
    )
    print("🔍 Computing uniqueness...")
    if not dataset.has_field("uniqueness"):
        fob.compute_uniqueness(dataset)

    print("🎯 Subsampling dataset...")
    subset = subsample_dataset(dataset, noise_ratio)

    print(f"⛵ Splitting train/val based on {split_by}...")
    subset = split_by_field(subset, split_by, val_ratio, tags_suffix)

    print("🏋️ Creating temporal training dataset...")
    tmp_name = f"tmp_{dataset_name}"
    try:
        export = subset.clone(tmp_name)
    except ValueError:
        print(f"Loading existing dataset {tmp_name}")
        export = fo.load_dataset(tmp_name)

    print("🗺️ Applying category map...")
    class_map = read_yaml(class_map)
    export = apply_category_map(export, class_map)

    classes = export.distinct("ground_truth_det.detections.label")
    missing_classes = set(class_map.values()) - set(classes) - {"None"}
    assert not missing_classes, f"Missing classes: {missing_classes}"
    print(f"🏷️ Classes:\n{classes}")

    print("📜 Replacing filepaths to point to raw data...")
    filepaths = export.values("filepath")
    filepaths = [fp.replace("8Bit", "16Bit").replace("jpg", "png") for fp in filepaths]
    export.set_values("filepath", filepaths)

    print("📤 Exporting YOLO dataset...")
    export_as_yolo_dataset(
        export,
        dataset_name,
        tags_suffix,
        export_dir,
        classes,
        debug,
    )

    print(f"🗑️ Delete temporal dataset {tmp_name}")
    fo.delete_dataset(tmp_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and split training data")
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset to fetch and split",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="/mnt/datasets/yolo",
        help="Directory to export the training data",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default="./yolov5/data/class_map.yaml",
        help="Path to the class map",
    )
    parser.add_argument(
        "--split-by",
        type=str,
        default="trip",
        help="DB field to split the training data",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.25,
        help="Ratio of noise samples to add to the training set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of validation samples to add to the training set",
    )
    parser.add_argument(
        "--tags-suffix",
        type=str,
        default="v0",
        help="Suffix to add to the tags",
    )

    parser.add_argument(
        "--fo-tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags to filter the dataset in FiftyOne (separated by space for multiple tags, by default None, ex --fo-tags tag1 tag2)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_dataset(**vars(args))


if __name__ == "__main__":
    main()