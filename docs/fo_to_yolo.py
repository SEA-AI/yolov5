"""
Export a FiftyOne dataset to YOLO format with optional Weights & Biases registration.

Steps:
1. (Optional) Filter by FiftyOne sample tags.
2. Subsample to a target annotated/background ratio.
3. Split into train/val, put all samples in one split, or let the caller decide.
4. Apply a class-mapping YAML (unmapped labels are dropped).
5. (Optional) Redirect filepaths to 16-bit PNG images.
6. Export in YOLOv5 format.
7. (Optional) Upload dataset, label-distribution plot, class map, and
   parameters to the W&B Dataset Registry.

Usage:
python fo_to_yolo.py \\
  --dataset-name "TRAIN_RL_SPLIT_THERMAL_2024_03" \\
  --export-dir "/mnt/datasets/yolo" \\
  --class-map "data/class_map.yaml" \\
  --split-mode split \\
  --split-by sequence \\
  --noise-ratio 0.25 \\
  --val-ratio 0.2 \\
  --label-field ground_truth_det \\
  --use-16bit \\
  --wandb --wandb-entity my-org --wandb-collection my-collection \\
  --tags-suffix v2 \\
  --fo-tags tag_one tag_two \\
  --seed 42
"""

import argparse
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import yaml
from fiftyone import ViewField as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_yaml(yaml_file: str) -> dict:
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_dataset(dataset_name: str, fo_tags: list[str] | None = None) -> fo.DatasetView:
    if dataset_name not in fo.list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' not found in FiftyOne")
    dataset = fo.load_dataset(dataset_name)
    dataset = dataset.match_tags(fo_tags) if fo_tags else dataset
    if len(dataset) == 0:
        tag_info = f" with tags {fo_tags}" if fo_tags else ""
        raise ValueError(f"Dataset '{dataset_name}'{tag_info} is empty")
    return dataset


def subsample_dataset(
    dataset: fo.DatasetView,
    noise_ratio: float,
    label_field: str,
    seed: int,
) -> fo.DatasetView:
    """Keep all annotated samples and up to noise_ratio * n_annotated background samples."""
    annotated = dataset.exists(f"{label_field}.detections", True)
    noise = dataset.exists(f"{label_field}.detections", False)
    n_noise = min(len(noise), int(len(annotated) * noise_ratio))
    return annotated + noise.sort_by("uniqueness", reverse=True).take(n_noise, seed=seed)


def split_by_field(
    dataset: fo.DatasetView,
    field: str,
    val_ratio: float,
    tags_suffix: str,
) -> fo.DatasetView:
    """Tag samples TRAIN/VAL by distributing values of *field* across splits."""
    counts = dict(sorted(dataset.count_values(field).items(), key=lambda x: x[1], reverse=True))
    val_keys = list(counts.keys())[:: int(1 / val_ratio)]

    train_n = sum(v for k, v in counts.items() if k not in val_keys)
    val_n = sum(v for k, v in counts.items() if k in val_keys)
    total_n = train_n + val_n
    print(f"  train/val split: {train_n/total_n:.2f} / {val_n/total_n:.2f}")

    dataset.untag_samples(f"TRAIN_{tags_suffix}")
    dataset.untag_samples(f"VAL_{tags_suffix}")
    dataset.match(~F(field).is_in(val_keys)).tag_samples(f"TRAIN_{tags_suffix}")
    dataset.match(F(field).is_in(val_keys)).tag_samples(f"VAL_{tags_suffix}")
    return dataset


def split_random(
    dataset: fo.DatasetView,
    val_ratio: float,
    tags_suffix: str,
    seed: int,
) -> fo.DatasetView:
    """Tag samples TRAIN/VAL via a random split."""
    import random

    random.seed(seed)
    ids = list(dataset.values("id"))
    random.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    total_n = len(ids)
    print(f"  train/val split: {len(train_ids)/total_n:.2f} / {n_val/total_n:.2f}")

    dataset.untag_samples(f"TRAIN_{tags_suffix}")
    dataset.untag_samples(f"VAL_{tags_suffix}")
    dataset.select(train_ids).tag_samples(f"TRAIN_{tags_suffix}")
    dataset.select(val_ids).tag_samples(f"VAL_{tags_suffix}")
    return dataset


def apply_category_map(
    dataset: fo.DatasetView,
    class_map: dict,
    label_field: str,
) -> fo.DatasetView:
    """Drop labels not in class_map or mapped to 'None', then remap remaining labels."""
    keep_labels = [k for k, v in class_map.items() if v != "None"]
    dataset = dataset.filter_labels(
        label_field,
        F("label").is_in(keep_labels),
        only_matches=False,
    )
    dataset.keep()
    dataset.save()
    return dataset.map_labels(label_field, class_map)


def plot_label_distribution(counts_by_split: dict[str, dict], save_path: str) -> plt.Figure:
    """Bar chart of label counts per split, saved to *save_path*."""
    splits = list(counts_by_split.keys())
    all_labels = sorted(set().union(*[set(c.keys()) for c in counts_by_split.values()]))
    n_splits = len(splits)
    bar_width = 0.6 / n_splits

    fig, ax = plt.subplots(figsize=(16, 4))
    xaxis = np.arange(len(all_labels))
    for i, split in enumerate(splits):
        counts = [counts_by_split[split].get(lbl, 0) for lbl in all_labels]
        ax.bar(xaxis + i * bar_width, counts, label=split, width=bar_width)

    title = "Label distribution"
    if len(splits) > 1:
        title += " — " + " / ".join(splits) + " split"
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylabel("count")
    ax.set_xticks(xaxis + bar_width * (n_splits - 1) / 2)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def export_splits(
    dataset: fo.DatasetView,
    tags_suffix: str,
    export_dir: str,
    dataset_name: str,
    label_field: str,
    classes: list[str],
    split_mode: str,
    debug: bool,
) -> None:
    """Export one or both splits to YOLOv5 format."""
    if split_mode == "split":
        items = [(f"TRAIN_{tags_suffix}", "train"), (f"VAL_{tags_suffix}", "val")]
    elif split_mode == "train":
        items = [(f"TRAIN_{tags_suffix}", "train")]
    else:
        items = [(f"VAL_{tags_suffix}", "val")]

    for fo_tag, yolo_split in items:
        print(f"  exporting '{yolo_split}' split...")
        split = dataset.match_tags(fo_tag)
        if debug:
            split = split.take(100, seed=51)
        split.export(
            export_dir=str(Path(export_dir) / dataset_name),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=yolo_split,
            export_media=True,
        )

    # Patch dataset.yaml: replace the absolute 'path' with '.' so the dataset
    # is portable (works wherever the folder is placed or downloaded from W&B).
    yaml_path = Path(export_dir) / dataset_name / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data["path"] = "."
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def register_in_wandb(
    export_dir: str,
    dataset_name: str,
    params: dict,
    wandb_entity: str,
    wandb_collection: str,
) -> None:
    """Upload dataset artifact to the W&B Dataset Registry."""
    import wandb

    # A run is required by W&B to upload artifacts. We use a fixed internal
    # project so it never appears alongside training experiments.
    with wandb.init(
        entity=wandb_entity,
        project="dataset-registry",
        name=wandb_collection,
        config=params,
        job_type="dataset-upload",
    ) as run:
        artifact = wandb.Artifact(
            name=wandb_collection,
            type="dataset",
            description=params.get("description", ""),
            metadata=params,
        )
        # Dataset files (images, labels, dataset.yaml, class_map.yaml,
        # label_distribution.png, description.txt) are all inside out_dir.
        artifact.add_dir(str(Path(export_dir) / dataset_name))

        logged = run.log_artifact(artifact)
        logged.wait()
        run.link_artifact(
            logged,
            target_path=f"{wandb_entity}/wandb-registry-dataset/{wandb_collection}",
        )

    print(f"  dataset linked to registry collection '{wandb_collection}'")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def export_dataset(
    dataset_name: str,
    export_dir: str,
    class_map_path: str,
    description: str = "",
    split_mode: str = "split",
    split_by: str = "trip",
    noise_ratio: float | None = None,
    val_ratio: float = 0.2,
    use_16bit: bool = False,
    label_field: str = "ground_truth_det",
    register_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_collection: str | None = None,
    tags_suffix: str = "v0",
    fo_tags: list[str] | None = None,
    seed: int = 42,
    debug: bool = False,
) -> None:
    if register_wandb and not wandb_entity:
        raise ValueError("--wandb-entity is required when --wandb is set")
    wandb_collection = wandb_collection or dataset_name

    # Clean up stale tmp datasets from previous interrupted runs
    for d in fo.list_datasets():
        if d.startswith("tmp_"):
            fo.delete_dataset(d)

    out_dir = Path(export_dir) / dataset_name
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print(f"[1/6] Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, fo_tags)
    print(f"  {len(dataset)} samples" + (f" (filtered by tags {fo_tags})" if fo_tags else ""))

    # --- Uniqueness ---
    if not dataset.has_field("uniqueness"):
        print("[2/6] Computing uniqueness (first time only)...")
        fob.compute_uniqueness(dataset)
    else:
        print("[2/6] Uniqueness already computed, skipping.")

    # --- Subsample ---
    if noise_ratio is not None:
        print(f"[3/6] Subsampling (noise_ratio={noise_ratio})...")
        subset = subsample_dataset(dataset, noise_ratio, label_field, seed)
        print(f"  {len(subset)} samples after subsampling")
    else:
        print("[3/6] Skipping subsampling (using all samples).")
        subset = dataset

    # --- Split ---
    print(f"[4/6] Tagging splits (split_mode='{split_mode}')...")
    if split_mode == "split":
        existing_tags = set(subset.count_values("tags").keys())
        train_tag, val_tag = f"TRAIN_{tags_suffix}", f"VAL_{tags_suffix}"
        if train_tag in existing_tags and val_tag in existing_tags:
            n_train = len(subset.match_tags(train_tag))
            n_val = len(subset.match_tags(val_tag))
            total = n_train + n_val
            print(f"  using existing split tags '{train_tag}' / '{val_tag}'")
            print(f"  train/val split: {n_train/total:.2f} / {n_val/total:.2f}")
        elif split_by == "random":
            subset = split_random(subset, val_ratio, tags_suffix, seed)
        else:
            subset = split_by_field(subset, split_by, val_ratio, tags_suffix)
    else:
        tag = f"TRAIN_{tags_suffix}" if split_mode == "train" else f"VAL_{tags_suffix}"
        subset.untag_samples(tag)
        subset.tag_samples(tag)
        print(f"  all {len(subset)} samples tagged as '{split_mode}'")

    # Clone to a tmp dataset so we can mutate it (filepath swap, label filter)
    # without touching the original.
    tmp_name = f"tmp_{dataset_name}"
    try:
        export = subset.clone(tmp_name)
    except ValueError:
        print(f"  tmp dataset '{tmp_name}' already exists, reusing it")
        export = fo.load_dataset(tmp_name)

    # --- Category map ---
    print("[5/6] Applying category map...")
    class_map = read_yaml(class_map_path)
    export = apply_category_map(export, class_map, label_field)

    classes = export.distinct(f"{label_field}.detections.label")
    missing = set(class_map.values()) - set(classes) - {"None"}
    if missing:
        print(f"  WARNING: the following mapped classes have no samples: {missing}")
    print(f"  classes: {classes}")

    # --- 16-bit filepath swap ---
    if use_16bit:
        print("  switching filepaths to 16-bit PNG...")
        filepaths = [
            fp.replace("8Bit", "16Bit").replace("jpg", "png")
            for fp in export.values("filepath")
        ]
        export.set_values("filepath", filepaths)

    # --- Export ---
    print("[6/6] Exporting to YOLO format...")
    export_splits(export, tags_suffix, export_dir, dataset_name, label_field, classes, split_mode, debug)

    # --- Label distribution plot ---
    counts_by_split: dict[str, dict] = {}
    if split_mode in ("split", "train"):
        counts_by_split["train"] = export.match_tags(f"TRAIN_{tags_suffix}").count_values(
            f"{label_field}.detections.label"
        )
    if split_mode in ("split", "val"):
        counts_by_split["val"] = export.match_tags(f"VAL_{tags_suffix}").count_values(
            f"{label_field}.detections.label"
        )
    plot_path = str(out_dir / "label_distribution.png")
    fig = plot_label_distribution(counts_by_split, plot_path)
    plt.close(fig)
    print(f"  label distribution saved → {plot_path}")

    # --- Class map YAML ---
    class_map_yaml = out_dir / "class_map.yaml"
    with open(class_map_yaml, "w", encoding="utf-8") as f:
        yaml.dump(class_map, f, default_flow_style=False, allow_unicode=True)
    print(f"  class map saved → {class_map_yaml}")

    # --- Description ---
    if description:
        desc_path = out_dir / "description.txt"
        desc_path.write_text(description)
        print(f"  description saved → {desc_path}")

    # --- W&B registration ---
    if register_wandb:
        print("Registering in Weights & Biases...")
        params = {
            "description": description,
            "dataset_name": dataset_name,
            "split_mode": split_mode,
            "split_by": split_by if split_mode == "split" else None,
            "noise_ratio": noise_ratio,
            "val_ratio": val_ratio if split_mode == "split" else None,
            "use_16bit": use_16bit,
            "label_field": label_field,
            "fo_tags": fo_tags,
            "tags_suffix": tags_suffix,
            "seed": seed,
        }
        register_in_wandb(
            export_dir=export_dir,
            dataset_name=dataset_name,
            params=params,
            wandb_entity=wandb_entity,
            wandb_collection=wandb_collection,
        )

    fo.delete_dataset(tmp_name)
    print(f"\nDone. Dataset exported to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a FiftyOne dataset to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--dataset-name", required=True, help="FiftyOne dataset name")
    parser.add_argument("--export-dir", required=True, help="Local directory to write the exported dataset")
    parser.add_argument("--class-map", required=True, dest="class_map_path", help="Path to class-mapping YAML file")

    # Split behaviour
    parser.add_argument(
        "--split-mode",
        choices=["split", "train", "val"],
        default="split",
        help="'split' → export train+val; 'train' → all samples go to train; 'val' → all samples go to val",
    )
    parser.add_argument(
        "--split-by",
        default="trip",
        help="Metadata field used to assign train/val groups, or 'random' for a random split. Only used with --split-mode split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for validation. Only used with --split-mode split.",
    )

    # Sampling
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=None,
        help="Max ratio of background (unannotated) samples relative to annotated samples. Omit to use all samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible subsampling and splits.")

    # Export options
    parser.add_argument("--label-field", default="ground_truth_det", help="FiftyOne label field to export.")
    parser.add_argument("--use-16bit", action="store_true", help="Redirect filepaths to 16-bit PNG images instead of 8-bit JPEG.")
    parser.add_argument("--tags-suffix", default="v0", help="Suffix appended to TRAIN_/VAL_ sample tags in FiftyOne.")
    parser.add_argument("--fo-tags", nargs="+", default=None, metavar="TAG", help="Pre-filter the FiftyOne dataset to samples with these tags.")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", dest="register_wandb", help="Upload dataset to the W&B Dataset Registry.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (username or org). Required when --wandb is set.")
    parser.add_argument("--wandb-collection", default=None, help="Registry collection name. Defaults to --dataset-name if not set.")

    # Misc
    parser.add_argument("--debug", action="store_true", help="Export only 100 samples per split (for quick testing).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.description = input("Dataset description (press Enter to skip): ").strip()
    export_dataset(**vars(args))


if __name__ == "__main__":
    main()
