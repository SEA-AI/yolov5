"""
Export a FiftyOne dataset to YOLO format with optional Weights & Biases registration.

Steps:
1. (Optional) Filter by FiftyOne sample tags.
2. (Optional) Subsample to a target annotated/background ratio; omit to use all samples.
3. Split into train/val, put all samples in one split, or use existing split tags.
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
  --split-mode split \\       # split | train | val
  --split-by sequence \\      # field name | random (ignored if split tags already exist)
  --noise-ratio 0.25 \\       # omit to use all samples
  --val-ratio 0.2 \\
  --label-field ground_truth_det \\
  --use-16bit \\
  --wandb --wandb-entity sea-ai --wandb-org sea-ai-org --wandb-collection my-collection \\  # omit --wandb to skip upload entirely
                                                                                          # omit --wandb-org to upload artifact but skip registry linking (it will not great a dataset in the registry, but the artifact will still be available in the project and can be linked manually later)
                                                                                          # omit --wandb-collection to use --dataset-name as the artifact name
  --tags-suffix v0 \\   #dataset version suffix for the TRAIN_/VAL_ tags (e.g. v0, v1, etc.); only relevant if --split-mode is split
  --fo-tags tag_one tag_two \\   # optional pre-filtering by FiftyOne sample tags; omit to use all samples in the dataset
  --seed 42
  # A description will be prompted interactively and is required to proceed.
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fiftyone as fo
import fiftyone.brain as fob
import matplotlib
from utils.general import LOGGER

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import yaml
from fiftyone import ViewField as F
from utils.dataset_utils import validate_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
) -> fo.DatasetView:
    """Keep all annotated samples and up to noise_ratio * n_annotated background samples."""
    annotated = dataset.exists(f"{label_field}.detections", True)
    noise = dataset.exists(f"{label_field}.detections", False)
    n_noise = min(len(noise), int(len(annotated) * noise_ratio))
    return annotated + noise.sort_by("uniqueness", reverse=True).limit(n_noise)


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
    LOGGER.info(f"  train/val split: {train_n/total_n:.2f} / {val_n/total_n:.2f}")

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
    LOGGER.info(f"  train/val split: {len(train_ids)/total_n:.2f} / {n_val/total_n:.2f}")

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
    """Drop labels not in class_map or mapped to 'None', then remap remaining labels.

    Returns a lazy view — the original dataset is never mutated.
    """
    keep_labels = [k for k, v in class_map.items() if v != "None"]
    return (
        dataset
        .filter_labels(label_field, F("label").is_in(keep_labels), only_matches=False)
        .map_labels(label_field, class_map)
    )



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
        LOGGER.info(f"  exporting '{yolo_split}' split...")
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
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def register_in_wandb(
    export_dir: str,
    dataset_name: str,
    params: dict,
    wandb_entity: str,
    wandb_collection: str,
    wandb_org: str | None = None,
) -> None:
    """Upload dataset artifact to W&B. If wandb_org is provided, also link to the Dataset Registry."""
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
        LOGGER.info(f"  artifact '{wandb_collection}' uploaded to project '{wandb_entity}/dataset-registry'")

        if wandb_org:
            run.link_artifact(
                logged,
                target_path=f"{wandb_org}/wandb-registry-dataset/{wandb_collection}",
            )
            LOGGER.info(f"  artifact linked to registry collection '{wandb_org}/{wandb_collection}'")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    dataset_name: str
    export_dir: str
    class_map_path: str
    description: str = ""
    split_mode: Literal["split", "train", "val"] = "split"
    split_by: str = "trip"
    noise_ratio: float | None = None
    val_ratio: float = 0.2
    use_16bit: bool = False
    label_field: str = "ground_truth_det"
    register_wandb: bool = False
    wandb_entity: str | None = None
    wandb_collection: str | None = None
    wandb_org: str | None = None
    tags_suffix: str = "v0"
    fo_tags: list[str] | None = None
    seed: int = 42
    debug: bool = False

    def __post_init__(self) -> None:
        valid_modes = {"split", "train", "val"}
        if self.split_mode not in valid_modes:
            raise ValueError(f"split_mode must be one of {valid_modes}, got '{self.split_mode}'")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def export_dataset(cfg: ExportConfig) -> None:
    if cfg.register_wandb and not cfg.wandb_entity:
        raise ValueError("--wandb-entity is required when --wandb is set")
    wandb_collection = cfg.wandb_collection or cfg.dataset_name

    folder_name = f"{cfg.dataset_name}_{cfg.tags_suffix}"
    out_dir = Path(cfg.export_dir) / folder_name
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    LOGGER.info(f"[1/6] Loading dataset '{cfg.dataset_name}'...")
    dataset = load_dataset(cfg.dataset_name, cfg.fo_tags)
    LOGGER.info(f"  {len(dataset)} samples" + (f" (filtered by tags {cfg.fo_tags})" if cfg.fo_tags else ""))

    # --- Uniqueness ---
    if cfg.noise_ratio is not None:
        if not dataset.has_field("uniqueness"):
            LOGGER.info("[2/6] Computing uniqueness (first time only)...")
            fob.compute_uniqueness(dataset)
        else:
            LOGGER.info("[2/6] Uniqueness already computed, skipping.")
    else:
        LOGGER.info("[2/6] Skipping uniqueness (no subsampling requested).")

    # --- Subsample ---
    if cfg.noise_ratio is not None:
        LOGGER.info(f"[3/6] Subsampling (noise_ratio={cfg.noise_ratio})...")
        subset = subsample_dataset(dataset, cfg.noise_ratio, cfg.label_field)
        LOGGER.info(f"  {len(subset)} samples after subsampling")
    else:
        LOGGER.info("[3/6] Skipping subsampling (using all samples).")
        subset = dataset

    # --- Split ---
    LOGGER.info(f"[4/6] Tagging splits (split_mode='{cfg.split_mode}')...")
    if cfg.split_mode == "split":
        existing_tags = set(subset.count_values("tags").keys())
        train_tag, val_tag = f"TRAIN_{cfg.tags_suffix}", f"VAL_{cfg.tags_suffix}"
        if train_tag in existing_tags and val_tag in existing_tags:
            n_train = len(subset.match_tags(train_tag))
            n_val = len(subset.match_tags(val_tag))
            total = n_train + n_val
            LOGGER.info(f"  using existing split tags '{train_tag}' / '{val_tag}'")
            LOGGER.info(f"  train/val split: {n_train/total:.2f} / {n_val/total:.2f}")
        elif cfg.split_by == "random":
            subset = split_random(subset, cfg.val_ratio, cfg.tags_suffix, cfg.seed)
        else:
            subset = split_by_field(subset, cfg.split_by, cfg.val_ratio, cfg.tags_suffix)
    else:
        tag = f"TRAIN_{cfg.tags_suffix}" if cfg.split_mode == "train" else f"VAL_{cfg.tags_suffix}"
        subset.untag_samples(tag)
        subset.tag_samples(tag)
        LOGGER.info(f"  all {len(subset)} samples tagged as '{cfg.split_mode}'")

    # --- Category map ---
    LOGGER.info("[5/6] Applying category map...")
    class_map = yaml.safe_load(Path(cfg.class_map_path).read_text(encoding="utf-8"))
    export = apply_category_map(subset, class_map, cfg.label_field)

    classes = export.distinct(f"{cfg.label_field}.detections.label")
    missing = set(class_map.values()) - set(classes) - {"None"}
    if missing:
        LOGGER.warning(f"  the following mapped classes have no samples: {missing}")
    LOGGER.info(f"  classes: {classes}")

    # --- 16-bit filepath swap ---
    # set_values mutates a dataset, so we clone into a tmp dataset only when needed.
    tmp_name = f"tmp_{cfg.dataset_name}"
    if cfg.use_16bit:
        LOGGER.info("  switching filepaths to 16-bit PNG...")
        export = export.clone(tmp_name)
        filepaths = [
            fp.replace("8Bit", "16Bit").replace("jpg", "png")
            for fp in export.values("filepath")
        ]
        export.set_values("filepath", filepaths)

    # --- Export ---
    LOGGER.info("[6/6] Exporting to YOLO format...")
    export_splits(export, cfg.tags_suffix, cfg.export_dir, folder_name, cfg.label_field, classes, cfg.split_mode, cfg.debug)

    # --- Validate ---
    LOGGER.info("  validating export...")
    if cfg.split_mode in ("split", "train"):
        validate_split(out_dir, "train")
    if cfg.split_mode in ("split", "val"):
        validate_split(out_dir, "val")

    # --- Label distribution plot ---
    counts_by_split: dict[str, dict] = {}
    if cfg.split_mode in ("split", "train"):
        counts_by_split["train"] = export.match_tags(f"TRAIN_{cfg.tags_suffix}").count_values(
            f"{cfg.label_field}.detections.label"
        )
    if cfg.split_mode in ("split", "val"):
        counts_by_split["val"] = export.match_tags(f"VAL_{cfg.tags_suffix}").count_values(
            f"{cfg.label_field}.detections.label"
        )
    plot_path = str(out_dir / "label_distribution.png")
    fig = plot_label_distribution(counts_by_split, plot_path)
    plt.close(fig)
    LOGGER.info(f"  label distribution saved → {plot_path}")

    # --- Class map YAML ---
    class_map_yaml = out_dir / "class_map.yaml"
    with open(class_map_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(class_map, f, default_flow_style=False, allow_unicode=True)
    LOGGER.info(f"  class map saved → {class_map_yaml}")

    # --- Description ---
    if cfg.description:
        desc_path = out_dir / "description.txt"
        desc_path.write_text(cfg.description)
        LOGGER.info(f"  description saved → {desc_path}")

    # --- W&B registration ---
    if cfg.register_wandb:
        LOGGER.info("Registering in Weights & Biases...")
        params = {
            "description": cfg.description,
            "dataset_name": cfg.dataset_name,
            "split_mode": cfg.split_mode,
            "split_by": cfg.split_by if cfg.split_mode == "split" else None,
            "noise_ratio": cfg.noise_ratio,
            "val_ratio": cfg.val_ratio if cfg.split_mode == "split" else None,
            "use_16bit": cfg.use_16bit,
            "label_field": cfg.label_field,
            "fo_tags": cfg.fo_tags,
            "tags_suffix": cfg.tags_suffix,
            "seed": cfg.seed,
        }
        register_in_wandb(
            export_dir=cfg.export_dir,
            dataset_name=folder_name,
            params=params,
            wandb_entity=cfg.wandb_entity,
            wandb_collection=wandb_collection,
            wandb_org=cfg.wandb_org,
        )

    if cfg.use_16bit:
        fo.delete_dataset(tmp_name)
    LOGGER.info(f"\nDone. Dataset exported to: {out_dir}")


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
    parser.add_argument("--wandb-entity", default=None, help="W&B team entity. Required when --wandb is set.")
    parser.add_argument("--wandb-collection", default=None, help="Artifact/collection name. Defaults to --dataset-name if not set.")
    parser.add_argument("--wandb-org", default=None, help="W&B organization entity for Dataset Registry linking. If omitted, artifact is uploaded to the project but not linked to the registry.")

    # Misc
    parser.add_argument("--debug", action="store_true", help="Export only 100 samples per split (for quick testing).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    description = input("Dataset description (required): ").strip()
    if not description:
        LOGGER.error("Error: a description is required. Aborting.")
        return
    export_dataset(ExportConfig(**vars(args), description=description))


if __name__ == "__main__":
    main()
