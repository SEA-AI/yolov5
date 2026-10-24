"""
Combine multiple exported YOLO datasets into a single self-contained dataset
that can be uploaded to W&B and used for training on any machine.

All source datasets must share the same class list (same names in the same order).
Images and labels from all source datasets are copied into a single output directory.
The combined dataset.yaml always uses path: . with relative train/val paths.

Each entry in --datasets can be either:
  - A local folder path  (e.g. /home/sea-ai/Documents/DATASET_A_v0)
  - A W&B artifact ref   (e.g. sea-ai/dataset-registry/DATASET_A:v0)

W&B artifact entries are downloaded automatically to --download-dir and are also
declared as lineage inputs in the uploaded artifact.

Usage (local paths):
python combine_datasets.py \\
  --datasets "/mnt/datasets/DATASET_A_v0" "/mnt/datasets/DATASET_B_v0" \\
  --output-dir "/mnt/datasets/COMBINED_v0" \\
  --wandb --wandb-entity sea-ai --wandb-collection "my-combined-dataset"

Usage (W&B artifacts — no local setup needed):
python combine_datasets.py \\
  --datasets "sea-ai/dataset-registry/DATASET_A:v0" "sea-ai/dataset-registry/DATASET_B:v1" \\
  --wandb --wandb-entity sea-ai --wandb-collection "my-combined-dataset"
  # artifacts are downloaded to a /tmp temp dir, W&B cache skipped

Usage (mixed):
python combine_datasets.py \\
  --datasets "/mnt/datasets/DATASET_A_v0" "sea-ai/dataset-registry/DATASET_B:v1" \\
  --wandb --wandb-entity sea-ai --wandb-collection "my-combined-dataset"

  # --output-dir "/mnt/datasets/COMBINED"  → where the combined dataset is written (default: temp dir)
  # --download-dir "/mnt/datasets"  → where W&B artifacts are downloaded (default: /tmp, cache skipped)
  # --wandb-org sea-ai-org  → also links to the Dataset Registry
  # W&B versions artifacts by content hash: a new version is only created when the combined files differ from the previous upload
  # A description will be prompted interactively and is required to proceed.
"""

import argparse
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

from utils.dataset_utils import log_dataset, validate_split

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_wandb_ref(entry: str) -> bool:
    """Return True if entry looks like a W&B artifact ref (entity/project/name:version)."""
    return not Path(entry).exists()


def names_as_list(names) -> list[str]:
    """Normalise names to a flat list for comparison, handling both list and dict formats."""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys())]
    return [str(n) for n in names]


def validate_classes(yamls: list[tuple[Path, dict]]) -> list[str]:
    """Ensure all datasets share the same class list. Returns the raw names from the first yaml."""
    all_names = [(str(d), tuple(names_as_list(y["names"]))) for d, y in yamls]
    unique = set(names for _, names in all_names)
    if len(unique) > 1:
        lines = "\n".join(f"  {d}: {list(names)}" for d, names in all_names)
        raise ValueError(f"Datasets have inconsistent class lists:\n{lines}")
    return yamls[0][1]["names"]  # return as-is from the source yaml


def download_artifact(ref: str, download_dir: str | None, run=None) -> tuple[str, str | None]:
    """Download a W&B artifact (through *run* for lineage, if given) and return (local_path, tmp_dir_to_cleanup).

    tmp_dir_to_cleanup is None when *download_dir* is given.
    """
    artifact_name = ref.split("/")[-1].split(":")[0]
    if run is not None:
        artifact = run.use_artifact(ref)
    else:
        import wandb

        artifact = wandb.Api().artifact(ref)

    if download_dir is None:
        tmp_parent = tempfile.mkdtemp(dir="/tmp", prefix="wandb_")
        dest = str(Path(tmp_parent) / artifact_name)
        artifact.download(root=dest, skip_cache=True)
        return dest, tmp_parent

    dest = str(Path(download_dir) / artifact_name)
    artifact.download(root=dest, skip_cache=True)
    return dest, None



def copy_split(
    dataset_dir: Path,
    yaml_data: dict,
    split: str,
    output_dir: Path,
    prefix: str,
) -> bool:
    """Copy images and labels for one split into the combined output directory.

    Files are prefixed with *prefix* (e.g. 'd0') to avoid name collisions.
    Returns True if the split existed and files were copied, False otherwise.
    """
    split_entry = yaml_data.get(split)
    if not split_entry:
        return False

    base = Path(yaml_data.get("path", "."))
    if not base.is_absolute():
        base = (dataset_dir / base).resolve()

    img_src = base / split_entry
    if not img_src.exists():
        print(f"  WARNING: image directory not found: {img_src}")
        return False

    # YOLO convention: labels mirror the images path with 'images' replaced by 'labels'
    lbl_src = Path(str(img_src).replace("/images/", "/labels/"))

    img_dst = output_dir / "images" / split
    lbl_dst = output_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_file in sorted(img_src.iterdir()):
        if not img_file.is_file():
            continue
        new_stem = f"{prefix}_{img_file.stem}"
        shutil.copy2(img_file, img_dst / f"{new_stem}{img_file.suffix}")
        lbl_file = lbl_src / f"{img_file.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, lbl_dst / f"{new_stem}.txt")
        else:
            (lbl_dst / f"{new_stem}.txt").touch()  # empty = background sample
        copied += 1

    print(f"    {copied} {split} samples copied (prefix: {prefix})")
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def combine_datasets(
    datasets: list[str],
    output_dir: str | None = None,
    description: str = "",
    download_dir: str | None = None,
    register_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_collection: str | None = None,
    wandb_org: str | None = None,
) -> None:
    if register_wandb and not wandb_entity:
        raise ValueError("--wandb-entity is required when --wandb is set.")

    # Resolve output directory
    if not output_dir:
        output_dir = tempfile.mkdtemp()
        print(f"No --output-dir set, using temp dir: {output_dir}")

    wandb_collection = wandb_collection or Path(output_dir).name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        run = None
        if register_wandb:
            import wandb

            # Download sources through the run so lineage records the same version.
            run = stack.enter_context(
                wandb.init(
                    entity=wandb_entity,
                    project="dataset-registry",
                    name=wandb_collection,
                    job_type="dataset-combine",
                    config={"description": description, "source_artifacts": [d for d in datasets if is_wandb_ref(d)]},
                )
            )

        # --- Download, copy, and immediately cleanup each dataset in turn ---
        classes = None
        has_train, has_val = False, False

        print(f"\nProcessing {len(datasets)} dataset(s)...")
        for i, entry in enumerate(datasets):
            if is_wandb_ref(entry):
                if not download_dir:
                    print(f"  No --download-dir set, downloading to /tmp.")
                print(f"  [{i}] Downloading artifact '{entry}'...")
                local_path, tmp_parent = download_artifact(entry, download_dir, run)
                print(f"      → {local_path}")
            else:
                local_path, tmp_parent = entry, None

            yaml_path = Path(local_path) / "dataset.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"dataset.yaml not found in '{local_path}'")
            y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            print(f"  [{i}] loaded {yaml_path}")

            # Validate classes incrementally against the first dataset
            dataset_names = names_as_list(y["names"])
            if classes is None:
                classes = y["names"]
            elif tuple(names_as_list(classes)) != tuple(dataset_names):
                raise ValueError(
                    f"Class mismatch in dataset {i} ('{local_path}'):\n"
                    f"  expected : {names_as_list(classes)}\n"
                    f"  got      : {dataset_names}"
                )

            prefix = f"d{i}"
            if copy_split(Path(local_path), y, "train", output_path, prefix):
                has_train = True
            if copy_split(Path(local_path), y, "val", output_path, prefix):
                has_val = True

            if tmp_parent:
                shutil.rmtree(tmp_parent, ignore_errors=True)
                print(f"  [{i}] temp download dir removed")

        # --- Validate ---
        print("  validating combined dataset...")
        if has_train:
            validate_split(output_path, "train")
        if has_val:
            validate_split(output_path, "val")

        # --- Write combined dataset.yaml ---
        combined: dict = {"path": ".", "names": classes}
        if has_train:
            combined["train"] = "./images/train/"
        if has_val:
            combined["val"] = "./images/val/"

        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(combined, f, default_flow_style=False, allow_unicode=True)

        print(f"\nCombined dataset written → {output_dir}")
        if has_train:
            n_train = len(list((output_path / "images" / "train").iterdir()))
            print(f"  train images : {n_train}")
        if has_val:
            n_val = len(list((output_path / "images" / "val").iterdir()))
            print(f"  val images   : {n_val}")
        print(f"  classes      : {names_as_list(classes)}")
        print(f"\nTrain with: python train.py --data {yaml_path}")

        # --- Description ---
        if description:
            (output_path / "description.txt").write_text(description)

        # --- W&B ---
        if run is not None:
            print("\nRegistering in Weights & Biases...")
            metadata = {"datasets": datasets, "classes": names_as_list(classes)}
            run.config.update(metadata)
            log_dataset(run, output_path, wandb_collection, description, metadata, wandb_org)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple exported YOLO datasets into a single self-contained dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="PATH_OR_ARTIFACT",
        help=(
            "Local folder paths or W&B artifact references "
            "(e.g. sea-ai/dataset-registry/DATASET_A:v0). Can be mixed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directory to write the combined dataset. Defaults to a temp directory.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory to download W&B artifacts into. Defaults to a /tmp temp directory (W&B cache skipped).",
    )

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", dest="register_wandb", help="Upload combined dataset to W&B.")
    parser.add_argument("--wandb-entity", default=None, help="W&B team entity. Required when --wandb is set.")
    parser.add_argument("--wandb-collection", default=None, help="Artifact name. Defaults to the output directory name.")
    parser.add_argument("--wandb-org", default=None, help="W&B organization entity for Dataset Registry linking.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.description = input("Description (required): ").strip()
    if not args.description:
        print("Error: a description is required. Aborting.")
        return
    combine_datasets(**vars(args))


if __name__ == "__main__":
    main()
