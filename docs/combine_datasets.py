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
  # artifacts are downloaded to a temp dir automatically

Usage (mixed):
python combine_datasets.py \\
  --datasets "/mnt/datasets/DATASET_A_v0" "sea-ai/dataset-registry/DATASET_B:v1" \\
  --wandb --wandb-entity sea-ai --wandb-collection "my-combined-dataset"

  # --output-dir "/mnt/datasets/COMBINED"  → where the combined dataset is written (default: temp dir)
  # --download-dir "/mnt/datasets"  → where W&B artifacts are downloaded (default: temp dir)
  # --wandb-org sea-ai-org  → also links to the Dataset Registry
  # A description will be prompted interactively and is required to proceed.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def download_artifact(ref: str, download_dir: str) -> str:
    """Download a W&B artifact and return its local path."""
    import wandb

    api = wandb.Api()
    artifact = api.artifact(ref)
    artifact_name = ref.split("/")[-1].split(":")[0]
    dest = str(Path(download_dir) / artifact_name)
    artifact.download(root=dest)
    return dest


def validate_split(out_dir: Path, split: str) -> None:
    """Raise if a split directory is missing, empty, or has mismatched image/label counts."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing directory for split '{split}': expected {img_dir} and {lbl_dir}")

    image_exts = {".jpg", ".jpeg", ".png"}
    n_images = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in image_exts)
    n_labels = sum(1 for f in lbl_dir.iterdir() if f.suffix == ".txt")

    if n_images == 0:
        raise ValueError(f"No images found in '{img_dir}'")
    if n_labels == 0:
        raise ValueError(f"No label files found in '{lbl_dir}'")
    if n_images != n_labels:
        raise ValueError(
            f"Image/label count mismatch in split '{split}': {n_images} images vs {n_labels} labels"
        )
    print(f"  {split}: {n_images} images, {n_labels} labels — OK")


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
# W&B registration
# ---------------------------------------------------------------------------


def register_in_wandb(
    output_dir: Path,
    description: str,
    dataset_dirs: list[str],
    classes: list[str],
    wandb_refs: list[str],
    wandb_entity: str,
    wandb_collection: str,
    wandb_org: str | None,
) -> None:
    """Upload combined dataset directory to W&B. W&B artifact entries are declared as lineage inputs."""
    import wandb

    with wandb.init(
        entity=wandb_entity,
        project="dataset-registry",
        name=wandb_collection,
        job_type="dataset-combine",
        config={
            "description": description,
            "datasets": dataset_dirs,
            "classes": names_as_list(classes),
            "source_artifacts": wandb_refs,
        },
    ) as run:
        if wandb_refs:
            print("  linking source artifacts for lineage...")
            for ref in wandb_refs:
                run.use_artifact(ref)

        artifact = wandb.Artifact(
            name=wandb_collection,
            type="dataset",
            description=description,
            metadata={"datasets": dataset_dirs, "classes": names_as_list(classes)},
        )
        artifact.add_dir(str(output_dir))

        logged = run.log_artifact(artifact)
        logged.wait()
        print(f"  artifact '{wandb_collection}' uploaded to '{wandb_entity}/dataset-registry'")

        if wandb_org:
            run.link_artifact(
                logged,
                target_path=f"{wandb_org}/wandb-registry-dataset/{wandb_collection}",
            )
            print(f"  artifact linked to registry '{wandb_org}/{wandb_collection}'")


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

    # --- Resolve entries: download W&B refs, keep local paths as-is ---
    local_dirs: list[str] = []
    wandb_refs: list[str] = []

    for entry in datasets:
        if is_wandb_ref(entry):
            if not download_dir:
                download_dir = tempfile.mkdtemp()
                print(f"No --download-dir set, using temp dir: {download_dir}")
            print(f"Downloading artifact '{entry}'...")
            local_path = download_artifact(entry, download_dir)
            print(f"  → {local_path}")
            local_dirs.append(local_path)
            wandb_refs.append(entry)
        else:
            local_dirs.append(entry)

    # Resolve output directory
    if not output_dir:
        output_dir = tempfile.mkdtemp()
        print(f"No --output-dir set, using temp dir: {output_dir}")

    wandb_collection = wandb_collection or Path(output_dir).name

    # --- Read and validate ---
    print(f"\nReading {len(local_dirs)} dataset(s)...")
    yamls: list[tuple[Path, dict]] = []
    for d in local_dirs:
        yaml_path = Path(d) / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"dataset.yaml not found in '{d}'")
        yamls.append((Path(d), read_yaml(yaml_path)))
        print(f"  loaded {yaml_path}")

    classes = validate_classes(yamls)
    print(f"  classes consistent across all datasets: {classes}")

    # --- Copy images and labels ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    has_train, has_val = False, False
    print(f"\nCopying dataset files to {output_dir}...")
    for i, (dataset_dir, y) in enumerate(yamls):
        prefix = f"d{i}"
        print(f"  dataset {i}: {dataset_dir}")
        if copy_split(dataset_dir, y, "train", output_path, prefix):
            has_train = True
        if copy_split(dataset_dir, y, "val", output_path, prefix):
            has_val = True

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
    if register_wandb:
        print("\nRegistering in Weights & Biases...")
        register_in_wandb(
            output_dir=output_path,
            description=description,
            dataset_dirs=local_dirs,
            classes=classes,
            wandb_refs=wandb_refs,
            wandb_entity=wandb_entity,
            wandb_collection=wandb_collection,
            wandb_org=wandb_org,
        )

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
        help="Directory to download W&B artifacts into. Defaults to a temp directory if any entry in --datasets is a W&B artifact ref.",
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
