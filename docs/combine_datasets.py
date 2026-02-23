"""
Combine multiple exported YOLO datasets into a single dataset.yaml for training.

All source datasets must share the same class list (same names in the same order).
The combined YAML lists each dataset's train/val image directories as separate paths,
which YOLOv5 natively supports.

Each entry in --datasets can be either:
  - A local folder path  (e.g. /home/sea-ai/Documents/DATASET_A_v0)
  - A W&B artifact ref   (e.g. sea-ai/dataset-registry/DATASET_A:v0)

W&B artifact entries are downloaded automatically to --download-dir and are also
declared as lineage inputs in the uploaded artifact.

Usage (local paths):
python combine_datasets.py \\
  --datasets "/mnt/datasets/DATASET_A_v0" "/mnt/datasets/DATASET_B_v0" \\
  --output "/mnt/datasets/combined.yaml" \\       # optional; defaults to a temp file
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

  # --download-dir "/mnt/datasets"  → override where W&B artifacts are downloaded
  # --output "/mnt/datasets/combined.yaml"  → override where the combined YAML is written
  # --wandb-org sea-ai-org  → also links to the Dataset Registry
  # A description will be prompted interactively and is required to proceed.
"""

import argparse
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


def resolve_split_path(dataset_dir: Path, yaml_data: dict, split: str) -> str | None:
    """Resolve an absolute path for a split (train/val) from a dataset.yaml."""
    split_entry = yaml_data.get(split)
    if not split_entry:
        return None
    base = Path(yaml_data.get("path", "."))
    if not base.is_absolute():
        base = dataset_dir / base
    return str((base / split_entry).resolve())


def validate_classes(yamls: list[tuple[Path, dict]]) -> list[str]:
    """Ensure all datasets share the same class list and return it."""
    all_names = [(str(d), tuple(y["names"])) for d, y in yamls]
    unique = set(names for _, names in all_names)
    if len(unique) > 1:
        lines = "\n".join(f"  {d}: {list(names)}" for d, names in all_names)
        raise ValueError(f"Datasets have inconsistent class lists:\n{lines}")
    return list(yamls[0][1]["names"])


def download_artifact(ref: str, download_dir: str) -> str:
    """Download a W&B artifact and return its local path."""
    import wandb

    api = wandb.Api()
    artifact = api.artifact(ref)
    artifact_name = ref.split("/")[-1].split(":")[0]
    dest = str(Path(download_dir) / artifact_name)
    artifact.download(root=dest)
    return dest


# ---------------------------------------------------------------------------
# W&B registration
# ---------------------------------------------------------------------------


def register_in_wandb(
    yaml_path: Path,
    description: str,
    dataset_dirs: list[str],
    classes: list[str],
    wandb_refs: list[str],
    wandb_entity: str,
    wandb_collection: str,
    wandb_org: str | None,
) -> None:
    """Upload combined YAML to W&B. W&B artifact entries are declared as lineage inputs."""
    import wandb

    with wandb.init(
        entity=wandb_entity,
        project="dataset-registry",
        name=wandb_collection,
        job_type="dataset-combine",
        config={
            "description": description,
            "datasets": dataset_dirs,
            "classes": classes,
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
            metadata={"datasets": dataset_dirs, "classes": classes},
        )
        artifact.add_file(str(yaml_path))

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
    output: str | None = None,
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

    # Resolve output path: default to <download_dir>/combined.yaml or cwd
    if not output:
        base = download_dir or tempfile.mkdtemp()
        output = str(Path(base) / "combined.yaml")
        print(f"No --output set, writing to: {output}")

    wandb_collection = wandb_collection or Path(output).stem

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

    # --- Build combined paths ---
    train_paths, val_paths = [], []
    for dataset_dir, y in yamls:
        t = resolve_split_path(dataset_dir, y, "train")
        v = resolve_split_path(dataset_dir, y, "val")
        if t:
            train_paths.append(t)
        if v:
            val_paths.append(v)

    combined: dict = {"path": ".", "nc": len(classes), "names": classes}
    if train_paths:
        combined["train"] = train_paths
    if val_paths:
        combined["val"] = val_paths

    # --- Write combined YAML ---
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(combined, f, default_flow_style=False, allow_unicode=True)

    print(f"\nCombined dataset.yaml saved → {output_path}")
    if train_paths:
        print(f"  train splits : {len(train_paths)}")
    if val_paths:
        print(f"  val splits   : {len(val_paths)}")
    print(f"  classes      : {classes}")
    print(f"\nTrain with: python train.py --data {output_path}")

    if description:
        output_path.with_suffix(".description.txt").write_text(description)

    # --- W&B ---
    if register_wandb:
        print("\nRegistering in Weights & Biases...")
        register_in_wandb(
            yaml_path=output_path,
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
        description="Combine multiple exported YOLO datasets into a single dataset.yaml.",
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
        "--output",
        default=None,
        help="Path to write the combined dataset.yaml. Defaults to <download-dir>/combined.yaml.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory to download W&B artifacts into. Defaults to a temp directory if any entry in --datasets is a W&B artifact ref.",
    )

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", dest="register_wandb", help="Upload combined YAML to W&B.")
    parser.add_argument("--wandb-entity", default=None, help="W&B team entity. Required when --wandb is set.")
    parser.add_argument("--wandb-collection", default=None, help="Artifact name. Defaults to the output filename stem.")
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
