import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from utils import LOGGER

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def validate_split(out_dir: Path, split: str) -> None:
    """Raise if a split directory is missing, empty, or has mismatched image/label counts."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing directory for split '{split}': expected {img_dir} and {lbl_dir}")

    n_images = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    n_labels = sum(1 for f in lbl_dir.iterdir() if f.suffix == ".txt")

    if n_images == 0:
        raise ValueError(f"No images found in '{img_dir}'")
    if n_labels == 0:
        raise ValueError(f"No label files found in '{lbl_dir}'")
    if n_images != n_labels:
        raise ValueError(
            f"Image/label count mismatch in split '{split}': {n_images} images vs {n_labels} labels"
        )
    LOGGER.info(f"  {split}: {n_images} images, {n_labels} labels — OK")


@contextmanager
def wandb_isolated_dirs():
    """Context manager that redirects W&B cache and data dirs to temp directories.

    Prevents ~/.cache/wandb/artifacts and ~/.local/share/wandb/artifacts from
    accumulating large dataset files after each upload. Both temp dirs are
    deleted when the context exits, and the original env vars are restored.
    """
    prev_cache = os.environ.get("WANDB_CACHE_DIR")
    prev_data = os.environ.get("WANDB_DATA_DIR")
    tmp_dir = tempfile.mkdtemp(prefix="wandb_tmp_")
    os.environ["WANDB_CACHE_DIR"] = tmp_dir
    os.environ["WANDB_DATA_DIR"] = tmp_dir
    try:
        yield
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if prev_cache is not None:
            os.environ["WANDB_CACHE_DIR"] = prev_cache
        else:
            os.environ.pop("WANDB_CACHE_DIR", None)
        if prev_data is not None:
            os.environ["WANDB_DATA_DIR"] = prev_data
        else:
            os.environ.pop("WANDB_DATA_DIR", None)


def log_dataset(run, folder: Path, name: str, description: str, metadata: dict, wandb_org: str | None = None) -> None:
    """Upload *folder* as a dataset artifact in *run*; if *wandb_org* is set, also link it to the Dataset Registry."""
    import wandb

    artifact = wandb.Artifact(name=name, type="dataset", description=description, metadata=metadata)
    # Skip W&B's staging/cache copies (slow for large datasets).
    # Safe because nothing writes to the folder during upload.
    artifact.add_dir(str(folder), skip_cache=True, policy="immutable")
    logged = run.log_artifact(artifact)
    logged.wait()
    LOGGER.info(f"  artifact '{name}' uploaded to '{run.entity}/{run.project}'")

    if wandb_org:
        try:
            run.link_artifact(logged, target_path=f"{wandb_org}/wandb-registry-dataset/{name}")
            LOGGER.info(f"  artifact linked to registry collection '{wandb_org}/{name}'")
        except wandb.errors.CommError as e:
            if "Duplicate entry" not in str(e):
                raise
            LOGGER.warning(
                f"  artifact version already linked to registry collection '{wandb_org}/{name}' "
                f"(content unchanged since last upload — no new version created)"
            )
