"""
Train horizon model on RGB or IR16bit images.

Example:
    CUDA_VISIBLE_DEVICES=1 python3 horizon/train.py \
        --dataset_name "TRAIN_RL_SPLIT_THERMAL_2024_03" \
        --train_tag "TRAIN_per_sequence" \
        --val_tag "VAL_per_sequence" \
        --device 0
"""

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import fiftyone as fo
import numpy as np
import torch
import wandb
from fiftyone import ViewField as F
from torch.cuda import amp
from torch.nn import CrossEntropyLoss, Dropout
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

wandb.login()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from horizon.dataloaders import (  # noqa: E402
    get_train_dataloader,
    get_val_dataloader
)
from models.custom import HorizonModel  # noqa: E402
from utils.autobatch import check_train_batch_size  # noqa: E402
from utils.downloads import attempt_download  # noqa: E402
from utils.general import LOGGER, TQDM_BAR_FORMAT  # noqa: E402
from utils.horizon import pitch_theta_to_points  # noqa: E402
from utils.torch_utils import ModelEMA, smart_optimizer  # noqa: E402


def get_dataloaders(
    dataset_name: str,
    train_tag: str,
    val_tag: str,
    imgsz: int,
    im_compression_prob: float,
    batch_size: int,
    field: str = "ground_truth_pl.polylines.closed",
):
    # TODO: add tag check
    train_dataloader = get_train_dataloader(
        dataset=(
            fo.load_dataset(dataset_name).match(F(field) == [False]).match_tags(train_tag)
            # .take(5000, seed=51)
        ),
        imgsz=imgsz,
        batch_size=batch_size if imgsz == 640 else 16,
        im_compression_prob=im_compression_prob,
    )

    val_dataloader = get_val_dataloader(
        dataset=(
            fo.load_dataset(dataset_name).match(F(field) == [False]).match_tags(val_tag)
            # .take(5000, seed=51)
        ),
        imgsz=imgsz,
        batch_size=batch_size if imgsz == 640 else 16,
    )

    return train_dataloader, val_dataloader


def update(
    model: HorizonModel,
    train_dataloader: DataLoader,
    loss_pitch: CrossEntropyLoss,
    loss_theta: CrossEntropyLoss,
    pitch_weight: float,
    theta_weight: float,
    scaler: amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    ema: ModelEMA,
    epoch: int,
    epochs: int,
):
    """Update model weights via back-propagation."""

    # initialize mean losses
    t_loss, t_ploss, t_tloss = 0.0, 0.0, 0.0

    s = ("\n" + "%11s" * 7) % (
        "Epoch",
        "GPU_mem",
        "total_loss",
        "pitch_loss",
        "theta_loss",
        "Instances",
        "Size",
    )
    LOGGER.info(s)

    # iterate over batches
    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        bar_format=TQDM_BAR_FORMAT,
    )

    for i, data in pbar:
        images, targets = data
        images, targets = images.to(model.device), targets.to(model.device)

        # process targets
        pitch_i, theta_i = model.to_discrete(pitch=targets[..., 0], theta=targets[..., 1])

        # forward
        x_pitch, x_theta = model(images)

        # backward
        _loss_pitch = loss_pitch(x_pitch, pitch_i)
        _loss_theta = loss_theta(x_theta, theta_i)
        loss = pitch_weight * _loss_pitch + theta_weight * _loss_theta
        scaler.scale(loss).backward()

        # Optimize
        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ema:
            ema.update(model)

        t_loss = (t_loss * i + loss.item()) / (i + 1)  # update mean losses
        t_ploss = (t_ploss * i + _loss_pitch.item()) / (i + 1)
        t_tloss = (t_tloss * i + _loss_theta.item()) / (i + 1)

        mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.2f} GB"
        pbar.set_description(
            ("%11s" * 2 + "%11.4g" * 5)
            % (
                f"{epoch}/{epochs - 1}",
                mem,
                t_loss,
                t_ploss,
                t_tloss,
                targets.shape[0],
                images.shape[-1],
            )
        )

    return t_loss, t_ploss, t_tloss


def evaluate(
    model: HorizonModel,
    val_dataloader: DataLoader,
    loss_pitch: CrossEntropyLoss,
    loss_theta: CrossEntropyLoss,
    pitch_weight: float,
    theta_weight: float,
    ema: ModelEMA,
):
    """Evaluate model on validation set."""

    # use MSE as metric
    criterion_pitch = torch.nn.MSELoss()
    criterion_theta = torch.nn.MSELoss()
    mse_pitch, mse_theta = 0.0, 0.0

    # initialize mean losses
    v_loss, v_ploss, v_tloss = 0.0, 0.0, 0.0

    s = ("\n" + "%22s" + "%11s" * 5) % (
        "",
        "total_mse",
        "pitch_mse",
        "theta_mse",
        "Instances",
        "Size",
    )
    LOGGER.info(s)
    # iterate over batches
    pbar = tqdm(
        enumerate(val_dataloader),
        total=len(val_dataloader),
        bar_format=TQDM_BAR_FORMAT,
    )

    for i, data in pbar:
        images, targets = data
        images, targets = images.to(model.device), targets.to(model.device)

        with torch.no_grad():
            x_pitch, x_theta = ema.ema(images)

        # process targets
        pitch_i, theta_i = model.to_discrete(pitch=targets[..., 0], theta=targets[..., 1])

        # store losses
        _loss_pitch = loss_pitch(x_pitch, pitch_i)
        _loss_theta = loss_theta(x_theta, theta_i)
        loss = pitch_weight * _loss_pitch + theta_weight * _loss_theta

        v_loss = (v_loss * i + loss.item()) / (i + 1)  # update mean losses
        v_ploss = (v_ploss * i + _loss_pitch.item()) / (i + 1)
        v_tloss = (v_tloss * i + _loss_theta.item()) / (i + 1)

        # update running mean of MSE
        (y_pitch, _), (y_theta, _) = model.postprocess(x_pitch, x_theta)
        mse_pitch = (mse_pitch * i + criterion_pitch(y_pitch, targets[..., 0]).item()) / (i + 1)
        mse_theta = (mse_theta * i + criterion_theta(y_theta, targets[..., 1]).item()) / (i + 1)

        pbar.set_description(
            ("%22s" + "%11.4g" * 5)
            % (
                "",
                mse_pitch + mse_theta,
                mse_pitch,
                mse_theta,
                targets.shape[0],
                images.shape[-1],
            )
        )

    return v_loss, v_ploss, v_tloss, mse_pitch, mse_theta


def run(
    dataset_name: str,  # fiftyone dataset name
    train_tag: str = "train",  # fiftyone dataset tag
    val_tag: str = "val",  # fiftyone dataset tag
    weights: str = "yolov5n.pt",  # initial weights path
    nc_pitch: int = 500,  # number of pitch classes
    nc_theta: int = 500,  # number of theta classes
    pitch_weight: float = 1.0,  # pitch loss weight
    theta_weight: float = 1.0,  # theta loss weight
    imgsz: int = 640,  # model input size (assumes squared input)
    epochs: int = 100,
    dropout: float = 0.25,  # dropout rate for classification heads
    im_compression_prob: float = 0.9,
    batch_size: int = -1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train a horizon model.

    Args:
        dataset_name (str): Fiftyone dataset name.
        train_tag (str): Fiftyone dataset tag for training data.
        val_tag (str): Fiftyone dataset tag for validation data.
        weights (str): Path to the initial model weights.
        nc_pitch (int): Number of pitch classes.
        nc_theta (int): Number of theta classes.
        pitch_weight (float): Weight for pitch loss.
        theta_weight (float): Weight for theta loss.
        imgsz (int): Model input size (assumes squared input).
        epochs (int): Number of training epochs.
        dropout (float): Dropout rate for classification heads.
        im_compression_prob (float): Probability for image compression augmentation.
        batch_size (int): Total batch size for the GPU, -1 for automatic batch size.
        device (str): Computing device, either 'cuda' or 'cpu'.

    Returns:
        None
    """
    # create dir to store checkpoints
    ckpt_dir = ROOT / "runs" / "horizon" / "train" / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"{ckpt_dir=}\n")

    if not os.path.exists(weights):
        weights = attempt_download(weights)  # download if not found locally

    # load as horizon model
    model = HorizonModel(weights, nc_pitch, nc_theta, device=device)
    for m in model.model:
        print(m.i, m.f, m.type)
    for m in model.modules():
        if isinstance(m, Dropout) and dropout is not None:
            m.p = dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # all params trainable
    model.imgsz = imgsz
    LOGGER.info(f"{model.nc_pitch=}, {model.nc_theta=}\n")

    # Batch size
    if batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
    else:
        LOGGER.info(f"Batch Size = {batch_size}")

    # load dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        dataset_name, train_tag, val_tag, imgsz, im_compression_prob, batch_size
    )
    LOGGER.info(f"{len(train_dataloader)=}, {len(val_dataloader)=}")

    optimizer = smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=0.0001)

    lrf = 0.001  # final lr (fraction of lr0)
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    loss_pitch = CrossEntropyLoss(label_smoothing=0.0)
    loss_theta = CrossEntropyLoss(label_smoothing=0.0)

    scaler = amp.GradScaler(enabled=model.device != "cpu")

    ema = ModelEMA(model)
    best_mse = 1e10

    model.info()
    LOGGER.info("Starting training...\n")

    wandb.init(
        project="yolo-horizon",
        entity="sea-ai",
        job_type="training",
        config={
            "dataset_name": dataset_name,
            "train_tag": train_tag,
            "val_tag": val_tag,
            "weights": weights,
            "nc_pitch": nc_pitch,
            "nc_theta": nc_theta,
            "pitch_weight": pitch_weight,
            "theta_weight": theta_weight,
            "imgsz": imgsz,
            "epochs": epochs,
            "dropout": dropout,
            "device": device,
            "batch_size": batch_size,
            "im_compression_prob": im_compression_prob,
        },
    )

    for epoch in range(epochs):
        t_loss, t_ploss, t_tloss = 0.0, 0.0, 0.0
        v_loss, v_ploss, v_tloss = 0.0, 0.0, 0.0

        model.train()
        t_loss, t_ploss, t_tloss = update(
            model,
            train_dataloader,
            loss_pitch,
            loss_theta,
            pitch_weight,
            theta_weight,
            scaler,
            optimizer,
            ema,
            epoch,
            epochs,
        )

        scheduler.step()

        model.eval()
        v_loss, v_ploss, v_tloss, mse_pitch, mse_theta = evaluate(
            model, val_dataloader, loss_pitch, loss_theta, pitch_weight, theta_weight, ema
        )

        if mse_pitch + mse_theta < best_mse:
            best_mse = mse_pitch + mse_theta
            ckpt = {
                "epoch": epoch,
                "model": deepcopy(ema.ema),  # deepcopy(de_parallel(model)).half(),
                "ema": None,  # deepcopy(ema.ema).half(),
                "updates": ema.updates,
                "optimizer": None,  # optimizer.state_dict(),
                "losses": dict(pitch=v_ploss, theta=v_tloss, total=v_loss),
                "date": datetime.now().isoformat(),
            }
            path = ckpt_dir / "best.pt"
            torch.save(ckpt, path)
            del ckpt

            wandb.run.summary["best/mse_sum"] = best_mse
            wandb.run.summary["best/mse_pitch"] = mse_pitch
            wandb.run.summary["best/mse_theta"] = mse_theta
            wandb.run.summary["best/epoch"] = epoch

        # Save latest checkpoint
        ckpt = {
            "epoch": epoch,
            "model": deepcopy(ema.ema),  # deepcopy(de_parallel(model)).half(),
            "ema": None,  # deepcopy(ema.ema).half(),
            "updates": ema.updates,
            "optimizer": None,  # optimizer.state_dict(),
            "losses": dict(pitch=v_ploss, theta=v_tloss, total=v_loss),
            "date": datetime.now().isoformat(),
        }
        path = ckpt_dir / "last.pt"
        torch.save(ckpt, path)
        del ckpt

        # log to wandb
        log_dict = {
            "metrics/mse_sum": mse_pitch + mse_theta,
            "metrics/mse_pitch": mse_pitch,
            "metrics/mse_theta": mse_theta,
            "train/total_loss": t_loss,
            "train/pitch_loss": t_ploss,
            "train/theta_loss": t_tloss,
            "val/total_loss": v_loss,
            "val/pitch_loss": v_ploss,
            "val/theta_loss": v_tloss,
        }

        if epoch % 5 == 0:
            log_dict["predictions"] = [wandb.Image(**img) for img in get_wb_images(model, val_dataloader, n=10)]

        wandb.run.log(log_dict)

    wandb.run.log_model(
        name=f"yolov5h-{wandb.run.id}",
        path=str(ckpt_dir / "best.pt"),
        aliases=["best"],
    )

    wandb.run.log_model(
        name=f"yolov5h-{wandb.run.id}",
        path=str(ckpt_dir / "last.pt"),
        aliases=["latest", "last"],
    )

    wandb.run.finish()


def get_wb_images(model: HorizonModel, dataloader: DataLoader, n=10):
    """Get n images with predictions and ground truth for wandb logging."""
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(dataloader.dataset), size=n, replace=False)
    model.eval()
    wb_images = []
    mask_labels = {
        0: "background",
        1: "horizon",
    }  # keys are class indices (pixel values)

    for i in indices:
        images, targets = dataloader.dataset[i]
        images = images.unsqueeze(0).to(model.device)
        with torch.no_grad():
            x_pitch, x_theta = model(images)
        (y_pitch, _), (y_theta, _) = model.postprocess(x_pitch, x_theta)
        y_pitch, y_theta = y_pitch.cpu().numpy(), y_theta.cpu().numpy()

        im = remove_black_padding(  # hack until dataloader is enhanced
            (images[0, 0, ...] * 255).cpu().numpy().astype(np.uint8)
        )

        gt_points = pitch_theta_to_points(targets[0], targets[1], input_hw=images.shape[-2:], orig_hw=im.shape[:2])
        gt_points = np.array(gt_points).astype(np.int32)
        gt_mask = np.zeros(im.shape, dtype=np.uint8)  # 0=background, 1=horizon
        cv2.line(gt_mask, gt_points[0], gt_points[1], color=1, thickness=4)

        y_points = pitch_theta_to_points(
            y_pitch.item(),
            y_theta.item(),
            input_hw=images.shape[-2:],
            orig_hw=im.shape[:2],
        )
        y_points = np.array(y_points).astype(np.int32)
        y_mask = np.zeros(im.shape, dtype=np.uint8)  # 0=background, 1=horizon
        cv2.line(y_mask, y_points[0], y_points[1], color=1, thickness=4)

        wb_images.append(
            dict(
                data_or_path=im,
                masks={
                    "predictions": {"mask_data": y_mask, "class_labels": mask_labels},
                    "ground_truth": {"mask_data": gt_mask, "class_labels": mask_labels},
                },
                caption=dataloader.dataset.filepaths[i],
            )
        )

    return wb_images


def remove_black_padding(image):
    """Remove black padding from an image."""

    # Apply a binary threshold to detect non-black areas
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding rect of the biggest contour (assumed to be the image content)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y : y + h, x : x + w]
        return cropped_image
    else:
        return image  # Return original if no contours found


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="dataset name", required=True)
    parser.add_argument("--train_tag", type=str, default="train", help="train tag")
    parser.add_argument("--val_tag", type=str, default="val", help="val tag")
    parser.add_argument("--weights", type=str, default="yolov5n.pt", help="initial weights path")
    parser.add_argument("--nc_pitch", type=int, default=500, help="number of pitch classes")
    parser.add_argument("--nc_theta", type=int, default=500, help="number of theta classes")
    parser.add_argument("--pitch_weight", type=float, default=1.0, help="pitch loss weight")
    parser.add_argument("--theta_weight", type=float, default=1.0, help="theta loss weight")
    parser.add_argument("--imgsz", type=int, default=640, help="train, val image size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    parser.add_argument(
        "--im_compression_prob",
        type=float,
        default=0.9,
        help="Image compression probability (data Augmentation). 0 to disable",
    )
    parser.add_argument("--batch_size", type=int, default=-1, help="batch size")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
