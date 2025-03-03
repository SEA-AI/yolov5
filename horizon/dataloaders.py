"""
SEA.AI
authors: Kevin Serrano,

This file contains the dataset classes used for training
YOLOv5 horizon detection model (RGB and ir16bit).
"""

from typing import List, Tuple

import cv2
import fiftyone as fo
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import horizon.transforms as T

def get_train_dataloader(
    dataset: fo.Dataset,
    imgsz: int,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = False,
    dataloader_kwargs: dict = None,
    im_compression_prob: float = 0.9,
):
    dataset = HorizonDataset(
        dataset=dataset,
        rgb_transform=T.horizon_augment_rgb(imgsz, im_compression_prob),
        ir16bit_transform=T.horizon_augment_ir16bit(imgsz, im_compression_prob),
        target_transform=T.points_to_normalised_pitch_theta
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        **dataloader_kwargs if dataloader_kwargs is not None else {},
    )
    
def get_val_dataloader(
    dataset: fo.Dataset,
    imgsz: int,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = False,
    pin_memory: bool = False,
    dataloader_kwargs: dict = None,
):
    dataset = HorizonDataset(
        dataset=dataset,
        rgb_transform=T.horizon_base_rgb(imgsz),
        ir16bit_transform=T.horizon_base_ir16bit(imgsz),
        target_transform=T.points_to_normalised_pitch_theta
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        **dataloader_kwargs if dataloader_kwargs is not None else {},
    )
    
def get_train_rgb_dataloader(
    dataset: fo.Dataset,
    imgsz: int,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = False,
    dataloader_kwargs: dict = None,
    im_compression_prob: float = 0.9,
):
    """Get training set dataloader for an RGB dataset."""

    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_augment_rgb(imgsz, im_compression_prob),
        target_transform=T.points_to_normalised_pitch_theta,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        **dataloader_kwargs if dataloader_kwargs is not None else {},
    )


def get_val_rgb_dataloader(
    dataset: fo.Dataset,
    imgsz: int,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = False,
    pin_memory: bool = False,
    dataloader_kwargs: dict = None,
):
    """Get validation set dataloader for an RGB dataset."""

    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_base_rgb(imgsz),
        target_transform=T.points_to_normalised_pitch_theta,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        **dataloader_kwargs if dataloader_kwargs is not None else {},
    )


def get_train_ir16bit_dataloader(
    dataset: fo.Dataset,
    imgsz: int,
    batch_size: int = 64,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = False,
    dataloader_kwargs: dict = None,
    im_compression_prob: float = 0.9,
):
    """Get training set dataloader for an ir16bit dataset."""

    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_augment_ir16bit(imgsz, im_compression_prob),
        target_transform=T.points_to_normalised_pitch_theta,
        replace_8bit_path=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        **dataloader_kwargs if dataloader_kwargs is not None else {},
    )


def get_val_ir16bit_dataloader(
    dataset: fo.Dataset,
    imgsz: int,
    batch_size: int = 64,
    num_workers: int = 8,
    shuffle: bool = False,
    pin_memory: bool = False,
    dataloader_kwargs: dict = None,
):
    """Get validation set dataloader for an ir16bit dataset."""

    dataset = HorizonDataset(
        dataset=dataset,
        transform=T.horizon_base_ir16bit(imgsz),
        target_transform=T.points_to_normalised_pitch_theta,
        replace_8bit_path=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        **dataloader_kwargs if dataloader_kwargs is not None else {},
    )


class FiftyOneBaseDataset(Dataset):
    """
    Base class for datasets using FiftyOne datasets. Use this as parent class for your custom dataset.

    You may want to extend __init__ and __getitem__.
    Example:
    ```
    class MyDataset(FiftyOneBaseDataset):
        def __init__(self, dataset : fo.Dataset, transform : callable):
            super().__init__(dataset)
            self.transform = transform
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            fpath = sample["filepath"]
            image = Image.open(fpath)
            image = self.transform(image)
            return image
    ```
    """

    def __init__(self, dataset: fo.Dataset):
        super().__init__()
        self.dataset = dataset

        # dataset cannot be accessed by index but by 'id'
        self.ids = dataset.values("id")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> fo.Sample:
        _id = self.ids[idx]
        sample = self.dataset[_id]
        return sample


class HorizonDataset(Dataset):
    """Dataset for horizon detection training."""

    def __init__(
        self,
        dataset: fo.Dataset,
        field: str = "ground_truth_pl.polylines.points",
        rgb_transform: callable = None,
        ir16bit_transform: callable = None,
        target_transform: callable = None,
    ):
        """
        Args:
            dataset (fo.Dataset): FiftyOne dataset
            transform (callable): transform to apply to image and target.
                Example usage: `transform(image=image, keypoints=target)`
            target_transform (callable): transform to apply to target
                Example usage: `target_transform(target, image_w, image_h)`

        Returns:
            image (np.ndarray): HxWxC
            target (np.ndarray): depends on target_transform
        """
        super().__init__()

        self.filepaths: List[str] = dataset.values("filepath")
        self.targets: np.ndarray = np.array(dataset.values(field)).squeeze()

        if rgb_transform is not None:
            assert callable(rgb_transform), "rgb transform must be callable"
        self.rgb_transform = rgb_transform

        if ir16bit_transform is not None:
            assert callable(ir16bit_transform), "ir16bit transform must be callable"
        self.ir16bit_transform = ir16bit_transform

        if target_transform is not None:
            assert callable(target_transform), "target_transform must be callable"
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        fpath = self.filepaths[idx]
        target = self.targets[idx]

        if any(["Thermal" in fpath, "thermal" in fpath]):
            transform = self.ir16bit_transform
            fpath = (
                fpath.replace("8Bit", "16Bit")
                .replace(".jpg", ".png")
                .replace("jpeg", "png")
            )
        else:
            transform = self.rgb_transform

        # read image (np.ndarray)
        image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if image.ndim > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w = image.shape[:2]  # numpy has (H,W,C)

        # 1 point per row, 2 columns (x,y)
        target = np.array(target).reshape(-1, 2)
        # denormalize points
        target[:, 0] *= image_w
        target[:, 1] *= image_h

        if transform:
            # albumentations does not like points in the border
            target = self.shift_points_in_border(target, image.shape[1], image.shape[0])
            augmented = transform(image=image, keypoints=target)
            image = augmented["image"]
            image_h, image_w = image.shape[1:]  # tensor has now (C,H,W)
            target = augmented["keypoints"]

        try:
            if self.target_transform:
                target = self.target_transform(target, image_w, image_h)
        except Exception as e:
            print(f"Error in target_transform for sample {fpath}: {e}")
            raise e

        return image, target

    def shift_points_in_border(self, points, img_w, img_h):
        """If points are in the border, shift them inside the image by 1 pixel."""
        points[:, 0] = np.clip(points[:, 0], 1, img_w - 1)
        points[:, 1] = np.clip(points[:, 1], 1, img_h - 1)
        return points
