import os
import typing
from os.path import join

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from albumentations import (ColorJitter, Compose, ImageCompression,
                            LongestMaxSize, Normalize, PadIfNeeded,
                            RandomResizedCrop, Resize, ShiftScaleRotate)
from albumentations.pytorch.transforms import ToTensorV2
from config import Config as C
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


def get_train_transforms(img_size: int = 224) -> torch.tensor:
    return Compose(
        [
            ColorJitter(),
            ImageCompression(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            LongestMaxSize(max_size=img_size),
            PadIfNeeded(img_size, img_size,
                        border_mode=cv2.BORDER_CONSTANT, mask_value=0),
            ShiftScaleRotate(rotate_limit=30),
            RandomResizedCrop(img_size, img_size, scale=(0.85, 1.0)),
            ToTensorV2()
        ]
    )


def get_val_transforms(img_size: int = 224) -> torch.tensor:
    return Compose(
        [
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            LongestMaxSize(max_size=img_size),
            PadIfNeeded(img_size, img_size,
                        border_mode=cv2.BORDER_CONSTANT, mask_value=0),
            ToTensorV2()
        ]
    )


class SakeDataset(Dataset):
    def __init__(
        self,
        image_filepaths: list,
        labels: list = None,
        is_train=False
    ) -> None:

        self.image_filepaths = image_filepaths
        self.labels = labels
        self.is_train = is_train
        if is_train:
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

    def __len__(self) -> int:
        return len(self.image_filepaths)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.tensor, torch.tensor]:
        item = dict()
        image_filepath = self.image_filepaths[idx]
        image = self.__read_image(image_filepath)
        image = self.transform(image=image)["image"]
        item["image"] = image

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            item["label"] = label
        return item

    def __read_image(self, path: str) -> None:
        with open(path, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb)
        return image


def create_tiled_image(dataset, n, save_path):
    """
    Create a tiled image from N samples of the dataset.
    """

    # DataLoaderを使用してデータセットからN枚の画像を取得
    dataloader = DataLoader(dataset, batch_size=n, shuffle=True)
    batch = next(iter(dataloader))
    images = batch["image"]

    # N枚の画像をタイル状に並べる
    tiled_image = vutils.make_grid(images, nrow=int(np.sqrt(n)))

    # [-1, 1] の範囲から [0, 1] の範囲に変更
    tiled_image = (tiled_image - tiled_image.min()) / \
        (tiled_image.max() - tiled_image.min())

    # 保存
    vutils.save_image(tiled_image, save_path)


if __name__ == "__main__":
    annotation_df = pd.read_csv(C.train_filepath)
    train_filenames = annotation_df["filename"].to_list()
    annotation_df["path"] = [str(C.query_images_dir.joinpath(filename))
                             for filename in train_filenames]

    train_dataset = SakeDataset(
        image_filepaths=annotation_df["path"].to_list(),
        is_train=True)

    test_dataset = SakeDataset(
        image_filepaths=annotation_df["path"].to_list(),
        is_train=False)

    os.makedirs(join(C.work_dir), exist_ok=True)
    create_tiled_image(train_dataset, 12, join(
        C.work_dir, "train_image_samples.png"))
    create_tiled_image(test_dataset, 12, join(
        C.work_dir, "test_image_samples.png"))
