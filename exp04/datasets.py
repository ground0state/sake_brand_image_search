
import typing

import numpy as np
import torch
from albumentations import Compose, Normalize, Resize, RandomResizedCrop
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


def get_train_transforms(img_size: int = 224) -> torch.tensor:
    return Compose(
        [
            # RandomResizedCrop(img_size, img_size, scale=(0.85, 1.0),
            Resize(img_size, img_size),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]
    )


def get_val_transforms(img_size: int = 224) -> torch.tensor:
    return Compose(
        [
            Resize(img_size, img_size),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
