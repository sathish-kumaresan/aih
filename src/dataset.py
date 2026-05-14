import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BreaKHisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].to_list()
        self.labels = df["label"].to_numpy(dtype="float32")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), float(self.labels[i])


def train_transform(
    img_size: int = 384,
    *,
    hflip: float = 0.5,
    vflip: float = 0.5,
    rotation_deg: float = 15.0,
    brightness: float = 0.10,
    contrast: float = 0.10,
    saturation: float = 0.05,
    hue: float = 0.02,
):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=hflip),
        transforms.RandomVerticalFlip(p=vflip),
        transforms.RandomRotation(
            degrees=rotation_deg, interpolation=InterpolationMode.BILINEAR, fill=0,
        ),
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int = 384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _seed_worker(worker_id: int) -> None:
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
