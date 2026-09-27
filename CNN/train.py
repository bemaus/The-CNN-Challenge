import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn import build_model


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_transforms(data_cfg):
    size = data_cfg["img_size"]
    aug = data_cfg["augment"]
    normalize = transforms.Normalize(data_cfg["mean"], data_cfg["std"])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=tuple(aug["random_resized_crop_scale"])),
        transforms.RandomHorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0.0),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.ColorJitter(*aug["color_jitter"]),
        transforms.RandomGrayscale(p=aug["random_grayscale_p"]),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=aug["random_erasing_p"]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf

