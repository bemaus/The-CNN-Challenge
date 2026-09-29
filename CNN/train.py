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
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets, transforms
from torchvision.datasets.folder import find_classes
from torchvision.transforms import v2

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

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/best.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    return parser.parse_args()

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

class CutMixDataset(Dataset):
    def __init__(self, base, num_classes, alpha=1.0, prob=0.5):
        self.base = base
        self.num_classes = num_classes
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.prob = prob

    def __len__(self):
        return len(self.base)

    def _one_hot(self, y):
        return nn.functional.one_hot(torch.tensor(y), self.num_classes).float()

    def __getitem__(self, i):
        x1, y1 = self.base[i]
        if torch.rand(1).item() >= self.prob:
            return x1, self._one_hot(y1)

        j = torch.randint(len(self.base), (1,)).item()
        x2, y2 = self.base[j]

        _, h, w = x1.shape
        lam = self.beta.sample().item()
        cut_h, cut_w = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
        y0, y1_ = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x0, x1_ = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)

        mixed = x1.clone()
        mixed[:, y0:y1_, x0:x1_] = x2[:, y0:y1_, x0:x1_]
        lam = 1 - (y1_ - y0) * (x1_ - x0) / (h * w)
        return mixed, lam * self._one_hot(y1) + (1 - lam) * self._one_hot(y2)

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
  
def build_loaders(cfg):
    d = cfg["data"]
    train_tf, eval_tf = build_transforms(d)

    train_view = datasets.ImageFolder(d["train_root"], transform=train_tf)
    val_view = datasets.ImageFolder(d["train_root"], transform=eval_tf)

    targets = np.array(train_view.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=d["val_fraction"], stratify=targets, random_state=cfg["seed"]
    )

    kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], pin_memory=torch.cuda.is_available())
    train_ds = Subset(train_view, train_idx)
    if d["augment"].get("cutmix_alpha", 0) > 0:
        train_ds = CutMixDataset(train_ds, num_classes=len(train_view.classes),
                                 alpha=d["augment"]["cutmix_alpha"], prob=d["augment"].get("cutmix_prob", 0.5))
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(Subset(val_view, val_idx), shuffle=False, **kw)
    return train_loader, val_loader, train_view.classes

def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_random_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, class_names = build_loaders(cfg)
    model = build_model(cfg, num_classes=len(class_names)).to(device)

    model, history, best_val_acc = fit(model, train_loader, val_loader, cfg, device)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # The config is stored inside the checkpoint so evaluate.py can rebuild the exact model.
    ckpt_path = Path(cfg["output"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "config": cfg,
        "history": history,
        "best_val_acc": best_val_acc,
    }, ckpt_path)
    print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()

