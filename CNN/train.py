import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

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
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "best.yaml"))
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


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device,
                label_smoothing=0.1, patience=6):
    from evaluate import evaluate
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc, best_epoch, epochs_no_improve = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Training loop {epoch}/{num_epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()               
            running_loss += loss.item() * labels.size(0)
            hard_labels = labels.argmax(1) if labels.ndim > 1 else labels 
            correct += (outputs.argmax(1) == hard_labels).sum().item()
            seen += labels.size(0)

        train_loss, train_acc = running_loss / seen, correct / seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])):
            history[k].append(v)

        flag = ''
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, epochs_no_improve = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            flag = '  <- best'
        else:
            epochs_no_improve += 1

        print(f'Epoch {epoch:02d}/{num_epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.0f}s{flag}')
        if epochs_no_improve >= patience:
            print(f'Early stopping: no validation improvement for {patience} epochs.')
            break

    model.load_state_dict(best_state)
    print(f'Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})')
    return model, history, best_val_acc


def make_optimizer(model, backbone_lr, head_lr, weight_decay):
    return optim.AdamW([
        {'params': model.features.parameters(), 'lr': backbone_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr},
    ], weight_decay=weight_decay)


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_random_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, class_names = build_loaders(cfg)
    model = build_model(cfg, num_classes=len(class_names)).to(device)

    optimizer = make_optimizer(model, cfg["optimizer"]["backbone_lr"], 
        cfg["optimizer"]["head_lr"], cfg["optimizer"]["weight_decay"])
    
    num_epochs = cfg["training"]["num_epochs"]
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, cfg["scheduler"]["warmup_epochs"] * steps_per_epoch, num_epochs * steps_per_epoch)

    model, history, best_val_acc = train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device,
                label_smoothing=cfg["training"]["label_smoothing"], patience=cfg["training"]["patience"])
    print(f"Best validation accuracy: {best_val_acc:.4f}")

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

