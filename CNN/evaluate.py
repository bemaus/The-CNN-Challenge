from pathlib import Path
from glob import glob
import random
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from cnn import build_model
from train import (parse_args, load_config, build_loaders, build_transforms,
                   make_optimizer, warmup_cosine, train_model)

def find_root():
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        PROJECT_ROOT = Path('/content/gdrive/MyDrive/CNN_Data')
    except ImportError:
        print('Not running in Google Colab; Drive mount skipped.')
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return PROJECT_ROOT

def load_data(PROJECT_ROOT):
    DATA_ROOT = PROJECT_ROOT / 'data'
    TRAIN_ROOT = DATA_ROOT / 'train'
    TEST_ROOT = DATA_ROOT / 'test'
    TEST2_ROOT = DATA_ROOT / 'test2'   
    CHECKPOINT_PATH = PROJECT_ROOT / 'best_efficientnet_b0_scenes.pt'
    SEED = 0

    def set_random_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_random_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Project root:', PROJECT_ROOT.resolve())
    print('Data root:', DATA_ROOT.resolve())
    print('Train root:', TRAIN_ROOT.resolve())
    print('Using device:', device)
    return TRAIN_ROOT, TEST_ROOT, TEST2_ROOT, CHECKPOINT_PATH, SEED, device


def sample_set(train_loader, class_names, IMAGENET_MEAN, IMAGENET_STD):
    images, labels = next(iter(train_loader))
    print('Batch shape:', images.shape, labels.shape)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, image, label in zip(axes.flat, images[:8], labels[:8]):
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.set_title(class_names[label.argmax().item() if label.ndim else label.item()])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def learning_curve(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Training loss')
    ax1.plot(history['val_loss'], label='Validation loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss over epochs'); ax1.legend()
    ax2.plot(history['train_acc'], label='Training accuracy (augmented)')
    ax2.plot(history['val_acc'], label='Validation accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy over epochs'); ax2.legend()
    plt.tight_layout()
    plt.show()


def confucion_matrix_report(model, val_loader, class_names, device, num_classes):
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, return_preds=True)
    val_loss_tta, val_acc_tta = evaluate(model, val_loader, device, tta=True)
    print(f'Validation accuracy: {val_acc:.4f} | with flip-TTA: {val_acc_tta:.4f}')
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=3))

    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Validation confusion matrix')
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


def final_evaluation(model, test_loader, device, class_names):
    test_loss, test_acc = evaluate(model, test_loader, device)
    test_loss_tta, test_acc_tta = evaluate(model, test_loader, device, tta=True)
    print(f'Final test loss: {test_loss:.4f}')
    print(f'Final test accuracy: {test_acc:.4f}')
    print(f'Final test accuracy with flip-TTA: {test_acc_tta:.4f}')


@torch.inference_mode()
def evaluate(model, loader, device, tta=False, return_preds=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if tta:
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        running_loss += criterion(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if return_preds:
            all_preds.append(preds.cpu()); all_labels.append(labels.cpu())
    if return_preds:
        return running_loss / total, correct / total, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()
    return running_loss / total, correct / total


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_predictions(original_image, probabilities, class_names, true_label=None):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 6))
    axarr[0].imshow(original_image, cmap='gray' if original_image.mode == 'L' else None)
    axarr[0].axis('off')
    pred = class_names[int(np.argmax(probabilities))]
    axarr[0].set_title(f'Pred: {pred}' + (f' | True: {true_label}' if true_label else ''))
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel('Probability')
    axarr[1].set_title('Class Predictions')
    axarr[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


def sample_predictions(model, test_loader, class_names, device, num_samples=5,
                       TEST_ROOT=None, eval_transform=None, SEED=0):
    test_images = sorted(glob(str(TEST_ROOT / '*' / '*.jpg')))
    rng = np.random.default_rng(SEED)
    for example in rng.choice(test_images, num_samples, replace=False):
        original_image, image_tensor = preprocess_image(example, eval_transform)
        probabilities = predict(model, image_tensor, device)
        visualize_predictions(original_image, probabilities, class_names, true_label=Path(example).parent.name)


def main():
    print('PyTorch version', torch.__version__)
    print('Torchvision version', torchvision.__version__)
    print('Numpy version', np.__version__)
    print('Pandas version', pd.__version__)
    print(f"CUDA available: {torch.cuda.is_available()}")

    args = parse_args()
    cfg = load_config(args.config, args.set)
    PROJECT_ROOT = find_root()
    TRAIN_ROOT, TEST_ROOT, TEST2_ROOT, CHECKPOINT_PATH, SEED, device = load_data(PROJECT_ROOT)
    cfg["data"]["train_root"] = str(TRAIN_ROOT)
    cfg["output"]["checkpoint"] = str(CHECKPOINT_PATH)

    d = cfg["data"]
    train_loader, val_loader, class_names = build_loaders(cfg)
    _, eval_transform = build_transforms(d)
    test_dataset = datasets.ImageFolder(TEST_ROOT, transform=eval_transform)
    assert test_dataset.classes == class_names, 'train/test class folders do not match'
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=d["num_workers"],
                             pin_memory=device.type == 'cuda')
    num_classes = len(class_names)
    print(f'Classes ({num_classes}): {class_names}')
    print(f'Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)} | Test: {len(test_dataset)}')
    sample_set(train_loader, class_names, d["mean"], d["std"])

    model = build_model(cfg, num_classes=num_classes).to(device)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    o, t = cfg["optimizer"], cfg["training"]
    optimizer = make_optimizer(model, o["backbone_lr"], o["head_lr"], o["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, cfg["scheduler"]["warmup_epochs"] * steps_per_epoch,
                              t["num_epochs"] * steps_per_epoch)
    model, history, best_val_acc = train_model(model, train_loader, val_loader, optimizer, scheduler,
                                               t["num_epochs"], device, label_smoothing=t["label_smoothing"],
                                               patience=t["patience"])
    torch.save({'model_state': model.state_dict(), 'class_names': class_names, 'config': cfg,
                'history': history, 'best_val_acc': best_val_acc}, CHECKPOINT_PATH)
    print('Saved checkpoint to', CHECKPOINT_PATH)

    learning_curve(history)
    confucion_matrix_report(model, val_loader, class_names, device, num_classes)
    final_evaluation(model, test_loader, device, class_names)
    sample_predictions(model, test_loader, class_names, device, num_samples=5,
                       TEST_ROOT=TEST_ROOT, eval_transform=eval_transform, SEED=SEED)


if __name__ == "__main__":
    main()