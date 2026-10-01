import torch
import torch.nn as nn
from torchvision import models
import argparse
import yaml
import os, ssl

try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

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

class TNet(nn.Module):
    def __init__(self, num_classes = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool2d((15, 15)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 15 * 15, num_classes),
        )

    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        x = self.features(x)
        return self.classifier(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes = 16, dropout=0.3, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(enet_out_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
    

def build_model(cfg, num_classes):
    m = cfg["model"]
    name = m["name"]
    if name == "TNet":
        return TNet(
            num_classes=num_classes,
        )
    if name == "efficientnet_b0":
        return EfficientNet(
            num_classes=num_classes,
            dropout=m.get("dropout", 0.3),
            pretrained=m.get("pretrained", True),
        )
    raise ValueError(f"Unknown model name: {name!r}")

def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    model = build_model(cfg, num_classes=16)
    out = model(torch.randn(4, 3, 224, 224))
    print(model)
    print("Output shape:", tuple(out.shape))
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    main()
