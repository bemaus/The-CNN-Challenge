import torch
import torch.nn as nn
from torchvision import models

class TNet(nn.Module):
    def __init__(self, num_classes = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 15 * 15, num_classes),
        )

    def forward(self, x):
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
    if name == "simple_cnn":
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



if __name__ == "__main__":
    model = TNet(num_classes=16)
    out = model(torch.randn(4, 3, 224, 224))
    print(model)
    print("Output shape:", tuple(out.shape))
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
