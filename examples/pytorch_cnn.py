"""Example Workload 2: PyTorch Vision CNN (Deep Learning)."""

import torch
import torch.nn as nn
import numpy as np


class SimpleVisionCNN(nn.Module):
    """Convolutional Neural Network for image classification (3x32x32)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = 10) -> nn.Module:
    """Instantiate and initialize PyTorch CNN."""
    torch.manual_seed(42)
    model = SimpleVisionCNN(num_classes=num_classes)
    model.eval()
    return model


def get_sample_input(batch_size: int = 1) -> torch.Tensor:
    """Generate synthetic batch of image tensors (B, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)


def get_test_data(n_samples: int = 100, num_classes: int = 10):
    """Generate synthetic evaluation dataset for accuracy checks."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 3, 32, 32).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(n_samples,)).astype(np.int64)
    return X, y


if __name__ == "__main__":
    model = get_model()
    x = get_sample_input(2)
    with torch.no_grad():
        out = model(x)
    print("PyTorch CNN loaded. Output shape:", out.shape)
