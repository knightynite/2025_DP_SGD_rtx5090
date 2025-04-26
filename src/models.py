"""Small CNNs for DP-SGD experiments.

DP training favors:
  - GroupNorm over BatchNorm (batchnorm leaks per-sample info)
  - simple architectures (per-sample-grad memory grows with depth)
  - bounded gradients (large gradients waste budget on clipping)
"""
import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    """Simple MNIST CNN — 2 conv blocks + MLP head, GroupNorm."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=2, padding=3)
        self.gn1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.gn2 = nn.GroupNorm(4, 32)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CIFARNet(nn.Module):
    """Small ResNet-flavored CNN for CIFAR-10 with GroupNorm.

    Avoids deep ResNet — DP-SGD per-sample grad memory makes very deep
    networks expensive even on Blackwell. This is a 6-conv backbone that
    reaches ~70% clean accuracy and ~55% at ε=8.
    """

    def __init__(self):
        super().__init__()
        def block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(8, out_c),
                nn.ReLU(inplace=True),
            )
        self.body = nn.Sequential(
            block(3, 32),
            block(32, 64, stride=2),
            block(64, 128, stride=2),
            block(128, 128),
            block(128, 256, stride=2),
            block(256, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(256, 10)

    def forward(self, x):
        return self.head(self.body(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
