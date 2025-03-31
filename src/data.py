"""Dataset loaders for MNIST and CIFAR-10 with DP-friendly batching.

DP-SGD wants per-sample gradients, which means batch_size = "logical batch
size" — we accumulate microbatches inside a step. The loader contract:

  - returns one (x, y) tuple per call
  - normalized to [0, 1]
  - shuffled with a deterministic generator for reproducibility
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path('./data')


def mnist_loaders(batch_size: int = 256, num_workers: int = 2,
                  generator=None) -> tuple:
    DATA_ROOT.mkdir(exist_ok=True)
    tx = transforms.Compose([
        transforms.ToTensor(),  # already in [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tx)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tx)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=generator,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def cifar10_loaders(batch_size: int = 256, num_workers: int = 2,
                    generator=None) -> tuple:
    DATA_ROOT.mkdir(exist_ok=True)
    tx_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])
    tx_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])
    train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tx_train)
    test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tx_test)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=generator,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
