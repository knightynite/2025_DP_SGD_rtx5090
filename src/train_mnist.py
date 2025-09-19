"""DP-SGD training on MNIST with explicit privacy budget tracking."""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dp_sgd import dp_step
from privacy_accountant import compute_rdp, get_privacy_spent, DEFAULT_ORDERS


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1.0)
    p.add_argument('--max-norm', type=float, default=1.0)
    p.add_argument('--sigma', type=float, default=1.1)
    p.add_argument('--target-epsilon', type=float, default=8.0)
    p.add_argument('--delta', type=float, default=1e-5)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST('./data', train=True, download=True, transform=tfm)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    n_train = len(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    model = CNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    q = args.batch_size / n_train
    steps_per_epoch = n_train // args.batch_size

    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            dp_step(model, opt, F.cross_entropy, x, y, args.max_norm, args.sigma)

        steps = (ep + 1) * steps_per_epoch
        rdp = compute_rdp(q, args.sigma, steps, DEFAULT_ORDERS)
        eps, alpha = get_privacy_spent(DEFAULT_ORDERS, rdp, args.delta)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(test_set)
        print(f'epoch {ep + 1:2d}  acc={acc:.4f}  '
              f'(eps={eps:.2f}, delta={args.delta}, optimal_alpha={alpha:.1f})')

        if eps >= args.target_epsilon:
            print(f'reached target epsilon {args.target_epsilon}, stopping')
            break


if __name__ == '__main__':
    main()
