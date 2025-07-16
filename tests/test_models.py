"""Smoke tests for MNIST/CIFAR DP-friendly nets."""
import unittest

import torch

from src.models import MNISTNet, CIFARNet, count_params


class TestMNISTNet(unittest.TestCase):
    def test_forward_shape(self):
        net = MNISTNet()
        x = torch.randn(4, 1, 28, 28)
        out = net(x)
        self.assertEqual(out.shape, (4, 10))

    def test_no_batch_norm(self):
        net = MNISTNet()
        for m in net.modules():
            self.assertFalse(isinstance(m, torch.nn.BatchNorm2d),
                             'DP training should not use BatchNorm')

    def test_param_count_reasonable(self):
        n = count_params(MNISTNet())
        self.assertLess(n, 100_000)
        self.assertGreater(n, 1_000)


class TestCIFARNet(unittest.TestCase):
    def test_forward_shape(self):
        net = CIFARNet()
        x = torch.randn(2, 3, 32, 32)
        out = net(x)
        self.assertEqual(out.shape, (2, 10))

    def test_uses_groupnorm(self):
        net = CIFARNet()
        gn = [m for m in net.modules() if isinstance(m, torch.nn.GroupNorm)]
        self.assertGreater(len(gn), 0)


if __name__ == '__main__':
    unittest.main()
