"""From-scratch DP-SGD using torch.func per-sample gradients.

Per step:
  1. compute g_i for each sample i (via vmap)
  2. clip each g_i to L2 norm C
  3. sum, add Gaussian noise sigma * C * I
  4. apply optimizer step on the noised mean
"""
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap


def make_per_sample_grad_fn(model: nn.Module, loss_fn):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_for_one(params_, x, y):
        out = functional_call(model, (params_, buffers), (x.unsqueeze(0),))
        return loss_fn(out, y.unsqueeze(0))

    grad_fn = grad(loss_for_one)
    per_sample = vmap(grad_fn, in_dims=(None, 0, 0))
    return per_sample, params


