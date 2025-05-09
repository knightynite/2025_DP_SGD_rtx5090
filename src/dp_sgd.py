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


def clip_and_noise(per_sample_grads: dict, max_norm: float, sigma: float, batch_size: int):
    """Per-sample clipping then sum + Gaussian noise."""
    flat = torch.cat([g.reshape(g.shape[0], -1) for g in per_sample_grads.values()], dim=1)
    norms = flat.norm(dim=1)
    scale = (max_norm / (norms + 1e-9)).clamp(max=1.0)

    clipped = {}
    for name, g in per_sample_grads.items():
        view_shape = (g.shape[0],) + (1,) * (g.dim() - 1)
        clipped[name] = g * scale.view(view_shape)

    summed = {name: g.sum(dim=0) for name, g in clipped.items()}
    noised = {name: g + torch.randn_like(g) * sigma * max_norm for name, g in summed.items()}
    return {name: g / batch_size for name, g in noised.items()}


def dp_step(model, optimizer, loss_fn, x, y, max_norm, sigma):
    per_sample_grads_fn, params = make_per_sample_grad_fn(model, loss_fn)
    grads = per_sample_grads_fn(params, x, y)
    private_grads = clip_and_noise(grads, max_norm, sigma, batch_size=x.shape[0])

    # write grads onto the model and step
    for name, p in model.named_parameters():
        p.grad = private_grads[name].detach()
    optimizer.step()
    optimizer.zero_grad()
