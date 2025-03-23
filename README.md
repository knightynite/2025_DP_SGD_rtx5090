# 2025 — DP-SGD on RTX 5090 (Blackwell, CUDA 13.1)

Differentially private SGD on MNIST and CIFAR-10, tuned for the Blackwell architecture
(RTX 5090, CUDA 13.1). Compares Opacus and a from-scratch DP-SGD implementation, and
tracks privacy budget with an RDP accountant.

## Approach

1. **From-scratch DP-SGD** (`src/dp_sgd.py`):
    - Compute per-sample gradients (via `torch.func.vmap` + `torch.func.grad`)
    - Clip per-sample to L2 norm `C`
    - Sum, add Gaussian noise σ·C
    - Step optimizer
2. **Opacus** comparison for sanity-check
3. **Privacy accountant** (`src/privacy_accountant.py`) — RDP-based ε, δ tracking
