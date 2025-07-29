"""Lightweight RDP accountant for the Subsampled Gaussian Mechanism.

References:
  - Mironov 2017 — Renyi Differential Privacy
  - Wang, Balle, Kasiviswanathan 2019 — Subsampled Renyi DP and Analytical Moments
"""
import math

import numpy as np
from scipy import special


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    """RDP at order alpha for sampled Gaussian with subsampling rate q, noise sigma."""
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * sigma ** 2)
    if alpha == 1.0:
        return q * (math.exp(1.0 / (2.0 * sigma ** 2)) - 1.0)
    if not float(alpha).is_integer():
        # binary-only fallback works fine for typical orders
        alpha = int(round(alpha))

    log_a = -np.inf
    for i in range(int(alpha) + 1):
        log_term = (
            math.log(special.binom(int(alpha), i))
            + i * math.log(q)
            + (int(alpha) - i) * math.log(1.0 - q)
            + (i * i - i) / (2.0 * sigma ** 2)
        )
        log_a = np.logaddexp(log_a, log_term)
    return float(log_a) / (alpha - 1)


def compute_rdp(q: float, sigma: float, steps: int, orders) -> np.ndarray:
    rdp = np.zeros(len(orders), dtype=np.float64)
    for i, alpha in enumerate(orders):
        rdp[i] = _compute_rdp(q, sigma, alpha) * steps
    return rdp


def get_privacy_spent(orders, rdp, target_delta: float):
    """Convert RDP curve to the tightest (epsilon, delta) pair."""
    orders = np.asarray(orders, dtype=np.float64)
    rdp = np.asarray(rdp, dtype=np.float64)
    eps = rdp - math.log(target_delta) / (orders - 1)
    idx = int(np.argmin(eps))
    return float(eps[idx]), float(orders[idx])


DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
