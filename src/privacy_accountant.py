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


