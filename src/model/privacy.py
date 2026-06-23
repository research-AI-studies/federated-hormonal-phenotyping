"""Differential-privacy accounting.

A compact Renyi differential privacy (RDP) accountant for the subsampled
Gaussian mechanism, plus a helper that calibrates the noise multiplier to a
target (epsilon, delta) budget by bisection. Used to bound the privacy cost of
DP-SGD updates in the federated VAE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _log_add(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _compute_rdp_single(q: float, sigma: float, alpha: int) -> float:
    """RDP of the subsampled Gaussian mechanism at integer order alpha."""
    log_terms = []
    for k in range(alpha + 1):
        log_coef = (
            math.lgamma(alpha + 1)
            - math.lgamma(k + 1)
            - math.lgamma(alpha - k + 1)
        )
        term = (
            log_coef
            + k * math.log(q)
            + (alpha - k) * math.log(1 - q)
            + (k * k - k) / (2 * sigma ** 2)
        )
        log_terms.append(term)
    log_sum = -np.inf
    for t in log_terms:
        log_sum = _log_add(log_sum, t)
    return float(log_sum / (alpha - 1))


@dataclass
class RDPAccountant:
    sample_rate: float
    noise_multiplier: float
    orders: tuple[int, ...] = tuple(range(2, 64))

    def epsilon(self, steps: int, delta: float) -> float:
        """Convert accumulated RDP over ``steps`` to an (epsilon, delta) bound."""
        best = np.inf
        for alpha in self.orders:
            rdp = steps * _compute_rdp_single(
                self.sample_rate, self.noise_multiplier, alpha
            )
            eps = rdp - math.log(delta) / (alpha - 1)
            best = min(best, eps)
        return float(best)


def calibrate_noise(
    sample_rate: float,
    steps: int,
    target_epsilon: float,
    target_delta: float,
    lo: float = 0.3,
    hi: float = 20.0,
    tol: float = 1e-2,
) -> float:
    """Bisection search for the smallest noise multiplier meeting the budget."""
    for _ in range(60):
        mid = (lo + hi) / 2
        eps = RDPAccountant(sample_rate, mid).epsilon(steps, target_delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
        if abs(eps - target_epsilon) < tol:
            break
    return hi
