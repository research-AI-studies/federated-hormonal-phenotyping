"""Self-organising map over the VAE latent space.

A light wrapper around a batch SOM with a Gaussian neighbourhood. It exposes a
``federated_merge`` hook so that codebooks trained on separate clients can be
averaged into a global map (see ``federated.py``).
"""
from __future__ import annotations

import numpy as np


class SOM:
    def __init__(
        self,
        grid: tuple[int, int] = (8, 8),
        input_dim: int = 8,
        sigma: float = 1.2,
        learning_rate: float = 0.5,
        iterations: int = 5000,
        neighbourhood: str = "gaussian",
        seed: int = 42,
    ) -> None:
        self.rows, self.cols = grid
        self.input_dim = input_dim
        self.sigma0 = sigma
        self.lr0 = learning_rate
        self.iterations = iterations
        self.neighbourhood = neighbourhood
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 1, (self.rows, self.cols, input_dim))
        self._coords = np.array(
            [[(i, j) for j in range(self.cols)] for i in range(self.rows)]
        )

    def _bmu(self, x: np.ndarray) -> tuple[int, int]:
        dists = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(dists), dists.shape)

    def _neighbourhood(self, bmu, sigma: float) -> np.ndarray:
        d2 = np.sum((self._coords - np.array(bmu)) ** 2, axis=2)
        if self.neighbourhood == "bubble":
            return (d2 <= sigma ** 2).astype(float)
        return np.exp(-d2 / (2 * sigma ** 2))

    def train(self, data: np.ndarray) -> "SOM":
        n = len(data)
        for t in range(self.iterations):
            decay = 1.0 - t / self.iterations
            sigma = max(self.sigma0 * decay, 1e-3)
            lr = self.lr0 * decay
            x = data[self.rng.integers(n)]
            bmu = self._bmu(x)
            h = self._neighbourhood(bmu, sigma)[..., None]
            self.weights += lr * h * (x - self.weights)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return a flat node index in [0, rows*cols) for each sample."""
        out = np.empty(len(data), dtype=int)
        for i, x in enumerate(data):
            r, c = self._bmu(x)
            out[i] = r * self.cols + c
        return out

    def quantisation_error(self, data: np.ndarray) -> float:
        errs = [np.linalg.norm(x - self.weights[self._bmu(x)]) for x in data]
        return float(np.mean(errs))

    def federated_merge(self, others: list["SOM"], weights: list[float] | None = None) -> "SOM":
        """Average this map's codebook with peers (FedAvg over SOM weights)."""
        maps = [self] + list(others)
        w = np.array(weights) if weights is not None else np.ones(len(maps))
        w = w / w.sum()
        self.weights = np.tensordot(w, np.stack([m.weights for m in maps]), axes=1)
        return self
