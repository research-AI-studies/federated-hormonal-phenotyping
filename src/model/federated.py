"""Federated orchestration (FedAvg) for the VAE + SOM pipeline.

A self-contained simulation: the cohort matrix is partitioned across virtual
clients (IID or Dirichlet non-IID), each client trains a local VAE for a few
epochs, and parameters are aggregated by sample-weighted averaging across
rounds. The aggregated encoder then produces the embedding consumed by a
federated SOM. The same client/aggregate contract maps onto a real Flower
(`flwr`) deployment; this module keeps the experiment reproducible offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .som import SOM
from .vae import VAE, vae_loss


def partition(
    x: np.ndarray,
    num_clients: int,
    scheme: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42,
) -> list[np.ndarray]:
    """Split row indices across clients, IID or Dirichlet non-IID."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    if scheme == "iid":
        return [p for p in np.array_split(idx, num_clients)]
    proportions = rng.dirichlet(np.repeat(alpha, num_clients))
    cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
    return [p for p in np.split(idx, cuts)]


def _local_train(model: VAE, data: np.ndarray, epochs: int, lr: float, beta: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tensor = torch.tensor(data, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon, mu, logvar = model(tensor)
        loss = vae_loss(recon, tensor, mu, logvar, beta)
        loss.backward()
        opt.step()
    return model


def _fedavg(global_model: VAE, locals_: list[VAE], sizes: list[int],
            noise_sigma: float = 0.0) -> VAE:
    """Sample-weighted FedAvg. DP noise is added to the aggregated *update*,
    scaled by the per-tensor update dispersion so that the privacy knob
    ``noise_sigma`` stays numerically stable across rounds."""
    total = sum(sizes)
    prev = global_model.state_dict()
    new_state = {}
    for key in prev:
        agg = sum(
            (loc.state_dict()[key] * (n / total)) for loc, n in zip(locals_, sizes)
        )
        if noise_sigma > 0 and agg.dtype.is_floating_point:
            delta = agg - prev[key]
            scale = float(delta.std()) if delta.numel() > 1 else float(delta.abs().mean())
            agg = agg + torch.randn_like(agg) * noise_sigma * (scale + 1e-8)
        new_state[key] = agg
    global_model.load_state_dict(new_state)
    return global_model


@dataclass
class FederatedConfig:
    num_clients: int = 3
    rounds: int = 20
    local_epochs: int = 2
    partition: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    lr: float = 1e-3
    beta: float = 1.0
    noise_sigma: float = 0.0  # Gaussian DP noise std added to aggregated params


def run_federated_vae(
    x: np.ndarray,
    cfg: FederatedConfig,
    hidden_dims: list[int],
    latent_dim: int,
    dropout: float,
    seed: int = 42,
) -> VAE:
    """Train a global VAE by FedAvg over partitioned clients."""
    torch.manual_seed(seed)
    parts = partition(x, cfg.num_clients, cfg.partition, cfg.dirichlet_alpha, seed)
    parts = [p for p in parts if len(p) > 0]

    global_model = VAE(x.shape[1], hidden_dims, latent_dim, dropout)
    for _ in range(cfg.rounds):
        locals_, sizes = [], []
        for p in parts:
            local = VAE(x.shape[1], hidden_dims, latent_dim, dropout)
            local.load_state_dict(global_model.state_dict())
            _local_train(local, x[p], cfg.local_epochs, cfg.lr, cfg.beta)
            locals_.append(local)
            sizes.append(len(p))
        global_model = _fedavg(global_model, locals_, sizes, cfg.noise_sigma)
    return global_model


def run_federated_som(
    latent: np.ndarray,
    parts: list[np.ndarray],
    grid: tuple[int, int],
    sigma: float,
    learning_rate: float,
    iterations: int,
    seed: int = 42,
) -> SOM:
    """Train one SOM per client on the shared latent space, then FedAvg codebooks."""
    client_maps = []
    for i, p in enumerate(parts):
        som = SOM(grid, latent.shape[1], sigma, learning_rate, iterations, seed=seed + i)
        som.train(latent[p])
        client_maps.append(som)
    global_som = client_maps[0]
    global_som.federated_merge(client_maps[1:], weights=[len(p) for p in parts])
    return global_som
