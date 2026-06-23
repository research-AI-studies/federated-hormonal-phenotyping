"""Variational autoencoder for tabular representation learning.

A symmetric MLP encoder/decoder with a Gaussian latent prior. The encoder
output (posterior mean) serves as the embedding consumed by the SOM. The same
module is trained either centrally or as the local model in the federated
setting (see ``federated.py``).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def train_vae(
    x: np.ndarray,
    hidden_dims: list[int] | None = None,
    latent_dim: int = 8,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    beta: float = 1.0,
    seed: int = 42,
    device: str | None = None,
) -> VAE:
    """Train a VAE centrally and return the fitted module."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tensor = torch.tensor(x, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model = VAE(x.shape[1], hidden_dims, latent_dim, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def encode_latent(model: VAE, x: np.ndarray, device: str | None = None) -> np.ndarray:
    """Return the posterior-mean latent embedding for ``x``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    mu, _ = model.encode(tensor)
    return mu.cpu().numpy()
