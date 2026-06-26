"""Smoke tests exercising the pipeline on a small example cohort."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.example.generate_example import generate  # noqa: E402
from src.preprocessing import drop_high_missing, encode_features, mice_impute, winsorize  # noqa: E402
from src.model import encode_latent, train_vae  # noqa: E402
from src.model.som import SOM  # noqa: E402
from src.model.privacy import RDPAccountant, calibrate_noise  # noqa: E402
from src.validation import consensus_stability, internal_metrics, sweep_k  # noqa: E402


def _encoded(n=120, seed=0):
    df = generate(n, seed).drop(columns=["patient_id", "diagnosis"])
    df = mice_impute(winsorize(drop_high_missing(df)), seed=seed)
    return encode_features(df)


def test_preprocessing_no_nans():
    enc = _encoded()
    assert not np.isnan(enc.matrix).any()
    assert enc.matrix.shape[0] == 120


def test_vae_latent_shape():
    enc = _encoded()
    model = train_vae(enc.matrix, hidden_dims=[16], latent_dim=4, epochs=3, batch_size=32)
    latent = encode_latent(model, enc.matrix)
    assert latent.shape == (120, 4)


def test_som_predict_range():
    enc = _encoded()
    latent = encode_latent(
        train_vae(enc.matrix, hidden_dims=[16], latent_dim=4, epochs=2), enc.matrix
    )
    som = SOM(grid=(4, 4), input_dim=4, iterations=200).train(latent)
    nodes = som.predict(latent)
    assert nodes.min() >= 0 and nodes.max() < 16


def test_validation_metrics():
    enc = _encoded()
    latent = encode_latent(
        train_vae(enc.matrix, hidden_dims=[16], latent_dim=4, epochs=2), enc.matrix
    )
    sweep = sweep_k(latent, [2, 3, 4])
    assert set(sweep) == {2, 3, 4}
    labels = np.zeros(len(latent), dtype=int)
    labels[: len(latent) // 2] = 1
    assert "silhouette" in internal_metrics(latent, labels)
    stab = consensus_stability(latent, k=3, resamples=10)
    assert 0.0 <= stab["mean_consensus"] <= 1.0


def test_privacy_accountant_monotone():
    nm = calibrate_noise(0.1, 100, target_epsilon=5.0, target_delta=1e-5)
    eps = RDPAccountant(0.1, nm).epsilon(100, 1e-5)
    assert eps <= 5.0 + 0.5
