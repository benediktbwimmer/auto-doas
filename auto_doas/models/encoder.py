"""Encoder network for Level-0 spectra."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SpectralEncoder(nn.Module):
    """Convolutional encoder that produces latent representations from spectra."""

    def __init__(self, num_wavelengths: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        x = counts[:, None, :]
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class AutoDOASEncoder(nn.Module):
    """Predict gas columns and instrument nuisances from Level-0 spectra."""

    def __init__(
        self,
        num_wavelengths: int,
        num_gases: int,
        latent_dim: int = 128,
        nuisance_dim: int = 64,
    ) -> None:
        super().__init__()
        self.spectral_encoder = SpectralEncoder(num_wavelengths, latent_dim)
        self.gas_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_gases),
            nn.Softplus(),
        )
        self.nuisance_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, nuisance_dim),
        )

    def forward(self, counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.spectral_encoder(counts)
        gas_columns = self.gas_head(embedding)
        nuisance = self.nuisance_head(embedding)
        return gas_columns, nuisance
