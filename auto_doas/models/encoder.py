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
        )
        self.nuisance_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, nuisance_dim),
        )
        self.log_calibration_scale = nn.Parameter(torch.zeros(num_gases))
        self.calibration_bias = nn.Parameter(torch.zeros(num_gases))

    def forward(self, counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.spectral_encoder(counts)
        raw_columns = self.gas_head(embedding)
        scale = torch.exp(self.log_calibration_scale)[None, :]
        gas_columns = scale * raw_columns + self.calibration_bias[None, :]
        nuisance = self.nuisance_head(embedding)
        return gas_columns, nuisance

    @torch.no_grad()
    def set_calibration(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        """Initialise calibration scale and bias parameters."""

        scale_tensor = torch.as_tensor(scale, dtype=self.log_calibration_scale.dtype, device=self.log_calibration_scale.device)
        bias_tensor = torch.as_tensor(bias, dtype=self.calibration_bias.dtype, device=self.calibration_bias.device)
        if scale_tensor.shape != self.log_calibration_scale.shape:
            raise ValueError(f"Expected scale shape {self.log_calibration_scale.shape}, got {scale_tensor.shape}")
        if bias_tensor.shape != self.calibration_bias.shape:
            raise ValueError(f"Expected bias shape {self.calibration_bias.shape}, got {bias_tensor.shape}")
        if torch.any(scale_tensor <= 0):
            raise ValueError("Calibration scale factors must be positive.")
        self.log_calibration_scale.copy_(torch.log(scale_tensor))
        self.calibration_bias.copy_(bias_tensor)

    @torch.no_grad()
    def get_calibration(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current calibration scale and bias as detached tensors."""

        scale = torch.exp(self.log_calibration_scale.detach())
        bias = self.calibration_bias.detach().clone()
        return scale, bias
