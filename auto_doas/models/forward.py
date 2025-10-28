"""Differentiable forward model for Level-0 spectral synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.cross_sections import CrossSectionDatabase


def gaussian_kernel(width_px: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Construct a Gaussian convolution kernel for each batch element."""

    device = width_px.device
    half = (kernel_size - 1) / 2
    positions = torch.linspace(-half, half, kernel_size, device=device, dtype=width_px.dtype)
    kernel = torch.exp(-0.5 * (positions[None, :] / (width_px[:, None] + 1e-6)) ** 2)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return kernel


@dataclass
class InstrumentParameters:
    """Physical parameters describing an instrument."""

    wavelength_offset_nm: float = 0.0
    wavelength_scale: float = 1.0
    lsf_width_px: float = 1.0
    stray_light_fraction: float = 0.0
    nonlinear_response: float = 0.0

    def clamp(self) -> "InstrumentParameters":
        """Clamp the parameters to conservative priors."""

        return InstrumentParameters(
            wavelength_offset_nm=float(torch.clamp(torch.tensor(self.wavelength_offset_nm), -0.05, 0.05)),
            wavelength_scale=float(torch.clamp(torch.tensor(self.wavelength_scale), 0.995, 1.005)),
            lsf_width_px=float(torch.clamp(torch.tensor(self.lsf_width_px), 0.2, 5.0)),
            stray_light_fraction=float(torch.clamp(torch.tensor(self.stray_light_fraction), 0.0, 0.05)),
            nonlinear_response=float(torch.clamp(torch.tensor(self.nonlinear_response), -0.02, 0.02)),
        )


class InstrumentEmbedding(nn.Module):
    """Embedding table storing instrument specific parameters."""

    def __init__(self, num_instruments: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_instruments, embedding_dim)

    def forward(self, instrument_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(instrument_ids)


class AutoDOASForwardModel(nn.Module):
    """Combine atmospheric latents and instrument parameters to reconstruct spectra."""

    def __init__(
        self,
        wavelengths_nm: torch.Tensor,
        cross_sections: CrossSectionDatabase,
        continuum_basis: Optional[torch.Tensor] = None,
        num_instruments: int = 1,
        embedding_dim: int = 128,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.register_buffer("wavelengths_nm", wavelengths_nm.float())
        self.cross_sections = cross_sections.resample(self.wavelengths_nm)
        self.gases = self.cross_sections.gases
        self.absorption_matrix = nn.Parameter(
            self.cross_sections.absorption_matrix(self.gases), requires_grad=False
        )
        if continuum_basis is None:
            continuum_basis = torch.stack(
                [torch.ones_like(self.wavelengths_nm), self.wavelengths_nm / self.wavelengths_nm.mean()],
                dim=0,
            )
        self.register_buffer("continuum_basis", continuum_basis)
        self.instrument_embedding = InstrumentEmbedding(num_instruments, embedding_dim)
        self.kernel_size = kernel_size
        self.nuisance_head = nn.Sequential(
            nn.Linear(embedding_dim + len(self.gases), 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        gas_columns: torch.Tensor,
        instrument_ids: torch.Tensor,
        nuisance_latent: torch.Tensor,
        instrument_parameters: Optional[Dict[int, InstrumentParameters]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct spectra and return diagnostics."""

        batch_size = gas_columns.shape[0]
        device = gas_columns.device
        base_wavelengths = self.wavelengths_nm.to(device)
        absorption = self.absorption_matrix.to(device)
        optical_depth = torch.matmul(gas_columns, absorption)
        continuum = torch.matmul(
            torch.cat([gas_columns, torch.ones(batch_size, 1, device=device)], dim=1),
            self.continuum_basis[: gas_columns.shape[1] + 1].to(device),
        )
        differential = optical_depth + continuum

        instrument_embed = self.instrument_embedding(instrument_ids)
        nuisance_input = torch.cat([nuisance_latent, instrument_embed], dim=-1)
        nuisance_params = self.nuisance_head(nuisance_input)
        gain = torch.nn.functional.softplus(nuisance_params[:, 0]) + 1e-3
        offset = nuisance_params[:, 1]
        stray_light = torch.sigmoid(nuisance_params[:, 2]) * 0.05

        if instrument_parameters is not None:
            offsets = torch.tensor(
                [instrument_parameters[int(i.item())].wavelength_offset_nm for i in instrument_ids],
                device=device,
            )
            scales = torch.tensor(
                [instrument_parameters[int(i.item())].wavelength_scale for i in instrument_ids],
                device=device,
            )
            lsf_width = torch.tensor(
                [instrument_parameters[int(i.item())].lsf_width_px for i in instrument_ids],
                device=device,
            )
        else:
            offsets = torch.zeros(batch_size, device=device)
            scales = torch.ones(batch_size, device=device)
            lsf_width = torch.ones(batch_size, device=device)

        wavelengths = base_wavelengths[None, :] * scales[:, None] + offsets[:, None]

        kernel = gaussian_kernel(lsf_width, self.kernel_size)
        padded = F.pad(differential[:, None, :], (self.kernel_size // 2, self.kernel_size // 2), mode="replicate")
        convolved = F.conv1d(padded, kernel[:, None, :], groups=batch_size)[:, 0, :]
        simulated = torch.exp(-convolved)
        simulated = gain[:, None] * simulated + offset[:, None]
        simulated = (1 - stray_light[:, None]) * simulated + stray_light[:, None] * simulated.mean(dim=1, keepdim=True)
        diagnostics = {
            "optical_depth": differential.detach(),
            "gain": gain.detach(),
            "offset": offset.detach(),
            "stray_light": stray_light.detach(),
            "wavelengths_nm": wavelengths.detach(),
        }
        return simulated, diagnostics
