"""Differentiable forward model for Level-0 spectral synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.atmosphere import rayleigh_optical_depth
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
        include_rayleigh: bool = True,
        rayleigh_pressure_hpa: float = 1013.25,
        rayleigh_temperature_k: float = 288.15,
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
            nn.Linear(64, 7),
        )
        rayleigh = rayleigh_optical_depth(
            self.wavelengths_nm,
            pressure_hpa=rayleigh_pressure_hpa,
            temperature_k=rayleigh_temperature_k,
        )
        if not include_rayleigh:
            rayleigh = torch.zeros_like(rayleigh)
        self.register_buffer("rayleigh_optical_depth", rayleigh.float())
        self.include_rayleigh = include_rayleigh

    def forward(
        self,
        gas_columns: torch.Tensor,
        instrument_ids: torch.Tensor,
        nuisance_latent: torch.Tensor,
        air_mass_factors: Optional[torch.Tensor] = None,
        instrument_parameters: Optional[Dict[int, InstrumentParameters]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct spectra and return diagnostics."""

        batch_size = gas_columns.shape[0]
        device = gas_columns.device
        base_wavelengths = self.wavelengths_nm.to(device)
        absorption = self.absorption_matrix.to(device)
        if air_mass_factors is None:
            air_mass = torch.ones(batch_size, 1, device=device, dtype=gas_columns.dtype)
        else:
            air_mass = air_mass_factors.to(device=device, dtype=gas_columns.dtype)
            if air_mass.ndim == 1:
                air_mass = air_mass[:, None]
            elif air_mass.ndim == 2 and air_mass.shape != gas_columns.shape:
                raise ValueError(
                    "air_mass_factors must have shape [batch] or match gas_columns"
                )
            elif air_mass.ndim > 2:
                raise ValueError("air_mass_factors must be 1D or 2D tensor")
        scaled_columns = gas_columns * air_mass
        optical_depth = torch.matmul(scaled_columns, absorption)
        continuum = torch.matmul(
            torch.cat([gas_columns, torch.ones(batch_size, 1, device=device)], dim=1),
            self.continuum_basis[: gas_columns.shape[1] + 1].to(device),
        )
        differential = optical_depth + continuum

        rayleigh_component = None
        if self.include_rayleigh:
            if air_mass.ndim == 1:
                rayleigh_amf = air_mass[:, None]
            elif air_mass.ndim == 2:
                rayleigh_amf = air_mass.mean(dim=1, keepdim=True)
            else:
                raise ValueError("air_mass_factors must be 1D or 2D tensor")
            rayleigh_component = rayleigh_amf * self.rayleigh_optical_depth.to(
                device=device, dtype=gas_columns.dtype
            )[None, :]
            differential = differential + rayleigh_component

        instrument_embed = self.instrument_embedding(instrument_ids)
        nuisance_input = torch.cat([nuisance_latent, instrument_embed], dim=-1)
        nuisance_params = self.nuisance_head(nuisance_input)
        gain = F.softplus(nuisance_params[:, 0]) + 1e-3
        offset = nuisance_params[:, 1]
        predicted_wavelength_offset = 0.05 * torch.tanh(nuisance_params[:, 2])
        predicted_wavelength_scale = 1.0 + 0.005 * torch.tanh(nuisance_params[:, 3])
        predicted_lsf_width = torch.clamp(F.softplus(nuisance_params[:, 4]) + 1e-3, 0.2, 5.0)
        predicted_stray = torch.sigmoid(nuisance_params[:, 5]) * 0.05
        predicted_nonlinear = 0.02 * torch.tanh(nuisance_params[:, 6])

        applied_offsets = predicted_wavelength_offset.clone()
        applied_scales = predicted_wavelength_scale.clone()
        applied_lsf = predicted_lsf_width.clone()
        instrument_stray = torch.zeros(batch_size, device=device, dtype=base_wavelengths.dtype)
        instrument_nonlinear = torch.zeros(
            batch_size, device=device, dtype=base_wavelengths.dtype
        )
        if instrument_parameters is not None:
            offsets_override = torch.zeros(
                batch_size, device=device, dtype=base_wavelengths.dtype
            )
            scales_override = torch.ones(
                batch_size, device=device, dtype=base_wavelengths.dtype
            )
            lsf_override = torch.zeros(
                batch_size, device=device, dtype=base_wavelengths.dtype
            )
            override_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            for idx, instrument_idx in enumerate(instrument_ids.tolist()):
                params = None
                key = int(instrument_idx)
                try:
                    params = instrument_parameters[key]
                except KeyError:
                    params = None
                if params is None:
                    continue
                clamped = params.clamp()
                override_mask[idx] = True
                offsets_override[idx] = clamped.wavelength_offset_nm
                scales_override[idx] = clamped.wavelength_scale
                lsf_override[idx] = clamped.lsf_width_px
                instrument_stray[idx] = clamped.stray_light_fraction
                instrument_nonlinear[idx] = clamped.nonlinear_response
            applied_offsets = torch.where(override_mask, offsets_override, applied_offsets)
            applied_scales = torch.where(override_mask, scales_override, applied_scales)
            applied_lsf = torch.where(override_mask, lsf_override, applied_lsf)

        wavelengths = base_wavelengths[None, :] * applied_scales[:, None] + applied_offsets[:, None]

        differential = self._resample_spectrum(differential, base_wavelengths, wavelengths)

        kernel = gaussian_kernel(applied_lsf, self.kernel_size)
        padded = F.pad(
            differential[:, None, :],
            (self.kernel_size // 2, self.kernel_size // 2),
            mode="replicate",
        )
        padded_group = padded.transpose(0, 1)
        convolved = F.conv1d(padded_group, kernel[:, None, :], groups=batch_size)
        convolved = convolved.transpose(0, 1)[:, 0, :]
        absorption_counts = torch.exp(-convolved)
        post_gain_counts = gain[:, None] * absorption_counts + offset[:, None]
        total_nonlinear = torch.clamp(predicted_nonlinear + instrument_nonlinear, -0.04, 0.04)
        nonlinear_counts = post_gain_counts + total_nonlinear[:, None] * post_gain_counts**2
        total_stray = torch.clamp(predicted_stray + instrument_stray, 0.0, 0.2)
        simulated = (1 - total_stray[:, None]) * nonlinear_counts + total_stray[:, None] * nonlinear_counts.mean(dim=1, keepdim=True)
        diagnostics = {
            "optical_depth_component": optical_depth.detach(),
            "optical_depth": differential.detach(),
            "gain": gain.detach(),
            "offset": offset.detach(),
            "stray_light": total_stray.detach(),
            "wavelengths_nm": wavelengths.detach(),
            "predicted_stray_light": predicted_stray.detach(),
            "instrument_stray_light": instrument_stray.detach(),
            "predicted_wavelength_offset": predicted_wavelength_offset.detach(),
            "predicted_wavelength_scale": predicted_wavelength_scale.detach(),
            "predicted_lsf_width": predicted_lsf_width.detach(),
            "applied_wavelength_offset": applied_offsets.detach(),
            "applied_wavelength_scale": applied_scales.detach(),
            "applied_lsf_width": applied_lsf.detach(),
            "predicted_nonlinearity": predicted_nonlinear.detach(),
            "instrument_nonlinearity": instrument_nonlinear.detach(),
            "total_nonlinearity": total_nonlinear.detach(),
            "post_gain_counts": post_gain_counts.detach(),
            "pre_stray_counts": nonlinear_counts.detach(),
            "air_mass_factor": air_mass.detach(),
            "effective_gas_columns": scaled_columns.detach(),
        }
        if rayleigh_component is not None:
            diagnostics["rayleigh_optical_depth"] = rayleigh_component.detach()
        return simulated, diagnostics

    def _resample_spectrum(
        self,
        spectrum: torch.Tensor,
        base_wavelengths: torch.Tensor,
        target_wavelengths: torch.Tensor,
    ) -> torch.Tensor:
        """Resample ``spectrum`` defined on ``base_wavelengths`` onto ``target_wavelengths``."""

        if spectrum.ndim != 2:
            raise ValueError("spectrum must have shape [batch, num_wavelengths]")
        base_min = float(base_wavelengths[0])
        base_max = float(base_wavelengths[-1])
        if base_max - base_min <= 0:
            raise ValueError("base_wavelengths must span a non-zero interval")

        normalized = (target_wavelengths - base_min) / (base_max - base_min)
        normalized = normalized * 2 - 1
        normalized = torch.clamp(normalized, -1.0, 1.0)
        zeros = torch.zeros_like(normalized)
        grid = torch.stack([zeros, normalized], dim=-1)[:, None, :, :]
        input_tensor = spectrum[:, None, None, :]
        resampled = F.grid_sample(
            input_tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return resampled[:, 0, 0, :]
