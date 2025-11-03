"""Differentiable forward model for Level-0 spectral synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.atmosphere import rayleigh_optical_depth
from ..physics.cross_sections import CrossSectionDatabase
from ..physics.geometry import (
    geometric_air_mass_factor,
    exponential_air_mass_factor,
)
from .context import ObservationContextConfig, ObservationContextEncoder


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


class AutoDOASForwardModel(nn.Module):
    """Combine atmospheric latents and instrument parameters to reconstruct spectra."""

    def __init__(
        self,
        wavelengths_nm: torch.Tensor,
        cross_sections: CrossSectionDatabase,
        continuum_basis: Optional[torch.Tensor] = None,
        num_instruments: int = 1,
        embedding_dim: int = 128,
        context_time_fourier_terms: int = 4,
        context_angle_fourier_terms: int = 3,
        context_metadata_scaler: float = 0.1,
        context_day_period_seconds: float = 86_400.0,
        kernel_size: int = 15,
        include_rayleigh: bool = True,
        rayleigh_pressure_hpa: float = 1013.25,
        rayleigh_temperature_k: float = 288.15,
        air_mass_mode: str = "geometric",
        air_mass_scale_height_km: float = 7.0,
        air_mass_max_altitude_km: float = 60.0,
        air_mass_earth_radius_km: float = 6371.0,
        air_mass_num_samples: int = 256,
        air_mass_max_angle_deg: float = 89.0,
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
        self.num_instruments = int(num_instruments)
        context_config = ObservationContextConfig(
            embedding_dim=embedding_dim,
            time_fourier_terms=context_time_fourier_terms,
            angle_fourier_terms=context_angle_fourier_terms,
            metadata_scaler=context_metadata_scaler,
            day_period_seconds=context_day_period_seconds,
        )
        self.context_encoder = ObservationContextEncoder(context_config)
        self.kernel_size = kernel_size
        self.nuisance_head = nn.Sequential(
            nn.LazyLinear(64),
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
        if air_mass_mode not in {"geometric", "chapman"}:
            raise ValueError("air_mass_mode must be 'geometric' or 'chapman'")
        self.air_mass_mode = air_mass_mode
        self.air_mass_scale_height_km = float(air_mass_scale_height_km)
        self.air_mass_max_altitude_km = float(air_mass_max_altitude_km)
        self.air_mass_earth_radius_km = float(air_mass_earth_radius_km)
        self.air_mass_num_samples = int(air_mass_num_samples)
        if self.air_mass_num_samples < 2:
            raise ValueError("air_mass_num_samples must be at least 2")
        self.air_mass_max_angle_deg = float(air_mass_max_angle_deg)

    def forward(
        self,
        gas_columns: torch.Tensor,
        instrument_ids: torch.Tensor,
        nuisance_latent: torch.Tensor,
        air_mass_factors: Optional[torch.Tensor] = None,
        instrument_parameters: Optional[Dict[int, InstrumentParameters]] = None,
        solar_zenith_angle: Optional[torch.Tensor] = None,
        viewing_zenith_angle: Optional[torch.Tensor] = None,
        relative_azimuth_angle: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        exposure_time: Optional[torch.Tensor] = None,
        ccd_temperature: Optional[torch.Tensor] = None,
        solar_reference: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reconstruct spectra and return diagnostics."""

        batch_size = gas_columns.shape[0]
        device = gas_columns.device
        dtype = gas_columns.dtype
        base_wavelengths = self.wavelengths_nm.to(device)
        absorption = self.absorption_matrix.to(device)
        solar_amf_component: Optional[torch.Tensor] = None
        viewing_amf_component: Optional[torch.Tensor] = None
        viewing_weight: Optional[torch.Tensor] = None
        air_mass: Optional[torch.Tensor] = None

        def _to_tensor(value: Optional[torch.Tensor | float | int]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=dtype)
            return torch.tensor(value, device=device, dtype=dtype)

        def _prepare(value: Optional[torch.Tensor | float | int]) -> Optional[torch.Tensor]:
            tensor = _to_tensor(value)
            if tensor is not None and tensor.ndim == 0:
                tensor = tensor.expand(batch_size)
            return tensor

        solar_angles = _prepare(solar_zenith_angle)
        viewing_angles = _prepare(viewing_zenith_angle)
        relative_angles = _prepare(relative_azimuth_angle)
        timestamp_values = _prepare(timestamps)
        exposure_values = _prepare(exposure_time)
        temperature_values = _prepare(ccd_temperature)

        if air_mass_factors is None:
            def _compute_component(angle_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if angle_tensor is None:
                    return None
                if self.air_mass_mode == "chapman":
                    component = exponential_air_mass_factor(
                        angle_tensor,
                        scale_height_km=self.air_mass_scale_height_km,
                        max_altitude_km=self.air_mass_max_altitude_km,
                        earth_radius_km=self.air_mass_earth_radius_km,
                        num_samples=self.air_mass_num_samples,
                        max_angle_deg=self.air_mass_max_angle_deg,
                    )
                else:
                    component = geometric_air_mass_factor(
                        angle_tensor, max_angle_deg=self.air_mass_max_angle_deg
                    )
                return component.to(device=device, dtype=dtype)

            if solar_angles is not None:
                solar_amf_component = _compute_component(solar_angles)
                solar_amf_component = solar_amf_component.reshape(batch_size, -1)[:, 0]
            if viewing_angles is not None:
                viewing_amf_component = _compute_component(viewing_angles)
                viewing_amf_component = viewing_amf_component.reshape(batch_size, -1)[:, 0]
            if solar_amf_component is not None and viewing_amf_component is not None:
                if relative_angles is not None:
                    relative = torch.clamp(relative_angles.reshape(batch_size, -1)[:, 0], 0.0, 180.0)
                    viewing_weight = 0.5 * (
                        1.0 + torch.cos(torch.deg2rad(relative))
                    )
                else:
                    viewing_weight = torch.ones(
                        batch_size, device=device, dtype=dtype
                    )
                viewing_weight = viewing_weight.to(dtype=dtype)
                air_mass = solar_amf_component + viewing_weight * viewing_amf_component
            elif solar_amf_component is not None:
                air_mass = solar_amf_component
            elif viewing_amf_component is not None:
                air_mass = viewing_amf_component
            else:
                air_mass = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            air_mass = air_mass_factors.to(device=device, dtype=dtype)
            if air_mass.ndim == 1:
                air_mass = air_mass[:, None]
            elif air_mass.ndim == 2 and air_mass.shape != gas_columns.shape:
                raise ValueError(
                    "air_mass_factors must have shape [batch] or match gas_columns"
                )
            elif air_mass.ndim > 2:
                raise ValueError("air_mass_factors must be 1D or 2D tensor")
        if air_mass.ndim == 1:
            air_mass = air_mass[:, None]
        scaled_columns = gas_columns * air_mass
        optical_depth = torch.matmul(scaled_columns, absorption)
        continuum = torch.matmul(
            torch.cat([gas_columns, torch.ones(batch_size, 1, device=device)], dim=1),
            self.continuum_basis[: gas_columns.shape[1] + 1].to(device),
        )
        differential = optical_depth + continuum
        solar_reference_component = None
        if solar_reference is not None:
            solar_reference_component = solar_reference.to(device=device, dtype=dtype)
            if solar_reference_component.ndim == 1:
                solar_reference_component = solar_reference_component[None, :]
            if solar_reference_component.shape[0] == 1:
                solar_reference_component = solar_reference_component.expand(batch_size, -1)
            differential = differential + solar_reference_component

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

        context_embedding = self.context_encoder(
            batch_size=batch_size,
            timestamps=timestamp_values,
            solar_zenith_angle=solar_angles,
            viewing_zenith_angle=viewing_angles,
            relative_azimuth_angle=relative_angles,
            exposure_time=exposure_values,
            ccd_temperature=temperature_values,
        ).to(device=device, dtype=dtype)
        nuisance_input = torch.cat([nuisance_latent, context_embedding], dim=-1)
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
            "context_embedding": context_embedding.detach(),
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
        if solar_amf_component is not None:
            diagnostics["solar_air_mass_factor"] = solar_amf_component.detach()
        if viewing_amf_component is not None:
            diagnostics["viewing_air_mass_factor"] = viewing_amf_component.detach()
        if viewing_weight is not None:
            diagnostics["viewing_air_mass_weight"] = viewing_weight.detach()
        if rayleigh_component is not None:
            diagnostics["rayleigh_optical_depth"] = rayleigh_component.detach()
        if solar_reference_component is not None:
            diagnostics["solar_reference_log"] = solar_reference_component.detach()
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
