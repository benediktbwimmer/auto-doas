"""Context encoders for physics informed DOAS models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _resolve_device_dtype(*tensors: Optional[torch.Tensor]) -> Tuple[torch.device, torch.dtype]:
    """Return common device and dtype inferred from provided tensors."""

    for tensor in tensors:
        if tensor is not None:
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float32


def _ensure_shape(
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    fill: float = 0.0,
) -> torch.Tensor:
    """Return ``value`` broadcast to ``[batch_size, 1]`` or create a filler tensor."""

    if value is None:
        return torch.full((batch_size, 1), fill, device=device, dtype=dtype)
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape[0] != batch_size:
        raise ValueError("Context tensor batch dimension mismatch")
    return tensor.reshape(batch_size, 1)


@dataclass
class ObservationContextConfig:
    """Configuration for :class:`ObservationContextEncoder`."""

    embedding_dim: int = 64
    time_fourier_terms: int = 4
    angle_fourier_terms: int = 3
    metadata_scaler: float = 0.1
    day_period_seconds: float = 86_400.0


class ObservationContextEncoder(nn.Module):
    """Encode observation metadata into an instrument-agnostic context embedding.

    The encoder maps solar geometry, acquisition timing and detector state into a shared
    representation that can be concatenated with encoder nuisance latents.  By relying on the
    shared solar illumination cycle instead of instrument identifiers the forward model captures
    long-term, instrument-agnostic structure that is consistent across the observing network.
    """

    def __init__(self, config: ObservationContextConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Sequential(
            nn.Linear(self._input_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    @property
    def _input_dim(self) -> int:
        cfg = self.config
        # Fourier terms generate sin/cos pairs.
        time_dim = 2 * cfg.time_fourier_terms
        angle_dim = 2 * cfg.angle_fourier_terms * 3  # solar, viewing, relative azimuth
        metadata_dim = 3  # exposure, ccd temperature, solar zenith mean
        return time_dim + angle_dim + metadata_dim

    def forward(
        self,
        batch_size: int,
        timestamps: Optional[torch.Tensor] = None,
        solar_zenith_angle: Optional[torch.Tensor] = None,
        viewing_zenith_angle: Optional[torch.Tensor] = None,
        relative_azimuth_angle: Optional[torch.Tensor] = None,
        exposure_time: Optional[torch.Tensor] = None,
        ccd_temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device, dtype = _resolve_device_dtype(
            timestamps,
            solar_zenith_angle,
            viewing_zenith_angle,
            relative_azimuth_angle,
            exposure_time,
            ccd_temperature,
        )
        if all(
            value is None
            for value in (
                timestamps,
                solar_zenith_angle,
                viewing_zenith_angle,
                relative_azimuth_angle,
                exposure_time,
                ccd_temperature,
            )
        ):
            param = next(self.parameters(), None)
            if param is not None:
                device = param.device
                dtype = param.dtype

        timestamps = _ensure_shape(timestamps, batch_size, device, dtype)
        solar = _ensure_shape(solar_zenith_angle, batch_size, device, dtype)
        viewing = _ensure_shape(viewing_zenith_angle, batch_size, device, dtype)
        relative = _ensure_shape(relative_azimuth_angle, batch_size, device, dtype)
        exposure = _ensure_shape(exposure_time, batch_size, device, dtype)
        ccd_temp = _ensure_shape(ccd_temperature, batch_size, device, dtype)

        cfg = self.config
        twopi = 2 * torch.pi

        day_phase = (timestamps % cfg.day_period_seconds) / cfg.day_period_seconds
        time_terms = torch.arange(
            1,
            cfg.time_fourier_terms + 1,
            device=device,
            dtype=dtype,
        )
        time_angles = day_phase * time_terms * twopi
        time_features = torch.cat(
            [torch.sin(time_angles), torch.cos(time_angles)],
            dim=-1,
        )

        def _fourier(angle: torch.Tensor) -> torch.Tensor:
            terms = torch.arange(
                1,
                cfg.angle_fourier_terms + 1,
                device=device,
                dtype=dtype,
            )
            rad = torch.deg2rad(angle)
            phases = rad * terms
            return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

        solar_features = _fourier(solar)
        viewing_features = _fourier(viewing)
        relative_features = _fourier(relative)

        normalized_exposure = exposure * cfg.metadata_scaler
        normalized_temperature = ccd_temp * cfg.metadata_scaler
        normalized_sza = solar / 90.0  # solar zenith in [0, 180]

        metadata = torch.cat(
            [normalized_exposure, normalized_temperature, normalized_sza],
            dim=-1,
        )

        raw_features = torch.cat(
            [
                time_features,
                solar_features,
                viewing_features,
                relative_features,
                metadata,
            ],
            dim=-1,
        )
        return self.project(raw_features)


__all__ = ["ObservationContextEncoder", "ObservationContextConfig"]
