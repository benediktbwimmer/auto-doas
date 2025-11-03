"""Shared solar reference models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SolarReferenceConfig:
    """Configuration for :class:`SharedSolarReference`."""

    decay: float = 0.01
    min_count: float = 1.0


class SharedSolarReference(nn.Module):
    """Maintain a shared log-irradiance reference across instruments.

    The module keeps an exponential moving average of the log-count spectra weighted by
    the cosine of the solar zenith angle.  This provides a stable, instrument-agnostic
    reference spectrum that can be injected into the forward model to anchor the
    continuum to the solar illumination rather than instrument specific offsets.
    """

    def __init__(
        self,
        wavelengths_nm: torch.Tensor,
        config: Optional[SolarReferenceConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or SolarReferenceConfig()
        self.decay = float(cfg.decay)
        self.min_count = float(cfg.min_count)
        self.register_buffer("wavelengths_nm", wavelengths_nm.clone().float())
        self.register_buffer("log_reference", torch.zeros_like(self.wavelengths_nm))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(
        self,
        counts: torch.Tensor,
        solar_zenith_angle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update the running solar reference and return the current log-reference."""

        device = self.log_reference.device
        dtype = self.log_reference.dtype
        spectra = counts.to(device=device, dtype=dtype)
        log_spectra = torch.log(torch.clamp(spectra, min=self.min_count))
        if solar_zenith_angle is not None:
            weights = torch.cos(torch.deg2rad(solar_zenith_angle.to(device=device, dtype=dtype)))
            weights = torch.clamp(weights, min=0.0)
        else:
            weights = torch.ones(spectra.shape[0], device=device, dtype=dtype)
        if torch.all(weights <= 0):
            return self.log_reference.clone()
        weighted_mean = (weights[:, None] * log_spectra).sum(dim=0) / (weights.sum() + 1e-6)
        if bool(self.initialized.item()):
            self.log_reference.mul_(1.0 - self.decay).add_(self.decay * weighted_mean)
        else:
            self.log_reference.copy_(weighted_mean)
            self.initialized.copy_(torch.tensor(True, device=device))
        return self.log_reference.clone()

    def reference(self) -> torch.Tensor:
        """Return the current log-reference spectrum."""

        return self.log_reference.clone()

    def irradiance(self) -> torch.Tensor:
        """Return the exponential of the log-reference."""

        return torch.exp(self.log_reference)


__all__ = ["SharedSolarReference", "SolarReferenceConfig"]

