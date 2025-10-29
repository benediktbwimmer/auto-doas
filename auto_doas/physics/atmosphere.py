"""Atmospheric physics utilities for DOAS retrievals."""

from __future__ import annotations

from typing import Union

import torch

TensorLike = Union[torch.Tensor, float, int]


def rayleigh_optical_depth(
    wavelengths_nm: TensorLike,
    pressure_hpa: float = 1013.25,
    temperature_k: float = 288.15,
) -> torch.Tensor:
    """Compute the molecular (Rayleigh) scattering optical depth.

    The implementation follows the parameterization by Bodhaine et al. (1999)
    which expresses the Rayleigh optical depth as a function of wavelength,
    surface pressure and temperature.  Wavelengths are provided in nanometers
    and the resulting optical depth is dimensionless.  The pressure dependence
    is linear while the temperature dependence scales with ``288.15 / T`` which
    reflects the variation of the molecular number density with temperature.

    Args:
        wavelengths_nm: Wavelength grid expressed in nanometers.  Scalars and
            tensors are accepted and automatically converted to tensors.
        pressure_hpa: Surface pressure in hectopascals (hPa).  Defaults to the
            standard atmosphere of 1013.25 hPa.
        temperature_k: Atmospheric temperature in Kelvin.  Defaults to 288.15 K.

    Returns:
        Torch tensor containing the Rayleigh optical depth evaluated at each
        wavelength.

    Raises:
        ValueError: If any wavelength is non-positive.
    """

    if not isinstance(wavelengths_nm, torch.Tensor):
        wavelengths = torch.tensor(wavelengths_nm, dtype=torch.float32)
    else:
        wavelengths = wavelengths_nm.to(dtype=torch.float32)

    if torch.any(wavelengths <= 0):
        raise ValueError("wavelengths_nm must be strictly positive")

    # Convert to micrometers for the Bodhaine et al. (1999) coefficients.
    wavelengths_um = wavelengths / 1000.0
    inv_wavelength_sq = wavelengths_um.reciprocal() ** 2

    # Polynomial fit for Rayleigh optical depth (dimensionless).
    base_optical_depth = 0.008569 * wavelengths_um.pow(-4) * (
        1.0 + 0.0113 * inv_wavelength_sq + 0.00013 * inv_wavelength_sq**2
    )

    pressure_scale = float(pressure_hpa) / 1013.25
    temperature_scale = 288.15 / float(temperature_k)
    optical_depth = base_optical_depth * pressure_scale * temperature_scale
    return optical_depth


__all__ = ["rayleigh_optical_depth"]
