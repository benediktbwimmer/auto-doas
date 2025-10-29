"""Geometric utilities for atmospheric radiative transfer."""

from __future__ import annotations

from typing import Optional, Union

import torch

TensorLike = Union[torch.Tensor, float, int]


def geometric_air_mass_factor(
    solar_zenith_angle_deg: TensorLike, max_angle_deg: float = 89.0
) -> torch.Tensor:
    """Compute the geometric air mass factor from the solar zenith angle.

    The implementation follows the empirical formulation by Kasten & Young (1989)
    which provides a smooth approximation of the optical path length through the
    atmosphere for solar zenith angles below 90 degrees.  Angles equal to or above
    the horizon are clipped to ``max_angle_deg`` to avoid numerical instabilities.

    Args:
        solar_zenith_angle_deg: Solar zenith angle in degrees.  Scalars and tensors
            are accepted and broadcast as necessary.
        max_angle_deg: Maximum allowed zenith angle.  Inputs larger than this value
            are clipped to maintain numerical stability.  The default corresponds to
            a sun that is one degree above the horizon.

    Returns:
        Torch tensor containing the geometric air mass factors with the same shape
        as the broadcast solar zenith angle tensor.
    """

    if not isinstance(solar_zenith_angle_deg, torch.Tensor):
        angles = torch.tensor(float(solar_zenith_angle_deg), dtype=torch.float32)
    else:
        angles = solar_zenith_angle_deg.to(dtype=torch.float32)

    max_angle = torch.tensor(max_angle_deg, dtype=angles.dtype, device=angles.device)
    clipped = torch.clamp(angles, 0.0, max_angle.item())

    radians = torch.deg2rad(clipped)
    cosine = torch.clamp(torch.cos(radians), min=1e-3)

    # Kasten & Young (1989) empirical fit.  The coefficients operate on degree values.
    denominator = cosine + 0.50572 * torch.pow(96.07995 - clipped, -1.6364)
    return 1.0 / denominator


def double_geometric_air_mass_factor(
    solar_zenith_angle_deg: TensorLike,
    viewing_zenith_angle_deg: TensorLike,
    relative_azimuth_angle_deg: Optional[TensorLike] = None,
    max_angle_deg: float = 89.0,
) -> torch.Tensor:
    """Two-leg geometric air mass factor for sun-to-surface and surface-to-sensor paths.

    The function approximates the effective air mass factor for passive DOAS retrievals
    where photons travel from the sun to the scattering point and then towards the
    instrument.  Both the incoming solar zenith angle (SZA) and the viewing zenith
    angle (VZA) are modeled using the same Kasten & Young (1989) formulation employed
    by :func:`geometric_air_mass_factor`.  When a relative azimuth angle is supplied we
    apply a cosine-based weight to the viewing path to mimic the reduced contribution
    for back-scattered geometries.

    Args:
        solar_zenith_angle_deg: Solar zenith angle(s) in degrees.
        viewing_zenith_angle_deg: Viewing zenith angle(s) in degrees.
        relative_azimuth_angle_deg: Optional relative azimuth angle(s) in degrees.
            ``0°`` corresponds to forward scattering whereas ``180°`` represents a
            pure back-scattering configuration.  When omitted the viewing path is
            fully counted.
        max_angle_deg: Maximum zenith angle applied to both SZA and VZA.

    Returns:
        Torch tensor with the same broadcast shape as the input angles containing
        the effective geometric air mass factor for a two-leg photon path.
    """

    solar = geometric_air_mass_factor(solar_zenith_angle_deg, max_angle_deg=max_angle_deg)
    viewing = geometric_air_mass_factor(viewing_zenith_angle_deg, max_angle_deg=max_angle_deg)

    if relative_azimuth_angle_deg is None:
        weight = torch.ones_like(viewing)
    else:
        if not isinstance(relative_azimuth_angle_deg, torch.Tensor):
            relative = torch.tensor(relative_azimuth_angle_deg, dtype=torch.float32, device=viewing.device)
        else:
            relative = relative_azimuth_angle_deg.to(device=viewing.device, dtype=torch.float32)
        relative = torch.clamp(relative, 0.0, 180.0)
        weight = 0.5 * (1.0 + torch.cos(torch.deg2rad(relative)))
        weight = weight.to(dtype=viewing.dtype)

    return solar + weight * viewing


def exponential_air_mass_factor(
    zenith_angle_deg: TensorLike,
    scale_height_km: float = 7.0,
    max_altitude_km: float = 60.0,
    earth_radius_km: float = 6371.0,
    num_samples: int = 256,
    max_angle_deg: float = 89.0,
) -> torch.Tensor:
    """Chapman-style air mass factor for an exponential atmosphere."""

    if num_samples < 2:
        raise ValueError("num_samples must be at least 2 for numerical integration")

    if not isinstance(zenith_angle_deg, torch.Tensor):
        angles = torch.tensor(float(zenith_angle_deg), dtype=torch.float32)
    else:
        angles = zenith_angle_deg.to(dtype=torch.float32)

    device = angles.device
    max_angle = torch.tensor(max_angle_deg, dtype=angles.dtype, device=device)
    angles = torch.clamp(angles, 0.0, max_angle.item())

    original_shape = angles.shape
    angles_flat = angles.reshape(-1)
    radians = torch.deg2rad(angles_flat)

    radius = torch.tensor(earth_radius_km, dtype=angles.dtype, device=device)
    scale_height = torch.tensor(scale_height_km, dtype=angles.dtype, device=device)
    top_of_atmosphere = radius + torch.tensor(max_altitude_km, dtype=angles.dtype, device=device)

    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    term = torch.clamp(top_of_atmosphere**2 - (radius**2) * sine**2, min=1e-6)
    path_length = -radius * cosine + torch.sqrt(term)
    path_length = torch.clamp(path_length, min=1e-6)

    samples = torch.linspace(
        0.0, 1.0, steps=num_samples, dtype=angles.dtype, device=device
    )
    distances = path_length[:, None] * samples[None, :]
    radial_distance = torch.sqrt(
        radius**2 + distances**2 + 2.0 * radius * distances * cosine[:, None]
    )
    altitude = radial_distance - radius
    density = torch.exp(-altitude / scale_height)
    slant_column = torch.trapz(density, distances, dim=1)

    vertical_column = scale_height * (
        1.0
        - torch.exp(
            -torch.tensor(max_altitude_km, dtype=angles.dtype, device=device)
            / scale_height
        )
    )
    air_mass = (slant_column / vertical_column).reshape(original_shape)
    return air_mass


def double_exponential_air_mass_factor(
    solar_zenith_angle_deg: TensorLike,
    viewing_zenith_angle_deg: TensorLike,
    relative_azimuth_angle_deg: Optional[TensorLike] = None,
    scale_height_km: float = 7.0,
    max_altitude_km: float = 60.0,
    earth_radius_km: float = 6371.0,
    num_samples: int = 256,
    max_angle_deg: float = 89.0,
) -> torch.Tensor:
    """Two-leg air mass factor using an exponential atmosphere approximation."""

    if not isinstance(solar_zenith_angle_deg, torch.Tensor):
        solar = torch.tensor(float(solar_zenith_angle_deg), dtype=torch.float32)
    else:
        solar = solar_zenith_angle_deg.to(dtype=torch.float32)
    if not isinstance(viewing_zenith_angle_deg, torch.Tensor):
        viewing = torch.tensor(float(viewing_zenith_angle_deg), dtype=torch.float32)
    else:
        viewing = viewing_zenith_angle_deg.to(dtype=torch.float32)

    solar, viewing = torch.broadcast_tensors(solar, viewing)
    solar_factor = exponential_air_mass_factor(
        solar,
        scale_height_km=scale_height_km,
        max_altitude_km=max_altitude_km,
        earth_radius_km=earth_radius_km,
        num_samples=num_samples,
        max_angle_deg=max_angle_deg,
    )
    viewing_factor = exponential_air_mass_factor(
        viewing,
        scale_height_km=scale_height_km,
        max_altitude_km=max_altitude_km,
        earth_radius_km=earth_radius_km,
        num_samples=num_samples,
        max_angle_deg=max_angle_deg,
    )

    if relative_azimuth_angle_deg is None:
        weight = torch.ones_like(viewing_factor)
    else:
        if not isinstance(relative_azimuth_angle_deg, torch.Tensor):
            relative = torch.tensor(
                float(relative_azimuth_angle_deg),
                dtype=torch.float32,
                device=viewing_factor.device,
            )
        else:
            relative = relative_azimuth_angle_deg.to(
                device=viewing_factor.device, dtype=torch.float32
            )
        relative, _ = torch.broadcast_tensors(relative, viewing_factor)
        relative = torch.clamp(relative, 0.0, 180.0)
        weight = 0.5 * (1.0 + torch.cos(torch.deg2rad(relative)))
    weight = weight.to(dtype=viewing_factor.dtype)
    return solar_factor + weight * viewing_factor


__all__ = [
    "geometric_air_mass_factor",
    "double_geometric_air_mass_factor",
    "exponential_air_mass_factor",
    "double_exponential_air_mass_factor",
]
