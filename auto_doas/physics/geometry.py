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


__all__ = ["geometric_air_mass_factor", "double_geometric_air_mass_factor"]
