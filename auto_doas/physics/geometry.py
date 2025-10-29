"""Geometric utilities for atmospheric radiative transfer."""

from __future__ import annotations

from typing import Union

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


__all__ = ["geometric_air_mass_factor"]
