import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.physics.atmosphere import rayleigh_optical_depth


def test_rayleigh_optical_depth_monotonic_and_positive():
    wavelengths = torch.tensor([320.0, 400.0, 500.0, 600.0])
    tau = rayleigh_optical_depth(wavelengths)
    assert torch.all(tau > 0)
    assert torch.all(tau[:-1] > tau[1:])


def test_rayleigh_optical_depth_scales_with_pressure_and_temperature():
    wavelengths = torch.linspace(430.0, 450.0, steps=5)
    tau_reference = rayleigh_optical_depth(wavelengths)
    tau_double_pressure = rayleigh_optical_depth(wavelengths, pressure_hpa=2 * 1013.25)
    torch.testing.assert_close(tau_double_pressure, tau_reference * 2.0)

    tau_cold = rayleigh_optical_depth(wavelengths, temperature_k=250.0)
    tau_warm = rayleigh_optical_depth(wavelengths, temperature_k=320.0)
    assert torch.all(tau_cold > tau_warm)


def test_rayleigh_optical_depth_rejects_non_positive_wavelengths():
    with pytest.raises(ValueError, match="wavelengths_nm must be strictly positive"):
        rayleigh_optical_depth(torch.tensor([400.0, 0.0]))
