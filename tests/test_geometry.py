import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.physics.geometry import (
    double_geometric_air_mass_factor,
    geometric_air_mass_factor,
    double_exponential_air_mass_factor,
    exponential_air_mass_factor,
)


def test_geometric_air_mass_factor_increases_with_angle():
    angles = torch.tensor([0.0, 45.0, 60.0, 80.0])
    factors = geometric_air_mass_factor(angles)

    torch.testing.assert_close(factors[0], torch.tensor(1.0), atol=5e-4, rtol=5e-4)
    assert torch.all(factors[1:] > factors[:-1])


def test_geometric_air_mass_factor_accepts_scalars():
    value = geometric_air_mass_factor(60.0)
    torch.testing.assert_close(value, geometric_air_mass_factor(torch.tensor(60.0)))


def test_double_geometric_air_mass_factor_combines_paths():
    solar = torch.tensor([30.0])
    viewing = torch.tensor([45.0])
    combined = double_geometric_air_mass_factor(solar, viewing)
    expected = geometric_air_mass_factor(solar) + geometric_air_mass_factor(viewing)
    torch.testing.assert_close(combined, expected)


def test_double_geometric_air_mass_factor_accounts_for_relative_azimuth():
    solar = torch.tensor([50.0])
    viewing = torch.tensor([40.0])
    forward = double_geometric_air_mass_factor(solar, viewing, relative_azimuth_angle_deg=0.0)
    backward = double_geometric_air_mass_factor(solar, viewing, relative_azimuth_angle_deg=180.0)
    assert torch.all(backward < forward)


def test_exponential_air_mass_factor_behaves_physically():
    angles = torch.tensor([0.0, 30.0, 60.0, 80.0])
    factors = exponential_air_mass_factor(angles)

    torch.testing.assert_close(factors[0], torch.tensor(1.0), atol=1e-3, rtol=1e-3)
    assert torch.all(factors[1:] > factors[:-1])
    geometric = geometric_air_mass_factor(angles)
    assert torch.all(factors <= geometric + 1e-3)


def test_double_exponential_air_mass_factor_matches_component_sum():
    solar = torch.tensor([35.0])
    viewing = torch.tensor([25.0])
    combined = double_exponential_air_mass_factor(solar, viewing)
    expected = exponential_air_mass_factor(solar) + exponential_air_mass_factor(viewing)
    torch.testing.assert_close(combined, expected)


def test_double_exponential_air_mass_factor_uses_relative_weighting():
    solar = torch.tensor([45.0])
    viewing = torch.tensor([50.0])
    forward = double_exponential_air_mass_factor(solar, viewing, relative_azimuth_angle_deg=0.0)
    backward = double_exponential_air_mass_factor(solar, viewing, relative_azimuth_angle_deg=180.0)
    assert torch.all(backward < forward)
