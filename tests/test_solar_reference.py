import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.physics.solar_reference import SharedSolarReference, SolarReferenceConfig


def test_shared_solar_reference_prefers_midday_counts():
    wavelengths = torch.linspace(430.0, 432.0, steps=5)
    reference = SharedSolarReference(wavelengths, SolarReferenceConfig(decay=0.5))

    morning_counts = torch.full((1, 5), 100.0)
    noon_counts = torch.full((1, 5), 200.0)
    morning_angle = torch.tensor([80.0])
    noon_angle = torch.tensor([10.0])

    reference.update(morning_counts, solar_zenith_angle=morning_angle)
    after_morning = reference.reference().clone()
    reference.update(noon_counts, solar_zenith_angle=noon_angle)
    after_noon = reference.reference()

    assert torch.all(after_noon > after_morning)


def test_shared_solar_reference_returns_irradiance():
    wavelengths = torch.linspace(430.0, 432.0, steps=3)
    reference = SharedSolarReference(wavelengths)
    counts = torch.full((1, 3), 150.0)
    reference.update(counts)
    irradiance = reference.irradiance()
    expected = torch.exp(reference.reference())
    torch.testing.assert_close(irradiance, expected)

