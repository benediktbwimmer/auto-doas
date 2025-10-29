import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.models.encoder import AutoDOASEncoder
from auto_doas.models.end_to_end import PhysicsInformedEndToEndModel
from auto_doas.models.forward import AutoDOASForwardModel
from auto_doas.physics.cross_sections import CrossSectionDatabase
from auto_doas.physics.geometry import double_geometric_air_mass_factor


def _make_components():
    wavelengths = np.linspace(430.0, 432.0, num=5, dtype=np.float32)
    absorption = {"NO2": np.linspace(1.0, 2.0, num=5, dtype=np.float32) * 1e-20}
    database = CrossSectionDatabase.from_arrays(wavelengths, absorption)
    forward = AutoDOASForwardModel(
        torch.from_numpy(wavelengths),
        database,
        num_instruments=1,
        embedding_dim=4,
        kernel_size=3,
    )
    encoder = AutoDOASEncoder(num_wavelengths=len(wavelengths), num_gases=len(database.gases))
    return encoder, forward


def test_end_to_end_pipeline_matches_forward_model():
    encoder, forward = _make_components()
    model = PhysicsInformedEndToEndModel(encoder, forward)

    counts = torch.ones(2, 5)
    instrument_ids = torch.zeros(2, dtype=torch.long)

    gas, nuisance, reconstruction, diagnostics = model(counts, instrument_ids)

    recon_ref, diagnostics_ref = forward(gas, instrument_ids, nuisance)

    torch.testing.assert_close(reconstruction, recon_ref)
    for key, value in diagnostics_ref.items():
        torch.testing.assert_close(diagnostics[key], value)


def test_gradients_flow_through_full_model():
    encoder, forward = _make_components()
    model = PhysicsInformedEndToEndModel(encoder, forward)

    counts = torch.ones(1, 5, requires_grad=True)
    instrument_ids = torch.zeros(1, dtype=torch.long)

    _, _, reconstruction, _ = model(counts, instrument_ids)
    loss = reconstruction.sum()
    loss.backward()

    assert counts.grad is not None


def test_end_to_end_model_propagates_geometry_arguments():
    encoder, forward = _make_components()
    model = PhysicsInformedEndToEndModel(encoder, forward)

    counts = torch.ones(1, 5)
    instrument_ids = torch.zeros(1, dtype=torch.long)
    solar = torch.tensor([45.0])
    viewing = torch.tensor([30.0])

    _, _, _, diagnostics = model(
        counts,
        instrument_ids,
        solar_zenith_angle=solar,
        viewing_zenith_angle=viewing,
    )

    expected = double_geometric_air_mass_factor(solar, viewing)
    torch.testing.assert_close(diagnostics["air_mass_factor"], expected[:, None])
