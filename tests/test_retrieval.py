import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.models.forward import AutoDOASForwardModel, InstrumentParameters
from auto_doas.physics.cross_sections import CrossSectionDatabase
from auto_doas.retrieval import PhysicsBasedDOASRetrieval


class DummyEncoder(torch.nn.Module):
    def __init__(self, num_gases: int, nuisance_dim: int) -> None:
        super().__init__()
        self.num_gases = num_gases
        self.nuisance_dim = nuisance_dim

    def forward(self, counts: torch.Tensor):
        batch = counts.shape[0]
        gas = torch.full((batch, self.num_gases), 0.1, device=counts.device)
        nuisance = torch.zeros((batch, self.nuisance_dim), device=counts.device)
        return gas, nuisance


def _make_forward_model(num_instruments: int = 2) -> AutoDOASForwardModel:
    wavelengths = np.linspace(430.0, 432.0, num=5, dtype=np.float32)
    absorption = {"NO2": np.linspace(1.0, 2.0, num=5, dtype=np.float32) * 1e-20}
    database = CrossSectionDatabase.from_arrays(wavelengths, absorption)
    model = AutoDOASForwardModel(
        torch.from_numpy(wavelengths),
        database,
        num_instruments=num_instruments,
        embedding_dim=4,
        kernel_size=3,
    )
    with torch.no_grad():
        model.instrument_embedding.embedding.weight.zero_()
        for module in model.nuisance_head:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
    model.eval()
    return model


def _make_retrieval(default_params=None) -> PhysicsBasedDOASRetrieval:
    forward = _make_forward_model()
    encoder = DummyEncoder(num_gases=len(forward.gases), nuisance_dim=len(forward.gases))
    retrieval = PhysicsBasedDOASRetrieval(
        encoder,
        forward,
        default_instrument_parameters=default_params,
    )
    return retrieval


def test_retrieval_uses_default_instrument_parameters():
    retrieval = _make_retrieval({0: InstrumentParameters(stray_light_fraction=0.02)})
    counts = torch.ones(1, 5)
    instrument_ids = torch.tensor([0])

    result = retrieval.run(counts, instrument_ids)
    torch.testing.assert_close(
        result.diagnostics["instrument_stray_light"],
        torch.full_like(result.diagnostics["instrument_stray_light"], 0.02),
    )

    override = {0: InstrumentParameters(stray_light_fraction=0.01)}
    result_override = retrieval.run(counts, instrument_ids, instrument_parameters=override)
    torch.testing.assert_close(
        result_override.diagnostics["instrument_stray_light"],
        torch.full_like(result_override.diagnostics["instrument_stray_light"], 0.01),
    )

    result_after_override = retrieval.run(counts, instrument_ids)
    torch.testing.assert_close(
        result_after_override.diagnostics["instrument_stray_light"],
        torch.full_like(result_after_override.diagnostics["instrument_stray_light"], 0.02),
    )

    retrieval.clear_instrument_parameters()
    result_cleared = retrieval.run(counts, instrument_ids)
    torch.testing.assert_close(
        result_cleared.diagnostics["instrument_stray_light"],
        torch.zeros_like(result_cleared.diagnostics["instrument_stray_light"]),
    )


def test_update_instrument_parameters_merges_defaults():
    retrieval = _make_retrieval({0: InstrumentParameters(stray_light_fraction=0.02)})
    counts = torch.ones(1, 5)

    retrieval.update_instrument_parameters({
        0: InstrumentParameters(stray_light_fraction=0.03),
        1: InstrumentParameters(stray_light_fraction=0.04),
    })

    result_updated = retrieval.run(counts, torch.tensor([0]))
    torch.testing.assert_close(
        result_updated.diagnostics["instrument_stray_light"],
        torch.full_like(result_updated.diagnostics["instrument_stray_light"], 0.03),
    )

    result_new = retrieval.run(counts, torch.tensor([1]))
    torch.testing.assert_close(
        result_new.diagnostics["instrument_stray_light"],
        torch.full_like(result_new.diagnostics["instrument_stray_light"], 0.04),
    )


def test_retrieval_computes_air_mass_from_solar_zenith_angle():
    retrieval = _make_retrieval()
    counts = torch.ones(1, 5)
    instrument_ids = torch.tensor([0])
    solar_zenith = torch.tensor([60.0])

    result = retrieval.run(counts, instrument_ids, solar_zenith_angle=solar_zenith)
    expected = 2.0  # approximately 1 / cos(60 deg)
    torch.testing.assert_close(
        result.diagnostics["air_mass_factor"],
        torch.full_like(result.diagnostics["air_mass_factor"], expected),
        atol=2e-1,
        rtol=1e-1,
    )
