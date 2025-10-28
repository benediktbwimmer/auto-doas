import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.models.forward import AutoDOASForwardModel, InstrumentParameters
from auto_doas.physics.cross_sections import CrossSectionDatabase


def _make_forward_model() -> AutoDOASForwardModel:
    wavelengths = np.linspace(430.0, 432.0, num=5, dtype=np.float32)
    absorption = {"NO2": np.linspace(1.0, 2.0, num=5, dtype=np.float32) * 1e-20}
    database = CrossSectionDatabase.from_arrays(wavelengths, absorption)
    model = AutoDOASForwardModel(
        torch.from_numpy(wavelengths),
        database,
        num_instruments=1,
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


def _dummy_inputs(model: AutoDOASForwardModel):
    batch_size = 1
    gas_columns = torch.full((batch_size, len(model.gases)), 0.1)
    nuisance = torch.zeros((batch_size, len(model.gases)))
    instrument_ids = torch.zeros(batch_size, dtype=torch.long)
    return gas_columns, instrument_ids, nuisance


def test_instrument_stray_light_adds_to_predicted_component():
    model = _make_forward_model()
    gas_columns, instrument_ids, nuisance = _dummy_inputs(model)

    _, diagnostics_without = model(gas_columns, instrument_ids, nuisance)
    torch.testing.assert_close(
        diagnostics_without["stray_light"],
        diagnostics_without["predicted_stray_light"],
    )

    instrument_parameters = {0: InstrumentParameters(stray_light_fraction=0.02)}
    _, diagnostics_with = model(
        gas_columns,
        instrument_ids,
        nuisance,
        instrument_parameters=instrument_parameters,
    )
    torch.testing.assert_close(
        diagnostics_with["instrument_stray_light"],
        torch.full_like(diagnostics_with["instrument_stray_light"], 0.02),
    )
    torch.testing.assert_close(
        diagnostics_with["stray_light"],
        diagnostics_with["predicted_stray_light"]
        + diagnostics_with["instrument_stray_light"],
    )


def test_instrument_nonlinearity_shapes_detector_response():
    model = _make_forward_model()
    gas_columns, instrument_ids, nuisance = _dummy_inputs(model)

    _, diagnostics_base = model(gas_columns, instrument_ids, nuisance)
    alpha = 0.02
    instrument_parameters = {0: InstrumentParameters(nonlinear_response=alpha)}
    _, diagnostics_nonlinear = model(
        gas_columns,
        instrument_ids,
        nuisance,
        instrument_parameters=instrument_parameters,
    )

    torch.testing.assert_close(
        diagnostics_nonlinear["instrument_nonlinearity"],
        torch.full_like(diagnostics_nonlinear["instrument_nonlinearity"], alpha),
    )
    torch.testing.assert_close(
        diagnostics_nonlinear["post_gain_counts"],
        diagnostics_base["post_gain_counts"],
    )
    torch.testing.assert_close(
        diagnostics_base["pre_stray_counts"],
        diagnostics_base["post_gain_counts"],
    )
    torch.testing.assert_close(
        diagnostics_nonlinear["pre_stray_counts"],
        diagnostics_nonlinear["post_gain_counts"]
        + alpha * diagnostics_nonlinear["post_gain_counts"] ** 2,
    )
    assert not torch.allclose(
        diagnostics_nonlinear["pre_stray_counts"],
        diagnostics_base["pre_stray_counts"],
    )
