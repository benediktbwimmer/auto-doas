import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.models.forward import AutoDOASForwardModel, InstrumentParameters
from auto_doas.physics.cross_sections import CrossSectionDatabase
from auto_doas.physics.geometry import (
    double_geometric_air_mass_factor,
    geometric_air_mass_factor,
)


def _make_forward_model(**kwargs) -> AutoDOASForwardModel:
    wavelengths = np.linspace(430.0, 432.0, num=5, dtype=np.float32)
    absorption = {"NO2": np.linspace(1.0, 2.0, num=5, dtype=np.float32) * 1e-20}
    database = CrossSectionDatabase.from_arrays(wavelengths, absorption)
    model = AutoDOASForwardModel(
        torch.from_numpy(wavelengths),
        database,
        num_instruments=1,
        embedding_dim=4,
        kernel_size=3,
        **kwargs,
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


def test_rayleigh_scattering_influences_reconstruction():
    model_with = _make_forward_model(include_rayleigh=True)
    model_without = _make_forward_model(include_rayleigh=False)

    gas_columns, instrument_ids, nuisance = _dummy_inputs(model_with)

    reconstruction_with, diagnostics_with = model_with(
        gas_columns, instrument_ids, nuisance
    )
    reconstruction_without, diagnostics_without = model_without(
        gas_columns, instrument_ids, nuisance
    )

    assert "rayleigh_optical_depth" in diagnostics_with
    assert torch.all(diagnostics_with["rayleigh_optical_depth"] > 0)
    assert "rayleigh_optical_depth" not in diagnostics_without
    assert not torch.allclose(reconstruction_with, reconstruction_without)


def test_forward_model_derives_air_mass_from_geometry():
    model = _make_forward_model()
    gas_columns, instrument_ids, nuisance = _dummy_inputs(model)
    solar = torch.tensor([60.0])
    viewing = torch.tensor([30.0])

    _, diagnostics = model(
        gas_columns,
        instrument_ids,
        nuisance,
        solar_zenith_angle=solar,
        viewing_zenith_angle=viewing,
    )

    expected_total = double_geometric_air_mass_factor(solar, viewing)
    expected_solar = geometric_air_mass_factor(solar)
    expected_viewing = geometric_air_mass_factor(viewing)

    torch.testing.assert_close(diagnostics["air_mass_factor"], expected_total[:, None])
    torch.testing.assert_close(diagnostics["solar_air_mass_factor"], expected_solar)
    torch.testing.assert_close(diagnostics["viewing_air_mass_factor"], expected_viewing)
    torch.testing.assert_close(
        diagnostics["viewing_air_mass_weight"], torch.ones_like(expected_viewing)
    )


def test_forward_model_uses_relative_azimuth_weighting():
    model = _make_forward_model()
    gas_columns, instrument_ids, nuisance = _dummy_inputs(model)
    solar = torch.tensor([55.0])
    viewing = torch.tensor([40.0])
    relative = torch.tensor([180.0])

    _, diagnostics = model(
        gas_columns,
        instrument_ids,
        nuisance,
        solar_zenith_angle=solar,
        viewing_zenith_angle=viewing,
        relative_azimuth_angle=relative,
    )

    expected_weight = 0.5 * (1.0 + torch.cos(torch.deg2rad(relative)))
    expected_total = (
        geometric_air_mass_factor(solar)
        + expected_weight * geometric_air_mass_factor(viewing)
    )

    torch.testing.assert_close(
        diagnostics["viewing_air_mass_weight"], expected_weight.to(diagnostics["viewing_air_mass_weight"].dtype)
    )
    torch.testing.assert_close(diagnostics["air_mass_factor"], expected_total[:, None])


def test_air_mass_factor_scales_optical_depth():
    model = _make_forward_model()
    gas_columns, instrument_ids, nuisance = _dummy_inputs(model)

    with torch.no_grad():
        model.absorption_matrix.copy_(model.absorption_matrix * 1e6)

    _, diagnostics_base = model(gas_columns, instrument_ids, nuisance)
    amf = torch.tensor([2.0])
    scaled, diagnostics_scaled = model(
        gas_columns,
        instrument_ids,
        nuisance,
        air_mass_factors=amf,
    )

    torch.testing.assert_close(
        diagnostics_scaled["air_mass_factor"], amf[:, None]
    )
    torch.testing.assert_close(
        diagnostics_scaled["effective_gas_columns"],
        gas_columns * amf[:, None],
    )
    assert torch.all(
        diagnostics_scaled["optical_depth_component"]
        > diagnostics_base["optical_depth_component"]
    )


def test_partial_instrument_parameter_overrides_only_apply_to_matching_ids():
    wavelengths = np.linspace(430.0, 432.0, num=5, dtype=np.float32)
    absorption = {"NO2": np.linspace(1.0, 2.0, num=5, dtype=np.float32) * 1e-20}
    database = CrossSectionDatabase.from_arrays(wavelengths, absorption)
    model = AutoDOASForwardModel(
        torch.from_numpy(wavelengths),
        database,
        num_instruments=2,
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

    gas_columns = torch.full((2, len(model.gases)), 0.1)
    nuisance = torch.zeros((2, len(model.gases)))
    instrument_ids = torch.tensor([0, 1], dtype=torch.long)
    overrides = {
        1: InstrumentParameters(
            wavelength_offset_nm=0.01,
            wavelength_scale=1.001,
            lsf_width_px=2.0,
            stray_light_fraction=0.015,
            nonlinear_response=0.01,
        )
    }

    _, diagnostics = model(
        gas_columns, instrument_ids, nuisance, instrument_parameters=overrides
    )

    torch.testing.assert_close(
        diagnostics["instrument_stray_light"],
        torch.tensor([0.0, 0.015], dtype=diagnostics["instrument_stray_light"].dtype),
    )
    torch.testing.assert_close(
        diagnostics["applied_wavelength_offset"][0],
        diagnostics["predicted_wavelength_offset"][0],
    )
    torch.testing.assert_close(
        diagnostics["applied_wavelength_offset"][1],
        torch.tensor(0.01, dtype=diagnostics["applied_wavelength_offset"].dtype),
    )
    torch.testing.assert_close(
        diagnostics["applied_wavelength_scale"],
        torch.tensor(
            [
                diagnostics["predicted_wavelength_scale"][0].item(),
                1.001,
            ],
            dtype=diagnostics["applied_wavelength_scale"].dtype,
        ),
    )
    torch.testing.assert_close(
        diagnostics["applied_lsf_width"][0],
        diagnostics["predicted_lsf_width"][0],
    )
    torch.testing.assert_close(
        diagnostics["applied_lsf_width"][1],
        torch.tensor(2.0, dtype=diagnostics["applied_lsf_width"].dtype),
    )
    torch.testing.assert_close(
        diagnostics["instrument_nonlinearity"],
        torch.tensor([0.0, 0.01], dtype=diagnostics["instrument_nonlinearity"].dtype),
    )
