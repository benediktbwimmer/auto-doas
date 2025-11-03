import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_doas.config import HyperParameters
from auto_doas.models.encoder import AutoDOASEncoder
from auto_doas.models.forward import AutoDOASForwardModel
from auto_doas.physics.cross_sections import CrossSectionDatabase
from auto_doas.train.continual import ContinualAutoDOASLearner


def _make_learner():
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
    encoder = AutoDOASEncoder(num_wavelengths=len(wavelengths), num_gases=len(database.gases), latent_dim=32, nuisance_dim=len(database.gases))
    hyper = HyperParameters(
        gases=database.gases,
        embedding_dim=4,
        batch_size=2,
        learning_rate=5e-4,
    )
    learner = ContinualAutoDOASLearner(encoder, forward, hyper)
    return learner


def _mock_batch(batch_size: int = 2):
    counts = torch.ones(batch_size, 5)
    return {
        "counts": counts,
        "instrument_id": torch.zeros(batch_size, dtype=torch.long),
        "timestamp": torch.zeros(batch_size),
        "solar_zenith_angle": torch.full((batch_size,), 40.0),
        "exposure_time": torch.ones(batch_size),
        "ccd_temperature": torch.zeros(batch_size),
    }


def test_continual_learner_updates_and_returns_losses():
    learner = _make_learner()
    losses = learner.observe_batch(_mock_batch())
    assert "total" in losses
    assert losses["total"] >= 0.0

