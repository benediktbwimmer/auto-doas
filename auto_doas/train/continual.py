"""Continual learning utilities for Auto-DOAS."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch

from ..config import HyperParameters
from ..models.encoder import AutoDOASEncoder
from ..models.forward import AutoDOASForwardModel, InstrumentParameters
from ..models.losses import AutoDOASLoss
from ..physics.solar_reference import SharedSolarReference


class ContinualAutoDOASLearner:
    """Lightweight continual learner that updates on streaming spectra."""

    def __init__(
        self,
        encoder: AutoDOASEncoder,
        forward_model: AutoDOASForwardModel,
        hyperparams: HyperParameters,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        solar_reference: Optional[SharedSolarReference] = None,
    ) -> None:
        self.encoder = encoder
        self.forward_model = forward_model
        self.hyperparams = hyperparams
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.forward_model.to(self.device)
        params = list(self.encoder.parameters()) + list(self.forward_model.parameters())
        self.optimizer = optimizer or torch.optim.AdamW(params, lr=hyperparams.learning_rate)
        self.loss_fn = AutoDOASLoss(
            lambda_high=hyperparams.lambda_high,
            lambda_c=hyperparams.lambda_c,
            lambda_cons=hyperparams.lambda_cons,
            lambda_theta=hyperparams.lambda_theta,
            lambda_nn=hyperparams.lambda_nn,
        )
        if solar_reference is None:
            solar_reference = SharedSolarReference(self.forward_model.wavelengths_nm.detach())
        self.solar_reference = solar_reference.to(self.device)
        self._previous_columns: Optional[torch.Tensor] = None

    def _to_device(
        self,
        tensor: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.device, dtype=dtype)

    def observe_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
    ) -> Dict[str, float]:
        """Update model parameters using a single collated batch."""

        self.encoder.train()
        self.forward_model.train()
        counts = batch["counts"].to(self.device)
        instrument_ids = batch["instrument_id"].to(self.device)
        dtype = counts.dtype
        solar_zenith = self._to_device(batch.get("solar_zenith_angle"), dtype=dtype)
        viewing_zenith = self._to_device(batch.get("viewing_zenith_angle"), dtype=dtype)
        relative_azimuth = self._to_device(batch.get("relative_azimuth_angle"), dtype=dtype)
        timestamps = self._to_device(batch.get("timestamp"), dtype=dtype)
        exposure_time = self._to_device(batch.get("exposure_time"), dtype=dtype)
        ccd_temperature = self._to_device(batch.get("ccd_temperature"), dtype=dtype)
        gas_columns, nuisance = self.encoder(counts)
        solar_reference_log = self.solar_reference.update(
            counts.detach(),
            solar_zenith_angle=solar_zenith.detach() if solar_zenith is not None else None,
        ).to(self.device, dtype=dtype)
        reconstruction, diagnostics = self.forward_model(
            gas_columns,
            instrument_ids,
            nuisance,
            instrument_parameters=instrument_parameters,
            solar_zenith_angle=solar_zenith,
            viewing_zenith_angle=viewing_zenith,
            relative_azimuth_angle=relative_azimuth,
            timestamps=timestamps,
            exposure_time=exposure_time,
            ccd_temperature=ccd_temperature,
            solar_reference=solar_reference_log,
        )
        if self._previous_columns is None or self._previous_columns.shape != gas_columns.shape:
            neighbor_columns = gas_columns.detach()
        else:
            neighbor_columns = self._previous_columns
        inst_reg = torch.stack(
            [
                diagnostics["gain"],
                diagnostics["offset"],
                diagnostics["stray_light"],
            ],
            dim=0,
        ).mean(dim=0)
        losses = self.loss_fn(
            reconstruction,
            counts,
            gas_columns,
            neighbor_columns.detach(),
            inst_reg,
        )
        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.forward_model.parameters()),
            self.hyperparams.gradient_clip,
        )
        self.optimizer.step()
        self._previous_columns = gas_columns.detach()
        return {key: float(value.detach()) for key, value in losses.items()}

    def observe_stream(
        self,
        stream: Iterable[Mapping[str, torch.Tensor]],
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
    ) -> Dict[str, float]:
        """Iterate over a stream of batches and update continually."""

        last_losses: Dict[str, float] = {}
        for batch in stream:
            last_losses = self.observe_batch(batch, instrument_parameters=instrument_parameters)
        return last_losses


__all__ = ["ContinualAutoDOASLearner"]
