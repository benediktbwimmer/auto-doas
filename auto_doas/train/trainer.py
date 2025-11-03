"""Training loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import HyperParameters
from ..data.dataset import Level0Dataset
from ..models.encoder import AutoDOASEncoder
from ..models.forward import AutoDOASForwardModel, InstrumentParameters
from ..models.losses import AutoDOASLoss
from ..physics.solar_reference import SharedSolarReference


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0


class AutoDOASTrainer:
    """Coordinate encoder, forward model and loss for unsupervised training."""

    def __init__(
        self,
        encoder: AutoDOASEncoder,
        forward_model: AutoDOASForwardModel,
        hyperparams: HyperParameters,
        device: Optional[torch.device] = None,
        solar_reference: Optional[SharedSolarReference] = None,
    ) -> None:
        self.encoder = encoder
        self.forward_model = forward_model
        self.hyperparams = hyperparams
        self.loss_fn = AutoDOASLoss(
            lambda_high=hyperparams.lambda_high,
            lambda_c=hyperparams.lambda_c,
            lambda_cons=hyperparams.lambda_cons,
            lambda_theta=hyperparams.lambda_theta,
            lambda_nn=hyperparams.lambda_nn,
            lambda_mean=hyperparams.lambda_mean,
            mean_target=hyperparams.mean_target,
            calibration_prior_scale=hyperparams.calibration_prior_scale,
            calibration_prior_bias=hyperparams.calibration_prior_bias,
            log_calibration_scale=encoder.log_calibration_scale,
            calibration_bias=encoder.calibration_bias,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.forward_model.to(self.device)
        if solar_reference is None:
            solar_reference = SharedSolarReference(self.forward_model.wavelengths_nm.detach())
        self.solar_reference = solar_reference.to(self.device)
        self.state = TrainerState()

    def create_optimizer(self) -> torch.optim.Optimizer:
        params = list(self.encoder.parameters()) + list(self.forward_model.parameters())
        return torch.optim.AdamW(params, lr=self.hyperparams.learning_rate)

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        instrument_parameters: Optional[Dict[int, InstrumentParameters]] = None,
        pgn_weight: float = 0.0,
    ) -> Dict[str, float]:
        self.encoder.train()
        self.forward_model.train()
        total_losses: Dict[str, float] = {}
        for batch in dataloader:
            counts = batch["counts"].to(self.device)
            instrument_ids = batch["instrument_id"].to(self.device)
            solar_zenith = batch.get("solar_zenith_angle")
            viewing_zenith = batch.get("viewing_zenith_angle")
            relative_azimuth = batch.get("relative_azimuth_angle")
            timestamps = batch.get("timestamp").to(self.device)
            exposure = batch.get("exposure_time").to(self.device)
            temperature = batch.get("ccd_temperature").to(self.device)
            solar_zenith = solar_zenith.to(self.device) if solar_zenith is not None else None
            viewing_zenith = viewing_zenith.to(self.device) if viewing_zenith is not None else None
            relative_azimuth = (
                relative_azimuth.to(self.device) if relative_azimuth is not None else None
            )
            gas_columns, nuisance = self.encoder(counts)
            solar_reference_log = None
            if self.solar_reference is not None:
                with torch.no_grad():
                    sza_for_ref = solar_zenith.detach() if solar_zenith is not None else None
                    solar_reference_log = self.solar_reference.update(
                        counts.detach(),
                        solar_zenith_angle=sza_for_ref,
                    ).to(self.device, dtype=counts.dtype)
            reconstruction, diagnostics = self.forward_model(
                gas_columns,
                instrument_ids,
                nuisance,
                instrument_parameters=instrument_parameters,
                solar_zenith_angle=solar_zenith,
                viewing_zenith_angle=viewing_zenith,
                relative_azimuth_angle=relative_azimuth,
                timestamps=timestamps,
                exposure_time=exposure,
                ccd_temperature=temperature,
                solar_reference=solar_reference_log,
            )
            neighbor_columns = gas_columns.roll(-1, dims=0)
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
            pgn_supervision = torch.zeros((), device=self.device, dtype=gas_columns.dtype)
            if pgn_weight > 0.0 and "pgn_no2_vcd" in batch:
                targets = batch["pgn_no2_vcd"].to(self.device)
                weights = batch.get("pgn_no2_weight")
                if weights is not None:
                    weights = weights.to(self.device)
                mask = torch.isfinite(targets)
                if mask.any():
                    preds = gas_columns[mask, 0]
                    targ = targets[mask]
                    if weights is not None:
                        w = torch.clamp(weights[mask], min=0.0)
                        pgn_supervision = torch.sum(w * (preds - targ) ** 2) / torch.clamp(w.sum(), min=1.0)
                    else:
                        pgn_supervision = F.mse_loss(preds, targ)
            losses["pgn_supervision"] = pgn_supervision
            optimizer.zero_grad()
            (losses["total"] + pgn_weight * pgn_supervision).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.forward_model.parameters()),
                self.hyperparams.gradient_clip,
            )
            optimizer.step()
            with torch.no_grad():
                for key, value in losses.items():
                    total_losses[key] = total_losses.get(key, 0.0) + float(value.detach())
            self.state.global_step += 1
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        self.state.epoch += 1
        return total_losses

    def fit(
        self,
        dataset: Level0Dataset,
        epochs: int,
        instrument_parameters: Optional[Dict[int, InstrumentParameters]] = None,
        pgn_weight: float = 0.0,
    ) -> Dict[str, float]:
        dataloader = DataLoader(
            dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=True,
            collate_fn=Level0Dataset.collate_fn,
        )
        optimizer = self.create_optimizer()
        last_losses: Dict[str, float] = {}
        for _ in range(epochs):
            last_losses = self.train_epoch(
                dataloader, optimizer, instrument_parameters, pgn_weight=pgn_weight
            )
        return last_losses
