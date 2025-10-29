"""High level physics-based DOAS retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from .data.dataset import Level0Dataset
from .models.encoder import AutoDOASEncoder
from .models.forward import AutoDOASForwardModel, InstrumentParameters
from .physics.geometry import geometric_air_mass_factor


@dataclass
class RetrievalResult:
    """Container with Level-2 products and Level-0 reconstructions.

    Attributes:
        level2_columns: Retrieved slant column densities for each gas.
        vertical_columns: Vertical column densities obtained by dividing the
            slant columns by the corresponding air mass factors.
        nuisance_latent: Latent nuisance parameters produced by the encoder.
        reconstruction: Forward model reconstruction of the Level-0 counts.
        diagnostics: Additional tensors emitted by the forward model.
        air_mass_factor: Air mass factors used to convert slant to vertical
            columns. The shape matches ``[batch, 1]`` for broadcasting.
    """

    level2_columns: torch.Tensor
    vertical_columns: torch.Tensor
    nuisance_latent: torch.Tensor
    reconstruction: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]
    air_mass_factor: torch.Tensor

    def to(self, device: torch.device) -> "RetrievalResult":
        """Move the tensors inside the result to ``device``."""

        return RetrievalResult(
            level2_columns=self.level2_columns.to(device),
            vertical_columns=self.vertical_columns.to(device),
            nuisance_latent=self.nuisance_latent.to(device),
            reconstruction=self.reconstruction.to(device),
            diagnostics={key: value.to(device) for key, value in self.diagnostics.items()},
            air_mass_factor=self.air_mass_factor.to(device),
        )


class PhysicsBasedDOASRetrieval:
    """Run the encoder and physics forward model to obtain Level-2 products.

    The retrieval follows the flow: Level-0 detector counts are ingested by the encoder,
    producing gas slant column estimates (Level-2) and nuisance parameters.  These are then
    passed through the differentiable forward model to synthesize Level-0 spectra, closing the
    loop between measurement and reconstruction.  Instrument calibration parameters can be
    registered with the retriever and are automatically merged with per-call overrides.
    """

    def __init__(
        self,
        encoder: AutoDOASEncoder,
        forward_model: AutoDOASForwardModel,
        device: Optional[torch.device] = None,
        default_instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
    ) -> None:
        self.encoder = encoder
        self.forward_model = forward_model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._instrument_parameters: Dict[int, InstrumentParameters] = {}
        if default_instrument_parameters is not None:
            self.set_instrument_parameters(default_instrument_parameters)
        self.to(self.device)
        self.eval()

    def to(self, device: torch.device) -> "PhysicsBasedDOASRetrieval":
        """Move the underlying modules to ``device``."""

        self.device = device
        self.encoder.to(device)
        self.forward_model.to(device)
        return self

    def eval(self) -> "PhysicsBasedDOASRetrieval":  # pragma: no cover - trivial wrapper
        self.encoder.eval()
        self.forward_model.eval()
        return self

    @torch.no_grad()
    def run(
        self,
        counts: torch.Tensor,
        instrument_ids: torch.Tensor,
        nuisance_latent: Optional[torch.Tensor] = None,
        solar_zenith_angle: Optional[torch.Tensor] = None,
        air_mass_factors: Optional[torch.Tensor] = None,
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
        detach: bool = True,
    ) -> RetrievalResult:
        """Execute the full retrieval for a batch of Level-0 counts."""

        counts = counts.to(self.device)
        instrument_ids = instrument_ids.to(self.device)
        if nuisance_latent is None:
            gas_columns, nuisance_latent = self.encoder(counts)
        else:
            gas_columns, _ = self.encoder(counts)
            nuisance_latent = nuisance_latent.to(self.device)
        if air_mass_factors is not None:
            air_mass_factors = air_mass_factors.to(self.device)
        elif solar_zenith_angle is not None:
            air_mass_factors = geometric_air_mass_factor(
                solar_zenith_angle.to(self.device)
            )
        instrument_parameters = self._resolve_instrument_parameters(instrument_parameters)
        reconstruction, diagnostics = self.forward_model(
            gas_columns,
            instrument_ids,
            nuisance_latent,
            air_mass_factors=air_mass_factors,
            instrument_parameters=instrument_parameters,
        )

        if "air_mass_factor" in diagnostics:
            air_mass_factor = diagnostics["air_mass_factor"]
        else:
            air_mass_factor = torch.ones(
                gas_columns.shape[0],
                1,
                device=self.device,
                dtype=gas_columns.dtype,
            )
        if air_mass_factor.ndim == 1:
            air_mass_factor = air_mass_factor[:, None]
        vertical_columns = gas_columns / torch.clamp(air_mass_factor, min=1e-6)
        if detach:
            gas_columns = gas_columns.detach()
            nuisance_latent = nuisance_latent.detach()
            reconstruction = reconstruction.detach()
            diagnostics = {key: value.detach() for key, value in diagnostics.items()}
            air_mass_factor = air_mass_factor.detach()
            vertical_columns = vertical_columns.detach()
        return RetrievalResult(
            gas_columns,
            vertical_columns,
            nuisance_latent,
            reconstruction,
            diagnostics,
            air_mass_factor,
        )

    @torch.no_grad()
    def retrieve_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
    ) -> RetrievalResult:
        """Retrieve Level-2 columns and reconstruction for a collated batch."""

        counts = batch["counts"]
        instrument_ids = batch["instrument_id"]
        solar_zenith_angle = batch.get("solar_zenith_angle")
        return self.run(
            counts,
            instrument_ids,
            solar_zenith_angle=solar_zenith_angle,
            instrument_parameters=instrument_parameters,
        )

    def set_instrument_parameters(
        self, instrument_parameters: Mapping[int, InstrumentParameters]
    ) -> "PhysicsBasedDOASRetrieval":
        """Replace the default instrument parameters used during retrieval."""

        self._instrument_parameters = {
            int(idx): params for idx, params in instrument_parameters.items()
        }
        return self

    def update_instrument_parameters(
        self, instrument_parameters: Mapping[int, InstrumentParameters]
    ) -> "PhysicsBasedDOASRetrieval":
        """Update (or add) instrument parameters used during retrieval."""

        for idx, params in instrument_parameters.items():
            self._instrument_parameters[int(idx)] = params
        return self

    def clear_instrument_parameters(self) -> "PhysicsBasedDOASRetrieval":
        """Remove all stored default instrument parameters."""

        self._instrument_parameters.clear()
        return self

    def _resolve_instrument_parameters(
        self, overrides: Optional[Mapping[int, InstrumentParameters]]
    ) -> Optional[Dict[int, InstrumentParameters]]:
        """Merge stored instrument parameters with overrides for a retrieval call."""

        if overrides is None and not self._instrument_parameters:
            return None
        merged: Dict[int, InstrumentParameters] = dict(self._instrument_parameters)
        if overrides is not None:
            merged.update({int(idx): params for idx, params in overrides.items()})
        return merged

    @torch.no_grad()
    def retrieve_dataset(
        self,
        dataset: Level0Dataset,
        batch_size: int = 32,
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
    ) -> Iterator[RetrievalResult]:
        """Iterate over retrieval results for an entire dataset."""

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=Level0Dataset.collate_fn,
        )
        for batch in dataloader:
            yield self.retrieve_batch(batch, instrument_parameters=instrument_parameters)


__all__ = [
    "PhysicsBasedDOASRetrieval",
    "RetrievalResult",
]
