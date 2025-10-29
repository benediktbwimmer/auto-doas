"""End-to-end physics informed DOAS model."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import AutoDOASEncoder
from .forward import AutoDOASForwardModel, InstrumentParameters


class PhysicsInformedEndToEndModel(nn.Module):
    """Compose encoder and physics forward model for end-to-end retrieval.

    The module jointly evaluates the neural encoder and the differentiable physics
    forward model.  It exposes a ``nn.Module`` compatible interface so it can be
    optimized directly using standard PyTorch tooling.  The encoder predicts gas
    slant columns and nuisance latents from Level-0 detector counts while the
    forward model synthesizes a reconstruction consistent with the instrument
    physics.  No tensors are detached which keeps the full pipeline
    differentiable and suitable for gradient-based learning.
    """

    def __init__(
        self,
        encoder: AutoDOASEncoder,
        forward_model: AutoDOASForwardModel,
        detach_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.forward_model = forward_model
        self.detach_diagnostics = detach_diagnostics

        self._expected_nuisance_dim = self._infer_forward_nuisance_dim()
        encoder_nuisance_dim = self._infer_encoder_nuisance_dim()
        if encoder_nuisance_dim != self._expected_nuisance_dim:
            self.nuisance_projection: nn.Module = nn.Linear(
                encoder_nuisance_dim, self._expected_nuisance_dim
            )
        else:  # pragma: no cover - trivial branch
            self.nuisance_projection = nn.Identity()

    def forward(
        self,
        counts: torch.Tensor,
        instrument_ids: torch.Tensor,
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
        air_mass_factors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run encoder and forward model on Level-0 counts.

        Args:
            counts: Detector counts tensor with shape ``[batch, num_wavelengths]``.
            instrument_ids: Integer tensor identifying the instrument for each
                element in the batch.
            instrument_parameters: Optional mapping that provides calibrated
                instrument parameters.  When provided they override the learned
                nuisance predictions in the forward model.
            air_mass_factors: Optional tensor containing per-sample (or per-gas)
                geometric air mass factors to scale the gas slant columns prior to
                spectral synthesis.

        Returns:
            Tuple containing gas slant column estimates, nuisance latent vectors,
            reconstructed spectra and diagnostic tensors from the forward model.
        """

        device = next(self.encoder.parameters()).device
        counts = counts.to(device)
        instrument_ids = instrument_ids.to(device)

        gas_columns, nuisance_latent = self.encoder(counts)
        nuisance_latent = self._project_nuisance(nuisance_latent)
        reconstruction, diagnostics = self.forward_model(
            gas_columns,
            instrument_ids,
            nuisance_latent,
            air_mass_factors=air_mass_factors,
            instrument_parameters=instrument_parameters,
        )

        if self.detach_diagnostics:
            reconstruction = reconstruction.detach()
            diagnostics = {key: value.detach() for key, value in diagnostics.items()}

        return gas_columns, nuisance_latent, reconstruction, diagnostics

    def encode(self, counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode Level-0 counts into gas columns and nuisance latents."""

        device = next(self.encoder.parameters()).device
        counts = counts.to(device)
        gas_columns, nuisance_latent = self.encoder(counts)
        nuisance_latent = self._project_nuisance(nuisance_latent)
        return gas_columns, nuisance_latent

    def reconstruct(
        self,
        gas_columns: torch.Tensor,
        instrument_ids: torch.Tensor,
        nuisance_latent: torch.Tensor,
        instrument_parameters: Optional[Mapping[int, InstrumentParameters]] = None,
        air_mass_factors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Synthesize Level-0 spectra using the physics forward model."""

        return self.forward_model(
            gas_columns,
            instrument_ids,
            nuisance_latent,
            air_mass_factors=air_mass_factors,
            instrument_parameters=instrument_parameters,
        )

    def _infer_forward_nuisance_dim(self) -> int:
        """Determine the latent dimension expected by the forward model."""

        instrument_dim = self.forward_model.instrument_embedding.embedding.embedding_dim
        for module in self.forward_model.nuisance_head:
            if isinstance(module, nn.Linear):
                return module.in_features - instrument_dim
        raise RuntimeError("Forward model nuisance head has no linear layer")

    def _infer_encoder_nuisance_dim(self) -> int:
        """Inspect the encoder to determine its nuisance latent dimension."""

        for module in reversed(self.encoder.nuisance_head):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise RuntimeError("Encoder nuisance head has no linear layer")

    def _project_nuisance(self, nuisance: torch.Tensor) -> torch.Tensor:
        """Project encoder nuisance latents to the expected dimension."""

        if nuisance.shape[-1] != self._expected_nuisance_dim:
            nuisance = self.nuisance_projection(nuisance)
        return nuisance


__all__ = ["PhysicsInformedEndToEndModel"]

