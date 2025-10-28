"""Loss functions for unsupervised DOAS training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class AutoDOASLoss:
    """Compute composite losses used during unsupervised training."""

    def __init__(
        self,
        lambda_high: float = 2.0,
        lambda_c: float = 0.1,
        lambda_cons: float = 0.2,
        lambda_theta: float = 0.05,
        lambda_nn: float = 0.01,
    ) -> None:
        self.lambda_high = lambda_high
        self.lambda_c = lambda_c
        self.lambda_cons = lambda_cons
        self.lambda_theta = lambda_theta
        self.lambda_nn = lambda_nn

    @staticmethod
    def high_pass(signal: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        kernel = torch.ones(1, 1, kernel_size, device=signal.device) / kernel_size
        smoothed = F.conv1d(signal[:, None, :], kernel, padding=kernel_size // 2)
        return signal[:, None, :] - smoothed

    def __call__(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        gas_columns: torch.Tensor,
        neighbor_columns: torch.Tensor,
        instrument_regularizer: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return a dictionary with individual loss contributions."""

        recon_loss = F.mse_loss(reconstruction, target)
        hp_recon_loss = F.mse_loss(
            self.high_pass(reconstruction).squeeze(1),
            self.high_pass(target).squeeze(1),
        )
        tv = torch.mean(torch.abs(gas_columns[:, 1:] - gas_columns[:, :-1])) if gas_columns.shape[1] > 1 else torch.tensor(0.0, device=gas_columns.device)
        consistency = F.l1_loss(gas_columns, neighbor_columns)
        sparsity = torch.mean(torch.abs(gas_columns))
        inst_reg = torch.mean(instrument_regularizer)
        total = (
            recon_loss
            + self.lambda_high * hp_recon_loss
            + self.lambda_c * tv
            + self.lambda_cons * consistency
            + self.lambda_theta * inst_reg
            + self.lambda_nn * sparsity
        )
        return {
            "total": total,
            "reconstruction": recon_loss,
            "high_pass": hp_recon_loss,
            "temporal_smoothness": tv,
            "consistency": consistency,
            "instrument_regularizer": inst_reg,
            "sparsity": sparsity,
        }
