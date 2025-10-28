"""Package level configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HyperParameters:
    """Container for default training hyper-parameters.

    The values mirror the blueprint specification in the project README and provide a
    centralized location to tweak learning settings during experimentation.
    """

    gases: tuple = ("NO2", "O3", "SO2", "HCHO")
    window_nm: tuple = (430.0, 460.0)
    embedding_dim: int = 128
    batch_size: int = 48
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    lambda_high: float = 2.0
    lambda_c: float = 0.1
    lambda_cons: float = 0.2
    lambda_theta: float = 0.05
    lambda_nn: float = 0.01

    def to_dict(self) -> Dict[str, float]:
        """Return a plain dictionary representation."""

        return {
            "gases": self.gases,
            "window_nm": self.window_nm,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "lambda_high": self.lambda_high,
            "lambda_c": self.lambda_c,
            "lambda_cons": self.lambda_cons,
            "lambda_theta": self.lambda_theta,
            "lambda_nn": self.lambda_nn,
        }


DEFAULT_HYPERPARAMS = HyperParameters()
