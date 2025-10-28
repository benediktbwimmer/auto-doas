"""auto_doas package.

Provides components for building an unsupervised Differential Optical Absorption Spectroscopy
(DOAS) retrieval system that operates directly on Level-0 detector counts.
"""

from .config import DEFAULT_HYPERPARAMS
from .physics.cross_sections import CrossSectionDatabase
from .models.forward import AutoDOASForwardModel, InstrumentParameters
from .models.encoder import AutoDOASEncoder
from .train.trainer import AutoDOASTrainer

__all__ = [
    "DEFAULT_HYPERPARAMS",
    "CrossSectionDatabase",
    "AutoDOASForwardModel",
    "InstrumentParameters",
    "AutoDOASEncoder",
    "AutoDOASTrainer",
]
