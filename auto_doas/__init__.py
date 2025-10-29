"""auto_doas package.

Provides components for building an unsupervised Differential Optical Absorption Spectroscopy
(DOAS) retrieval system that operates directly on Level-0 detector counts.
"""

from .config import DEFAULT_HYPERPARAMS
from .physics.cross_sections import CrossSectionDatabase
from .physics.geometry import geometric_air_mass_factor
from .models.forward import AutoDOASForwardModel, InstrumentParameters
from .models.encoder import AutoDOASEncoder
from .models.end_to_end import PhysicsInformedEndToEndModel
from .retrieval import PhysicsBasedDOASRetrieval, RetrievalResult
from .train.trainer import AutoDOASTrainer

__all__ = [
    "DEFAULT_HYPERPARAMS",
    "CrossSectionDatabase",
    "geometric_air_mass_factor",
    "AutoDOASForwardModel",
    "InstrumentParameters",
    "AutoDOASEncoder",
    "PhysicsInformedEndToEndModel",
    "PhysicsBasedDOASRetrieval",
    "RetrievalResult",
    "AutoDOASTrainer",
]
