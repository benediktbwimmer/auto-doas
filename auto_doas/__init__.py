"""auto_doas package.

Provides components for building an unsupervised Differential Optical Absorption Spectroscopy
(DOAS) retrieval system that operates directly on Level-0 detector counts.
"""

from .config import DEFAULT_HYPERPARAMS
from .physics.atmosphere import rayleigh_optical_depth
from .physics.cross_sections import CrossSectionDatabase
from .physics.geometry import geometric_air_mass_factor
from .physics.solar_reference import SharedSolarReference, SolarReferenceConfig
from .models.context import ObservationContextConfig, ObservationContextEncoder
from .models.forward import AutoDOASForwardModel, InstrumentParameters
from .models.encoder import AutoDOASEncoder
from .models.end_to_end import PhysicsInformedEndToEndModel
from .retrieval import PhysicsBasedDOASRetrieval, RetrievalResult
from .train.continual import ContinualAutoDOASLearner
from .train.trainer import AutoDOASTrainer

__all__ = [
    "DEFAULT_HYPERPARAMS",
    "CrossSectionDatabase",
    "geometric_air_mass_factor",
    "rayleigh_optical_depth",
    "SharedSolarReference",
    "SolarReferenceConfig",
    "ObservationContextEncoder",
    "ObservationContextConfig",
    "AutoDOASForwardModel",
    "InstrumentParameters",
    "AutoDOASEncoder",
    "PhysicsInformedEndToEndModel",
    "PhysicsBasedDOASRetrieval",
    "RetrievalResult",
    "ContinualAutoDOASLearner",
    "AutoDOASTrainer",
]
