# Auto-DOAS

This repository provides a starting implementation of an unsupervised, physics-aware retrieval
system for converting Level-0 Differential Optical Absorption Spectroscopy (DOAS) spectra into gas
slant columns.  The implementation follows the blueprint outlined in the project brief and combines
learned encoders with a differentiable forward model that honors instrument physics.

## Package layout

```
auto_doas/
├── config.py              # Default hyper-parameters
├── data/dataset.py        # Level-0 spectrum containers and dataset helpers
├── models/encoder.py      # Neural encoder that predicts gas and nuisance latents
├── models/forward.py      # Differentiable forward model for spectral reconstruction
├── models/losses.py       # Composite unsupervised training losses
├── physics/cross_sections.py  # Cross-section data loading utilities
└── train/trainer.py       # High-level training orchestrator
```

## Usage

1. Create a `CrossSectionDatabase` either from numpy arrays or a directory of `.npz` files.
2. Instantiate the encoder and forward model using a shared wavelength grid.
3. Wrap Level-0 observations in a `Level0Dataset` and train using `AutoDOASTrainer`.

Example (pseudo-code):

```python
from auto_doas import (
    AutoDOASEncoder,
    AutoDOASForwardModel,
    AutoDOASTrainer,
    CrossSectionDatabase,
    DEFAULT_HYPERPARAMS,
)

wavelengths = ...  # torch tensor [num_pixels]
cross_sections = CrossSectionDatabase.from_arrays(wavelengths.numpy(), {...})
encoder = AutoDOASEncoder(num_wavelengths=len(wavelengths), num_gases=len(cross_sections.gases))
forward = AutoDOASForwardModel(wavelengths, cross_sections, num_instruments=num_instruments)
trainer = AutoDOASTrainer(encoder, forward, DEFAULT_HYPERPARAMS)
losses = trainer.fit(dataset, epochs=10)
```

This scaffolding is intentionally modular: additional physics terms, regularizers, or curriculum
stages can be implemented by extending the provided modules.
