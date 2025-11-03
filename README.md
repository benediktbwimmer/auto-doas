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
├── models/context.py      # Instrument-agnostic observation context encoder
├── models/encoder.py      # Neural encoder that predicts gas and nuisance latents
├── models/forward.py      # Differentiable forward model for spectral reconstruction
├── models/losses.py       # Composite unsupervised training losses
├── retrieval.py           # High-level retrieval orchestration (Level-0 ↔ Level-2)
├── physics/cross_sections.py  # Cross-section data loading utilities
├── physics/solar_reference.py # Shared solar irradiance reference model
└── train/trainer.py       # High-level training orchestrator
└── train/continual.py     # Continual learning façade built on the trainer
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

## Shared-Sun Continual Retrieval

The forward model now conditions on an observation context embedding rather than instrument ids.
Temporal Fourier features of the timestamp and solar/viewing geometry capture the shared solar
illumination seen by every spectrometer, which makes the retrieval instrument-agnostic by design.

A `SharedSolarReference` module tracks an exponential moving average of log-irradiance using the
cosine of the solar zenith angle as weight.  Passing this reference into the forward model anchors
the continuum to the true solar spectrum rather than instrument-specific offsets.  The trainer
updates the reference each mini-batch and the retriever can optionally continue updating it during
inference by setting `update_solar_reference=True`.

For continual deployment, use `ContinualAutoDOASLearner` which wraps the encoder, forward model and
solar reference into a single streaming learner:

```python
from auto_doas import (
    ContinualAutoDOASLearner,
    AutoDOASEncoder,
    AutoDOASForwardModel,
    CrossSectionDatabase,
    DEFAULT_HYPERPARAMS,
)

learner = ContinualAutoDOASLearner(encoder, forward, DEFAULT_HYPERPARAMS)
for batch in live_stream:
    learner.observe_batch(batch)
```

Every observed spectrum therefore updates both the retrieval parameters and the shared solar
reference, enabling a network of instruments to learn from each other through the sun they
collectively observe.
