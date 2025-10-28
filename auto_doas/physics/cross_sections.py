"""Cross section utilities.

This module provides a thin abstraction around tabulated absorption cross sections.  It supports
loading cross section tables from disk, interpolating them onto a target wavelength grid and
sampling differentiable optical depths during training.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch


@dataclass
class CrossSection:
    """Holds cross section data for a single gas."""

    name: str
    wavelengths_nm: torch.Tensor
    absorption: torch.Tensor

    def to(self, device: torch.device) -> "CrossSection":
        """Return a copy of the cross section moved to ``device``."""

        return CrossSection(
            name=self.name,
            wavelengths_nm=self.wavelengths_nm.to(device),
            absorption=self.absorption.to(device),
        )


class CrossSectionDatabase:
    """In memory repository of cross sections for multiple gases."""

    def __init__(self, sections: Mapping[str, CrossSection]):
        self._sections = dict(sections)

    def __contains__(self, gas: str) -> bool:  # pragma: no cover - trivial
        return gas in self._sections

    def __getitem__(self, gas: str) -> CrossSection:
        return self._sections[gas]

    @property
    def gases(self) -> Tuple[str, ...]:
        return tuple(self._sections.keys())

    @classmethod
    def from_arrays(
        cls, wavelengths_nm: np.ndarray, absorption: Mapping[str, np.ndarray]
    ) -> "CrossSectionDatabase":
        """Create a database from numpy arrays.

        Args:
            wavelengths_nm: Shared wavelength grid in nanometers.
            absorption: Mapping from gas name to absorption cross section values with the same
                length as ``wavelengths_nm``.
        """

        wl = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
        sections = {
            gas: CrossSection(gas, wl, torch.as_tensor(values, dtype=torch.float32))
            for gas, values in absorption.items()
        }
        return cls(sections)

    @classmethod
    def from_directory(
        cls,
        directory: pathlib.Path,
        pattern: str = "*.npz",
        wavelength_key: str = "wavelengths_nm",
    ) -> "CrossSectionDatabase":
        """Load a database from ``.npz`` files stored in a directory.

        Each file is expected to contain the keys ``wavelengths_nm`` and ``absorption``.  The file
        stem is used as the gas identifier.
        """

        sections: Dict[str, CrossSection] = {}
        for path in directory.glob(pattern):
            data = np.load(path)
            wl = torch.as_tensor(data[wavelength_key], dtype=torch.float32)
            absorption = torch.as_tensor(data["absorption"], dtype=torch.float32)
            sections[path.stem.upper()] = CrossSection(path.stem.upper(), wl, absorption)
        if not sections:
            raise FileNotFoundError(f"No cross section files found in {directory!s}")
        return cls(sections)

    def resample(self, wavelengths_nm: torch.Tensor) -> "CrossSectionDatabase":
        """Resample the database onto a new wavelength grid using linear interpolation."""

        sections: Dict[str, CrossSection] = {}
        for gas, section in self._sections.items():
            absorption = torch.interp(
                wavelengths_nm,
                section.wavelengths_nm.to(wavelengths_nm.device),
                section.absorption.to(wavelengths_nm.device),
            )
            sections[gas] = CrossSection(gas, wavelengths_nm, absorption)
        return CrossSectionDatabase(sections)

    def absorption_matrix(self, gases: Iterable[str]) -> torch.Tensor:
        """Return a stacked tensor ``[len(gases), len(wavelengths)]`` of absorption values."""

        tensors = [self._sections[gas].absorption for gas in gases]
        return torch.stack(tensors, dim=0)

    def normalize(self, scale: float = 1e-20) -> "CrossSectionDatabase":
        """Return a new database with absorption values scaled by ``scale``."""

        sections = {
            gas: CrossSection(section.name, section.wavelengths_nm, section.absorption / scale)
            for gas, section in self._sections.items()
        }
        return CrossSectionDatabase(sections)
