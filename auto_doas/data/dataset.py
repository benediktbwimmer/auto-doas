"""Dataset utilities for Level-0 spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class Level0Spectrum:
    """Container describing a single Level-0 measurement."""

    counts: torch.Tensor
    wavelengths_nm: torch.Tensor
    instrument_id: int
    timestamp: float
    solar_zenith_angle: float
    exposure_time: float
    ccd_temperature: float


class Level0Dataset(Dataset):
    """Minimal in-memory dataset that stores Level-0 spectra and metadata."""

    def __init__(self, records: Dict[str, Level0Spectrum]):
        self._keys = list(records.keys())
        self._records = records

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self._keys)

    def __getitem__(self, index: int) -> Level0Spectrum:
        key = self._keys[index]
        return self._records[key]

    @staticmethod
    def collate_fn(batch: list[Level0Spectrum]) -> Dict[str, torch.Tensor]:
        """Collate a batch of ``Level0Spectrum`` objects into a dictionary of tensors."""

        counts = torch.stack([item.counts for item in batch], dim=0)
        wavelengths = torch.stack([item.wavelengths_nm for item in batch], dim=0)
        instrument_ids = torch.tensor([item.instrument_id for item in batch], dtype=torch.long)
        timestamps = torch.tensor([item.timestamp for item in batch], dtype=torch.float32)
        sza = torch.tensor([item.solar_zenith_angle for item in batch], dtype=torch.float32)
        exposure = torch.tensor([item.exposure_time for item in batch], dtype=torch.float32)
        temperature = torch.tensor([item.ccd_temperature for item in batch], dtype=torch.float32)
        return {
            "counts": counts,
            "wavelengths_nm": wavelengths,
            "instrument_id": instrument_ids,
            "timestamp": timestamps,
            "solar_zenith_angle": sza,
            "exposure_time": exposure,
            "ccd_temperature": temperature,
        }
