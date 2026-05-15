from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ams.dsp.embedding import band_energy_embedding


@dataclass(frozen=True)
class EmbedderInfo:
    backend: str
    embed_dim: int
    device: str
    model_id: str | None = None


class SpeakerEmbedder:
    def __init__(self, backend: str, device: str, embed_dim: int, model_id: str | None = None):
        if backend != "simple_band_energy":
            raise ImportError(f"Backend {backend!r} is not installed in the clean core build. Use simple_band_energy or reinstall optional ML extras.")
        self.backend = backend
        self.device = device
        self.embed_dim = int(embed_dim)
        self.model_id = model_id

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(self.backend, self.embed_dim, self.device, self.model_id)

    def embed(self, y: np.ndarray, sr_hz: int) -> np.ndarray:
        return band_energy_embedding(y, sr_hz=sr_hz, embed_dim=self.embed_dim)

    def embed_batch(self, wavs: List[np.ndarray], srs: List[int]) -> np.ndarray:
        return np.stack([self.embed(y, sr) for y, sr in zip(wavs, srs)], axis=0).astype(np.float32)


def choose_device(prefer_gpu: bool) -> str:
    return "cpu"
