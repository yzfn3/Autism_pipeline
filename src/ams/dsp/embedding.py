from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def band_energy_embedding(y: np.ndarray, sr_hz: int, embed_dim: int = 24) -> np.ndarray:
    if y.size == 0:
        return np.zeros((embed_dim,), dtype=np.float32)
    y = y.astype(np.float32)
    n = int(2 ** np.ceil(np.log2(max(len(y), 256))))
    spec = np.abs(np.fft.rfft(y, n=n)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr_hz))
    edges = np.linspace(80.0, min(7600.0, sr_hz / 2.0), embed_dim + 1)
    vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        vals.append(float(np.log1p(spec[mask].mean() if np.any(mask) else 0.0)))
    v = np.asarray(vals, dtype=np.float32)
    v = v - float(np.mean(v))
    nrm = float(np.linalg.norm(v))
    return (v / max(nrm, 1e-12)).astype(np.float32)
