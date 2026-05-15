from __future__ import annotations

import numpy as np


def rms_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(y.astype(np.float32)))))
    return 20.0 * np.log10(max(rms, 1e-12))


def frame_rms_db(y: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if frame_len <= 0 or hop <= 0 or y.size == 0:
        return np.asarray([], dtype=np.float32)
    vals = []
    for start in range(0, max(1, len(y) - frame_len + 1), hop):
        vals.append(rms_db(y[start:start + frame_len]))
    return np.asarray(vals, dtype=np.float32)
