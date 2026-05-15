from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": float(info.frames / info.samplerate),
        "format": info.format,
        "subtype": info.subtype,
    }


def load_mono(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y1 = y.mean(axis=1).astype(np.float32)
    if int(sr) != int(target_sr):
        y1 = resample_poly(y1, int(target_sr), int(sr)).astype(np.float32)
        sr = int(target_sr)
    return y1, int(sr)
