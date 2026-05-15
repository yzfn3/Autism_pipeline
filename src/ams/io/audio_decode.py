from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class DecodedAudio:
    y: np.ndarray
    sr_hz: int
    channels: int
    warnings: List[str]


def _decode_via_soundfile(path: Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return y, int(sr)


def _decode_via_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "decoded.wav"
        subprocess.check_call(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path), "-acodec", "pcm_f32le", str(out_wav)])
        return _decode_via_soundfile(out_wav)


def decode_audio(path: Path, use_ffmpeg: bool) -> DecodedAudio:
    warnings: List[str] = []
    try:
        y2d, sr = _decode_via_ffmpeg(path) if use_ffmpeg else _decode_via_soundfile(path)
    except FileNotFoundError:
        warnings.append("ffmpeg_not_found_falling_back_to_soundfile")
        y2d, sr = _decode_via_soundfile(path)
    except Exception as exc:
        if not use_ffmpeg:
            raise
        warnings.append(f"ffmpeg_decode_failed_falling_back_to_soundfile:{type(exc).__name__}")
        y2d, sr = _decode_via_soundfile(path)
    return DecodedAudio(y=y2d, sr_hz=sr, channels=int(y2d.shape[1]), warnings=warnings)


def to_canonical(decoded: DecodedAudio, canonical_sr_hz: int, channel_mode: str, normalize_peak_dbfs: float):
    y2d = decoded.y
    if channel_mode not in ("mono", "stereo"):
        raise ValueError(f"Unsupported channel_mode: {channel_mode}")
    y = y2d.mean(axis=1) if channel_mode == "mono" else y2d[:, :2]
    channels = 1 if channel_mode == "mono" else y.shape[1]
    if decoded.sr_hz != canonical_sr_hz:
        if y.ndim == 1:
            y = resample_poly(y, canonical_sr_hz, decoded.sr_hz).astype(np.float32)
        else:
            y = np.stack([resample_poly(y[:, c], canonical_sr_hz, decoded.sr_hz).astype(np.float32) for c in range(channels)], axis=1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    target = 10.0 ** (normalize_peak_dbfs / 20.0)
    gain = min(10.0, target / peak) if peak > 0 else 1.0
    y = (y * gain).astype(np.float32)
    stats = {"peak": peak, "gain_applied": float(gain), "rms": float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0}
    return y, int(canonical_sr_hz), int(channels), stats


def write_wav(path: Path, y: np.ndarray, sr_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr_hz, subtype="PCM_16")
