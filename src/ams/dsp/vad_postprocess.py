from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Segment:
    id: str
    start_sec: float
    end_sec: float

    def to_dict(self):
        return {"id": self.id, "start_sec": float(self.start_sec), "end_sec": float(self.end_sec)}


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    out = np.empty_like(values, dtype=np.float32)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _hysteresis(t_centers: np.ndarray, p_speech: np.ndarray, on_thresh: float, off_thresh: float) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    active = False
    start = 0.0
    step = float(np.median(np.diff(t_centers))) if len(t_centers) > 1 else 0.01
    for t, p in zip(t_centers, p_speech):
        if not active and p >= on_thresh:
            active = True
            start = float(t)
        elif active and p <= off_thresh:
            ranges.append((start, float(t)))
            active = False
    if active and len(t_centers):
        ranges.append((start, float(t_centers[-1] + step)))
    return ranges


def merge_ranges(ranges: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    rs = sorted((s, e) for s, e in ranges if e > s)
    if not rs:
        return []
    out = [rs[0]]
    for s, e in rs[1:]:
        ps, pe = out[-1]
        if s - pe <= gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def postprocess_vad(t_centers: np.ndarray, p_speech: np.ndarray, on_thresh: float, off_thresh: float, min_speech_sec: float, merge_gap_sec: float, pad_sec: float, clip_lo_sec: float, clip_hi_sec: float):
    raw = _hysteresis(t_centers, p_speech, on_thresh, off_thresh)
    padded = [(max(clip_lo_sec, s - pad_sec), min(clip_hi_sec, e + pad_sec)) for s, e in raw]
    merged = merge_ranges(padded, merge_gap_sec)
    kept = [(s, e) for s, e in merged if e - s >= min_speech_sec]
    segs = [Segment(f"speech_{i:04d}", s, e) for i, (s, e) in enumerate(kept)]
    stats = {"raw": len(raw), "merged": len(merged), "kept": len(kept)}
    return segs, stats
