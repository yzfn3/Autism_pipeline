from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    start_sec: float
    end_sec: float


def sliding_windows(start_sec: float, end_sec: float, win_dur_sec: float, hop_sec: float) -> list[Window]:
    start_sec = float(start_sec)
    end_sec = float(end_sec)
    if end_sec <= start_sec:
        return []
    if end_sec - start_sec <= win_dur_sec:
        return [Window(start_sec, end_sec)]
    out: list[Window] = []
    t = start_sec
    while t + win_dur_sec <= end_sec + 1e-9:
        out.append(Window(t, min(end_sec, t + win_dur_sec)))
        t += hop_sec
    if out and out[-1].end_sec < end_sec:
        out.append(Window(max(start_sec, end_sec - win_dur_sec), end_sec))
    return out
