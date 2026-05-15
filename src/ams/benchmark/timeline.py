from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


Range = Tuple[float, float]


@dataclass(frozen=True)
class ReferenceTimeline:
    ranges: List[Range]
    source: str
    fps: float
    target_speaker: str = ""

    @property
    def total_sec(self) -> float:
        return total_duration(self.ranges)


def timecode_to_sec(tc: str, fps: float) -> float:
    parts = tc.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Expected HH:MM:SS:FF timecode, got {tc!r}")
    h, m, s, f = (int(x) for x in parts)
    return h * 3600.0 + m * 60.0 + s + f / float(fps)


def merge_ranges(ranges: Iterable[Range], gap: float = 0.0) -> List[Range]:
    rs = sorted((float(s), float(e)) for s, e in ranges if float(e) > float(s))
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


def clip_ranges(ranges: Iterable[Range], lo: float, hi: float) -> List[Range]:
    return merge_ranges((max(lo, s), min(hi, e)) for s, e in ranges)


def total_duration(ranges: Iterable[Range]) -> float:
    return float(sum(max(0.0, e - s) for s, e in ranges))


def overlap_duration(a: List[Range], b: List[Range]) -> float:
    a = merge_ranges(a)
    b = merge_ranges(b)
    i = j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            overlap += e - s
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return float(overlap)


def subtract_collar(ranges: List[Range], collar_sec: float) -> List[Range]:
    if collar_sec <= 0:
        return merge_ranges(ranges)
    out: List[Range] = []
    for s, e in ranges:
        ns, ne = s + collar_sec, e - collar_sec
        if ne > ns:
            out.append((ns, ne))
    return merge_ranges(out)


def parse_main_timeline_csv(path: Path) -> ReferenceTimeline:
    """Parse DaVinci/EDL marker CSV or a simple start_sec,end_sec CSV."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    fields = set(reader.fieldnames or [])

    if {"start_sec", "end_sec"}.issubset(fields):
        ranges = [(float(row["start_sec"]), float(row["end_sec"])) for row in reader]
        return ReferenceTimeline(merge_ranges(ranges), source="start_end_csv", fps=0.0)

    if "Source In" not in fields:
        raise ValueError(f"Unsupported timeline CSV columns: {sorted(fields)}")

    timestamps: List[float] = []
    fps = 24.0
    for row in reader:
        tc = (row.get("Source In") or "").strip()
        if not tc:
            continue
        raw_fps = (row.get("Source FPS") or "").strip()
        fps = float(raw_fps) if raw_fps else fps
        timestamps.append(timecode_to_sec(tc, fps))

    timestamps.sort()
    if len(timestamps) % 2:
        raise ValueError(f"Reference marker CSV has {len(timestamps)} markers; expected start/end pairs")
    ranges = [(timestamps[i], timestamps[i + 1]) for i in range(0, len(timestamps), 2)]
    return ReferenceTimeline(merge_ranges(ranges), source="edl_marker_pairs", fps=fps)


def parse_rttm(path: Path, target_speaker: str | None = None) -> ReferenceTimeline:
    """Parse an RTTM file and return one speaker timeline.

    When target_speaker is omitted, the speaker with the largest annotated
    duration is selected. This keeps short VoxConverse fixtures deterministic
    while still allowing explicit target-speaker evaluation later.
    """
    by_speaker: dict[str, List[Range]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]
        if dur > 0:
            by_speaker.setdefault(speaker, []).append((start, start + dur))
    if not by_speaker:
        raise ValueError(f"No SPEAKER rows found in RTTM: {path}")
    if target_speaker:
        if target_speaker not in by_speaker:
            raise ValueError(f"Target speaker {target_speaker!r} not found in RTTM; available: {sorted(by_speaker)}")
        speaker = target_speaker
    else:
        speaker = sorted(by_speaker, key=lambda spk: (-total_duration(by_speaker[spk]), spk))[0]
    return ReferenceTimeline(merge_ranges(by_speaker[speaker]), source="rttm", fps=0.0, target_speaker=speaker)


def parse_reference_timeline(path: Path, target_speaker: str | None = None) -> ReferenceTimeline:
    if path.suffix.lower() == ".rttm":
        return parse_rttm(path, target_speaker=target_speaker)
    return parse_main_timeline_csv(path)


def write_ranges_csv(path: Path, ranges: List[Range]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec"])
        for s, e in ranges:
            writer.writerow([f"{s:.3f}", f"{e:.3f}"])
