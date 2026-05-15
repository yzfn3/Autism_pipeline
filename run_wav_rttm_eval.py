from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ams.config.schema import load_resolved_config
from ams.stages.stage_e_finalize import run_stage_e


@dataclass(frozen=True)
class RttmSegment:
    file_id: str
    start_sec: float
    end_sec: float
    speaker: str


def parse_rttm(path: Path) -> List[RttmSegment]:
    out: List[RttmSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        if dur > 0:
            out.append(RttmSegment(parts[1], start, start + dur, parts[7]))
    return sorted(out, key=lambda x: (x.start_sec, x.end_sec, x.speaker))


def pick_first_speaker(rttm: List[RttmSegment]) -> Optional[str]:
    return rttm[0].speaker if rttm else None


def merge_ranges(ranges: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    rs = sorted((float(s), float(e)) for s, e in ranges if e > s)
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


def ranges_for_speaker(rttm: List[RttmSegment], speaker: str) -> List[Tuple[float, float]]:
    return merge_ranges([(s.start_sec, s.end_sec) for s in rttm if s.speaker == speaker], 0.0)


def read_final_csv(path: Path) -> List[Tuple[float, float]]:
    if not path.exists():
        return []
    out: List[Tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            s, e = line.split(",")[:2]
            out.append((float(s), float(e)))
    return merge_ranges(out, 0.0)


def total_duration(ranges: List[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, e - s) for s, e in ranges))


def overlap_duration(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
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


def compute_metrics(ref: List[Tuple[float, float]], hyp: List[Tuple[float, float]]) -> Dict[str, float | None]:
    ref = merge_ranges(ref, 0.0)
    hyp = merge_ranges(hyp, 0.0)
    ref_dur = total_duration(ref)
    hyp_dur = total_duration(hyp)
    ov = overlap_duration(ref, hyp)
    union = ref_dur + hyp_dur - ov
    precision = ov / hyp_dur if hyp_dur > 0 else None
    recall = ov / ref_dur if ref_dur > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall > 0 else None
    return {
        "ref_total_sec": ref_dur,
        "hyp_total_sec": hyp_dur,
        "overlap_sec": ov,
        "missed_ref_sec": max(0.0, ref_dur - ov),
        "extra_hyp_sec": max(0.0, hyp_dur - ov),
        "union_sec": union,
        "hyp_minus_ref_sec": hyp_dur - ref_dur,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": ov / union if union > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run AMS and evaluate final timeline against an RTTM.")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--rttm", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--preset", type=str, default="noisy")
    ap.add_argument("--write-json", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_resolved_config(args.preset, None, [])
    run_stage_e(input_path=args.wav, out_root=args.out, cfg=cfg, final_csv_path=None)
    ref_segs = parse_rttm(args.rttm)
    speaker = pick_first_speaker(ref_segs)
    ref = ranges_for_speaker(ref_segs, speaker) if speaker else []
    hyp = read_final_csv(args.out / "final_main_timeline.csv")
    result = {"main_speaker": speaker, "metrics": compute_metrics(ref, hyp)}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
