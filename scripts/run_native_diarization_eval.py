from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ams.benchmark.artifacts import read_segmenter_result
from ams.benchmark.audio import audio_info
from ams.benchmark.metrics import score_main_timeline
from ams.benchmark.runner import _enrollment_ranges, _native_diarization_main
from ams.benchmark.timeline import clip_ranges, parse_reference_timeline, write_ranges_csv


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate native speaker labels from one diarization artifact.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--segmenter-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-main", type=Path, required=True)
    parser.add_argument("--target-speaker")
    parser.add_argument("--enrollment-sec", type=float, default=12.0)
    parser.add_argument("--enrollment-strategy", default="first")
    parser.add_argument("--collar-sec", type=float, default=0.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    seg = read_segmenter_result(args.segmenter_json)
    row = {
        "segmenter": seg.model,
        "embedding": "native_diarization",
        "status": "not_available",
        "mode": seg.mode,
        "error": "",
        "wall_time_sec": None,
    }
    try:
        if not seg.speaker_turns:
            raise RuntimeError(f"{seg.model} did not produce speaker turns")
        ref = parse_reference_timeline(args.ref, target_speaker=args.target_speaker)
        info = audio_info(args.wav)
        ref_ranges = clip_ranges(ref.ranges, 0.0, info["duration_sec"])
        enroll_ranges = _enrollment_ranges(ref_ranges, args.enrollment_sec, args.enrollment_strategy)
        hyp = _native_diarization_main(seg.speaker_turns, enroll_ranges)
        metrics = score_main_timeline(
            ref_ranges,
            hyp,
            audio_duration_sec=info["duration_sec"],
            collar_sec=args.collar_sec,
        )
        write_ranges_csv(args.out_main, hyp)
        row.update(metrics)
        row.update({"status": "ok", "output_csv": str(args.out_main)})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["wall_time_sec"] = round(time.perf_counter() - t0, 3)
        write_row(args.out_csv, row)
    return 0 if row["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
