from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from ams.benchmark.artifacts import read_segmenter_result
from ams.benchmark.audio import audio_info, load_mono
from ams.benchmark.embeddings import embedding_factories
from ams.benchmark.metrics import score_main_timeline
from ams.benchmark.runner import _centroid, _enrollment_ranges, classify_main_ranges
from ams.benchmark.timeline import clip_ranges, parse_reference_timeline, write_ranges_csv


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one embedding classifier against a precomputed segmenter artifact.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--segmenter-json", type=Path, required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-main", type=Path, required=True)
    parser.add_argument("--target-speaker")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--hop-sec", type=float, default=0.75)
    parser.add_argument("--enrollment-sec", type=float, default=12.0)
    parser.add_argument("--enrollment-strategy", default="first")
    parser.add_argument("--collar-sec", type=float, default=0.0)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    t0 = time.perf_counter()
    ref = parse_reference_timeline(args.ref, target_speaker=args.target_speaker)
    info = audio_info(args.wav)
    ref_ranges = clip_ranges(ref.ranges, 0.0, info["duration_sec"])
    enroll_ranges = _enrollment_ranges(ref_ranges, args.enrollment_sec, args.enrollment_strategy)
    seg = read_segmenter_result(args.segmenter_json)

    row = {
        "segmenter": seg.model,
        "embedding": args.embedding,
        "status": "not_available",
        "mode": seg.mode,
        "error": "",
        "wall_time_sec": None,
    }
    try:
        y, sr = load_mono(args.wav, 16000)
        embedder = embedding_factories(prefer_gpu=True)[args.embedding]()
        centroid = _centroid(embedder, y, sr, enroll_ranges)
        hyp = classify_main_ranges(
            speech_ranges=seg.speech_ranges,
            embedder=embedder,
            y=y,
            sr=sr,
            centroid=centroid,
            threshold=args.threshold,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
        )
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
