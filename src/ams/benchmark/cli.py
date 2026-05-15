from __future__ import annotations

import argparse
from pathlib import Path

from ams.benchmark.runner import BenchmarkConfig, default_embeddings, default_segmenters, run_benchmark


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark segmentation/diarization and speaker-embedding combinations against a MAIN-speaker timeline CSV.")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--ref-csv", type=Path, required=True)
    ap.add_argument("--target-speaker", type=str, default=None, help="Target speaker id for RTTM references. Default: longest speaker.")
    ap.add_argument("--out", type=Path, default=Path("benchmark_out"))
    ap.add_argument("--segmenters", type=str, default=None, help="Comma-separated subset. Default: requested models plus energy_vad_internal.")
    ap.add_argument("--embeddings", type=str, default=None, help="Comma-separated subset. Default: requested models plus band_energy_internal.")
    ap.add_argument("--collar-sec", type=float, default=0.0)
    ap.add_argument("--enrollment-sec", type=float, default=12.0)
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--window-sec", type=float, default=1.5)
    ap.add_argument("--hop-sec", type=float, default=0.75)
    args = ap.parse_args()

    cfg = BenchmarkConfig(
        wav_path=args.wav,
        reference_csv=args.ref_csv,
        out_dir=args.out,
        segmenters=_split_csv(args.segmenters, default_segmenters()),
        embeddings=_split_csv(args.embeddings, default_embeddings()),
        collar_sec=args.collar_sec,
        enrollment_sec=args.enrollment_sec,
        threshold=args.threshold,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        target_speaker=args.target_speaker,
    )
    report = run_benchmark(cfg)
    ok = [r for r in report["rows"] if r["status"] == "ok"]
    print(f"Wrote {report['results_csv']}")
    if ok:
        best = ok[0]
        print(f"Best: {best['segmenter']} + {best['embedding']} DER={best.get('der')} F1={best.get('f1')}")
    else:
        print("No model combination completed; see benchmark_report.json for adapter errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
