from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from multi_recording_benchmark import RecordingPair, discover_pairs


def write_manifest(path: Path, pairs: list[RecordingPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["recording_id", "wav_path", "ref_path", "pairing_rule"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "recording_id": pair.recording_id,
                    "wav_path": pair.wav_path,
                    "ref_path": pair.ref_path,
                    "pairing_rule": pair.pairing_rule,
                }
            )


def run_multi(manifest: Path, out_dir: Path, config: dict, plot_limit: int) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    for proxy in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        env.pop(proxy, None)
    cmd = [
        sys.executable,
        "multi_recording_benchmark.py",
        "--manifest",
        str(manifest),
        "--out",
        str(out_dir),
        "--plot-limit",
        str(plot_limit),
        "--diarization-pipelines",
        "none",
        "--segmenters",
        config["segmenters"],
        "--embeddings",
        config["embeddings"],
        "--threshold",
        str(config["threshold"]),
        "--window-sec",
        str(config["window_sec"]),
        "--hop-sec",
        str(config["hop_sec"]),
        "--enrollment-sec",
        str(config["enrollment_sec"]),
        "--enrollment-strategy",
        config["enrollment_strategy"],
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "threshold_sweep_run.log").open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def objective(row: pd.Series) -> float:
    der = float(row.get("der_mean", np.nan))
    ste_pct = float(row.get("abs_speaking_time_error_sec_mean", np.nan))
    ok = float(row.get("ok_recordings", 0.0))
    n = float(row.get("n_recordings", 1.0))
    failure_penalty = max(0.0, n - ok) / max(n, 1.0)
    if not np.isfinite(der):
        return 999.0 + failure_penalty
    return der + 0.0025 * (ste_pct if np.isfinite(ste_pct) else 0.0) + failure_penalty


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_window_hops(raw: str) -> list[tuple[float, float]]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        window, hop = item.split(":")
        out.append((float(window), float(hop)))
    return out


def parse_strings(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def candidate_grid(args: argparse.Namespace) -> list[dict]:
    out = []
    for threshold in parse_floats(args.thresholds):
        for window_sec, hop_sec in parse_window_hops(args.window_hops):
            for enrollment_strategy in parse_strings(args.enrollment_strategies):
                out.append(
                    {
                        "threshold": threshold,
                        "window_sec": window_sec,
                        "hop_sec": hop_sec,
                        "enrollment_sec": 12.0,
                        "enrollment_strategy": enrollment_strategy,
                        "segmenters": args.segmenters,
                        "embeddings": args.embeddings,
                    }
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune threshold/window/hop on 5 random recordings and test on 5 held-out recordings.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("threshold_sweep_out"))
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--dev-n", type=int, default=5)
    parser.add_argument("--test-n", type=int, default=5)
    parser.add_argument("--plot-limit", type=int, default=60)
    parser.add_argument("--thresholds", default="0.45,0.55,0.62,0.70")
    parser.add_argument("--window-hops", default="1.0:0.5,1.5:0.75,2.0:1.0")
    parser.add_argument("--enrollment-strategies", default="first,longest,spread")
    parser.add_argument("--segmenters", default="energy_vad_internal,webrtc_vad")
    parser.add_argument("--embeddings", default="resemblyzer_dvector")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs, pairing_report = discover_pairs(args.input_dir)
    if len(pairs) < args.dev_n + args.test_n:
        raise SystemExit(f"Need at least {args.dev_n + args.test_n} paired recordings; found {len(pairs)}")
    rng = random.Random(args.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    dev_pairs = shuffled[: args.dev_n]
    test_pairs = shuffled[args.dev_n : args.dev_n + args.test_n]

    dev_manifest = args.out / "dev_manifest.csv"
    test_manifest = args.out / "test_manifest.csv"
    write_manifest(dev_manifest, dev_pairs)
    write_manifest(test_manifest, test_pairs)

    sweep_rows = []
    for i, config in enumerate(candidate_grid(args), start=1):
        config_id = f"cfg_{i:03d}_thr{config['threshold']}_win{config['window_sec']}_hop{config['hop_sec']}".replace(".", "p")
        run_dir = args.out / "dev_runs" / config_id
        code = run_multi(dev_manifest, run_dir, config, plot_limit=0)
        agg_path = run_dir / "aggregate_by_model.csv"
        if agg_path.exists():
            agg = pd.read_csv(agg_path)
            for row in agg.to_dict("records"):
                score = objective(pd.Series(row))
                sweep_rows.append({"config_id": config_id, "returncode": code, "objective": score, **config, **row})
        else:
            sweep_rows.append({"config_id": config_id, "returncode": code, "objective": 999.0, **config, "status": "failed"})

    sweep = pd.DataFrame(sweep_rows).sort_values("objective", na_position="last")
    sweep.to_csv(args.out / "dev_sweep_results.csv", index=False)
    best = sweep.iloc[0].to_dict()
    best_config = {
        "threshold": float(best["threshold"]),
        "window_sec": float(best["window_sec"]),
        "hop_sec": float(best["hop_sec"]),
        "enrollment_sec": float(best["enrollment_sec"]),
        "enrollment_strategy": str(best["enrollment_strategy"]),
        "segmenters": str(best["segmenter"]),
        "embeddings": str(best["embedding"]),
    }
    test_dir = args.out / "test_best"
    run_multi(test_manifest, test_dir, best_config, plot_limit=args.plot_limit)

    test_agg = pd.read_csv(test_dir / "aggregate_by_model.csv") if (test_dir / "aggregate_by_model.csv").exists() else pd.DataFrame()
    summary = [
        "# Threshold sweep",
        "",
        f"Seed: `{args.seed}`",
        f"Dev recordings: {args.dev_n}",
        f"Test recordings: {args.test_n}",
        "",
        "## Dev recordings",
        "\n".join(f"- {p.recording_id}" for p in dev_pairs),
        "",
        "## Test recordings",
        "\n".join(f"- {p.recording_id}" for p in test_pairs),
        "",
        "## Best dev configuration",
        json.dumps(best_config, indent=2),
        "",
        "## Best dev row",
        pd.DataFrame([best]).to_string(index=False),
        "",
        "## Held-out test aggregate",
        test_agg.to_string(index=False) if not test_agg.empty else "(test aggregate missing)",
        "",
        "## Files",
        "- `dev_sweep_results.csv`",
        "- `dev_manifest.csv`",
        "- `test_manifest.csv`",
        "- `test_best/aggregate_by_model.csv`",
        "- `test_best/all_recording_results.csv`",
        "- `test_best/timeline_plots/*.png`",
    ]
    (args.out / "THRESHOLD_SWEEP.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (args.out / "best_config.json").write_text(json.dumps(best_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print((args.out / "THRESHOLD_SWEEP.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
