from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ams.benchmark.audio import audio_info
from ams.benchmark.timeline import parse_reference_timeline, total_duration, write_ranges_csv


DIARIZATION_PIPELINES = [
    "nemo_sortformer",
    "pyannote_community_1",
    "silero_campp_clustering",
]

SEGMENTATION_MODELS = [
    "nemo_sortformer",
    "pyannote_community_1",
    "silero_campp_clustering",
    "silero_vad",
    "webrtc_vad",
    "energy_vad_internal",
]

EMBEDDING_MODELS = [
    "titanet",
    "ecapa_tdnn",
    "wavlm",
    "campp",
    "speechbrain_resnet",
    "resemblyzer_dvector",
]


def parse_model_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip() and x.strip().lower() != "none"]


def clean_env(out_dir: Path, hf_token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env["PYTHONPATH"] = str(Path("src").resolve())
    env["WANDB_MODE"] = "disabled"
    env["TEMP"] = str((out_dir / "_tmp").resolve())
    env["TMP"] = env["TEMP"]
    env["HF_HOME"] = str((out_dir / "_hf_home").resolve())
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


def run_command(cmd: list[str], *, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    return proc.returncode, time.perf_counter() - t0


def read_single_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rows_if_valid(path: Path, min_rows: int = 1) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        rows = read_rows(path)
    except Exception:
        return None
    if len(rows) < min_rows:
        return None
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_table(df: pd.DataFrame, cols: list[str], limit: int = 30) -> str:
    if df.empty:
        return "(none)"
    return df[cols].head(limit).to_string(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated orchestrator for diarization, segmentation, and embedding comparisons.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("benchmark_orchestrated"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    parser.add_argument("--target-speaker")
    parser.add_argument("--diarization-pipelines", default=",".join(DIARIZATION_PIPELINES))
    parser.add_argument("--segmenters", default=",".join(SEGMENTATION_MODELS))
    parser.add_argument("--embeddings", default=",".join(EMBEDDING_MODELS))
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--hop-sec", type=float, default=0.75)
    parser.add_argument("--enrollment-sec", type=float, default=12.0)
    parser.add_argument("--enrollment-strategy", default="first")
    parser.add_argument("--collar-sec", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing artifacts and row CSVs when resuming a partially completed recording.")
    args = parser.parse_args()
    diarization_pipelines = parse_model_list(args.diarization_pipelines)
    segmentation_models = parse_model_list(args.segmenters)
    embedding_models = parse_model_list(args.embeddings)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for child in ["artifacts", "rows", "logs", "timelines", "heatmaps", "_tmp", "_hf_home"]:
        (out / child).mkdir(exist_ok=True)

    env = clean_env(out, args.hf_token)
    info = audio_info(args.wav)
    ref = parse_reference_timeline(args.ref, target_speaker=args.target_speaker)
    write_ranges_csv(out / "reference_main_timeline.csv", ref.ranges)

    segmenter_artifacts: dict[str, Path] = {}
    segmenter_rows: list[dict] = []
    for model in segmentation_models:
        artifact = out / "artifacts" / f"{model}.json"
        meta = out / "artifacts" / f"{model}.meta"
        if args.skip_existing and artifact.exists():
            segmenter_artifacts[model] = artifact
            segmenter_rows.append({"segmenter": model, "status": "ok", "artifact": str(artifact), "wall_time_sec": 0.0, "error": "", "note": "reused_existing_artifact"})
            continue
        code, wall = run_command(
            [
                sys.executable,
                "scripts/run_segmenter.py",
                "--wav",
                str(args.wav),
                "--model",
                model,
                "--out",
                str(artifact),
                "--meta",
                str(meta),
            ],
            env=env,
            log_path=out / "logs" / f"segmenter__{model}.log",
        )
        if code == 0 and artifact.exists():
            segmenter_artifacts[model] = artifact
            segmenter_rows.append({"segmenter": model, "status": "ok", "artifact": str(artifact), "wall_time_sec": round(wall, 3), "error": ""})
        else:
            log_tail = (out / "logs" / f"segmenter__{model}.log").read_text(encoding="utf-8", errors="replace")[-1000:]
            segmenter_rows.append({"segmenter": model, "status": "not_available", "artifact": "", "wall_time_sec": round(wall, 3), "error": log_tail.strip()})
    write_rows(out / "segmentation_model_status.csv", segmenter_rows)

    e2e_rows: list[dict] = []
    for model in diarization_pipelines:
        artifact = segmenter_artifacts.get(model)
        row_csv = out / "rows" / f"native__{model}.csv"
        cached = read_rows_if_valid(row_csv) if args.skip_existing else None
        if cached:
            e2e_rows.append(cached[0])
            continue
        if not artifact:
            e2e_rows.append({"segmenter": model, "embedding": "native_diarization", "status": "not_available", "error": "segmenter did not run"})
            continue
        code, _ = run_command(
            [
                sys.executable,
                "scripts/run_native_diarization_eval.py",
                "--wav",
                str(args.wav),
                "--ref",
                str(args.ref),
                "--segmenter-json",
                str(artifact),
                "--out-csv",
                str(row_csv),
                "--out-main",
                str(out / "timelines" / f"{model}.native_main.csv"),
                "--enrollment-sec",
                str(args.enrollment_sec),
                "--enrollment-strategy",
                args.enrollment_strategy,
                "--collar-sec",
                str(args.collar_sec),
            ],
            env=env,
            log_path=out / "logs" / f"native__{model}.log",
        )
        e2e_rows.append(read_single_row(row_csv) if row_csv.exists() else {"segmenter": model, "embedding": "native_diarization", "status": "not_available", "error": "native evaluation failed"})
    write_rows(out / "diarization_pipelines.csv", e2e_rows)

    matrix_rows: list[dict] = []
    specs = [f"{name}={path}" for name, path in segmenter_artifacts.items()]
    for embedding in embedding_models:
        row_csv = out / "rows" / f"embedding__{embedding}.csv"
        cached = read_rows_if_valid(row_csv, min_rows=len(specs)) if args.skip_existing else None
        if cached:
            matrix_rows.extend(cached)
            continue
        cmd = [
            sys.executable,
            "scripts/run_embedding_batch.py",
            "--wav",
            str(args.wav),
            "--ref",
            str(args.ref),
            "--embedding",
            embedding,
            "--out-csv",
            str(row_csv),
            "--main-dir",
            str(out / "timelines"),
            "--threshold",
            str(args.threshold),
            "--window-sec",
            str(args.window_sec),
            "--hop-sec",
            str(args.hop_sec),
            "--enrollment-sec",
            str(args.enrollment_sec),
            "--enrollment-strategy",
            args.enrollment_strategy,
            "--collar-sec",
            str(args.collar_sec),
        ]
        for spec in specs:
            cmd.extend(["--segmenter-json", spec])
        run_command(cmd, env=env, log_path=out / "logs" / f"embedding__{embedding}.log")
        if row_csv.exists():
            matrix_rows.extend(read_rows(row_csv))
        else:
            matrix_rows.extend(
                {
                    "segmenter": name,
                    "embedding": embedding,
                    "status": "not_available",
                    "error": "embedding batch failed before writing rows",
                }
                for name in segmenter_artifacts
            )

    write_rows(out / "segmentation_embedding_matrix.csv", matrix_rows)
    all_rows = e2e_rows + matrix_rows
    write_rows(out / "all_results.csv", all_rows)

    all_df = pd.DataFrame(all_rows)
    ok = all_df[all_df.get("status", "") == "ok"].copy()
    for col in ["der", "f1", "recall", "precision", "speaking_time_error_sec", "wall_time_sec"]:
        if col in ok:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    top_cols = ["segmenter", "embedding", "der", "f1", "recall", "precision", "speaking_time_error_sec", "wall_time_sec"]
    summary = [
        "# Orchestrated benchmark results",
        "",
        f"Audio: `{args.wav}`",
        f"Reference: `{args.ref}`",
        f"Duration: {info['duration_sec']:.3f} sec",
        f"Reference speaker: `{ref.target_speaker}`",
        f"Reference main speech: {total_duration(ref.ranges):.3f} sec",
        "",
        "## Model sets",
        f"End-to-end diarization pipelines: {', '.join(diarization_pipelines)}",
        f"Segmentation models: {', '.join(segmentation_models)}",
        f"Embedding models: {', '.join(embedding_models)}",
        "",
        "## Status counts",
        all_df.get("status", pd.Series(dtype=str)).value_counts().to_string(),
        "",
        "## Best completed rows",
        format_table(ok.sort_values(["der", "wall_time_sec"], na_position="last"), top_cols, 25),
        "",
        "## Files",
        "- `diarization_pipelines.csv`: native diarization pipeline comparison",
        "- `segmentation_embedding_matrix.csv`: 6 x 6 segmentation/embedding comparison",
        "- `all_results.csv`: combined rows",
        "- `heatmaps/`: PNG metric heatmaps",
    ]
    (out / "RESULTS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "wav": str(args.wav),
                "ref": str(args.ref),
                "audio": info,
                "reference": {
                    "source": ref.source,
                    "target_speaker": ref.target_speaker,
                    "total_main_sec": total_duration(ref.ranges),
                    "num_ranges": len(ref.ranges),
                },
                "diarization_pipelines": diarization_pipelines,
                "segmentation_models": segmentation_models,
                "embedding_models": embedding_models,
                "threshold": args.threshold,
                "window_sec": args.window_sec,
                "hop_sec": args.hop_sec,
                "enrollment_sec": args.enrollment_sec,
                "enrollment_strategy": args.enrollment_strategy,
                "collar_sec": args.collar_sec,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if Path("plot_benchmark_heatmaps.py").exists():
        shutil.copyfile(out / "segmentation_embedding_matrix.csv", out / "benchmark_results.csv")
        run_command(
            [
                sys.executable,
                "plot_benchmark_heatmaps.py",
                "--results",
                str(out / "benchmark_results.csv"),
                "--out",
                str(out / "heatmaps"),
            ],
            env=env,
            log_path=out / "logs" / "heatmaps.log",
        )

    print(out.resolve())
    print((out / "RESULTS.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
