from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ams.benchmark.audio import audio_info
from ams.benchmark.metrics import score_main_timeline
from ams.benchmark.timeline import Range, merge_ranges, parse_reference_timeline, write_ranges_csv


@dataclass(frozen=True)
class RecordingPair:
    recording_id: str
    wav_path: Path
    ref_path: Path
    pairing_rule: str


def norm_key(path: Path) -> str:
    name = path.name.lower()
    for suffix in [".wav.csv", ".csv", ".wav"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.startswith("vad_"):
        name = name[4:]
    return name


def discover_pairs(folder: Path) -> tuple[list[RecordingPair], list[dict]]:
    wavs = sorted(folder.rglob("*.wav"), key=lambda p: p.name.lower())
    csvs = sorted(folder.rglob("*.csv"), key=lambda p: p.name.lower())
    csv_by_key: dict[str, list[Path]] = {}
    for ref in csvs:
        csv_by_key.setdefault(norm_key(ref), []).append(ref)

    pairs: list[RecordingPair] = []
    report: list[dict] = []
    used_refs: set[Path] = set()
    for wav in wavs:
        key = norm_key(wav)
        candidates = csv_by_key.get(key, [])
        if len(candidates) == 1:
            ref = candidates[0]
            used_refs.add(ref)
            pairs.append(RecordingPair(wav.stem, wav, ref, "normalized_stem"))
            report.append({"recording_id": wav.stem, "status": "paired", "wav": str(wav), "ref": str(ref), "note": ""})
        elif len(candidates) > 1:
            report.append({"recording_id": wav.stem, "status": "ambiguous", "wav": str(wav), "ref": "", "note": "; ".join(str(x) for x in candidates)})
        else:
            report.append({"recording_id": wav.stem, "status": "missing_csv", "wav": str(wav), "ref": "", "note": ""})

    for ref in csvs:
        if ref not in used_refs:
            key = norm_key(ref)
            if not any(norm_key(wav) == key for wav in wavs):
                report.append({"recording_id": key, "status": "orphan_csv", "wav": "", "ref": str(ref), "note": ""})
    return pairs, report


def read_manifest(path: Path) -> list[RecordingPair]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            wav = Path(row["wav_path"])
            ref = Path(row["ref_path"])
            recording_id = row.get("recording_id") or wav.stem
            rows.append(RecordingPair(recording_id, wav, ref, row.get("pairing_rule") or "manifest"))
    return rows


def write_dicts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ranges(path: Path) -> list[Range]:
    rows = pd.read_csv(path)
    if not {"start_sec", "end_sec"}.issubset(rows.columns):
        return []
    return [(float(row.start_sec), float(row.end_sec)) for row in rows.itertuples()]


def subtract_ranges(base: list[Range], remove: list[Range]) -> list[Range]:
    out: list[Range] = []
    for start, end in merge_ranges(base):
        pieces = [(start, end)]
        for rs, re in merge_ranges(remove):
            new_pieces: list[Range] = []
            for ps, pe in pieces:
                if re <= ps or rs >= pe:
                    new_pieces.append((ps, pe))
                else:
                    if rs > ps:
                        new_pieces.append((ps, rs))
                    if re < pe:
                        new_pieces.append((re, pe))
            pieces = new_pieces
        out.extend(pieces)
    return merge_ranges(out)


def intersection_ranges(a: list[Range], b: list[Range]) -> list[Range]:
    out: list[Range] = []
    a = merge_ranges(a)
    b = merge_ranges(b)
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def plot_timeline(ref_ranges: list[Range], hyp_ranges: list[Range], out_png: Path, title: str, duration_sec: float) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlap = intersection_ranges(ref_ranges, hyp_ranges)
    missed = subtract_ranges(ref_ranges, hyp_ranges)
    false = subtract_ranges(hyp_ranges, ref_ranges)
    fig, ax = plt.subplots(figsize=(14, 3.8))

    def draw(ranges: list[Range], y: float, color: str, label: str, height: float = 0.28) -> None:
        first = True
        for start, end in ranges:
            ax.broken_barh([(start, end - start)], (y - height / 2, height), facecolors=color, label=label if first else None)
            first = False

    draw(ref_ranges, 3.0, "#3b82f6", "Reference")
    draw(hyp_ranges, 2.0, "#f59e0b", "Prediction")
    draw(overlap, 1.0, "#22c55e", "Overlap")
    draw(missed, 0.3, "#ef4444", "Missed")
    draw(false, -0.4, "#a855f7", "False alarm")
    ax.set_xlim(0, max(duration_sec, 1.0))
    ax.set_ylim(-0.8, 3.4)
    ax.set_yticks([3.0, 2.0, 1.0, 0.3, -0.4])
    ax.set_yticklabels(["ref", "hyp", "overlap", "miss", "false"])
    ax.set_xlabel("seconds")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_results(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["der", "f1", "recall", "precision", "speaking_time_error_sec", "abs_speaking_time_error_sec"]
    out = []
    for (segmenter, embedding), group in rows.groupby(["segmenter", "embedding"], dropna=False):
        ok = group[group["status"] == "ok"].copy()
        item = {
            "segmenter": segmenter,
            "embedding": embedding,
            "n_recordings": int(group["recording_id"].nunique()),
            "ok_recordings": int(ok["recording_id"].nunique()),
            "failure_rows": int((group["status"] != "ok").sum()),
        }
        for metric in metric_cols:
            vals = pd.to_numeric(ok.get(metric), errors="coerce").dropna().to_numpy(dtype=float) if metric in ok else np.array([])
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            item[f"{metric}_median"] = float(np.median(vals)) if vals.size else np.nan
            lo, hi = bootstrap_ci(vals)
            item[f"{metric}_ci95_low"] = lo
            item[f"{metric}_ci95_high"] = hi
        out.append(item)
    return pd.DataFrame(out).sort_values(["der_mean", "f1_mean"], ascending=[True, False], na_position="last")


def write_partial_outputs(out_dir: Path, all_rows: list[pd.DataFrame], run_rows: list[dict], plot_count: int = 0) -> None:
    write_dicts(out_dir / "recording_run_status.csv", run_rows)
    if not all_rows:
        return
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(out_dir / "all_recording_results.csv", index=False)
    aggregate = aggregate_results(all_df)
    aggregate.to_csv(out_dir / "aggregate_by_model.csv", index=False)
    write_aggregate_heatmaps(aggregate, out_dir / "aggregate_heatmaps")
    summary = [
        "# Multi-recording benchmark",
        "",
        f"Completed recording result tables: {all_df['recording_id'].nunique()}",
        f"Result rows: {len(all_df)}",
        f"Timeline plots so far: {plot_count}",
        "",
        "## Best aggregate rows so far",
        aggregate.head(25).to_string(index=False),
        "",
        "This file is updated incrementally while long runs are in progress.",
    ]
    (out_dir / "MULTI_RESULTS_PARTIAL.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def write_aggregate_heatmaps(aggregate: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("der_mean", "DER mean", "lower"),
        ("f1_mean", "F1 mean", "higher"),
        ("recall_mean", "Recall mean", "higher"),
        ("precision_mean", "Precision mean", "higher"),
        ("abs_speaking_time_error_sec_mean", "Abs speaking-time error mean (sec)", "lower"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, direction in metrics:
        if metric not in aggregate.columns:
            continue
        pivot = aggregate.pivot(index="segmenter", columns="embedding", values=metric)
        if pivot.empty:
            continue
        fig_w = max(8.0, 1.2 * len(pivot.columns) + 3)
        fig_h = max(4.5, 0.7 * len(pivot.index) + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        data = pivot.to_numpy(dtype=float)
        cmap = "viridis" if direction == "higher" else "viridis_r"
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                label = "" if not np.isfinite(value) else f"{value:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if np.isfinite(value) and value > np.nanmean(data) else "black")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"aggregate_heatmap_{metric}.png", dpi=160)
        plt.close(fig)


def run_orchestrator(pair: RecordingPair, out_dir: Path, args: argparse.Namespace, env: dict[str, str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "orchestrate_benchmark.py",
        "--wav",
        str(pair.wav_path),
        "--ref",
        str(pair.ref_path),
        "--out",
        str(out_dir),
        "--diarization-pipelines",
        args.diarization_pipelines,
        "--segmenters",
        args.segmenters,
        "--embeddings",
        args.embeddings,
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
    if args.hf_token:
        cmd.extend(["--hf-token", args.hf_token])
    if args.skip_existing:
        cmd.append("--skip-existing")
    log = out_dir / "multi_recording_orchestrator.log"
    with log.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run orchestrated benchmarks over a folder or manifest and aggregate results.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmark_manifest"))
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--sort-by-duration", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plot-limit", type=int, default=120)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    parser.add_argument("--diarization-pipelines", default="nemo_sortformer,pyannote_community_1,silero_campp_clustering")
    parser.add_argument("--segmenters", default="nemo_sortformer,pyannote_community_1,silero_campp_clustering,silero_vad,webrtc_vad,energy_vad_internal")
    parser.add_argument("--embeddings", default="titanet,ecapa_tdnn,wavlm,campp,speechbrain_resnet,resemblyzer_dvector")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--hop-sec", type=float, default=0.75)
    parser.add_argument("--enrollment-sec", type=float, default=12.0)
    parser.add_argument("--enrollment-strategy", default="first")
    parser.add_argument("--collar-sec", type=float, default=0.0)
    args = parser.parse_args()

    if not args.input_dir and not args.manifest:
        raise SystemExit("Provide --input-dir or --manifest")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        pairs = read_manifest(args.manifest)
        pairing_report = [{"recording_id": p.recording_id, "status": "paired", "wav": str(p.wav_path), "ref": str(p.ref_path), "note": p.pairing_rule} for p in pairs]
    else:
        pairs, pairing_report = discover_pairs(args.input_dir)
    if args.sort_by_duration:
        pairs = sorted(pairs, key=lambda p: audio_info(p.wav_path)["duration_sec"])
    else:
        pairs = sorted(pairs, key=lambda p: p.recording_id.lower())
    if args.max_recordings is not None:
        pairs = pairs[: args.max_recordings]

    write_dicts(args.out / "discovered_manifest.csv", [{"recording_id": p.recording_id, "wav_path": str(p.wav_path), "ref_path": str(p.ref_path), "pairing_rule": p.pairing_rule} for p in pairs])
    write_dicts(args.out / "pairing_report.csv", pairing_report)
    if args.discover_only:
        (args.out / "MULTI_RESULTS.md").write_text(
            "# Multi-recording benchmark discovery\n\n"
            f"Paired recordings: {len(pairs)}\n\n"
            "No models were run because `--discover-only` was used.\n",
            encoding="utf-8",
        )
        print((args.out / "MULTI_RESULTS.md").resolve())
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    for proxy in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        env.pop(proxy, None)

    all_rows: list[pd.DataFrame] = []
    run_rows: list[dict] = []
    for pair in tqdm(pairs, desc="recordings"):
        rec_out = args.out / "recordings" / pair.recording_id
        results_path = rec_out / "all_results.csv"
        if args.skip_existing and results_path.exists():
            code = 0
        else:
            code = run_orchestrator(pair, rec_out, args, env)
        run_row = {"recording_id": pair.recording_id, "status": "ok" if code == 0 and results_path.exists() else "failed", "returncode": code, "out_dir": str(rec_out), "wav_path": str(pair.wav_path), "ref_path": str(pair.ref_path)}
        try:
            info = audio_info(pair.wav_path)
            ref = parse_reference_timeline(pair.ref_path)
            run_row.update({"duration_sec": info["duration_sec"], "ref_main_sec": ref.total_sec, "ref_ranges": len(ref.ranges)})
        except Exception as exc:
            run_row["metadata_error"] = f"{type(exc).__name__}: {exc}"
        run_rows.append(run_row)
        if results_path.exists():
            df = pd.read_csv(results_path)
            df["recording_id"] = pair.recording_id
            df["wav_path"] = str(pair.wav_path)
            df["ref_path"] = str(pair.ref_path)
            all_rows.append(df)
        write_partial_outputs(args.out, all_rows, run_rows, plot_count=0)

    write_dicts(args.out / "recording_run_status.csv", run_rows)
    if not all_rows:
        raise SystemExit("No per-recording result tables were produced")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(args.out / "all_recording_results.csv", index=False)
    aggregate = aggregate_results(all_df)
    aggregate.to_csv(args.out / "aggregate_by_model.csv", index=False)
    write_aggregate_heatmaps(aggregate, args.out / "aggregate_heatmaps")

    plot_count = 0
    for row in all_df[all_df["status"] == "ok"].itertuples():
        if plot_count >= args.plot_limit:
            break
        output_csv = getattr(row, "output_csv", "")
        if not isinstance(output_csv, str) or not output_csv:
            continue
        try:
            ref = parse_reference_timeline(Path(row.ref_path))
            hyp = load_ranges(Path(output_csv))
            info = audio_info(Path(row.wav_path))
            title = f"{row.recording_id}: {row.segmenter} + {row.embedding}"
            safe_name = f"{row.recording_id}__{row.segmenter}__{row.embedding}".replace(" ", "_")
            plot_timeline(ref.ranges, hyp, args.out / "timeline_plots" / f"{safe_name}.png", title, info["duration_sec"])
            plot_count += 1
        except Exception:
            continue

    summary = [
        "# Multi-recording benchmark",
        "",
        f"Paired recordings run: {len(pairs)}",
        f"Successful recording runs: {sum(1 for row in run_rows if row['status'] == 'ok')}",
        f"Result rows: {len(all_df)}",
        f"Timeline plots: {plot_count}",
        "",
        "## Best aggregate rows",
        aggregate.head(25).to_string(index=False),
        "",
        "## Files",
        "- `discovered_manifest.csv`",
        "- `pairing_report.csv`",
        "- `recording_run_status.csv`",
        "- `all_recording_results.csv`",
        "- `aggregate_by_model.csv`",
        "- `aggregate_heatmaps/*.png`",
        "- `timeline_plots/*.png`",
    ]
    (args.out / "MULTI_RESULTS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (args.out / "multi_manifest.json").write_text(json.dumps({"recordings": len(pairs), "plot_count": plot_count}, indent=2) + "\n", encoding="utf-8")
    print((args.out / "MULTI_RESULTS.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
