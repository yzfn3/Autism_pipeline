from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = [
    "f1",
    "recall",
    "precision",
    "iou",
    "der",
    "miss_rate",
    "false_alarm_rate_vs_ref",
    "speaking_time_error_pct_ref",
    "abs_speaking_time_error_pct_ref",
]


def _float_or_nan(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _matrix(rows: list[dict[str, str]], metric: str):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("segmenter") and r.get("embedding")]
    segmenters = sorted({r["segmenter"] for r in ok})
    embeddings = sorted({r["embedding"] for r in ok})
    data = np.full((len(segmenters), len(embeddings)), np.nan, dtype=float)
    index_s = {name: i for i, name in enumerate(segmenters)}
    index_e = {name: i for i, name in enumerate(embeddings)}
    for row in ok:
        data[index_s[row["segmenter"]], index_e[row["embedding"]]] = _float_or_nan(row.get(metric, ""))
    return segmenters, embeddings, data


def _plot_metric(rows: list[dict[str, str]], metric: str, out_dir: Path) -> Path | None:
    segmenters, embeddings, data = _matrix(rows, metric)
    if data.size == 0 or np.all(np.isnan(data)):
        return None

    lower_is_better = metric in {
        "der",
        "miss_rate",
        "false_alarm_rate_vs_ref",
        "abs_speaking_time_error_pct_ref",
    }
    cmap = "magma_r" if lower_is_better else "viridis"
    if metric in {"f1", "recall", "precision", "iou", "miss_rate"}:
        vmin, vmax = 0.0, 1.0
    else:
        finite = data[np.isfinite(data)]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0

    fig_w = max(8.0, 1.5 * len(embeddings))
    fig_h = max(5.0, 0.65 * len(segmenters))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(embeddings)))
    ax.set_yticks(np.arange(len(segmenters)))
    ax.set_xticklabels(embeddings, rotation=35, ha="right")
    ax.set_yticklabels(segmenters)
    ax.set_title(metric)
    ax.set_xlabel("Embedding / speaker model")
    ax.set_ylabel("Segmentation / diarization model")

    for i in range(len(segmenters)):
        for j in range(len(embeddings)):
            val = data[i, j]
            label = "NA" if np.isnan(val) else f"{val:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric, rotation=270, labelpad=14)
    fig.tight_layout()

    out_path = out_dir / f"heatmap_{_safe_name(metric)}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render heatmaps from AMS benchmark_results.csv.")
    parser.add_argument("--results", type=Path, required=True, help="Path to benchmark_results.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output folder. Defaults to <results folder>/heatmaps")
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS), help="Comma-separated metric columns")
    args = parser.parse_args()

    rows = _read_rows(args.results)
    out_dir = args.out or (args.results.parent / "heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    written = []
    for metric in metrics:
        path = _plot_metric(rows, metric, out_dir)
        if path:
            written.append(path)

    print(f"Wrote {len(written)} heatmap(s) to {out_dir}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
