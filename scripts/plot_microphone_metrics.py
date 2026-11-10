"""Plot supplied precision/recall percentages and their calculated F1."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def percentage(value):
    number = float(value)
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("expected a percentage from 0 to 100")
    return number


def metrics(precision, recall):
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piezo", nargs=2, type=percentage, metavar=("PRECISION", "RECALL"), required=True)
    parser.add_argument("--phone", nargs=2, type=percentage, metavar=("PRECISION", "RECALL"), required=True)
    parser.add_argument("--title", default="Microphone comparison")
    parser.add_argument("--hypothetical", action="store_true", help="Mark all scores as simulated, not measured.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(3)
    for i, (label, color, values) in enumerate([
        ("Piezo", "#268054", args.piezo), ("Air phone", "#3879b8", args.phone)
    ]):
        scores = metrics(*values)
        bars = ax.bar(x + (i - .5) * .32, scores, .32, color=color, label=label)
        ax.bar_label(bars, labels=[f"{score:.1f}%" for score in scores], padding=5, fontsize=12)
        print(f"{label} F1: {scores[2]:.2f}%")
    ax.set_xticks(x, ["Precision", "Recall", "F1"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent")
    ax.set_title(f"{args.title} | HYPOTHETICAL" if args.hypothetical else args.title, fontsize=16)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.12))
    note = "Simulated scores, not measured performance." if args.hypothetical else "F1 calculated from the supplied precision and recall."
    fig.text(.5, .015, note, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
