"""Diarize two already aligned WAVs and optionally score one shared target."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from time import perf_counter

import soundfile as sf

from ams.benchmark.metrics import score_main_timeline
from ams.benchmark.segmentation import PyannoteDiarizationSegmenter
from ams.benchmark.timeline import clip_ranges, overlap_duration, parse_main_timeline_csv, total_duration


MODEL_ID = "pyannote/speaker-diarization-community-1"


def longest(clusters):
    return min(clusters, key=lambda key: (-total_duration(clusters[key]), key), default=None)


def match_timeline(anchor, clusters):
    """Match using temporal Dice only, with no reference labels."""
    anchor_sec = total_duration(anchor)
    if not anchor_sec or not clusters:
        return None, None

    def dice(key):
        spans = clusters[key]
        return 2 * overlap_duration(anchor, spans) / (anchor_sec + total_duration(spans))

    best = min(clusters, key=lambda key: (-dice(key), key))
    agreement = dice(best)
    return (best, agreement) if agreement > 0 else (None, agreement)


def read_aligned_pair(piezo, phone):
    audio = [sf.read(path, dtype="float32", always_2d=True) for path in (piezo, phone)]
    (a, sr_a), (b, sr_b) = audio
    if sr_a != sr_b or len(a) != len(b) or len(a) == 0:
        raise ValueError("Inputs must already be aligned, nonempty, and have equal sample rates and frame counts.")
    return (a.mean(axis=1), b.mean(axis=1)), sr_a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piezo", type=Path, required=True)
    parser.add_argument("--phone", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional target CSV with start_sec,end_sec on the aligned timebase.")
    parser.add_argument("--out", type=Path, required=True, help="New output directory for private results.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    try:
        waveforms, sr = read_aligned_pair(args.piezo, args.phone)
        duration_sec = len(waveforms[0]) / sr
        reference = parse_main_timeline_csv(args.reference).ranges if args.reference else None
        if reference is not None and (not reference or any(
            not (math.isfinite(s) and math.isfinite(e) and 0 <= s < e <= duration_sec) for s, e in reference
        )):
            raise ValueError("Reference must contain target speech intervals within the aligned audio duration.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    import torch
    from pyannote.audio import Pipeline

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available in this PyTorch installation")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(device)
    all_clusters = {}
    elapsed = {}
    for mic, waveform in zip(("piezo", "phone"), waveforms):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        result = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr, "uri": mic})
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed[mic] = perf_counter() - start
        groups = {}
        for segment, label in PyannoteDiarizationSegmenter._iter_pyannote_tracks(result):
            groups.setdefault(str(label), []).append((float(segment.start), float(segment.end)))
        all_clusters[mic] = {label: clipped for label, spans in groups.items()
                             if (clipped := clip_ranges(spans, 0, duration_sec))}

    longest_labels = {mic: longest(groups) for mic, groups in all_clusters.items()}
    anchor = all_clusters["piezo"].get(longest_labels["piezo"], [])
    matched, agreement = match_timeline(anchor, all_clusters["phone"])
    selections = [("piezo", "longest", longest_labels["piezo"]),
                  ("phone", "longest", longest_labels["phone"]), ("phone", "paired_match", matched)]
    rows = []
    for mic, rule, label in selections:
        spans = all_clusters[mic].get(label, [])
        row = dict(microphone=mic, selection=rule, cluster=label, speech_sec=total_duration(spans),
                   precision=None, recall=None, f1=None, missed_sec=None, extra_sec=None)
        if reference is not None and (rule != "paired_match" or label is not None):
            scores = score_main_timeline(reference, spans)
            row.update(precision=scores["precision"], recall=scores["recall"], f1=scores["f1"] or 0.0,
                       missed_sec=scores["missed_main_sec"], extra_sec=scores["false_alarm_main_sec"])
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "comparison.json").write_text(json.dumps(dict(
        model=MODEL_ID, device=str(device), duration_sec=duration_sec, sample_rate=sr,
        inference_sec=elapsed, realtime_factor={mic: t / duration_sec for mic, t in elapsed.items()},
        paired_timeline_dice=agreement, reference_supplied=reference is not None,
        clusters=all_clusters, selections=rows,
    ), indent=2), encoding="utf-8")
    print(f"Wrote {args.out / 'comparison.csv'} and {args.out / 'comparison.json'}")


if __name__ == "__main__":
    main()
