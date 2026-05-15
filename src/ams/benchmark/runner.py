from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from tqdm.auto import tqdm

from ams.benchmark.audio import audio_info, load_mono
from ams.benchmark.embeddings import REQUESTED_EMBEDDINGS, EmbeddingModel, embedding_factories
from ams.benchmark.metrics import score_main_timeline
from ams.benchmark.segmentation import REQUESTED_SEGMENTERS, SegmenterResult, segmenter_factories
from ams.benchmark.timeline import Range, clip_ranges, merge_ranges, overlap_duration, parse_reference_timeline, total_duration, write_ranges_csv
from ams.dsp.embedding import cosine_similarity
from ams.dsp.segmentation import sliding_windows


@dataclass(frozen=True)
class BenchmarkConfig:
    wav_path: Path
    reference_csv: Path
    out_dir: Path
    segmenters: List[str]
    embeddings: List[str]
    collar_sec: float = 0.0
    enrollment_sec: float = 12.0
    threshold: float = 0.62
    window_sec: float = 1.5
    hop_sec: float = 0.75
    enrollment_strategy: str = "first"
    run_unavailable: bool = True
    native_diarization_baselines: bool = True
    target_speaker: str | None = None


def _slice_audio(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return y[max(0, int(start * sr)):max(0, int(end * sr))]


def _embedding_clip(y: np.ndarray, sr: int, start: float, end: float, min_sec: float = 1.0) -> np.ndarray:
    clip = _slice_audio(y, sr, start, end)
    min_samples = max(1, int(float(min_sec) * sr))
    if clip.size == 0:
        return clip
    if clip.size >= min_samples:
        return clip
    pad = min_samples - clip.size
    mode = "reflect" if clip.size > 1 else "edge"
    return np.pad(clip, (0, pad), mode=mode).astype(np.float32)


def _take_from_ranges(ranges: List[Range], enrollment_sec: float) -> List[Range]:
    out: List[Range] = []
    need = float(enrollment_sec)
    for s, e in ranges:
        if need <= 0:
            break
        dur = min(e - s, need)
        if dur > 0:
            out.append((s, s + dur))
            need -= dur
    return out


def _enrollment_ranges(ref_ranges: List[Range], enrollment_sec: float, strategy: str = "first") -> List[Range]:
    ranges = merge_ranges(ref_ranges)
    strategy = (strategy or "first").lower()
    if strategy == "first":
        return _take_from_ranges(ranges, enrollment_sec)
    if strategy == "longest":
        ordered = sorted(ranges, key=lambda r: (-(r[1] - r[0]), r[0]))
        return _take_from_ranges(ordered, enrollment_sec)
    if strategy == "spread":
        if not ranges:
            return []
        usable = [(s, e) for s, e in ranges if e - s > 0]
        total = total_duration(usable)
        if total <= 0:
            return []
        target = min(float(enrollment_sec), total)
        chunks = max(1, min(len(usable), int(np.ceil(target / 3.0))))
        per_chunk = target / chunks
        selected: List[Range] = []
        if chunks == 1:
            s, e = max(usable, key=lambda r: r[1] - r[0])
            mid = (s + e) / 2.0
            half = min(per_chunk, e - s) / 2.0
            return [(max(s, mid - half), min(e, mid + half))]
        for i in range(chunks):
            q = i / max(chunks - 1, 1)
            idx = int(round(q * (len(usable) - 1)))
            s, e = usable[idx]
            dur = min(per_chunk, e - s)
            if dur <= 0:
                continue
            center = s + (e - s) * 0.5
            selected.append((max(s, center - dur / 2.0), min(e, center + dur / 2.0)))
        return merge_ranges(selected)
    raise ValueError(f"Unknown enrollment strategy {strategy!r}; expected first, longest, or spread")


def _centroid(embedder: EmbeddingModel, y: np.ndarray, sr: int, ranges: List[Range]) -> np.ndarray:
    clips = [_embedding_clip(y, sr, s, e) for s, e in ranges]
    clips = [clip for clip in clips if clip.size]
    if clips:
        batch_size = _embedding_batch_size()
        with tqdm(total=len(clips), desc=f"enroll:{embedder.name}", leave=False) as pbar:
            vectors = []
            for i in range(0, len(clips), batch_size):
                batch = clips[i : i + batch_size]
                vectors.extend(result.vector for result in embedder.embed_many(batch, sr))
                pbar.update(len(batch))
    else:
        vectors = []
    if not vectors:
        raise RuntimeError("No enrollment audio was available")
    c = np.stack(vectors, axis=0).mean(axis=0)
    return c / max(float(np.linalg.norm(c)), 1e-12)


def _embedding_batch_size() -> int:
    raw = os.environ.get("AMS_EMBED_BATCH_SIZE", "16")
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def classify_main_ranges(
    *,
    speech_ranges: List[Range],
    embedder: EmbeddingModel,
    y: np.ndarray,
    sr: int,
    centroid: np.ndarray,
    threshold: float,
    window_sec: float,
    hop_sec: float,
) -> List[Range]:
    main_windows: List[Range] = []
    windows: List[Range] = []
    for s, e in speech_ranges:
        windows.extend((w.start_sec, w.end_sec) for w in sliding_windows(s, e, window_sec, hop_sec))
    total_windows = len(windows)
    batch_size = _embedding_batch_size()
    with tqdm(total=total_windows, desc=f"classify:{embedder.name}", leave=False) as pbar:
        for i in range(0, total_windows, batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_clips = [_embedding_clip(y, sr, s, e) for s, e in batch_windows]
            usable = [(window, clip) for window, clip in zip(batch_windows, batch_clips) if clip.size]
            if usable:
                vectors = embedder.embed_many([clip for _, clip in usable], sr)
                for (start, end), result in zip((window for window, _ in usable), vectors):
                    score = cosine_similarity(result.vector, centroid)
                    if score >= threshold:
                        main_windows.append((start, end))
            pbar.update(len(batch_windows))
    return merge_ranges(main_windows, gap=hop_sec)


def _native_diarization_main(turns: List[dict], enrollment_ranges: List[Range]) -> List[Range]:
    if not turns:
        return []
    by_spk: Dict[str, float] = {}
    for t in turns:
        spk = str(t["speaker"])
        tr = [(float(t["start_sec"]), float(t["end_sec"]))]
        by_spk[spk] = by_spk.get(spk, 0.0) + overlap_duration(tr, enrollment_ranges)
    if not by_spk or max(by_spk.values()) <= 0:
        first = sorted(turns, key=lambda x: (float(x["start_sec"]), float(x["end_sec"])))[0]
        main_spk = str(first["speaker"])
    else:
        main_spk = sorted(by_spk, key=lambda spk: (-by_spk[spk], spk))[0]
    return merge_ranges((float(t["start_sec"]), float(t["end_sec"])) for t in turns if str(t["speaker"]) == main_spk)


def _row(status: str, segmenter: str, embedding: str, metrics: Optional[dict] = None, **extra) -> dict:
    base = {
        "status": status,
        "segmenter": segmenter,
        "embedding": embedding,
        "error": extra.pop("error", ""),
        "wall_time_sec": extra.pop("wall_time_sec", None),
        "note": extra.pop("note", ""),
    }
    if metrics:
        base.update(metrics)
    base.update(extra)
    return base


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ref = parse_reference_timeline(cfg.reference_csv, target_speaker=cfg.target_speaker)
    info = audio_info(cfg.wav_path)
    y, sr = load_mono(cfg.wav_path, 16000)
    ref_ranges = clip_ranges(ref.ranges, 0.0, info["duration_sec"])
    enroll_ranges = _enrollment_ranges(ref_ranges, cfg.enrollment_sec, cfg.enrollment_strategy)
    if not enroll_ranges:
        raise RuntimeError("Reference CSV did not provide any usable enrollment ranges")

    seg_factories = segmenter_factories()
    emb_factories = embedding_factories(prefer_gpu=True)
    rows: List[dict] = []
    segmenter_cache: Dict[str, SegmenterResult] = {}
    embedder_cache: Dict[str, EmbeddingModel] = {}

    write_ranges_csv(cfg.out_dir / "reference_main_timeline.csv", ref_ranges)

    for seg_name in tqdm(cfg.segmenters, desc="segmenters"):
        t0 = time.perf_counter()
        try:
            segmenter = seg_factories[seg_name]()
            tqdm.write(f"[segmenter] {seg_name}")
            seg_result = segmenter.run(cfg.wav_path)
            segmenter_cache[seg_name] = seg_result
            seg_wall = time.perf_counter() - t0
            write_ranges_csv(cfg.out_dir / f"{seg_name}.speech.csv", seg_result.speech_ranges)
        except Exception as exc:
            seg_wall = time.perf_counter() - t0
            message = f"{type(exc).__name__}: {exc}"
            for emb_name in cfg.embeddings:
                rows.append(_row("not_available", seg_name, emb_name, error=message, wall_time_sec=round(seg_wall, 3)))
            if cfg.native_diarization_baselines:
                rows.append(_row("not_available", seg_name, "native_diarization", error=message, wall_time_sec=round(seg_wall, 3)))
            continue

        if cfg.native_diarization_baselines and seg_result.speaker_turns:
            hyp = _native_diarization_main(seg_result.speaker_turns, enroll_ranges)
            metrics = score_main_timeline(ref_ranges, hyp, audio_duration_sec=info["duration_sec"], collar_sec=cfg.collar_sec)
            write_ranges_csv(cfg.out_dir / f"{seg_name}.native_main.csv", hyp)
            rows.append(_row("ok", seg_name, "native_diarization", metrics, wall_time_sec=round(seg_wall, 3), mode=seg_result.mode, note=seg_result.note))

        for emb_name in tqdm(cfg.embeddings, desc=f"embeddings:{seg_name}", leave=False):
            t1 = time.perf_counter()
            try:
                embedder = embedder_cache.get(emb_name)
                if embedder is None:
                    tqdm.write(f"[embedding:init] {emb_name}")
                    embedder = emb_factories[emb_name]()
                    embedder_cache[emb_name] = embedder
                tqdm.write(f"[pipeline] {seg_name} + {emb_name}")
                c = _centroid(embedder, y, sr, enroll_ranges)
                hyp = classify_main_ranges(
                    speech_ranges=seg_result.speech_ranges,
                    embedder=embedder,
                    y=y,
                    sr=sr,
                    centroid=c,
                    threshold=cfg.threshold,
                    window_sec=cfg.window_sec,
                    hop_sec=cfg.hop_sec,
                )
                metrics = score_main_timeline(ref_ranges, hyp, audio_duration_sec=info["duration_sec"], collar_sec=cfg.collar_sec)
                path = cfg.out_dir / f"{seg_name}__{emb_name}.main.csv"
                write_ranges_csv(path, hyp)
                rows.append(_row("ok", seg_name, emb_name, metrics, wall_time_sec=round(seg_wall + time.perf_counter() - t1, 3), mode=seg_result.mode, output_csv=str(path), note=seg_result.note))
            except Exception as exc:
                rows.append(_row("not_available", seg_name, emb_name, error=f"{type(exc).__name__}: {exc}", wall_time_sec=round(seg_wall + time.perf_counter() - t1, 3), mode=seg_result.mode))

    rows_sorted = sorted(rows, key=lambda r: (r.get("status") != "ok", r.get("der") if r.get("der") is not None else 999.0, -(r.get("f1") or 0.0), r["segmenter"], r["embedding"]))
    csv_path = cfg.out_dir / "benchmark_results.csv"
    fieldnames = sorted({k for row in rows_sorted for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    report = {
        "wav": str(cfg.wav_path),
        "reference_csv": str(cfg.reference_csv),
        "audio": info,
        "reference": {"source": ref.source, "fps": ref.fps, "target_speaker": ref.target_speaker, "num_ranges": len(ref_ranges), "total_main_sec": total_duration(ref_ranges)},
        "config": cfg.__dict__ | {"wav_path": str(cfg.wav_path), "reference_csv": str(cfg.reference_csv), "out_dir": str(cfg.out_dir)},
        "results_csv": str(csv_path),
        "rows": rows_sorted,
    }
    (cfg.out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def default_segmenters(include_internal: bool = True) -> List[str]:
    out = list(REQUESTED_SEGMENTERS)
    if include_internal:
        out.append("energy_vad_internal")
    return out


def default_embeddings(include_internal: bool = True) -> List[str]:
    out = list(REQUESTED_EMBEDDINGS)
    if include_internal:
        out.append("band_energy_internal")
    return out
