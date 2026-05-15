from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ams.config.schema import Config
from ams.dsp.embedding import cosine_similarity
from ams.dsp.metrics import rms_db
from ams.dsp.segmentation import sliding_windows
from ams.io.hashing import sha256_file
from ams.io.serialization import read_json, read_vector, write_csv_rows, write_json
from ams.ml.speaker_embedding import SpeakerEmbedder, choose_device
from ams.stages.stage_a_audio_prep import run_stage_a
from ams.stages.stage_b_speech_detect import run_stage_b
from ams.stages.stage_c_enroll_main import run_stage_c
from ams.stages.stage_utils import artifact_exists, make_header, params_hash, speaker_embedding_runtime_params, stage_artifacts_dir, upstream_c_params_hash_from_enrollment, write_artifact_json


def _aggregate(scores: list[float], agg: str) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if agg == "mean":
        return float(arr.mean())
    if agg == "p90":
        return float(np.percentile(arr, 90))
    return float(np.median(arr))


def run_stage_d(*, input_path: Path, out_root: Path, cfg: Config, b_artifact_path: Path | None = None) -> Path:
    input_hash = sha256_file(input_path)
    a_path = run_stage_a(input_path=input_path, out_root=out_root, cfg=cfg)
    b_path = b_artifact_path or run_stage_b(input_path=input_path, out_root=out_root, cfg=cfg)
    c_path = run_stage_c(input_path=input_path, out_root=out_root, cfg=cfg, b_artifact_path=b_path)
    a_art = read_json(a_path)
    c_art = read_json(c_path)
    stage_params = {"d": cfg.d.__dict__, "upstream_c_params_hash": upstream_c_params_hash_from_enrollment(c_art), "runtime": speaker_embedding_runtime_params(cfg.runtime)}
    ph = params_hash(stage_params)
    art_dir = stage_artifacts_dir(out_root, cfg.runtime.cache_dirname, input_hash, "D", ph)
    artifact_path = art_dir / "classification.json"
    win_json = art_dir / "window_classification.json"
    if artifact_exists(artifact_path) and win_json.exists():
        return artifact_path
    y, sr = sf.read(str(a_art["normalized_audio_path"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    centroid = np.asarray(read_vector(Path(c_art["centroid_path"])), dtype=np.float32)
    embedder = SpeakerEmbedder(cfg.runtime.speaker_embedding_backend, choose_device(cfg.runtime.prefer_gpu), len(centroid) or cfg.c.embed_dim)
    segments = read_json(Path(b_path).with_name("speech_segments.json")).get("segments") or []
    per_segment = []
    per_window = []
    rows = []
    for seg in segments:
        scores = []
        for w in sliding_windows(float(seg["start_sec"]), float(seg["end_sec"]), cfg.d.win_dur_sec, cfg.d.hop_sec):
            chunk = y[int(w.start_sec * sr):int(w.end_sec * sr)]
            if chunk.size == 0 or rms_db(chunk) < cfg.c.rms_floor_db:
                continue
            score = cosine_similarity(embedder.embed(chunk, sr), centroid)
            scores.append(score)
            dec = "MAIN" if score >= cfg.d.main_thresh else "OTHER"
            per_window.append({"segment_id": seg["id"], "start_sec": w.start_sec, "end_sec": w.end_sec, "score": score, "decision": dec})
        score = _aggregate(scores, cfg.d.agg)
        frac = float(np.mean(np.asarray(scores) >= cfg.d.main_thresh)) if scores else 0.0
        decision = "MAIN" if len(scores) >= cfg.d.min_windows_used and score >= cfg.d.main_thresh and frac >= cfg.d.require_frac_above_thresh else "OTHER"
        rec = {**seg, "score": score, "decision": decision, "n_windows_used": len(scores)}
        per_segment.append(rec)
        rows.append((seg["id"], f"{float(seg['start_sec']):.3f}", f"{float(seg['end_sec']):.3f}", f"{score:.6f}", decision, len(scores)))
    write_csv_rows(art_dir / "segment_scores.csv", ["segment_id", "start_sec", "end_sec", "score", "decision", "n_windows_used"], rows)
    write_json(win_json, {"per_window": per_window})
    write_json(art_dir / "segment_classification.json", {"per_segment": per_segment})
    summary = {"num_segments": len(per_segment), "num_windows": len(per_window), "main": sum(1 for s in per_segment if s["decision"] == "MAIN")}
    art = {"header": make_header("D", input_hash, stage_params, ph, {"C.enrollment.json": sha256_file(c_path)}), "per_segment": per_segment, "summary": summary}
    write_artifact_json(artifact_path, art)
    return artifact_path
