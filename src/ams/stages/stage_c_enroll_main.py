from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ams.config.schema import Config
from ams.dsp.metrics import rms_db
from ams.dsp.segmentation import sliding_windows
from ams.io.hashing import sha256_file
from ams.io.serialization import read_json, write_json, write_vector
from ams.ml.speaker_embedding import SpeakerEmbedder, choose_device
from ams.stages.stage_a_audio_prep import run_stage_a
from ams.stages.stage_b_speech_detect import run_stage_b
from ams.stages.stage_utils import artifact_exists, make_header, params_hash, speaker_embedding_runtime_params, stage_artifacts_dir, write_artifact_json


def run_stage_c(*, input_path: Path, out_root: Path, cfg: Config, b_artifact_path: Path | None = None) -> Path:
    input_hash = sha256_file(input_path)
    a_path = run_stage_a(input_path=input_path, out_root=out_root, cfg=cfg)
    b_path = b_artifact_path or run_stage_b(input_path=input_path, out_root=out_root, cfg=cfg)
    a_art = read_json(a_path)
    b_seg = read_json(Path(b_path).with_name("speech_segments.json"))
    stage_params = {"c": cfg.c.__dict__, "runtime": speaker_embedding_runtime_params(cfg.runtime)}
    ph = params_hash(stage_params)
    art_dir = stage_artifacts_dir(out_root, cfg.runtime.cache_dirname, input_hash, "C", ph)
    artifact_path = art_dir / "enrollment.json"
    centroid_path = art_dir / "main_centroid.vec"
    quality_path = art_dir / "enroll_quality.json"
    if artifact_exists(artifact_path) and centroid_path.exists():
        return artifact_path
    y, sr = sf.read(str(a_art["normalized_audio_path"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    embedder = SpeakerEmbedder(cfg.runtime.speaker_embedding_backend, choose_device(cfg.runtime.prefer_gpu), cfg.c.embed_dim)
    wavs = []
    srs = []
    for seg in b_seg.get("segments") or []:
        s = float(seg["start_sec"])
        e = min(float(seg["end_sec"]), float(cfg.c.enrollment_window_sec))
        if s >= cfg.c.enrollment_window_sec:
            break
        for w in sliding_windows(s, e, cfg.c.win_dur_sec, cfg.c.hop_sec):
            i0, i1 = int(w.start_sec * sr), int(w.end_sec * sr)
            chunk = y[i0:i1]
            if chunk.size and rms_db(chunk) >= cfg.c.rms_floor_db:
                wavs.append(chunk)
                srs.append(sr)
    if wavs:
        emb = embedder.embed_batch(wavs, srs)
        centroid = emb.mean(axis=0).astype(np.float32)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    else:
        centroid = np.zeros((int(cfg.c.embed_dim),), dtype=np.float32)
    write_vector(centroid_path, centroid.tolist())
    quality = {"status": "ok" if wavs else "no_enrollment_windows", "num_windows_used": len(wavs), "high_confidence": len(wavs) >= cfg.c.min_windows}
    write_json(quality_path, quality)
    art = {
        "header": make_header("C", input_hash, stage_params, ph, {"A.audio_prep.json": sha256_file(a_path), "B.speech_detect.json": sha256_file(Path(b_path))}),
        "centroid_path": str(centroid_path),
        "quality": {"quality_path": str(quality_path), **quality},
        "embed_dim": int(len(centroid)),
    }
    write_artifact_json(artifact_path, art)
    return artifact_path
