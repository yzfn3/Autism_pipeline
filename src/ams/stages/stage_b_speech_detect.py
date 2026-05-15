from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ams.config.schema import Config
from ams.dsp.metrics import frame_rms_db
from ams.dsp.vad_postprocess import ema_smooth, postprocess_vad
from ams.io.hashing import sha256_file
from ams.io.serialization import read_json, write_csv_rows, write_json
from ams.stages.stage_a_audio_prep import run_stage_a
from ams.stages.stage_utils import artifact_exists, make_header, params_hash, stage_artifacts_dir, write_artifact_json


def _energy_to_prob(rms_db: np.ndarray, floor: float) -> np.ndarray:
    x = (rms_db - (floor + 10.0)) / 3.0
    p = 1.0 / (1.0 + np.exp(-x))
    return np.where(rms_db < floor, p * 0.15, p).astype(np.float32)


def run_stage_b(*, input_path: Path, out_root: Path, cfg: Config) -> Path:
    input_hash = sha256_file(input_path)
    a_path = run_stage_a(input_path=input_path, out_root=out_root, cfg=cfg)
    a_art = read_json(a_path)
    norm_wav = Path(a_art["normalized_audio_path"])
    duration_sec = float(a_art["duration_sec"])
    stage_params = {"b": cfg.b.__dict__, "vad": "energy_only"}
    ph = params_hash(stage_params)
    art_dir = stage_artifacts_dir(out_root, cfg.runtime.cache_dirname, input_hash, "B", ph)
    artifact_path = art_dir / "speech_detect.json"
    seg_json = art_dir / "speech_segments.json"
    probs_csv = art_dir / "speech_probs.csv"
    if artifact_exists(artifact_path) and seg_json.exists() and probs_csv.exists():
        return artifact_path
    y, sr = sf.read(str(norm_wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame_len = max(1, int(round(cfg.b.frame_ms * 0.001 * sr)))
    hop = max(1, int(round(cfg.b.hop_ms * 0.001 * sr)))
    rms = frame_rms_db(y, frame_len, hop)
    p = _energy_to_prob(rms, cfg.b.rms_floor_db)
    alpha = float(1.0 - np.exp(-(cfg.b.hop_ms * 0.001) / max(1e-3, cfg.b.smooth_ms * 0.001)))
    p_sm = ema_smooth(p, alpha)
    t = (np.arange(len(p_sm), dtype=np.float32) * hop + 0.5 * frame_len) / float(sr)
    segments, stats = postprocess_vad(t, p_sm, cfg.b.on_thresh, cfg.b.off_thresh, cfg.b.min_speech_ms * 0.001, cfg.b.merge_gap_ms * 0.001, cfg.b.pad_ms * 0.001, 0.0, duration_sec)
    write_csv_rows(probs_csv, ["t_center_sec", "p_speech"], zip(t.tolist(), p_sm.tolist()))
    write_json(seg_json, {"segments": [s.to_dict() for s in segments], "source": "energy_only"})
    art = {"header": make_header("B", input_hash, stage_params, ph, {"A.audio_prep.json": sha256_file(a_path)}), "segments": [s.to_dict() for s in segments], "postprocess_stats": stats}
    write_artifact_json(artifact_path, art)
    return artifact_path
