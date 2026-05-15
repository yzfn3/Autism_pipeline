from __future__ import annotations

from pathlib import Path

from ams.config.schema import Config
from ams.io.audio_decode import decode_audio, to_canonical, write_wav
from ams.io.hashing import sha256_file
from ams.stages.stage_utils import artifact_exists, make_header, params_hash, stage_artifacts_dir, write_artifact_json


def run_stage_a(*, input_path: Path, out_root: Path, cfg: Config) -> Path:
    input_hash = sha256_file(input_path)
    stage_params = {"a": cfg.a.__dict__, "runtime": {"use_ffmpeg": cfg.runtime.use_ffmpeg}}
    ph = params_hash(stage_params)
    art_dir = stage_artifacts_dir(out_root, cfg.runtime.cache_dirname, input_hash, "A", ph)
    artifact_path = art_dir / "audio_prep.json"
    normalized_path = art_dir / "normalized.wav"
    if artifact_exists(artifact_path) and normalized_path.exists():
        return artifact_path
    decoded = decode_audio(input_path, use_ffmpeg=cfg.runtime.use_ffmpeg)
    y, sr, channels, stats = to_canonical(decoded, cfg.a.canonical_sr_hz, cfg.a.channel_mode, cfg.a.normalize_peak_dbfs)
    write_wav(normalized_path, y, sr)
    duration_sec = float(len(y) / sr) if y.ndim == 1 else float(y.shape[0] / sr)
    art = {
        "header": make_header("A", input_hash, stage_params, ph),
        "normalized_audio_path": str(normalized_path),
        "canonical_sr_hz": int(sr),
        "canonical_channels": int(channels),
        "duration_sec": duration_sec,
        "input_path": str(input_path),
        "input_hash": input_hash,
        "normalized_audio_hash": sha256_file(normalized_path),
        "decode_warnings": decoded.warnings,
        "stats": stats,
    }
    write_artifact_json(artifact_path, art)
    return artifact_path
