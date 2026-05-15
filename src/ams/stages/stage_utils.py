from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ams.config.schema import RuntimeConfig
from ams.io.hashing import sha256_bytes, sha256_file
from ams.io.serialization import stable_json_dumps, write_json


def speaker_embedding_runtime_params(runtime: RuntimeConfig) -> Dict[str, Any]:
    return {
        "speaker_embedding_backend": runtime.speaker_embedding_backend,
        "prefer_gpu": runtime.prefer_gpu,
        "speechbrain_model_id": runtime.speechbrain_model_id,
        "wavlm_model_id": runtime.wavlm_model_id,
        "titanet_model_id": runtime.titanet_model_id,
        "ecapa2_model_id": runtime.ecapa2_model_id,
        "wespeaker_campp_model_id": runtime.wespeaker_campp_model_id,
    }


def params_hash(stage_params: Dict[str, Any]) -> str:
    return sha256_bytes(stable_json_dumps(stage_params).encode("utf-8"))


def artifact_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def stage_artifacts_dir(out_root: Path, cache_dirname: str, audio_hash: str, stage_name: str, params_hash_value: str) -> Path:
    return out_root / cache_dirname / audio_hash / stage_name / params_hash_value / "artifacts"


def write_artifact_json(path: Path, obj: Dict[str, Any]) -> str:
    write_json(path, obj)
    return sha256_file(path)


def make_header(stage: str, audio_hash: str, params: Dict[str, Any], params_hash_value: str, upstream: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    return {
        "contract_version": "0.2",
        "stage": stage,
        "audio_content_hash": audio_hash,
        "params_hash": params_hash_value,
        "params": params,
        "upstream_artifact_hashes": upstream or {},
    }


def upstream_c_params_hash_from_enrollment(c_art: Dict[str, Any]) -> str:
    h = (c_art or {}).get("header") or {}
    return str(h.get("params_hash") or "missing_c_params_hash")
