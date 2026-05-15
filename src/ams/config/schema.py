from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ams.io.serialization import stable_json_dumps


@dataclass(frozen=True)
class StageAConfig:
    canonical_sr_hz: int = 16000
    channel_mode: str = "mono"
    normalize_peak_dbfs: float = -1.0


@dataclass(frozen=True)
class StageBConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    smooth_ms: float = 200.0
    on_thresh: float = 0.60
    off_thresh: float = 0.40
    rms_floor_db: float = -55.0
    min_speech_ms: float = 300.0
    merge_gap_ms: float = 250.0
    pad_ms: float = 100.0


@dataclass(frozen=True)
class StageCConfig:
    enrollment_window_sec: float = 30.0
    min_enroll_dur_sec: float = 2.0
    win_dur_sec: float = 1.0
    hop_sec: float = 0.5
    min_windows: int = 3
    rms_floor_db: float = -50.0
    embed_dim: int = 24
    self_consistency_filter: bool = True
    self_consistency_min_cos: float = 0.65
    enrollment_policy: str = "first_30s_anchor"
    enroll_scan_max_sec: float = 180.0
    speaker_change_min_consecutive: int = 2
    anchor_windows: int = 5
    enroll_sim_thresh: float = 0.72
    min_accepted_sec: float = 8.0
    min_accepted_frac: float = 0.80


@dataclass(frozen=True)
class StageDConfig:
    win_dur_sec: float = 1.0
    hop_sec: float = 0.5
    agg: str = "median"
    main_thresh: float = 0.65
    main_thresh_floor: float = 0.25
    uncertain_band: float = 0.05
    min_windows_used: int = 2
    require_frac_above_thresh: float = 0.7
    adaptive_thresh: bool = True
    adaptive_method: str = "otsu"
    adaptive_percentile: float = 88.0
    low_confidence_delta_thresh: float = 0.08
    low_confidence_delta_require_frac: float = 0.10


@dataclass(frozen=True)
class StageEConfig:
    merge_gap_sec: float = 0.4
    min_main_dur_sec: float = 0.5


@dataclass(frozen=True)
class RuntimeConfig:
    cache_dirname: str = "cache"
    artifacts_dirname: str = "artifacts"
    use_ffmpeg: bool = True
    prefer_gpu: bool = True
    speaker_embedding_backend: str = "simple_band_energy"
    speechbrain_model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    wavlm_model_id: str = "microsoft/wavlm-base-plus-sv"
    titanet_model_id: str = "titanet_large"
    ecapa2_model_id: str = "Jenthe/ECAPA2"
    wespeaker_campp_model_id: str = "Wespeaker/wespeaker-voxceleb-campplus"


@dataclass(frozen=True)
class Config:
    a: StageAConfig = field(default_factory=StageAConfig)
    b: StageBConfig = field(default_factory=StageBConfig)
    c: StageCConfig = field(default_factory=StageCConfig)
    d: StageDConfig = field(default_factory=StageDConfig)
    e: StageEConfig = field(default_factory=StageEConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def params_hashable(self) -> str:
        return stable_json_dumps(self.to_dict())


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _parse_set_overrides(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set override (expected key=value): {item}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except Exception:
            value = raw
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


def _dict_to_config(d: Dict[str, Any]) -> Config:
    return Config(
        a=StageAConfig(**(d.get("a") or {})),
        b=StageBConfig(**(d.get("b") or {})),
        c=StageCConfig(**(d.get("c") or {})),
        d=StageDConfig(**(d.get("d") or {})),
        e=StageEConfig(**(d.get("e") or {})),
        runtime=RuntimeConfig(**(d.get("runtime") or {})),
    )


def load_resolved_config(preset: Optional[str], config_path: Optional[Path], set_overrides: Optional[List[str]]) -> Config:
    merged = Config().to_dict()
    if preset:
        with resources.files("ams.config.presets").joinpath(f"{preset}.yaml").open("r", encoding="utf-8") as f:
            merged = _deep_update(merged, yaml.safe_load(f) or {})
    if config_path:
        merged = _deep_update(merged, _load_yaml(config_path))
    merged = _deep_update(merged, _parse_set_overrides(set_overrides or []))
    return _dict_to_config(merged)
