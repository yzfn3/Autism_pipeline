from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ams.config.schema import Config
from ams.io.hashing import sha256_file
from ams.io.serialization import read_json, write_csv_rows, write_json
from ams.stages.stage_d_classify import run_stage_d
from ams.stages.stage_utils import artifact_exists, make_header, params_hash, stage_artifacts_dir, write_artifact_json


def _merge_ranges(ranges: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    rs = sorted((float(s), float(e)) for s, e in ranges if e > s)
    if not rs:
        return []
    out = [rs[0]]
    for s, e in rs[1:]:
        ps, pe = out[-1]
        if s - pe <= gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _drop_short(ranges: List[Tuple[float, float]], min_dur: float):
    kept = [(s, e) for s, e in ranges if e - s >= min_dur]
    return kept, len(ranges) - len(kept)


def run_stage_e(*, input_path: Path, out_root: Path, cfg: Config, final_csv_path: Path | None, b_artifact_path: Path | None = None) -> Path:
    input_hash = sha256_file(input_path)
    d_path = run_stage_d(input_path=input_path, out_root=out_root, cfg=cfg, b_artifact_path=b_artifact_path)
    stage_params = {"e": cfg.e.__dict__, "upstream_d_hash": sha256_file(d_path)}
    ph = params_hash(stage_params)
    art_dir = stage_artifacts_dir(out_root, cfg.runtime.cache_dirname, input_hash, "E", ph)
    artifact_path = art_dir / "finalize.json"
    final_csv = final_csv_path or (out_root / "final_main_timeline.csv")
    if artifact_exists(artifact_path) and final_csv.exists():
        return artifact_path
    wins = read_json(Path(d_path).with_name("window_classification.json")).get("per_window") or []
    main_ranges = [(float(w["start_sec"]), float(w["end_sec"])) for w in wins if w.get("decision") == "MAIN"]
    merged = _merge_ranges(main_ranges, cfg.e.merge_gap_sec)
    cleaned, dropped = _drop_short(merged, cfg.e.min_main_dur_sec)
    write_csv_rows(final_csv, ["start_sec", "end_sec"], ((f"{s:.3f}", f"{e:.3f}") for s, e in cleaned))
    write_json(art_dir / "final_clean.json", {"main_ranges": [{"start_sec": s, "end_sec": e} for s, e in cleaned]})
    summary = {"num_main_ranges_final": len(cleaned), "dropped_short": dropped, "total_main_sec": sum(e - s for s, e in cleaned), "final_csv": str(final_csv)}
    art = {"header": make_header("E", input_hash, stage_params, ph, {"D.classification.json": sha256_file(d_path)}), "main_ranges": [{"start_sec": s, "end_sec": e} for s, e in cleaned], "summary": summary}
    write_artifact_json(artifact_path, art)
    return artifact_path
