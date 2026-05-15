from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from ams.config.schema import load_resolved_config
from ams.stages.stage_a_audio_prep import run_stage_a
from ams.stages.stage_b_speech_detect import run_stage_b
from ams.stages.stage_c_enroll_main import run_stage_c
from ams.stages.stage_d_classify import run_stage_d
from ams.stages.stage_e_finalize import run_stage_e

app = typer.Typer(add_completion=False, help="Audio MAIN Speaker (AMS) pipeline.")


def _cfg(preset: Optional[str], config: Optional[Path], set_kv: Optional[List[str]]):
    return load_resolved_config(preset=preset, config_path=config, set_overrides=set_kv or [])


@app.command("print-config")
def print_config(
    preset: Optional[str] = typer.Option(None, "--preset"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, dir_okay=False),
    set_kv: List[str] = typer.Option(None, "--set"),
):
    typer.echo(json.dumps(_cfg(preset, config, set_kv).to_dict(), indent=2, sort_keys=True))


@app.command("stage-a")
def stage_a(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False), out: Path = typer.Option(Path("out"), "--out"), preset: Optional[str] = None, config: Optional[Path] = None, set_kv: List[str] = typer.Option(None, "--set")):
    run_stage_a(input_path=input, out_root=out, cfg=_cfg(preset, config, set_kv))


@app.command("stage-b")
def stage_b(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False), out: Path = typer.Option(Path("out"), "--out"), preset: Optional[str] = None, config: Optional[Path] = None, set_kv: List[str] = typer.Option(None, "--set")):
    run_stage_b(input_path=input, out_root=out, cfg=_cfg(preset, config, set_kv))


@app.command("stage-c")
def stage_c(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False), out: Path = typer.Option(Path("out"), "--out"), preset: Optional[str] = None, config: Optional[Path] = None, set_kv: List[str] = typer.Option(None, "--set")):
    run_stage_c(input_path=input, out_root=out, cfg=_cfg(preset, config, set_kv))


@app.command("stage-d")
def stage_d(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False), out: Path = typer.Option(Path("out"), "--out"), preset: Optional[str] = None, config: Optional[Path] = None, set_kv: List[str] = typer.Option(None, "--set")):
    run_stage_d(input_path=input, out_root=out, cfg=_cfg(preset, config, set_kv))


@app.command("stage-e")
def stage_e(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False), out: Path = typer.Option(Path("out"), "--out"), final_csv: Optional[Path] = typer.Option(None, "--final-csv"), preset: Optional[str] = None, config: Optional[Path] = None, set_kv: List[str] = typer.Option(None, "--set")):
    run_stage_e(input_path=input, out_root=out, cfg=_cfg(preset, config, set_kv), final_csv_path=final_csv)


@app.command("all")
def all_stages(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False), out: Path = typer.Option(Path("out"), "--out"), preset: Optional[str] = None, config: Optional[Path] = None, set_kv: List[str] = typer.Option(None, "--set")):
    cfg = _cfg(preset, config, set_kv)
    run_stage_a(input_path=input, out_root=out, cfg=cfg)
    run_stage_b(input_path=input, out_root=out, cfg=cfg)
    run_stage_c(input_path=input, out_root=out, cfg=cfg)
    run_stage_d(input_path=input, out_root=out, cfg=cfg)
    run_stage_e(input_path=input, out_root=out, cfg=cfg, final_csv_path=None)
