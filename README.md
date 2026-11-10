# Audio MAIN Speaker Benchmark

This project benchmarks diarization, speech segmentation, and speaker embedding
pipelines for finding the MAIN speaker timeline in conversational audio.

It was built for studies where each recording has:

- a WAV file containing the conversation audio
- a reference timeline file marking the target/main speaker

The benchmark compares end-to-end diarization pipelines and segmentation +
speaker-embedding combinations using DER-style MAIN-speaker error, F1, recall,
precision, IoU, miss/false-alarm rates, and speaking-time error. It writes both
CSV tables and PNG heatmaps so results can be audited and understood quickly.

## What Is In This Repo

```text
.
├── src/ams/                         # Installable Python package
│   ├── benchmark/                   # Benchmark adapters, metrics, timelines
│   ├── config/                      # Presets for the older staged AMS pipeline
│   ├── dsp/                         # DSP helpers
│   ├── io/                          # Audio and serialization helpers
│   ├── ml/                          # Internal embedding helpers
│   └── stages/                      # Staged MAIN-speaker pipeline
├── scripts/                         # Isolated subprocess runners
├── tests/                           # Unit tests
├── orchestrate_benchmark.py         # One-recording 39-row benchmark runner
├── multi_recording_benchmark.py     # Folder/manifest multi-recording runner
├── threshold_sweep.py               # Dev/test threshold sweep runner
├── plot_benchmark_heatmaps.py       # Heatmap generation helper
├── run_wav_rttm_eval.py             # WAV + RTTM evaluation helper
├── pyproject.toml                   # Package metadata and dependencies
└── .gitignore                       # Keeps data, outputs, caches, secrets out
```

## What Is Not In This Repo

Do not commit private data, generated outputs, model caches, or secrets.

The `.gitignore` excludes:

- audio files: `*.wav`, `*.flac`, `*.mp3`, `*.m4a`
- reference/output tables: `*.csv`, `*.rttm`, spreadsheets, parquet, jsonl
- benchmark output folders: `benchmark_*`, `results*`, `outputs*`
- generated PNG heatmaps and timeline plots
- Python caches and virtual environments
- downloaded model/checkpoint files
- `.env` and token files

If you need to share data, use a private data store, a separate private repo, or
an institutional storage location. Keep this repository code-only.

## System Requirements

Recommended:

- Windows 10/11, Linux, or macOS
- Python 3.10 to 3.12
- Git
- FFmpeg available on `PATH`
- Enough disk space for model downloads and benchmark outputs
- CPU is sufficient for the built-in smoke test and the microphone comparison.
  The benchmark's local Pyannote and NeMo diarization adapters require CUDA.

Install FFmpeg:

- Windows: install from <https://ffmpeg.org/download.html> or use `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

Check:

```powershell
ffmpeg -version
python --version
```

## Setup From Scratch

Clone the repository:

```powershell
git clone https://github.com/yzfn3/Autism_pipeline.git
cd Autism_pipeline
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

Install the core package and plotting dependencies:

```powershell
python -m pip install -e ".[viz]"
python -m unittest discover -s tests
```

For the optional model adapters, install the benchmark extra in an environment
with a compatible PyTorch build. NeMo and FunASR are separate dependencies;
install them only for the adapters you intend to run. The full model list is not
needed for the smoke test.

```powershell
python -m pip install -e ".[viz,benchmark]"
```

## Quick CPU Smoke Test

Use any short WAV and a reference CSV with `start_sec,end_sec` columns on the
same zero-based audio timebase. The intervals should mark the target speaker,
not all speakers. This command uses built-in baselines and downloads no models:

```powershell
python orchestrate_benchmark.py `
  --wav "C:\path\to\recording.wav" `
  --ref "C:\path\to\target.csv" `
  --out outputs/smoke `
  --diarization-pipelines none `
  --segmenters energy_vad_internal `
  --embeddings band_energy_internal
```

Check `outputs/smoke/RESULTS.md` and `outputs/smoke/all_results.csv`. The
orchestrated benchmark uses labeled reference intervals to enroll the target;
its scores are not a standalone identification test on unlabeled recordings.

## Credentials

Pyannote and some Hugging Face-hosted models require a Hugging Face token.

Create a token at <https://huggingface.co/settings/tokens>, accept any required
model licenses on Hugging Face, then set the token in your terminal:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Never commit tokens. Do not put real tokens in README examples, source code, or
tracked config files.

## Compare Two Microphones

The paired comparison uses Pyannote Community-1. Install the narrow extra,
accept the [model terms](https://huggingface.co/pyannote/speaker-diarization-community-1),
and set `HF_TOKEN` as above. The first run needs the model weights; a cached
model can be used with `HF_HUB_OFFLINE=1`.

```powershell
python -m pip install -e ".[viz,pyannote]"
python scripts/compare_microphones.py `
  --piezo "C:\path\to\aligned_piezo.wav" `
  --phone "C:\path\to\aligned_phone.wav" `
  --reference "C:\path\to\target.csv" `
  --out outputs/microphone_pair `
  --device auto
```

Inputs must already be time-aligned and have identical sample rates and frame
counts. Equal lengths alone do not establish alignment. Omit `--reference` for
an unlabeled pair; accuracy fields will be blank. The output directory must be
new. `comparison.json` includes all anonymous clusters and inference timings;
`comparison.csv` compares each microphone's longest cluster and the phone
cluster with the greatest temporal overlap with the piezo anchor, measured by
Dice. Paired matching uses both recordings and assumes the piezo's longest
cluster is the wearer. It does not validate standalone phone identification.

To plot supplied precision and recall as percentages with calculated F1:

```powershell
python scripts/plot_microphone_metrics.py `
  --piezo 90 85 --phone 80 75 `
  --title "Example comparison" --hypothetical `
  --out outputs/example_microphone_scores.png
```

The example values are illustrative. Keep `--hypothetical` for simulated scores;
omit it only when plotting observed results. The repository does not include
private recordings, labels, or past study results.

## Data Layout

The multi-recording runner accepts one folder containing mixed WAV and CSV files.
It pairs files by normalized stem:

```text
data/
├── session_01.wav
├── VAD_session_01.wav.csv
├── session_02.wav
└── VAD_session_02.wav.csv
```

Supported CSV references include simple `start_sec,end_sec` rows and the
project's VAD/timeline CSV formats. RTTM references are supported by the
single-recording runner for standardized diarization datasets.

Before running models, check pairing:

```powershell
$env:PYTHONPATH = "src"
python multi_recording_benchmark.py `
  --input-dir "C:\path\to\wav_csv_folder" `
  --out benchmark_manifest_discovery `
  --discover-only
```

Review:

- `benchmark_manifest_discovery/discovered_manifest.csv`
- `benchmark_manifest_discovery/pairing_report.csv`

## Model Sets

Default end-to-end diarization pipelines:

- `nemo_sortformer`
- `pyannote_community_1`
- `silero_campp_clustering`

Default segmentation models:

- `nemo_sortformer`
- `pyannote_community_1`
- `silero_campp_clustering`
- `silero_vad`
- `webrtc_vad`
- `energy_vad_internal`

Default speaker embedding models:

- `titanet`
- `ecapa_tdnn`
- `wavlm`
- `campp`
- `speechbrain_resnet`
- `resemblyzer_dvector`

The default full benchmark produces:

- 3 native/end-to-end diarization rows
- 6 segmentation models x 6 embedding models = 36 rows
- 39 rows per recording


## Run One Recording

```powershell
$env:PYTHONPATH = "src"
$env:HF_TOKEN = "hf_your_token_here"

python orchestrate_benchmark.py `
  --wav "C:\path\to\recording.wav" `
  --ref "C:\path\to\reference.csv" `
  --out benchmark_one_recording `
  --enrollment-strategy spread `
  --skip-existing
```

Important flags:

- `--skip-existing` reuses completed artifacts and row CSVs
- `--enrollment-strategy spread` samples enrollment audio across the reference
- `--threshold` sets the speaker verification threshold, default `0.62`
- `--window-sec` and `--hop-sec` control sliding classification windows
- `--collar-sec` applies evaluation collar, default `0.0`

Outputs:

- `RESULTS.md`
- `all_results.csv`
- `diarization_pipelines.csv`
- `segmentation_embedding_matrix.csv`
- `heatmaps/*.png`
- `timelines/*.main.csv`
- `logs/*.log`
- `artifacts/*.json`

## Run A Multi-Recording Benchmark

```powershell
$env:PYTHONPATH = "src"
$env:HF_TOKEN = "hf_your_token_here"
$env:AMS_EMBED_BATCH_SIZE = "16"

python multi_recording_benchmark.py `
  --input-dir "C:\path\to\wav_csv_folder" `
  --out benchmark_full_study `
  --sort-by-duration `
  --skip-existing `
  --plot-limit 2000 `
  --enrollment-strategy spread
```

Use `--sort-by-duration` so shorter files finish first and you get useful
partial results quickly.

Use `--skip-existing` every time you resume. It prevents rerunning recordings
that already have `all_results.csv`, and it passes resume behavior down to the
single-recording orchestrator.

The runner updates partial aggregate outputs after each completed recording:

- `MULTI_RESULTS_PARTIAL.md`
- `recording_run_status.csv`
- `all_recording_results.csv`
- `aggregate_by_model.csv`
- `aggregate_heatmaps/*.png`

Final outputs:

- `MULTI_RESULTS.md`
- `multi_manifest.json`
- `discovered_manifest.csv`
- `pairing_report.csv`
- `recording_run_status.csv`
- `all_recording_results.csv`
- `aggregate_by_model.csv`
- `aggregate_heatmaps/*.png`
- `timeline_plots/*.png`
- `recordings/<recording_id>/...`

## Heatmaps

Aggregate study-level heatmaps are written to:

```text
benchmark_full_study/aggregate_heatmaps/
```

The most useful files are:

- `aggregate_heatmap_der_mean.png`: lower is better
- `aggregate_heatmap_f1_mean.png`: higher is better
- `aggregate_heatmap_recall_mean.png`: higher is better
- `aggregate_heatmap_precision_mean.png`: higher is better
- `aggregate_heatmap_abs_speaking_time_error_sec_mean.png`: lower is better

Each recording also has per-recording heatmaps under:

```text
benchmark_full_study/recordings/<recording_id>/heatmaps/
```

## Resume A Long Run

If a long run stops, rerun the same command with the same output folder and
`--skip-existing`:

```powershell
python multi_recording_benchmark.py `
  --input-dir "C:\path\to\wav_csv_folder" `
  --out benchmark_full_study `
  --sort-by-duration `
  --skip-existing `
  --plot-limit 2000 `
  --enrollment-strategy spread
```

Resume behavior:

- completed recordings with `all_results.csv` are skipped
- existing segmenter artifacts are reused
- completed embedding row CSVs are reused
- missing rows continue from the incomplete point where possible

## Run In The Background On Windows

You can create a small PowerShell launcher like:

```powershell
$ErrorActionPreference = 'Continue'
Set-Location 'C:\path\to\repo'
$env:PYTHONPATH='src'
$env:HF_TOKEN='hf_your_token_here'
$env:AMS_EMBED_BATCH_SIZE='16'
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:http_proxy -ErrorAction SilentlyContinue
Remove-Item Env:https_proxy -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force benchmark_full_study | Out-Null
python multi_recording_benchmark.py --input-dir "C:\path\to\wav_csv_folder" --out benchmark_full_study --sort-by-duration --skip-existing --plot-limit 2000 --enrollment-strategy spread *> benchmark_full_study\background_run.log
```

Start it:

```powershell
Start-Process -FilePath powershell.exe `
  -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','C:\path\to\repo\run_full39_background.ps1') `
  -WindowStyle Hidden
```

Check running processes:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*benchmark_full_study*' } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
```

## Threshold Sweep

Use the threshold sweep to tune thresholds/windows on a random dev split and
test on a held-out split:

```powershell
$env:PYTHONPATH = "src"

python threshold_sweep.py `
  --input-dir "C:\path\to\wav_csv_folder" `
  --out benchmark_threshold_sweep `
  --dev-n 5 `
  --test-n 5 `
  --thresholds "0.45,0.55,0.62,0.70" `
  --window-hops "1.0:0.5,1.5:0.75,2.0:1.0" `
  --enrollment-strategies "first,longest,spread"
```

This is useful for pipeline development, but for a publishable study you should
avoid repeatedly tuning on the same evaluation cohort.

## Metrics

The benchmark reports:

- `precision`
- `recall`
- `f1`
- `iou`
- `der`
- `miss_rate`
- `false_alarm_rate_vs_ref`
- `speaking_time_error_sec`
- `speaking_time_error_pct_ref`
- `abs_speaking_time_error_sec`
- `abs_speaking_time_error_pct_ref`
- `wall_time_sec`

For aggregate results, `aggregate_by_model.csv` includes mean, median, and
bootstrap 95% confidence intervals for key metrics.

## Reproducibility Notes

For a study, record:

- repository commit hash
- Python version
- OS and hardware
- GPU/CPU details
- exact model list
- model checkpoint versions
- threshold/window/hop/enrollment parameters
- whether HF model licenses were accepted
- exact train/dev/test split if threshold tuning was used

Recommended command to capture commit:

```powershell
git rev-parse HEAD
```

Recommended command to run tests:

```powershell
python -m unittest discover -s tests
```

## Common Issues

### Hugging Face authentication

If Pyannote fails with access errors, set `HF_TOKEN` and accept the model terms
on Hugging Face.

### FFmpeg not found

Install FFmpeg and confirm `ffmpeg -version` works in the same terminal.

### Windows import/library issues

Some audio ML packages have Windows-specific native dependency issues. Check the
per-model logs under `recordings/<recording_id>/logs/`. Failed models are kept
in result tables as `not_available` rather than replaced by proxy scores.

### Long runtime

Full 39-pipeline studies are expensive. Use:

- `--sort-by-duration`
- `--skip-existing`
- `AMS_EMBED_BATCH_SIZE=16`
- a small smoke run first

Smoke run example:

```powershell
python multi_recording_benchmark.py `
  --input-dir "C:\path\to\wav_csv_folder" `
  --out benchmark_smoke `
  --sort-by-duration `
  --max-recordings 3 `
  --diarization-pipelines none `
  --segmenters energy_vad_internal,webrtc_vad `
  --embeddings resemblyzer_dvector `
  --skip-existing
```

## Development

Run tests:

```powershell
python -m unittest discover -s tests
```

Compile-check changed files:

```powershell
python -m py_compile orchestrate_benchmark.py multi_recording_benchmark.py
```

Before committing:

```powershell
git status --short
git status --ignored --short
```

Make sure data, generated benchmark folders, caches, tokens, and model files are
not staged.

## Citation / Study Use

This repository is a benchmarking and analysis scaffold. If you use it in a
study, cite the upstream model/tool authors as appropriate, including Pyannote,
NVIDIA NeMo, SpeechBrain, Microsoft WavLM, Silero VAD, WebRTC VAD, FunASR/CAM++,
Resemblyzer, and any standardized datasets used for validation.
