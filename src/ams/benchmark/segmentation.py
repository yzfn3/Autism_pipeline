from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from tqdm.auto import tqdm

from ams.benchmark.audio import load_mono
from ams.benchmark.timeline import Range, merge_ranges
from ams.dsp.metrics import frame_rms_db
from ams.dsp.vad_postprocess import ema_smooth, postprocess_vad


@dataclass(frozen=True)
class SegmenterResult:
    speech_ranges: List[Range]
    speaker_turns: List[dict]
    model: str
    mode: str
    note: str = ""


class Segmenter:
    name: str

    def run(self, wav_path: Path) -> SegmenterResult:
        raise NotImplementedError


class EnergyVadSegmenter(Segmenter):
    name = "energy_vad_internal"

    def run(self, wav_path: Path) -> SegmenterResult:
        y, sr = load_mono(wav_path, 16000)
        tqdm.write("[energy_vad_internal] loading and scoring frames")
        frame_ms, hop_ms = 20.0, 10.0
        frame_len = int(sr * frame_ms / 1000.0)
        hop = int(sr * hop_ms / 1000.0)
        rms = frame_rms_db(y, frame_len, hop)
        x = (rms - (-55.0 + 10.0)) / 3.0
        p = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
        p = np.where(rms < -55.0, p * 0.15, p)
        p = ema_smooth(p, alpha=float(1.0 - np.exp(-(hop_ms / 1000.0) / 0.35)))
        t = (np.arange(len(p), dtype=np.float32) * hop + 0.5 * frame_len) / float(sr)
        segs, _ = postprocess_vad(t, p, 0.65, 0.45, 0.4, 0.4, 0.15, 0.0, len(y) / sr)
        ranges = [(s.start_sec, s.end_sec) for s in segs]
        return SegmenterResult(ranges, [], self.name, "vad")


class SileroVadSegmenter(Segmenter):
    name = "silero_vad"

    def run(self, wav_path: Path) -> SegmenterResult:
        import torch

        y, sr = load_mono(wav_path, 16000)
        tqdm.write("[silero_vad] loading torch.hub model and running VAD")
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
        get_speech_timestamps = utils[0]
        stamps = get_speech_timestamps(y, model, sampling_rate=sr)
        ranges = [(float(x["start"]) / sr, float(x["end"]) / sr) for x in stamps if x.get("end", 0) > x.get("start", 0)]
        return SegmenterResult(merge_ranges(ranges), [], self.name, "vad")


class WebRtcVadSegmenter(Segmenter):
    name = "webrtc_vad"

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30, min_speech_sec: float = 0.25, min_silence_sec: float = 0.2):
        self.aggressiveness = int(aggressiveness)
        self.frame_ms = int(frame_ms)
        self.min_speech_sec = float(min_speech_sec)
        self.min_silence_sec = float(min_silence_sec)

    def run(self, wav_path: Path) -> SegmenterResult:
        import webrtcvad

        from scipy.signal import resample_poly

        y, sr = load_mono(wav_path, 16000)
        if sr != 16000:
            y = resample_poly(y.astype(np.float32), 16000, int(sr)).astype(np.float32)
            sr = 16000
        pcm = np.clip(y, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
        frame_len = int(sr * self.frame_ms / 1000.0)
        bytes_per_frame = frame_len * 2
        vad = webrtcvad.Vad(self.aggressiveness)
        flags = []
        for offset in range(0, len(pcm16) - bytes_per_frame + 1, bytes_per_frame):
            flags.append(vad.is_speech(pcm16[offset:offset + bytes_per_frame], sr))

        ranges: List[Range] = []
        active_start: float | None = None
        last_speech_end = 0.0
        for i, flag in enumerate(flags):
            start = i * self.frame_ms / 1000.0
            end = start + self.frame_ms / 1000.0
            if flag:
                if active_start is None:
                    active_start = start
                last_speech_end = end
            elif active_start is not None and start - last_speech_end >= self.min_silence_sec:
                if last_speech_end - active_start >= self.min_speech_sec:
                    ranges.append((active_start, last_speech_end))
                active_start = None
        if active_start is not None and last_speech_end - active_start >= self.min_speech_sec:
            ranges.append((active_start, last_speech_end))
        return SegmenterResult(merge_ranges(ranges, gap=self.min_silence_sec), [], self.name, "vad")


class PyannoteDiarizationSegmenter(Segmenter):
    def __init__(self, name: str, pipeline_id: str):
        self.name = name
        self.pipeline_id = pipeline_id

    @staticmethod
    def _iter_pyannote_tracks(diarization):
        """Yield (turn, speaker) pairs from old Annotation or new DiarizeOutput."""
        candidates = [diarization]
        for attr in (
            "speaker_diarization",
            "exclusive_speaker_diarization",
            "diarization",
            "annotation",
            "output",
        ):
            if hasattr(diarization, attr):
                candidates.append(getattr(diarization, attr))
        if isinstance(diarization, dict):
            for key in (
                "speaker_diarization",
                "exclusive_speaker_diarization",
                "diarization",
                "annotation",
            ):
                if key in diarization:
                    candidates.append(diarization[key])

        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "itertracks"):
                for item in candidate.itertracks(yield_label=True):
                    if len(item) == 3:
                        turn, _, speaker = item
                    elif len(item) == 2:
                        turn, speaker = item
                    else:
                        continue
                    yield turn, speaker
                return
            if hasattr(candidate, "itersegments"):
                for turn in candidate.itersegments():
                    yield turn, "SPEAKER_00"
                return

        attrs = sorted(a for a in dir(diarization) if not a.startswith("_"))
        raise AttributeError(
            "Could not extract pyannote tracks from "
            f"{type(diarization).__name__}; available attributes: {attrs[:40]}"
        )

    def run(self, wav_path: Path) -> SegmenterResult:
        from pyannote.audio import Pipeline

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        pipeline_id = self.pipeline_id
        tqdm.write(f"[{self.name}] loading {self.pipeline_id}")
        pipeline = Pipeline.from_pretrained(pipeline_id, token=token)
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:
            pass
        tqdm.write(f"[{self.name}] running diarization")
        if "precision-2" in self.pipeline_id:
            diar = pipeline(str(wav_path))
        else:
            import torch

            y, sr = load_mono(wav_path, 16000)
            waveform = torch.from_numpy(y).unsqueeze(0)
            diar = pipeline({"waveform": waveform, "sample_rate": sr})
        turns: List[dict] = []
        ranges: List[Range] = []
        for turn, speaker in self._iter_pyannote_tracks(diar):
            s, e = float(turn.start), float(turn.end)
            ranges.append((s, e))
            turns.append({"start_sec": s, "end_sec": e, "speaker": str(speaker)})
        return SegmenterResult(merge_ranges(ranges), turns, self.name, "diarization")


class NemoSortformerSegmenter(Segmenter):
    name = "nemo_sortformer"

    def __init__(self, model_id: str = "nvidia/diar_sortformer_4spk-v1"):
        os.environ.setdefault("WANDB_MODE", "disabled")
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model_id = model_id
        tqdm.write(f"[{self.name}] loading {model_id}")
        self.model = SortformerEncLabelModel.from_pretrained(model_id, map_location="cpu")
        try:
            self.model.eval()
        except Exception:
            pass

    @staticmethod
    def _parse_line(line: str) -> dict:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Unexpected Sortformer segment line: {line!r}")
        return {"start_sec": float(parts[0]), "end_sec": float(parts[1]), "speaker": str(parts[2])}

    def run(self, wav_path: Path) -> SegmenterResult:
        tqdm.write(f"[{self.name}] running diarization")
        out = self.model.diarize(audio=str(wav_path), batch_size=1, num_workers=0, verbose=False)
        lines = out[0] if out and isinstance(out[0], list) else out
        turns = [self._parse_line(line) for line in lines]
        ranges = [(t["start_sec"], t["end_sec"]) for t in turns]
        return SegmenterResult(merge_ranges(ranges), turns, self.name, "diarization")


class SileroCamppClusteringDiarizer(Segmenter):
    name = "silero_campp_clustering"

    def __init__(self, window_sec: float = 1.5, hop_sec: float = 0.75, cluster_threshold: float = 0.45):
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.cluster_threshold = float(cluster_threshold)

    def run(self, wav_path: Path) -> SegmenterResult:
        from sklearn.cluster import AgglomerativeClustering

        from ams.benchmark.embeddings import FunASRCAMPPEmbedding
        from ams.dsp.segmentation import sliding_windows

        y, sr = load_mono(wav_path, 16000)
        vad = SileroVadSegmenter().run(wav_path)
        windows = [
            (window.start_sec, window.end_sec)
            for start, end in vad.speech_ranges
            for window in sliding_windows(start, end, self.window_sec, self.hop_sec)
        ]
        if not windows:
            return SegmenterResult([], [], self.name, "diarization", "Silero VAD found no speech")

        embedder = FunASRCAMPPEmbedding()
        vectors = []
        for start, end in tqdm(windows, desc=f"cluster:{self.name}", leave=False):
            clip = y[max(0, int(start * sr)):max(0, int(end * sr))]
            if clip.size:
                vectors.append(embedder.embed(clip, sr).vector)
        if not vectors:
            return SegmenterResult(vad.speech_ranges, [], self.name, "diarization", "No embedding windows were usable")

        if len(vectors) == 1:
            labels = np.array([0])
        else:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.cluster_threshold,
                    metric="cosine",
                    linkage="average",
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.cluster_threshold,
                    affinity="cosine",
                    linkage="average",
                )
            labels = clustering.fit_predict(np.stack(vectors, axis=0))

        raw_turns = [
            {"start_sec": float(start), "end_sec": float(end), "speaker": f"spk{int(label):02d}"}
            for (start, end), label in zip(windows, labels)
        ]
        turns: List[dict] = []
        for turn in sorted(raw_turns, key=lambda item: (item["speaker"], item["start_sec"], item["end_sec"])):
            if turns and turns[-1]["speaker"] == turn["speaker"] and turn["start_sec"] - turns[-1]["end_sec"] <= self.hop_sec:
                turns[-1]["end_sec"] = max(turns[-1]["end_sec"], turn["end_sec"])
            else:
                turns.append(dict(turn))
        turns = sorted(turns, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker"]))
        ranges = [(turn["start_sec"], turn["end_sec"]) for turn in turns]
        return SegmenterResult(merge_ranges(ranges), turns, self.name, "diarization")


class ExternalSegmenter(Segmenter):
    """Adapter for diarizers/segmenters provided outside this Python package.

    Command contract:
      command <wav_path> <out_json>
    JSON must contain either {"speech_ranges": [[s,e], ...]} or
    {"speaker_turns": [{"start_sec": s, "end_sec": e, "speaker": "..."}]}.
    """

    def __init__(self, name: str, env_var: str):
        self.name = name
        self.env_var = env_var
        self.command = os.environ.get(env_var)

    def run(self, wav_path: Path) -> SegmenterResult:
        if not self.command:
            raise RuntimeError(f"Real {self.name} adapter is not configured. Set {self.env_var} to a command that writes model output JSON.")
        out = wav_path.with_suffix(f".{self.name}.json")
        subprocess.check_call([*shlex.split(self.command), str(wav_path), str(out)])
        data = json.loads(out.read_text(encoding="utf-8"))
        turns = data.get("speaker_turns") or []
        ranges = data.get("speech_ranges") or [(t["start_sec"], t["end_sec"]) for t in turns]
        return SegmenterResult(merge_ranges([(float(s), float(e)) for s, e in ranges]), turns, self.name, "external")


def segmenter_factories() -> Dict[str, Callable[[], Segmenter]]:
    return {
        "pyannote_community_1": lambda: PyannoteDiarizationSegmenter("pyannote_community_1", "pyannote/speaker-diarization-community-1"),
        "pyannote_3_1": lambda: PyannoteDiarizationSegmenter("pyannote_3_1", "pyannote/speaker-diarization-3.1"),
        "nemo_sortformer": lambda: NemoSortformerSegmenter(),
        "silero_campp_clustering": lambda: SileroCamppClusteringDiarizer(),
        "nemo_msdd": lambda: ExternalSegmenter("nemo_msdd", "AMS_NEMO_MSDD_CMD"),
        "eend_m2f": lambda: ExternalSegmenter("eend_m2f", "AMS_EEND_M2F_CMD"),
        "mamba_speaker_segmentation": lambda: ExternalSegmenter("mamba_speaker_segmentation", "AMS_MAMBA_SEG_CMD"),
        "silero_vad": lambda: SileroVadSegmenter(),
        "webrtc_vad": lambda: WebRtcVadSegmenter(),
        "energy_vad_internal": lambda: EnergyVadSegmenter(),
    }


REQUESTED_SEGMENTERS: List[str] = [
    "nemo_sortformer",
    "pyannote_community_1",
    "silero_campp_clustering",
    "silero_vad",
    "webrtc_vad",
    "energy_vad_internal",
]
