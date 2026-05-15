from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ams.dsp.embedding import band_energy_embedding


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    model: str


class EmbeddingModel:
    name: str

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        raise NotImplementedError

    def embed_many(self, clips: List[np.ndarray], sr: int) -> List[EmbeddingResult]:
        return [self.embed(clip, sr) for clip in clips]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


class BandEnergyEmbedding(EmbeddingModel):
    name = "band_energy_internal"

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        return EmbeddingResult(band_energy_embedding(y, sr, embed_dim=32), self.name)


class SpeechBrainEcapaEmbedding(EmbeddingModel):
    name = "ecapa_tdnn_speechbrain"

    def __init__(self, device: str = "cpu", model_id: str = "speechbrain/spkrec-ecapa-voxceleb"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        self.model_id = model_id
        self.classifier = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        return self.embed_many([y], sr)[0]

    def embed_many(self, clips: List[np.ndarray], sr: int) -> List[EmbeddingResult]:
        import torch
        from scipy.signal import resample_poly

        fixed = []
        for clip in clips:
            y = clip
            if sr != 16000:
                y = resample_poly(y.astype(np.float32), 16000, int(sr)).astype(np.float32)
            fixed.append(torch.from_numpy(y.astype(np.float32)))
        wav = torch.nn.utils.rnn.pad_sequence(fixed, batch_first=True).to(self.device)
        with torch.no_grad():
            emb = self.classifier.encode_batch(wav).squeeze()
        arr = emb.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return [EmbeddingResult(_normalize(row), self.name) for row in arr]


class SpeechBrainXVectorEmbedding(EmbeddingModel):
    name = "speechbrain_xvector_voxceleb"

    def __init__(self, device: str = "cpu", model_id: str = "speechbrain/spkrec-xvect-voxceleb"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        self.model_id = model_id
        self.classifier = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        return self.embed_many([y], sr)[0]

    def embed_many(self, clips: List[np.ndarray], sr: int) -> List[EmbeddingResult]:
        import torch
        from scipy.signal import resample_poly

        fixed = []
        for clip in clips:
            y = clip
            if sr != 16000:
                y = resample_poly(y.astype(np.float32), 16000, int(sr)).astype(np.float32)
            fixed.append(torch.from_numpy(y.astype(np.float32)))
        wav = torch.nn.utils.rnn.pad_sequence(fixed, batch_first=True).to(self.device)
        with torch.no_grad():
            emb = self.classifier.encode_batch(wav).squeeze()
        arr = emb.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return [EmbeddingResult(_normalize(row), self.name) for row in arr]


class SpeechBrainResNetEmbedding(EmbeddingModel):
    name = "speechbrain_resnet_voxceleb"

    def __init__(self, device: str = "cpu", model_id: str = "speechbrain/spkrec-resnet-voxceleb"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        self.model_id = model_id
        self.classifier = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        return self.embed_many([y], sr)[0]

    def embed_many(self, clips: List[np.ndarray], sr: int) -> List[EmbeddingResult]:
        import torch
        from scipy.signal import resample_poly

        fixed = []
        for clip in clips:
            y = clip
            if sr != 16000:
                y = resample_poly(y.astype(np.float32), 16000, int(sr)).astype(np.float32)
            fixed.append(torch.from_numpy(y.astype(np.float32)))
        wav = torch.nn.utils.rnn.pad_sequence(fixed, batch_first=True).to(self.device)
        with torch.no_grad():
            emb = self.classifier.encode_batch(wav).squeeze()
        arr = emb.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return [EmbeddingResult(_normalize(row), self.name) for row in arr]


class ResemblyzerEmbedding(EmbeddingModel):
    name = "resemblyzer_dvector"

    def __init__(self):
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder()

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        from scipy.signal import resample_poly

        if sr != 16000:
            y = resample_poly(y.astype(np.float32), 16000, int(sr)).astype(np.float32)
        emb = self.encoder.embed_utterance(y.astype(np.float32))
        return EmbeddingResult(_normalize(np.asarray(emb, dtype=np.float32)), self.name)


class WavLMSpeakerEmbedding(EmbeddingModel):
    name = "wavlm_speaker_verification"

    def __init__(self, device: str = "cpu", model_id: str = "microsoft/wavlm-base-plus-sv"):
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.device = device
        self.model_id = model_id
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioXVector.from_pretrained(model_id).to(device)
        self.model.eval()

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        return self.embed_many([y], sr)[0]

    def embed_many(self, clips: List[np.ndarray], sr: int) -> List[EmbeddingResult]:
        import torch
        from scipy.signal import resample_poly

        fixed = []
        for clip in clips:
            y = clip
            if sr != 16000:
                y = resample_poly(y.astype(np.float32), 16000, int(sr)).astype(np.float32)
            fixed.append(y.astype(np.float32))
        inputs = self.extractor(fixed, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        emb = getattr(out, "embeddings", None)
        if emb is None:
            emb = out[0]
        arr = emb.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return [EmbeddingResult(_normalize(row), self.name) for row in arr]


class NemoTitaNetEmbedding(EmbeddingModel):
    name = "nemo_titanet_large"

    def __init__(self, model_id: str = "titanet_large"):
        os.environ.setdefault("WANDB_MODE", "disabled")
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.model_id = model_id
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_id, map_location="cpu")
        try:
            self.model.eval()
        except Exception:
            pass

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "clip.wav"
            sf.write(str(wav), y.astype(np.float32), sr)
            emb = self.model.get_embedding(str(wav))
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        return EmbeddingResult(_normalize(np.asarray(emb, dtype=np.float32)), self.name)


class FunASRCAMPPEmbedding(EmbeddingModel):
    name = "funasr_campp"

    def __init__(self, model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common"):
        os.environ.setdefault("WANDB_MODE", "disabled")
        from funasr import AutoModel

        self.model_id = model_id
        self.model = AutoModel(model=model_id, hub="ms", disable_update=True)

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "clip.wav"
            sf.write(str(wav), y.astype(np.float32), sr)
            out = self.model.generate(input=str(wav))
        if not out or "spk_embedding" not in out[0]:
            raise RuntimeError(f"CAM++ did not return spk_embedding; got {type(out).__name__}")
        emb = out[0]["spk_embedding"]
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        return EmbeddingResult(_normalize(np.asarray(emb, dtype=np.float32)), self.name)


class ExternalEmbedding(EmbeddingModel):
    """Adapter for models that do not have a stable local Python API.

    Command contract:
      command <wav_path> <out_vector_txt>
    The output file must contain whitespace-separated floats.
    """

    def __init__(self, name: str, env_var: str):
        self.name = name
        self.env_var = env_var
        self.command = os.environ.get(env_var)

    def embed(self, y: np.ndarray, sr: int) -> EmbeddingResult:
        import soundfile as sf

        if not self.command:
            raise RuntimeError(f"Real {self.name} adapter is not configured. Set {self.env_var} to a command that writes one embedding vector.")
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "clip.wav"
            vec = Path(td) / "embedding.txt"
            sf.write(str(wav), y, sr)
            subprocess.check_call([*shlex.split(self.command), str(wav), str(vec)])
            values = [float(x) for x in vec.read_text(encoding="utf-8").split()]
        return EmbeddingResult(_normalize(np.asarray(values, dtype=np.float32)), self.name)


def _torch_device(prefer_gpu: bool = True) -> str:
    try:
        import torch

        return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embedding_factories(prefer_gpu: bool = True) -> Dict[str, Callable[[], EmbeddingModel]]:
    device = _torch_device(prefer_gpu)
    return {
        "ecapa_tdnn": lambda: SpeechBrainEcapaEmbedding(device=device),
        "titanet": lambda: NemoTitaNetEmbedding(),
        "wavlm": lambda: WavLMSpeakerEmbedding(device=device),
        "campp": lambda: FunASRCAMPPEmbedding(),
        "speechbrain_xvector": lambda: SpeechBrainXVectorEmbedding(device=device),
        "speechbrain_resnet": lambda: SpeechBrainResNetEmbedding(device=device),
        "resemblyzer_dvector": lambda: ResemblyzerEmbedding(),
        "eres2net": lambda: ExternalEmbedding("eres2net", "AMS_ERES2NET_EMBED_CMD"),
        "kaldi_xvector": lambda: ExternalEmbedding("kaldi_xvector", "AMS_KALDI_XVECTOR_CMD"),
        "band_energy_internal": lambda: BandEnergyEmbedding(),
    }


REQUESTED_EMBEDDINGS: List[str] = [
    "titanet",
    "ecapa_tdnn",
    "wavlm",
    "campp",
    "speechbrain_resnet",
    "resemblyzer_dvector",
]
