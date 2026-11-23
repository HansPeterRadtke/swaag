"""Optional offline voice-similarity evidence; never owner authentication.

A voiceprint match is not live-person authentication. Replaying a genuine
recording will match the original speaker. No result from this component may
submit turns, execute tools, or authorize external effects. SWAAG's separate
server-side voice authority and genuine audio anti-replay still gate actions.

This module is deliberately importable on Python 3.11 without heavyweight
optional dependencies or pretrained files. Inference only occurs after a
caller explicitly constructs an isolated CPU-only model with a SHA-pinned
third-party ONNX file and a private audio root. No download, API request,
service startup, biometric file writing, or authorization is performed.

The fbank implementation follows published WeSpeaker inference parameters:
16-kHz mono PCM16, 80-bin Kaldi fbank, hamming, 25-ms frame, 10-ms shift,
zero dither, sample amplitude scaling by 2^15 and mean subtraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from array import array
import hashlib
import math
import os
from pathlib import Path
import stat
import sys
from typing import Iterable
import wave


# Exact independently inspected public CAM++ LM ONNX sample; keep the model
# file under private, separately licensed runtime storage, never in Git.
WESPEAKER_CAMPPLUS_LM_SHA256 = (
    "1068e4ac3a76bb9c769e6816ef30bf89363f6e966f1d938210cb8ed4038f8e93"
)
WESPEAKER_CAMPPLUS_LM_SIZE = 29_292_449


class SpeakerSimilarityUnavailable(RuntimeError):
    """A malformed/unknown input is refused, not assumed to be trusted."""


@dataclass(frozen=True)
class SpeakerComparison:
    """Pure evidence, intentionally without any usable permission flag."""

    cosine_similarity: float
    identity_claim_is_verified: bool = False
    live_human_proven: bool = False
    recording_replay_excluded: bool = False
    can_auto_submit: bool = False
    can_execute_tools: bool = False
    reason: str = "speaker_similarity_only_no_authority"

    def __bool__(self) -> bool:
        raise TypeError(
            "Voice similarity is evidence only; it MUST NOT be used as a boolean authorization"
        )


def cosine_similarity(first: Iterable[float], second: Iterable[float]) -> float:
    """Strict finite cosine, never a decision threshold or owner identity."""
    try:
        left = tuple(float(v) for v in first)
        right = tuple(float(v) for v in second)
    except (ValueError, TypeError, OverflowError) as exc:
        raise SpeakerSimilarityUnavailable("Invalid speaker embedding values") from exc
    if not left or len(left) != len(right):
        raise SpeakerSimilarityUnavailable("Speaker embedding size mismatch")
    if len(left) > 8192 or any(not math.isfinite(x) for x in left + right):
        raise SpeakerSimilarityUnavailable("Nonfinite or oversized speaker embedding")
    a = math.fsum(x * x for x in left)
    b = math.fsum(x * x for x in right)
    if a < 1.0e-16 or b < 1.0e-16:
        raise SpeakerSimilarityUnavailable("Empty speaker embedding vector")
    result = math.fsum(x * y for x, y in zip(left, right)) / math.sqrt(a * b)
    if not math.isfinite(result):
        raise SpeakerSimilarityUnavailable("Invalid cosine similarity")
    return max(-1.0, min(1.0, result))


def compare_voice_embeddings(first: Iterable[float], second: Iterable[float]) -> SpeakerComparison:
    return SpeakerComparison(cosine_similarity=cosine_similarity(first, second))


def sha256_regular_file(path: Path, *, maximum_bytes: int) -> str:
    if not path.is_absolute() or path.is_symlink():
        raise SpeakerSimilarityUnavailable("A regular absolute private file is required")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or not (0 < info.st_size <= maximum_bytes):
            raise SpeakerSimilarityUnavailable("Missing or oversized private source file")
        total = hashlib.sha256()
        with path.open("rb") as reader:
            while part := reader.read(1024 * 1024):
                total.update(part)
        # Refuse if the source changed during verification.
        now = path.stat()
        if (info.st_size, info.st_mtime_ns, info.st_ino) != (
            now.st_size,
            now.st_mtime_ns,
            now.st_ino,
        ):
            raise SpeakerSimilarityUnavailable("Private source changed during checksum")
        return total.hexdigest()
    except OSError as exc:
        raise SpeakerSimilarityUnavailable("Private source cannot be read") from exc


class OfflineSpeakerSimilarity:
    """Strict SHA-pinned CPU scoring, with no tool/turn authorization API."""

    def __init__(
        self,
        *,
        model_path: Path,
        audio_root: Path,
        model_sha256: str = WESPEAKER_CAMPPLUS_LM_SHA256,
        model_bytes: int = WESPEAKER_CAMPPLUS_LM_SIZE,
        cpu_threads: int = 2,
    ) -> None:
        if not (1 <= cpu_threads <= 4):
            raise SpeakerSimilarityUnavailable("Unbounded model CPU threading is forbidden")
        if not audio_root.is_absolute() or audio_root.is_symlink() or not audio_root.is_dir():
            raise SpeakerSimilarityUnavailable("The private audio directory is unavailable")
        if len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
            raise SpeakerSimilarityUnavailable("Expected model digest is malformed")
        if not (1 <= model_bytes <= 100_000_000):
            raise SpeakerSimilarityUnavailable("Model byte size is unbounded")
        if model_path.is_symlink() or not model_path.is_absolute():
            raise SpeakerSimilarityUnavailable("Only an absolute pinned ONNX model is accepted")
        try:
            if model_path.stat().st_size != model_bytes or sha256_regular_file(
                model_path, maximum_bytes=100_000_000
            ) != model_sha256:
                raise SpeakerSimilarityUnavailable("Model weights do not match the expected digest")
        except OSError as exc:
            raise SpeakerSimilarityUnavailable("Pinned speaker model is missing") from exc
        self._model_sha256 = model_sha256
        self._model_bytes = model_bytes
        self._audio_root = audio_root.resolve(strict=True)
        self._cpu_threads = cpu_threads
        self._model_path = model_path
        self._session = None
        self._numpy = None
        self._torch = None
        self._kaldi = None

    def _validated_pcm16(self, path: Path):
        """An immutable-size, bounded audio file; no raw voice content logged."""
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise SpeakerSimilarityUnavailable("Invalid private audio file")
        try:
            path.resolve(strict=True).relative_to(self._audio_root)
        except (ValueError, OSError) as exc:
            raise SpeakerSimilarityUnavailable("Voice sample escaped its private root") from exc
        # Guard against very long audio or unverified hidden format changes.
        sha256_regular_file(path, maximum_bytes=600_000)
        try:
            with wave.open(str(path), "rb") as recorded:
                if (
                    recorded.getnchannels(), recorded.getsampwidth(), recorded.getframerate()
                ) != (1, 2, 16000):
                    raise SpeakerSimilarityUnavailable("Audio must be 16k mono PCM16")
                count = recorded.getnframes()
                if not (32000 <= count <= 240000):
                    raise SpeakerSimilarityUnavailable("Speaker sample must be 2 to 15 seconds")
                body = recorded.readframes(count)
        except (wave.Error, EOFError, OSError) as exc:
            raise SpeakerSimilarityUnavailable("Cannot decode genuine PCM audio") from exc
        if len(body) != count * 2:
            raise SpeakerSimilarityUnavailable("Truncated PCM input")
        return body, count

    def _load_private_cpu_model(self) -> None:
        if self._session is not None:
            return
        # Refuse a substituted model BEFORE importing heavyweight optional
        # libraries, touching the GPU, or starting an ONNX session.
        try:
            if (
                self._model_path.stat().st_size != self._model_bytes
                or sha256_regular_file(self._model_path, maximum_bytes=100_000_000)
                != self._model_sha256
            ):
                raise SpeakerSimilarityUnavailable("Pinned model changed before CPU inference")
        except OSError as exc:
            raise SpeakerSimilarityUnavailable("Private model source disappeared") from exc
        try:
            import numpy as np  # intentionally optional
            import torch  # intentionally optional
            import torchaudio.compliance.kaldi as kaldi  # intentionally optional
            import onnxruntime as ort  # intentionally optional
        except (ImportError, OSError) as exc:
            raise SpeakerSimilarityUnavailable("Missing optional offline speaker runtime") from exc
        # Explicitly forbid remote providers, GPU allocation or model downloads.
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = self._cpu_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(
                str(self._model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise SpeakerSimilarityUnavailable("Pinned model cannot load on local CPU") from exc
        if (
            {x.name for x in session.get_inputs()} != {"feats"}
            or {x.name for x in session.get_outputs()} != {"embs"}
        ):
            raise SpeakerSimilarityUnavailable("Unexpected speaker embedding model signature")
        # Do not mutate torch's process-global thread setting: other local
        # services may use it. Run this optional component in an isolated
        # worker with OMP_NUM_THREADS bounded when needed.
        self._session = session
        self._numpy = np
        self._torch = torch
        self._kaldi = kaldi

    def embed_voice(self, path: Path) -> tuple[float, ...]:
        pcm, count = self._validated_pcm16(path)
        # Reject silence before imports or allocating a model session.
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if len(samples) != count or math.fsum(int(v) * int(v) for v in samples) < count * 3.2**2:
            raise SpeakerSimilarityUnavailable("Silent or nearly silent speaker sample")
        self._load_private_cpu_model()
        assert self._numpy is not None and self._torch is not None and self._kaldi is not None
        wave_np = self._numpy.frombuffer(pcm, dtype="<i2").astype(self._numpy.float32)
        signal = self._torch.from_numpy(wave_np.copy()).unsqueeze(0)
        features = self._kaldi.fbank(
            signal,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=16000,
            window_type="hamming",
            use_energy=False,
        )
        features = features - features.mean(dim=0)
        if features.shape[1] != 80 or features.shape[0] < 150:
            raise SpeakerSimilarityUnavailable("Too few speech features to compare")
        try:
            raw = self._session.run(
                ["embs"], {"feats": features.unsqueeze(0).numpy()}
            )[0]
        except Exception as exc:
            raise SpeakerSimilarityUnavailable("Local speaker model inference failed") from exc
        vector = self._numpy.asarray(raw, dtype=self._numpy.float64).reshape(-1)
        if not (128 <= vector.size <= 8192) or not self._numpy.isfinite(vector).all():
            raise SpeakerSimilarityUnavailable("Malformed voice embedding from model")
        length = float(self._numpy.linalg.norm(vector))
        if not (math.isfinite(length) and length > 1.0e-9):
            raise SpeakerSimilarityUnavailable("Model produced a zero voice embedding")
        # In-memory normalized vector; never persist owner voice embeddings.
        return tuple(float(x) for x in (vector / length))

    def compare_audio(self, reference: Path, candidate: Path) -> SpeakerComparison:
        return compare_voice_embeddings(self.embed_voice(reference), self.embed_voice(candidate))
