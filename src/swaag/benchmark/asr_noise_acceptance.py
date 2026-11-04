"""Opt-in ASR word-error and false-positive bench on source-hash-pinned real speech.

The private user-supplied manifest holds the real WAVs and reference transcripts.
Synthesized additive noise is solely a repeatable laboratory perturbation.
This benchmark never submits a conversation turn or restarts an ASR process.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
import urllib.request
import wave
from typing import Any, Callable, Sequence


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", str(text).casefold(), flags=re.UNICODE)


def word_edits(reference: str, hypothesis: str) -> tuple[int, int]:
    a, b = words(reference), words(hypothesis)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, prev[j] + 1, prev[j - 1] + int(x != y)))
        prev = current
    return prev[-1], len(a)


def pcm_wav(samples: Sequence[float]) -> bytes:
    from array import array
    import sys

    raw = array("h", [int(max(-1, min(1, x)) * 32767) for x in samples])
    if sys.byteorder != "little":
        raw.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setframerate(16000)
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.writeframes(raw.tobytes())
    return output.getvalue()


def wav_samples(source: bytes) -> list[float]:
    from array import array
    import sys

    with wave.open(io.BytesIO(source), "rb") as media:
        if (media.getframerate(), media.getnchannels(), media.getsampwidth()) != (16000, 1, 2):
            raise ValueError("source audio must be 16000 Hz mono signed PCM16 WAV")
        n = media.getnframes()
        if not 16000 <= n <= 16000 * 45:
            raise ValueError("source audio must have 1-45 seconds of samples")
        pcm = array("h")
        pcm.frombytes(media.readframes(n))
        if sys.byteorder != "little":
            pcm.byteswap()
        return [x / 32768.0 for x in pcm]


def noisy(samples: Sequence[float], snr_db: int, random_source: random.Random) -> tuple[bytes, float]:
    if snr_db not in {20, 10, 0}:
        raise ValueError("allowed laboratory SNR is 20, 10 or 0 dB")
    count = len(samples)
    rms = math.sqrt(sum(x * x for x in samples) / count)
    if rms < 1e-6:
        raise ValueError("original labeled speech contains no measurable signal")
    background = [random_source.gauss(0, 1) for _ in samples]
    b_rms = math.sqrt(sum(x * x for x in background) / count)
    gain = rms * (10 ** (-snr_db / 20)) / b_rms
    perturbed = [x + gain * n for x, n in zip(samples, background)]
    clipping = sum(abs(x) >= 1 for x in perturbed) / count
    return pcm_wav(perturbed), round(clipping * 100, 5)


def local_stt(url: str, data: bytes, lang: str) -> str:
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "audio/wav", "X-Voice-Language": lang},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict) or value.get("ok") is False or not isinstance(value.get("text"), str):
        raise ValueError("resident recognizer supplied no valid final transcription")
    return value["text"]


def load_source_manifest(path: Path, max_cases: int = 32) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= max_cases:
        raise ValueError("source manifest must provide 1 to max_cases records")
    seen = set()
    loaded = []
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("invalid source manifest record")
        identifier = str(row.get("id") or "")
        language = str(row.get("language") or "")
        label = str(row.get("reference") or "")
        digest = str(row.get("sha256") or "")
        if (not identifier or len(identifier) > 128 or identifier in seen
                or language not in {"en", "de"} or not words(label) or len(label) > 10000):
            raise ValueError("duplicate, empty or unsupported labeled recording")
        seen.add(identifier)
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("source audio SHA256 missing or invalid")
        file_path = Path(str(row.get("wav_path") or ""))
        if not file_path.is_absolute() or file_path.is_symlink():
            raise ValueError("source audio must be a real absolute non-symlink path")
        size = file_path.stat().st_size
        if not 32000 <= size <= 1500000:
            raise ValueError("source audio must be a bounded 1-45s PCM16 WAV")
        raw = file_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("source media SHA256 mismatch for " + identifier)
        signal = wav_samples(raw)
        rms = math.sqrt(sum(x * x for x in signal) / len(signal))
        loaded.append({"id": identifier, "lang": language, "reference": label,
                       "sha256": digest, "signal": signal, "source_rms": round(rms, 7)})
    return loaded


def _summary(rows: list[dict]) -> dict:
    result = {}
    for lang in ("en", "de"):
        for snr in (20, 10, 0):
            selected = [x for x in rows if x["language"] == lang and x["snr_db"] == snr]
            if not selected:
                continue
            clean = [x for x in selected if x["success"]]
            n = sum(x["reference_words"] for x in clean)
            result[f"{lang}_{snr}dB"] = {
                "count": len(selected), "successful": len(clean),
                "wer": round(sum(x["edits"] for x in clean) / n, 5) if n else None,
                "median_seconds": round(statistics.median(x["seconds"] for x in clean), 3) if clean else None,
            }
    return result


def run(*, manifest_path: Path, output: Path, seed: int = 20261104,
        snr_levels: Sequence[int] = (20, 10, 0),
        max_cases: int = 32,
        transcriber: Callable[[str, bytes, str], str] = local_stt,
        endpoint: str = "http://127.0.0.1:15703/transcribe",
        include_noise_only: bool = True) -> dict:
    if endpoint != "http://127.0.0.1:15703/transcribe":
        raise ValueError("only the known Thor resident loopback STT endpoint is allowed")
    if len(set(snr_levels)) != len(snr_levels) or not snr_levels or set(snr_levels) - {20, 10, 0}:
        raise ValueError("SNR must be a nonempty unique subset of 20, 10, 0 dB")
    if output.exists():
        raise FileExistsError("existing or uncertain STT evaluation receipt; never replay")
    loaded = load_source_manifest(manifest_path, max_cases=max_cases)
    rng = random.Random(seed)
    receipt = {"status": "started", "source_manifest_sha256":
               hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "seed": seed, "snr_db": list(snr_levels),
               "origin": "verified real PCM16 WAV + synthetic Gaussian noise",
               "source_case_count": len(loaded),
               "cases": [], "noise_only": [], "no_swaag_actions": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(json.dumps(receipt) + "\n")
        file.flush()
        os.fsync(file.fileno())
    try:
        for case in loaded:
            for level in snr_levels:
                wav, clipped = noisy(case["signal"], level, rng)
                began = time.monotonic()
                try:
                    actual = transcriber(endpoint, wav, case["lang"])
                    edits, n = word_edits(case["reference"], actual)
                    result = {"success": True, "recognized": actual, "edits": edits,
                              "reference_words": n}
                except Exception as exc:
                    result = {"success": False, "recognized": "", "edits": None,
                              "reference_words": len(words(case["reference"])),
                              "error_type": type(exc).__name__}
                receipt["cases"].append({
                    "id": case["id"], "language": case["lang"], "snr_db": level,
                    "source_sha256": case["sha256"],
                    "source_rms": case["source_rms"],
                    "derived_sha256": hashlib.sha256(wav).hexdigest(),
                    "clipped_percent": clipped,
                    "seconds": round(time.monotonic() - began, 3), **result})
                output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        if include_noise_only:
            for lang in sorted({x["lang"] for x in loaded}):
                for amplitude in (0.01, 0.05, 0.15):
                    samples = [rng.gauss(0, 1) for _ in range(16000 * 3)]
                    rms = math.sqrt(sum(x*x for x in samples) / len(samples))
                    wav = pcm_wav([x * amplitude / rms for x in samples])
                    began = time.monotonic()
                    try:
                        transcript = transcriber(endpoint, wav, lang)
                        result = {"success": True, "nonempty": bool(words(transcript)),
                                  "recognized": transcript}
                    except Exception as exc:
                        result = {"success": False, "nonempty": None,
                                  "error_type": type(exc).__name__}
                    receipt["noise_only"].append({
                        "language": lang, "noise_rms": amplitude,
                        "wav_sha256": hashlib.sha256(wav).hexdigest(),
                        "seconds": round(time.monotonic() - began, 3), **result})
                    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        receipt["summary"] = _summary(receipt["cases"])
        failures = sum(not x["success"] for x in receipt["cases"] + receipt["noise_only"])
        false_speech = sum(x["nonempty"] is True for x in receipt["noise_only"])
        receipt["negative_hallucinations"] = false_speech
        receipt["status"] = "passed" if not failures and not false_speech else "failed"
    except BaseException:
        receipt["status"] = "interrupted_or_failed"
        output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        raise
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--snr", type=int, action="append")
    args = parser.parse_args(argv)
    report = run(manifest_path=args.manifest, output=args.output, seed=args.seed,
                 snr_levels=args.snr if args.snr else (20, 10, 0))
    print("asr_noisy_real_speech={} {} speech_cases, {} noise_only".format(
        report["status"], len(report["cases"]), len(report["noise_only"])))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
