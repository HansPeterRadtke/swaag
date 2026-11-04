from __future__ import annotations

import hashlib
import io
import json
import math
import random
import wave
from pathlib import Path

import pytest

from swaag.benchmark.asr_noise_acceptance import (
    load_source_manifest, noisy, pcm_wav, run, wav_samples, word_edits,
)


def _fixture(tmp_path: Path) -> Path:
    cases = []
    for i, (lang, label) in enumerate((("en", "hello world"), ("de", "guten tag"))):
        audio = pcm_wav([0.04 * math.sin(x / 10.0) for x in range(16000)])
        file = tmp_path / f"{lang}.wav"
        file.write_bytes(audio)
        cases.append({"id": f"case-{i}", "language": lang,
                      "reference": label, "wav_path": str(file),
                      "sha256": hashlib.sha256(audio).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}))
    return manifest


def _fake_transcriber(url: str, raw: bytes, lang: str) -> str:
    assert url == "http://127.0.0.1:15703/transcribe"
    with wave.open(io.BytesIO(raw), "rb") as audio:
        frames = audio.getnframes()
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
    return "" if frames == 48000 else ("hello world" if lang == "en" else "guten tag")


def test_word_edits_are_reference_weighted_and_unicode_aware():
    assert word_edits("Hello world", "hello world") == (0, 2)
    assert word_edits("A B C", "A B D") == (1, 3)
    assert word_edits("A B C", "A B C D") == (1, 3)
    assert word_edits("Grüß Gott.", "grüß gott") == (0, 2)


def test_lab_snr_is_deterministic_and_pcm16_format():
    source = [0.05 * math.sin(i / 10) for i in range(16000)]
    first, clipped = noisy(source, 10, random.Random(42))
    second, clipped2 = noisy(source, 10, random.Random(42))
    assert first == second and clipped == clipped2
    assert clipped == 0
    assert len(wav_samples(first)) == 16000
    assert len(first) == 32044
    with pytest.raises(ValueError):
        noisy(source, 5, random.Random(42))


def test_real_audio_manifest_requires_exact_source_sha256(tmp_path):
    p = _fixture(tmp_path)
    assert len(load_source_manifest(p)) == 2
    content = json.loads(p.read_text())
    content["cases"][0]["sha256"] = "0" * 64
    p.write_text(json.dumps(content))
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(manifest_path=p, output=output, transcriber=_fake_transcriber)
    assert not output.exists()


def test_duplicate_case_identity_rejected_before_receipt(tmp_path):
    p = _fixture(tmp_path)
    content = json.loads(p.read_text())
    content["cases"][1]["id"] = content["cases"][0]["id"]
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="duplicate"):
        load_source_manifest(p)


def test_full_fake_asr_offline_summary_and_noise_only_positives(tmp_path):
    p = _fixture(tmp_path)
    out = tmp_path / "passed.json"
    report = run(manifest_path=p, output=out, seed=20261104, snr_levels=(20, 0),
                 transcriber=_fake_transcriber)
    assert report["status"] == "passed"
    assert len(report["cases"]) == 4
    assert len(report["noise_only"]) == 6
    assert report["negative_hallucinations"] == 0
    assert all(row["edits"] == 0 and row["source_sha256"] for row in report["cases"])
    assert all(x["wer"] == 0 for x in report["summary"].values())
    assert json.loads(out.read_text())["status"] == "passed"
    with pytest.raises(FileExistsError):
        run(manifest_path=p, output=out, transcriber=_fake_transcriber)


def test_noise_only_hallucination_is_failing_evidence(tmp_path):
    p = _fixture(tmp_path)

    def bad_transcriber(url, raw, lang):
        with wave.open(io.BytesIO(raw)) as audio:
            return "Thank you" if audio.getnframes() == 48000 else (
                "hello world" if lang == "en" else "guten tag")

    receipt = run(manifest_path=p, output=tmp_path / "negative.json", seed=5,
                  snr_levels=(20,), transcriber=bad_transcriber)
    assert receipt["status"] == "failed"
    assert receipt["negative_hallucinations"] == 6


def test_nonloopback_and_invalid_snr_are_refused(tmp_path):
    p = _fixture(tmp_path)
    for args in ({"endpoint": "https://external.example/transcribe"},
                 {"snr_levels": (20, 20)}, {"snr_levels": (5,)}):
        with pytest.raises(ValueError):
            run(manifest_path=p, output=tmp_path / "negative.json",
                transcriber=_fake_transcriber, **args)
    assert not (tmp_path / "negative.json").exists()


def test_source_audio_bounds_and_very_quiet_real_speech_are_handled(tmp_path):
    p = _fixture(tmp_path)
    manifest = json.loads(p.read_text())
    source = tmp_path / "en.wav"
    quiet = pcm_wav([0.0005 * math.sin(i / 10) for i in range(16000)])
    source.write_bytes(quiet)
    manifest["cases"][0]["sha256"] = hashlib.sha256(quiet).hexdigest()
    p.write_text(json.dumps(manifest))
    loaded = load_source_manifest(p)
    assert 0 < loaded[0]["source_rms"] < 0.002
    source.write_bytes(b"0" * 1500001)
    manifest["cases"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bounded"):
        load_source_manifest(p)
