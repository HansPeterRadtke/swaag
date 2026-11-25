"""Offline, synthetic negative tests for the real W99 evidence adjudicator."""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/demos"))
try:
    import verify_integrated_manual_voice_evidence as audit
finally:
    sys.path.pop(0)


def trace():
    t, words, reply = "test-turn", "recorded words", "test answer"
    hello = lambda gen: {"type": "hello", "connection_generation": gen,
                         "unverified_speaker_requires_review": True}
    asr = {"type": "asr", "event": "final", "turn_id": t, "text": words,
           "auto_send": False}
    audio = {"type": "audio", "event": "start", "turn_id": t,
             "stream_id": "first", "generation": 0}
    last = {"turn_id": t, "assistant": reply, "audio_completed": True,
            "status": "completed"}
    return t, words, reply, [
        hello(1), {"type": "history", "turns": []}, asr,
        hello(2), {"type": "history", "turns": [
            {"turn_id": t, "status": "draft", "recognized_text": words,
             "assistant": ""}]},
        dict(asr, restored=True),
        {"type": "turn", "event": "submitted", "automatic": False,
         "turn_id": t, "text": words},
        audio, hello(3),
        dict(audio, stream_id="second", resumed=True),
        dict(audio, event="end", stream_id="second"),
        {"type": "history", "turns": [last]},
    ]


def test_exact_normal_recovery_trace():
    turn, words, reply, events = trace()
    assert audit.audit_events(events, turn, words, reply) == "second"


@pytest.mark.parametrize("change", [
    lambda e: e[3].update(unverified_speaker_requires_review=False),
    lambda e: e[5].update(text="other transcription"),
    lambda e: e[6].update(automatic=True),
    lambda e: e.insert(7, copy.deepcopy(e[6])),
    lambda e: e[9].update(resumed=False),
    lambda e: e[10].update(stream_id="wrong"),
    lambda e: e[11]["turns"][0].update(audio_completed=False),
    lambda e: e.insert(2, {"type": "turn_proposal"}),
    lambda e: e.insert(6, {"type": "error"}),
    lambda e: e[4]["turns"][0].update(assistant="premature response"),
])
def test_mutated_trace_rejected(change):
    turn, words, reply, events = trace()
    change(events)
    with pytest.raises(ValueError):
        audit.audit_events(events, turn, words, reply)


def test_pinned_evidence_refuses_tampering_and_links(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    f = root / "data"
    f.write_bytes(b"genuine")
    assert audit.verified_file(root, "data", audit.sha(b"genuine")) == b"genuine"
    f.write_bytes(b"forged!")
    with pytest.raises(ValueError, match="altered"):
        audit.verified_file(root, "data", audit.sha(b"genuine"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"test")
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="external"):
        audit.verified_file(root, "link/secret", audit.sha(b"test"))


def test_framed_media_rejects_partial_or_extra_bytes():
    good = b"\x00\x00\x00\x04" + b"\x01\x02\x03\x04" + b"\x00"*4
    assert audit.decode_stream(good) == b"\x01\x02\x03\x04"
    for raw in (b"", good[:-1], good+b"extra", b"\x00"*4):
        with pytest.raises(ValueError):
            audit.decode_stream(raw)
