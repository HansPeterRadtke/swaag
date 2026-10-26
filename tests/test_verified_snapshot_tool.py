"""Live SWAAG runtime integration and persisted lineage of opt-in source binding.

Only checks exact immutable PNG *byte* snapshots under a trusted host attestor;
never tests general visual quality. Uses fresh temp-backed SWAAG sessions and
pure tools, with NO real external I/O or production runtime service changes.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import struct
import zlib

import pytest

from swaag.attachments import AttachmentStore
from swaag.perception.acceptance_policy import RenderSourceBinding
from swaag.runtime import AgentRuntime
from swaag.tools.registry import ToolRegistry
from swaag.tools.verified_snapshots import HostAttestedSnapshot, VerifyImmutableSnapshotTool


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag+data))


def _real_png(red: int) -> bytes:
    width, height = 3, 2
    raw = b"".join(b"\x00" + bytes((red, 31, 43)) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))
            + _chunk(b"IDAT",zlib.compress(raw)) + _chunk(b"IEND",b""))


def _identity(*, activity: str = "MainActivity", orientation: str = "portrait", width: int = 3) -> RenderSourceBinding:
    return RenderSourceBinding("original-android-screen", "pinned-source-release-7", "example.app",
                               activity, orientation, width, 2)


def _host_attestor(approved, actual, *, source=None, target=None, **changes):
    values = dict(approved_attachment_id=approved.attachment_id,
                  actual_attachment_id=actual.attachment_id,
                  approved_identity=source or _identity(), actual_identity=target or _identity(),
                  manifest_sha256="a"*64, reference_manifest_attested=True,
                  rendered_state_attested=True, immutable_snapshot_contract=True)
    return HostAttestedSnapshot(**{**values,**changes})


def _new_runtime(make_config, tmp_path):
    config = make_config(tools__enabled=["list_attachments", "verify_immutable_snapshot"])
    config.sessions.root = tmp_path / "only-research-sessions"
    return AgentRuntime(config, model_client=object())


def _attach_pair(runtime, session_id, *, mutation=False):
    raw = _real_png(143)
    reference = runtime.add_attachment(raw, original_name="approved.png",media_type="image/png",source="host-source-test",session_id=session_id)
    final = runtime.add_attachment(_real_png(144) if mutation else raw,original_name="actual.png",media_type="image/png",source="host-render-test",session_id=session_id)
    return reference, final


def _run(runtime, sid, reference, actual):
    return runtime.execute_tool_once("verify_immutable_snapshot", {"approved_attachment_id":reference.attachment_id,
                                                                    "actual_attachment_id":actual.attachment_id},session_id=sid)


def test_registered_not_enabled_or_capable_by_default(make_config, tmp_path):
    config = make_config()
    registry = ToolRegistry()
    assert registry.get("verify_immutable_snapshot").kind == "pure"
    assert "verify_immutable_snapshot" not in registry.tool_names(config)
    runtime = _new_runtime(make_config,tmp_path)
    state = runtime.create_or_load_session()
    assert "verify_immutable_snapshot" not in runtime.tools.tool_names(runtime.config)
    reference, actual = _attach_pair(runtime,state.session_id)
    attempt = _run(runtime,state.session_id,reference,actual)
    assert attempt.tool_result is None
    assert attempt.error and "required runtime capability" in attempt.error["error"]
    assert any(x.event_type=="tool_error" for x in runtime.history.read_history(state.session_id))


def test_real_persisted_tool_call_and_reload(make_config, tmp_path):
    runtime = _new_runtime(make_config,tmp_path)
    state=runtime.create_or_load_session()
    reference,actual=_attach_pair(runtime,state.session_id)
    runtime.bind_tool_runtime_capability(state.session_id,"immutable_snapshot_attestor",lambda r,a:_host_attestor(r,a))
    assert "verify_immutable_snapshot" in runtime.tools.tool_names(runtime.config,runtime_capabilities=runtime.tool_runtime_capabilities(state.session_id))
    pass_result=_run(runtime,state.session_id,reference,actual)
    assert pass_result.error is None
    assert pass_result.tool_result is not None
    output=pass_result.tool_result.output
    assert output["verdict"]=="pass"
    assert output["property_id"]=="exact_immutable_png_snapshot_bytes"
    assert output["approved_sha256"]==reference.sha256==output["actual_sha256"]
    assert output["universal_media_release_approved"] is False
    assert output["semantic_visual_quality_evaluated"] is False
    assert len(output["source_event_references"])==2
    assert all(x["hash"] and x["sequence"] for x in output["source_event_references"])
    history=runtime.history.read_history(state.session_id)
    assert any(x.event_type=="tool_result" and x.payload.get("tool_name")=="verify_immutable_snapshot" for x in history)

    # Reentry into persisted session is real: attachments, exact result/source
    # events survive reconstruction, but volatile trust intentionally does NOT.
    restarted = _new_runtime(make_config,tmp_path)
    reloaded=restarted.create_or_load_session(state.session_id)
    assert {a.attachment_id for a in reloaded.attachments}=={reference.attachment_id,actual.attachment_id}
    assert all(a.metadata.get("source_event_hash") for a in reloaded.attachments)
    assert not restarted.tool_runtime_capabilities(state.session_id)
    without_rebinding=_run(restarted,state.session_id,reference,actual)
    assert without_rebinding.error and without_rebinding.tool_result is None
    restarted.bind_tool_runtime_capability(state.session_id,"immutable_snapshot_attestor",lambda r,a:_host_attestor(r,a))
    confirmed=_run(restarted,state.session_id,reference,actual)
    assert confirmed.error is None and confirmed.tool_result.output["verdict"]=="pass"
    complete=restarted.history.rebuild_from_history(state.session_id,write_projections=False)
    assert any(m.name=="verify_immutable_snapshot" for m in complete.messages if m.role=="tool")
    assert len([x for x in restarted.history.read_history(state.session_id) if x.event_type=="tool_result" and x.payload.get("tool_name")=="verify_immutable_snapshot"])==2


def test_changed_final_output_fails_only_immutable_bytes_property(make_config,tmp_path):
    runtime=_new_runtime(make_config,tmp_path)
    s=runtime.create_or_load_session();ref,changed=_attach_pair(runtime,s.session_id,mutation=True)
    runtime.bind_tool_runtime_capability(s.session_id,"immutable_snapshot_attestor",lambda r,a:_host_attestor(r,a))
    run=_run(runtime,s.session_id,ref,changed)
    assert run.error is None
    assert run.tool_result.output["verdict"]=="fail"
    assert run.tool_result.output["reason"]=="verified_immutable_png_bytes_changed"
    assert run.tool_result.output["universal_media_release_approved"] is False


def test_wrong_activity_stale_attestation_and_nonimmutable_are_review(make_config,tmp_path):
    runtime=_new_runtime(make_config,tmp_path)
    s=runtime.create_or_load_session();ref,changed=_attach_pair(runtime,s.session_id,mutation=True)
    cases=[
        {"target":_identity(activity="SettingsActivity")},
        {"target":_identity(orientation="landscape")},
        {"target":_identity(width=4)},
        {"manifest_sha256":"not-a-sha"},
        {"actual_attachment_id":"different-attachment"},
        {"rendered_state_attested":False},
        {"reference_manifest_attested":False},
        {"immutable_snapshot_contract":False},
    ]
    for overrides in cases:
        runtime.bind_tool_runtime_capability(s.session_id,"immutable_snapshot_attestor",lambda r,a,opts=overrides:_host_attestor(r,a,**opts))
        run=_run(runtime,s.session_id,ref,changed)
        assert run.error is None
        assert run.tool_result.output["verdict"]=="review", overrides
    runtime.bind_tool_runtime_capability(s.session_id,"immutable_snapshot_attestor",lambda r,a:None)
    assert _run(runtime,s.session_id,ref,changed).tool_result.output["verdict"]=="review"


def test_corrupted_attachment_blob_returns_review_not_verified_fail(make_config,tmp_path):
    runtime=_new_runtime(make_config,tmp_path)
    s=runtime.create_or_load_session();ref,actual=_attach_pair(runtime,s.session_id,mutation=True)
    runtime.bind_tool_runtime_capability(s.session_id,"immutable_snapshot_attestor",lambda r,a:_host_attestor(r,a))
    store=AttachmentStore(runtime.config.sessions.root,max_upload_bytes=runtime.config.attachments.max_upload_bytes)
    p=store.path_for(actual)
    p.write_bytes(b"corrupted")  # ONLY under test temp session root
    run=_run(runtime,s.session_id,ref,actual)
    assert run.error is None
    assert run.tool_result.output["verdict"]=="review"
    assert run.tool_result.output["reason"]=="attachment_bytes_unavailable_or_corrupt"


def test_extra_untrusted_fields_are_not_accepted(make_config, tmp_path):
    runtime=_new_runtime(make_config,tmp_path)
    s=runtime.create_or_load_session();ref,actual=_attach_pair(runtime,s.session_id)
    runtime.bind_tool_runtime_capability(s.session_id,"immutable_snapshot_attestor",lambda r,a:_host_attestor(r,a))
    result=runtime.execute_tool_once("verify_immutable_snapshot", {"approved_attachment_id":ref.attachment_id,
                                 "actual_attachment_id":actual.attachment_id,"source_state":"MainActivity",
                                 "verified":True},session_id=s.session_id)
    assert result.tool_result is None
    assert result.error["error_type"]=="ToolValidationError"
    assert "attestation/state metadata" in result.error["error"]
