"""Opt-in, host-attested immutable PNG snapshot property check.

No model judges the pixels. An external trusted host must independently bind
source revision, exact Activity/state, dimensions and immutable-byte intent;
ordinary model/tool inputs cannot supply or replace that binding. The runtime
capability is deliberately not configured by default or persisted in sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Callable

from swaag.attachments import AttachmentStore, find_attachment
from swaag.perception.acceptance_policy import (
    RenderSourceBinding,
    Verdict,
    decide_bound_immutable_snapshot,
)
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import AttachmentReference, ToolExecutionResult
from swaag.utils import stable_json_dumps


CAPABILITY = "immutable_snapshot_attestor"
_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HostAttestedSnapshot:
    """A binding returned by trusted host code, never an LLM-provided dict.

    `manifest_sha256` is a host-attested external identity receipt; this tool
    validates its shape but not the issuer's authority. The host is responsible
    for binding an independently obtained real rendering state and source
    contract. The capability must be rebound after a process restart.
    """

    approved_attachment_id: str
    actual_attachment_id: str
    approved_identity: RenderSourceBinding
    actual_identity: RenderSourceBinding
    manifest_sha256: str
    approved_source_sha256: str
    actual_capture_sha256: str
    reference_manifest_attested: bool
    rendered_state_attested: bool
    immutable_snapshot_contract: bool


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read the PNG signature and IHDR size before stricter full decode."""
    try:
        with path.open("rb") as fp:
            header = fp.read(24)
    except OSError:
        return None
    if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[8:16] != b"\x00\x00\x00\x0dIHDR"):
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0 or width * height > 100_000_000:
        return None
    return width, height


def _png_full_decode(path: Path, expected_size: tuple[int, int]) -> bool:
    """Validate actual source pixels, CRCs and exact final IEND boundary.

    Pillow is an intentionally optional dependency for this opt-in tool. If it
    is absent, return REVIEW rather than trusting only a PNG filename/header.
    Use strict image load instead of an LLM/visual-quality heuristic.
    """
    if _png_dimensions(path) != expected_size:
        return False
    try:
        from PIL import Image
        with path.open("rb") as fp:
            fp.seek(-12, 2)
            if fp.read(12) != b"\x00\x00\x00\x00IEND\xaeB`\x82":
                return False
        # .verify() checks PNG chunk integrity, while .load() actually
        # decompresses pixels and detects missing/corrupt image data.
        with Image.open(path) as probe:
            if probe.format != "PNG" or probe.size != expected_size:
                return False
            probe.verify()
        with Image.open(path) as decoded:
            if decoded.format != "PNG" or decoded.size != expected_size:
                return False
            decoded.load()
        return True
    except Exception:  # noqa: BLE001 - malformed/unsupported images must abstain.
        # PIL raises several distinct decoder/CRC/decompression-bomb errors.
        # Never promote an unverified final image to snapshot PASS or FAIL.
        return False


class VerifyImmutableSnapshotTool(Tool):
    name = "verify_immutable_snapshot"
    description = (
        "Compare two persisted PNG attachments under a host-attested exact "
        "immutable screenshot contract, returning PASS/FAIL only for byte identity "
        "of the verified source and actual output; otherwise REVIEW."
    )
    usage_guidance = (
        "Requires an independently supplied host attestor, bound at runtime. "
        "Use only for an explicit immutable screenshot byte-equality contract. "
        "It does NOT judge clipped/overlapping text, contrast, semantics, "
        "user intent or the quality of 3D/video; never treat PASS as release approval. "
        "The model cannot supply source state, revision or attestation flags. "
        "Requires the optional Pillow decoder dependency when enabled."
    )
    kind = "pure"
    required_runtime_capability = CAPABILITY
    input_schema = {
        "type": "object",
        "properties": {
            "approved_attachment_id": {"type": "string"},
            "actual_attachment_id": {"type": "string"},
        },
        "required": ["approved_attachment_id", "actual_attachment_id"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict) -> dict:
        if not isinstance(raw_input, dict):
            raise ToolValidationError("verify_immutable_snapshot requires an object")
        unknown = set(raw_input) - {"approved_attachment_id", "actual_attachment_id"}
        if unknown:
            raise ToolValidationError("attestation/state metadata cannot be supplied as tool input")
        fields: dict[str, str] = {}
        for key in ("approved_attachment_id", "actual_attachment_id"):
            value = raw_input.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ToolValidationError(f"{key} must be a non-empty attachment ID <=256 chars")
            fields[key] = value.strip()
        return fields

    def execute(self, validated_input: dict, context: ToolContext) -> ToolExecutionResult:
        approved: AttachmentReference = find_attachment(
            context.session_state.attachments, validated_input["approved_attachment_id"]
        )
        actual: AttachmentReference = find_attachment(
            context.session_state.attachments, validated_input["actual_attachment_id"]
        )
        if (approved.media_type.lower() != "image/png"
                or actual.media_type.lower() != "image/png"):
            raise ToolValidationError("exact snapshot check requires PNG attachments")

        # The attached hash is itself durable in the session's attachment_added
        # history. Recompute the bytes on each invocation, do not trust captions.
        store = AttachmentStore(
            context.config.sessions.root,
            max_upload_bytes=context.config.attachments.max_upload_bytes,
        )
        source_events = []
        from swaag.tools.attachments import _source_references
        source_events.extend(_source_references(approved))
        source_events.extend(_source_references(actual))
        output = {
            "property_id": "exact_immutable_png_snapshot_bytes",
            "approved_attachment_id": approved.attachment_id,
            "actual_attachment_id": actual.attachment_id,
            "approved_sha256": approved.sha256,
            "actual_sha256": actual.sha256,
            "verdict": Verdict.REVIEW.value,
            "reason": "unverified_source_identity",
            "source_event_references": source_events,
            "source_state_attestation": "not_authenticated_by_this_tool",
            "semantic_visual_quality_evaluated": False,
            "universal_media_release_approved": False,
        }
        attestor: Callable = context.runtime_capabilities.get(CAPABILITY)
        try:
            attestation = attestor(approved, actual) if callable(attestor) else None
        except Exception:  # host attestation unavailability may not auto-reject content
            attestation = None
        if (not isinstance(attestation, HostAttestedSnapshot)
                or attestation.approved_attachment_id != approved.attachment_id
                or attestation.actual_attachment_id != actual.attachment_id
                or not isinstance(attestation.manifest_sha256, str)
                or not _SHA.fullmatch(attestation.manifest_sha256)
                or not isinstance(attestation.approved_source_sha256, str)
                or not _SHA.fullmatch(attestation.approved_source_sha256)
                or not isinstance(attestation.actual_capture_sha256, str)
                or not _SHA.fullmatch(attestation.actual_capture_sha256)
                or not isinstance(attestation.approved_identity, RenderSourceBinding)
                or not isinstance(attestation.actual_identity, RenderSourceBinding)):
            output["reason"] = "missing_or_invalid_host_attestation"
            return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))

        output["host_manifest_sha256"] = attestation.manifest_sha256
        # Unlike a self-reported attachment hash, the external host's expected
        # SOURCE bytes must come from its independent manifest and expected
        # TARGET bytes from its independent captured final output. Mismatches
        # cannot be treated as a semantic GUI failure or screenshot PASS.
        if (attestation.approved_source_sha256 != approved.sha256
                or attestation.actual_capture_sha256 != actual.sha256):
            output["reason"] = "host_source_or_capture_sha256_mismatch"
            return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))
        # If either blob is missing/corrupt or dimensions differ, this is
        # unverified target integrity, not evidence of an actual GUI defect.
        try:
            approved_path = store.path_for(approved)
            actual_path = store.path_for(actual)
        except (OSError, ValueError):
            output["reason"] = "attachment_bytes_unavailable_or_corrupt"
            return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))
        expected_approved_size = (attestation.approved_identity.width_px, attestation.approved_identity.height_px)
        expected_actual_size = (attestation.actual_identity.width_px, attestation.actual_identity.height_px)
        if (_png_dimensions(approved_path) != expected_approved_size
                or _png_dimensions(actual_path) != expected_actual_size):
            output["reason"] = "png_dimensions_or_header_do_not_match_attested_source"
            return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))
        if (not _png_full_decode(approved_path, expected_approved_size)
                or not _png_full_decode(actual_path, expected_actual_size)):
            output["reason"] = "png_final_pixels_or_end_boundary_not_verified"
            return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))

        verdict = decide_bound_immutable_snapshot(
            approved_identity=attestation.approved_identity,
            actual_identity=attestation.actual_identity,
            approved_file=approved_path,
            actual_file=actual_path,
            expected_approved_sha256=approved.sha256,
            expected_actual_sha256=actual.sha256,
            reference_manifest_attested=attestation.reference_manifest_attested,
            rendered_state_attested=attestation.rendered_state_attested,
            immutable_snapshot_contract=attestation.immutable_snapshot_contract,
        )
        output["verdict"] = verdict.value
        output["reason"] = (
            "attested_source_state_or_contract_unavailable"
            if verdict is Verdict.REVIEW
            else "verified_immutable_png_bytes_equal"
            if verdict is Verdict.PASS
            else "verified_immutable_png_bytes_changed"
        )
        output["source_state_attestation"] = (
            "host_asserted_matching_identity_and_bytes"
            if verdict is not Verdict.REVIEW else "not_authenticated_by_this_tool"
        )
        return ToolExecutionResult(self.name, output, stable_json_dumps(output, indent=2))


VERIFIED_SNAPSHOT_TOOLS = [VerifyImmutableSnapshotTool()]
