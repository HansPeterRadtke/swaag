"""Conservative per-property perceptual acceptance policy.

A source check can fail only a property it authoritatively verifies. Vision
proposals are never silently promoted to validated pass/fail decisions.
"""
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    PASS = 'pass'
    FAIL = 'fail'
    REVIEW = 'review'


@dataclass(frozen=True)
class Observation:
    property_id: str
    check_id: str
    outcome: str  # pass | fail | inconclusive
    provenance: str  # verified_source | unverified_source | visual_proposal | human_reference
    reference_verified: bool = False
    actual_output_verified: bool = False
    measured_property: bool = False


def decide_property(observations: list[Observation]) -> Verdict:
    """Evaluate one explicitly named property; avoid universal image approval.

    Even a verified check only covers its own property. A human/reference
    observation is trusted only when its exact reference and final output
    were independently verified. Conflicting trusted observations need review.
    """
    if not observations:
        return Verdict.REVIEW
    ids = {o.property_id for o in observations}
    if len(ids) != 1 or any(not o.property_id or not o.check_id for o in observations):
        raise ValueError('Each decision needs a single named property and check identities')
    trusted = []
    for o in observations:
        if o.outcome not in ('pass', 'fail', 'inconclusive'):
            raise ValueError('Unknown outcome')
        if o.provenance not in ('verified_source', 'unverified_source', 'visual_proposal', 'human_reference'):
            raise ValueError('Unknown provenance')
        if (o.provenance in ('verified_source', 'human_reference')
                and o.reference_verified and o.actual_output_verified and o.measured_property
                and o.outcome != 'inconclusive'):
            trusted.append(o.outcome)
    if len(set(trusted)) > 1:
        return Verdict.REVIEW
    if trusted:
        return Verdict.FAIL if trusted[0] == 'fail' else Verdict.PASS
    return Verdict.REVIEW


@dataclass(frozen=True)
class RenderSourceBinding:
    """An exact *authoritative* source and render-state identity.

    The caller must obtain these fields from an independently verified source
    manifest and render-state capture, not image captions, OCR or guessed paths.
    A state includes such matters as the actual Activity/UI state or film frame.
    """
    source_kind: str
    source_revision: str
    subject_id: str
    state_id: str
    orientation: str
    width_px: int
    height_px: int


def _binding_complete(binding: RenderSourceBinding) -> bool:
    if not isinstance(binding, RenderSourceBinding):
        return False
    if any(not isinstance(x, str) or not x.strip() for x in
           (binding.source_kind, binding.source_revision, binding.subject_id,
            binding.state_id, binding.orientation)):
        return False
    return all(type(x) is int and x > 0 for x in (binding.width_px, binding.height_px))


def _actual_file_sha_matches(path, expected_sha256: str) -> bool:
    """Bounded-memory check of original source bytes; never fetch substitutes."""
    import hashlib
    from pathlib import Path
    if (not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(ch not in '0123456789abcdef' for ch in expected_sha256)):
        return False
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size <= 0:
            return False
        digest = hashlib.sha256()
        with p.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha256
    except (OSError, TypeError, ValueError):
        return False


def decide_bound_immutable_snapshot(
    *,
    approved_identity: RenderSourceBinding,
    actual_identity: RenderSourceBinding,
    approved_file,
    actual_file,
    expected_approved_sha256: str,
    expected_actual_sha256: str,
    reference_manifest_attested: bool,
    rendered_state_attested: bool,
    immutable_snapshot_contract: bool,
) -> Verdict:
    """Fail-closed decision on ONLY an exact byte-for-byte snapshot contract.

    Addresses a demonstrated source-binding failure: screenshots from a
    different Activity in the same app/viewport were all falsely accused by
    pixel differencing. Source/version/state match must be independently
    attested, and the actual bytes of BOTH artifacts must match their prior
    independently frozen SHA-256 receipts. A mismatched or unverifiable source,
    a permissible redraw or an unavailable file is REVIEW, never FAIL/PASS.

    When and only when exact-byte equivalence is explicitly the required
    property, equal bytes means PASS and changed bytes means FAIL for *that
    property*. This says nothing about semantic clipping, user intention,
    photorealism, or a naturally changing GUI. Not a live SWAAG tool.

    The manifest/observed state attestation MUST come from trusted source and
    runtime capture. Caller-supplied flags or filenames are not themselves a
    cryptographic identity proof; this function cannot authenticate them.
    """
    if (reference_manifest_attested is not True
            or rendered_state_attested is not True
            or immutable_snapshot_contract is not True):
        return Verdict.REVIEW
    if not _binding_complete(approved_identity) or not _binding_complete(actual_identity):
        return Verdict.REVIEW
    if approved_identity != actual_identity:
        return Verdict.REVIEW
    if not _actual_file_sha_matches(approved_file, expected_approved_sha256):
        return Verdict.REVIEW
    if not _actual_file_sha_matches(actual_file, expected_actual_sha256):
        return Verdict.REVIEW
    return (Verdict.PASS if expected_approved_sha256 == expected_actual_sha256
            else Verdict.FAIL)
