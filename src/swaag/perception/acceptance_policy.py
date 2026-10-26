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
