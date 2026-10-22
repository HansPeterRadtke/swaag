"""Tiny evidence-bound guard over generated first speech, not an intent router.

The user-facing model makes the semantic answer/delegation decision. These
mechanical checks repair unsupported *claims in its proposed speech* that we
can disprove from the input boundary: a received transcript does not prove
microphone/audio playback, and a proposed delegation cannot already have run
any tool. They never start actions or grant tool permissions.
"""
from __future__ import annotations

import re

_UNPERFORMED_ACTION = re.compile(
    r"\b(?:I(?:\s+am|'m)\s+unable\s+to|I\s+cannot|I\s+can't|"
    r"unable\s+to|cannot)\s+(?:check|inspect|verify|run|access|read|retrieve|"
    r"look\s+up|start|stop|modify|edit|write|delete)\b", re.IGNORECASE,
)
_UNSUPPORTED_AUDIO = re.compile(
    r"\b(?:I\s+(?:can\s+)?hear\s+you|I\s+heard\s+(?:you|your\s+voice)|"
    r"your\s+microphone\s+(?:works|is\s+working)|"
    r"your\s+(?:audio|speaker)\s+(?:played|is\s+playing|works)|"
    r"ich\s+(?:kann\s+)?dich\s+hören|ich\s+höre\s+dich|"
    r"dein\s+mikrofon\s+funktioniert)\b", re.IGNORECASE,
)
_PRE_EFFECT_CLAIM = re.compile(
    r"^(?:\s*(?:Now\s+)?(?:Stopping|Starting|Deleting|Changing|Updating|"
    r"Running|Checking|Executing|Canceling|Cancelling|Submitting|"
    r"Resetting|Restarting|Saving|Deploying|Publishing|Uploading|Downloading|"
    r"Opening|Closing|Creating|Moving|Removing|Sending|Writing)\b|"
    r"\s*Successfully\s+(?:started|stopped|deleted|completed|updated|run|submitted)\b|"
    r"\s*I\s+(?:(?:have|already|just)\s+(?:started|stopped|deleted|completed|"
    r"updated|run|ran|executed|submitted)|did\s+that)\b|"
    r"\s*(?:The\s+)?(?:job|task|worker|plan|request)\s+(?:was|has\s+been)\s+"
    r"(?:just\s+)?(?:started|stopped|deleted|completed|submitted|cancelled|canceled)\b)",
    re.IGNORECASE,
)
_CONTRADICTED_ACTIVE_STATUS = re.compile(
    r"\b(?:there\s+(?:are|is)\s+no\s+active\s+workers?|"
    r"no\s+active\s+workers?\s+(?:are\s+)?left|"
    r"(?:the\s+)?(?:[\w-]+\s+){0,3}worker\b[^.!?\n]{0,140}"
    r"\b(?:has\s+(?:finished|stopped|completed)|is\s+(?:finished|stopped|completed)|"
    r"is\s+no\s+longer\s+(?:active|running|working)))", re.IGNORECASE,
)
_DIRECT_MIC_CHECK = frozenset({
    "can you hear me", "can you hear me now", "do you hear me",
    "kannst du mich hören", "hörst du mich",
})
_WORD_COUNT = re.compile(
    r"^\s*(?:please\s+)?reply\s+with\s+exactly\s+(\d{1,3}|[a-z-]+)\s+words?\s*:",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_DELEGATED_UNCERTAIN = "I'll check the available evidence before reporting a result."
_DELEGATED_INTENTION = "I'll review the request and report what I can verify."
_RECEIPT_ONLY = (
    "I received your message, but that alone does not verify microphone input "
    "or audio playback."
)
_RECEIPT_ONLY_DE = (
    "Deine Nachricht ist bei SWAAG angekommen. Das bestätigt weder den Mikrofoneingang "
    "noch die Audiowiedergabe."
)
_CURRENT_WORKER_CLAIM_WITHOUT_WORKER = re.compile(
    r"\b(?:the|my|your|a|an)\s+(?:[\w-]+\s+){0,3}worker\s+"
    r"(?:is|has\s+been)\s+(?:currently\s+)?(?:working|running|active)\b",
    re.IGNORECASE,
)

# This is a claim *in proposed speech*, not parsing the user's intent. Only
# count a narrowly recognizable present-tense inventory assertion, never an
# estimate based on a historical assistant utterance or absent observations.
_LIVE_ACTIVE_WORKER_COUNT_CLAIM = re.compile(
    r"^\s*(?:there\s+(?:are|is)\s+|(?:we|i)\s+have\s+)?"
    r"(?P<count>no|zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|\d{1,5})\s+active\s+workers?\b",
    re.IGNORECASE,
)


def normalize_fast_foreground_claim(
    mode: str, reply: str, authoritative_message: str,
    *, current_world: dict | None = None,
) -> tuple[str, str]:
    """Enforce a narrow evidential contract; preserve all other model prose.

    All text inputs must be exact strings and modes validated by the caller.
    A reply conflicting with evidence is conservatively delegated or replaced
    by a fact guaranteed by receiving the finalized message. It is never
    interpreted as a tool instruction. A recognized exact word count can be
    checked mechanically without a second model call.
    """
    if mode not in ("answer", "delegate"):
        raise ValueError("fast reply mode must be answer or delegate")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("fast reply text must be nonempty")
    if not isinstance(authoritative_message, str):
        raise TypeError("authoritative message must be text")
    reply = reply.strip()
    if mode == "delegate":
        # The fast layer has deliberately *not* inspected worker files, tool
        # permissions or client audio. Premature refusals or requests for the
        # user's file access are not justified: the full controller owns tools
        # and exact history and will raise any genuine missing prerequisite.
        if (_PRE_EFFECT_CLAIM.search(reply)
                or _UNPERFORMED_ACTION.search(reply)
                or re.search(r"\bplease\s+(?:provide|upload|send|share)\b",reply,re.IGNORECASE)
                or re.search(r"\bI\s+(?:do\s+not|don't)\s+have\s+access\b",reply,re.IGNORECASE)):
            return mode, _DELEGATED_INTENTION
        return mode, reply
    # No device/audio input receipt is available to this text-only boundary.
    if _UNSUPPORTED_AUDIO.search(reply):
        normalized = re.sub(r"[?!.\s]+$", "", authoritative_message.strip().lower())
        if normalized in _DIRECT_MIC_CHECK:
            return "answer", (_RECEIPT_ONLY_DE if normalized in {
                "kannst du mich hören", "hörst du mich"
            } else _RECEIPT_ONLY)
        return "delegate", _DELEGATED_UNCERTAIN
    if _UNPERFORMED_ACTION.search(reply):
        return "delegate", _DELEGATED_UNCERTAIN
    worker_claim = _LIVE_ACTIVE_WORKER_COUNT_CLAIM.match(reply)
    live_world = (current_world if isinstance(current_world, dict)
                  and current_world.get("snapshot_complete") is True else None)
    if live_world is None:
        # A fast model might confidently report an old or imaginary inventory
        # despite being told the source was unavailable. Do not bless an
        # explicit present-tense count or worker-is-running claim as observed.
        # Past-tense quotations remain the semantic model's responsibility.
        if worker_claim or _CURRENT_WORKER_CLAIM_WITHOUT_WORKER.search(reply):
            return "delegate", _DELEGATED_UNCERTAIN
    else:
        workers = live_world.get("workers")
        count = live_world.get("active_worker_count")
        # A snapshot_complete label is not itself evidence: the compact world
        # must have a well-formed inventory and a matching independently
        # exposed count. Fail closed only for live worker claims; ordinary
        # conceptual or historical answers do not depend on this inventory.
        valid_inventory = (
            isinstance(workers, list)
            and type(count) is int
            and count == len(workers)
            and all(isinstance(w, dict) and isinstance(w.get("status"), str)
                    and bool(w["status"]) for w in workers)
        )
        if (worker_claim or _CURRENT_WORKER_CLAIM_WITHOUT_WORKER.search(reply)) and not valid_inventory:
            return "delegate", _DELEGATED_UNCERTAIN
        if isinstance(workers, list):
            if worker_claim:
                raw = worker_claim.group("count").lower()
                claimed = int(raw) if raw.isdigit() else (
                    0 if raw in {"zero", "no"} else _NUMBER_WORDS[raw]
                )
                explicit_count = live_world.get("active_worker_count")
                # The compact worker-world builder produces the count from
                # exactly these rows. If that invariant is broken, never
                # certify the generated claim against contradictory evidence.
                if (claimed != len(workers)
                        or (type(explicit_count) is int and explicit_count != len(workers))):
                    return "delegate", _DELEGATED_UNCERTAIN
            # These are narrow factual contradiction checks on output, not a
            # semantic intent router: they can only reject an unsupported
            # first-stage answer, never execute a tool or mutate worker state.
            visibly_working = any(
                isinstance(w, dict) and w.get("status") in {"working", "running"}
                for w in workers
            )
            if (workers and _CONTRADICTED_ACTIVE_STATUS.search(reply)) or (
                not visibly_working and _CURRENT_WORKER_CLAIM_WITHOUT_WORKER.search(reply)
            ):
                return "delegate", _DELEGATED_UNCERTAIN
    strict = _WORD_COUNT.match(authoritative_message)
    if strict:
        raw = strict.group(1).lower()
        expected = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)
        if expected is None or len(reply.split()) != expected:
            return "delegate", _DELEGATED_INTENTION
    return "answer", reply
