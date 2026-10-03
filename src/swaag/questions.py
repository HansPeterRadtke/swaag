"""Question lifecycle derived from canonical history, never a second authority."""
from __future__ import annotations

from typing import Any

RESOLUTIONS = ("answered_by_user", "answered_by_research", "superseded", "irrelevant", "expired")


def validate_question_capacity(state, questions, config) -> None:
    """Reject overflow before committing an action or executing its tools."""
    if not questions:
        return
    if len(state.open_questions) + len(questions) > config.runtime.max_open_questions:
        raise ValueError("Open-question capacity reached; resolve existing questions before adding more")
    existing = sum(len(str(item.get(key, ""))) for item in state.open_questions
                   for key in ("question", "reason", "assumption_if_unanswered"))
    added = sum(len(getattr(item, key)) for item in questions
                for key in ("question", "reason", "assumption_if_unanswered"))
    if existing + added > config.runtime.max_open_question_chars:
        raise ValueError("Open-question character budget reached; resolve existing questions first")


def question_record(event) -> dict[str, Any]:
    return normalize_question({"question_id": event.id, "source_event_sequence": event.sequence,
            "created_at": event.timestamp,
            **{key: event.payload[key] for key in
               ("question", "criticality", "reason", "assumption_if_unanswered")},
            **({"importance": event.payload["importance"]} if "importance" in event.payload else {})})


IMPORTANCE_RANK = {"minor": 1, "normal": 2, "major": 3, "critical": 4}
QUESTION_FIELDS = ("question", "criticality", "importance", "reason", "assumption_if_unanswered")
REVISION_FIELDS = (*QUESTION_FIELDS, "question_id", "expected_revision", "revision_reason")


def normalize_question(item):
    return {"revision": 0, "updated_at": item.get("created_at"),
            "importance": "critical" if item.get("criticality") == "blocking" else "normal", **item}


def validate_revision(payload):
    if not isinstance(payload, dict) or set(payload) != set(REVISION_FIELDS):
        raise ValueError("Question revision requires exactly its declared fields")
    result = dict(payload)
    for key in ("question_id", "question", "reason", "revision_reason"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Question revision requires nonempty {key}")
        result[key] = result[key].strip()
    revision = result["expected_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer")
    if result["criticality"] not in ("optional", "blocking") or result["importance"] not in IMPORTANCE_RANK:
        raise ValueError("Invalid question criticality or importance")
    assumption = result["assumption_if_unanswered"]
    if not isinstance(assumption, str):
        raise ValueError("assumption_if_unanswered must be a string")
    result["assumption_if_unanswered"] = assumption.strip()
    if (result["criticality"] == "optional") != bool(assumption.strip()):
        raise ValueError("Optional questions need an explicit provisional assumption; blocking questions need an empty assumption")
    return result


def revision_event(state, payload, config, *, actor):
    revision = validate_revision(payload)
    current = next((normalize_question(item) for item in state.open_questions
                    if item['question_id'] == revision['question_id']), None)
    if current is None:
        raise ValueError("Unknown or already resolved question")
    if current['revision'] != revision['expected_revision']:
        raise ValueError("Stale question revision; read the current question before revising")
    chars = sum(len(str(item.get(key, ''))) for item in state.open_questions
                if item['question_id'] != current['question_id']
                for key in ('question', 'reason', 'assumption_if_unanswered'))
    chars += sum(len(revision[key]) for key in ('question', 'reason', 'assumption_if_unanswered'))
    if chars > config.runtime.max_open_question_chars or len(revision['revision_reason']) > config.runtime.max_open_question_chars:
        raise ValueError("Question revision exceeds configured character budget")
    return {**revision, 'revision': current['revision'] + 1, 'actor': actor}


def apply_revision_controls(runtime, state):
    """Apply typed edits only while the caller owns the session execution lock."""
    import json
    controls = [item for item in runtime.history.list_pending_control_messages(state.session_id)
                if item.get('source') == 'question_revision']
    if not controls:
        return []
    blocking_edits = []
    # A crash after the event commit but before acknowledging its inbox item must
    # not apply the edit twice or turn a successful edit into a stale rejection.
    ids = {item['control_id'] for item in controls}
    committed = {event.payload.get('control_id') for event in runtime.history.iter_history_reverse(state.session_id,
                    event_types=('agent_question_revised', 'agent_question_revision_rejected'))
                 if event.payload.get('control_id') in ids}
    for control in controls:
        control_id = control['control_id']
        if control_id not in committed:
            try:
                command = json.loads(control['message'])
                payload = revision_event(state, command['revision'], runtime.config, actor=command['actor'])
                runtime.history.record_event(state, 'agent_question_revised', {**payload, 'control_id': control_id})
                if payload['criticality'] == 'blocking':
                    blocking_edits.append(payload)
            except (ValueError, KeyError, TypeError) as exc:
                runtime.history.record_event(state, 'agent_question_revision_rejected',
                    {'control_id': control_id, 'reason': str(exc)})
        runtime.history.mark_control_message_processed(state.session_id, control_id)
    return blocking_edits


def question_inventory(manager):
    """Complete exact inventory across configured worker stores, with no top-N cut."""
    from swaag.progress import worker_progress
    questions, pending, workers, seen = [], [], [], set()
    for route, candidate in manager.worker_managers.items():
        for worker in candidate.list(include_archived=True):
            identity = (str(candidate.runtime.config.sessions.root), worker.worker_id)
            if identity in seen:
                continue
            seen.add(identity)
            owner = manager.worker_managers.get(worker.model_key, candidate)
            history = owner.runtime.history
            state = history.rebuild_from_history(worker.session_id, write_projections=False)
            workers.append({'worker_id': worker.worker_id, 'session_id': worker.session_id,
                            'model_key': worker.model_key, 'status': worker.status,
                            'objective': worker.objective, 'progress': worker_progress(worker, state),
                            'through_sequence': state.event_count})
            for item in state.open_questions:
                questions.append({**normalize_question(item), 'worker_id': worker.worker_id,
                                  'session_id': worker.session_id, 'model_key': worker.model_key})
            for control in history.list_pending_control_messages(worker.session_id):
                if control.get('source') == 'question_revision':
                    pending.append({'worker_id': worker.worker_id, **control})
    questions.sort(key=lambda item: (-int(item['criticality'] == 'blocking'),
        -IMPORTANCE_RANK[item['importance']], item['created_at'], item['question_id']))
    return {'complete': True, 'scope': 'all configured worker stores including archived workers',
            'questions': questions, 'pending_revisions': pending, 'workers': workers}
