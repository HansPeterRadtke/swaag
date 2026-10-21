"""Indexed optional exact old user-message candidates for first speech.

This is navigation evidence, not a semantic history selector or a fresh user
instruction. A bounded SQLite read avoids the slower global FTS ranking over
large prompt/tool payloads. Missing candidates NEVER prove that an old fact or
constraint does not exist; the full orchestrator owns further retrieval.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any

from swaag.history import HistoryStore

# Search terms are navigation hints only, NOT intent classification. Excluding
# generic connector words reduces irrelevant matches without deleting sources.
_STOP = frozenset({
    'about', 'after', 'again', 'also', 'before', 'could', 'does', 'from',
    'have', 'just', 'latest', 'more', 'please', 'that', 'their', 'them',
    'then', 'there', 'these', 'this', 'those', 'what', 'when', 'where',
    'which', 'while', 'with', 'would', 'your', 'were', 'into',
    'aber', 'also', 'dass', 'diese', 'dieser', 'einem', 'einen', 'einer',
    'etwas', 'haben', 'jetzt', 'kann', 'mehr', 'nicht', 'noch', 'oder',
    'schon', 'sein', 'sind', 'über', 'wann', 'warum', 'welche', 'welcher',
})


def history_query_terms(query: str, *, limit: int = 10) -> list[str]:
    tokens = re.findall(r'[^\W_]+', query.casefold(), re.UNICODE)
    uniques: list[str] = []
    for token in tokens:
        if len(token) < 4 and not (token.isdigit() and len(token) >= 3):
            continue
        if token in _STOP or token in uniques:
            continue
        uniques.append(token)
    # Prefer longer, specific terms while retaining the source request order
    # for deterministic tie-breaking. No grammar, intent or relevance claims.
    chosen = set(sorted(uniques, key=lambda x: (-len(x), uniques.index(x)))[:limit])
    return [token for token in uniques if token in chosen]


def old_user_history_candidates(
    history: HistoryStore, *, session_id: str, request: str,
    before_sequence: int, indexed_through_required: int,
    max_results: int = 4, max_chars: int = 1600,
    wall_budget_seconds: float = 0.07,
) -> dict[str, Any]:
    """Read exact older user words, never truncating a governing sentence.

    Returns incomplete/timeout transparently. The SQL walks the existing
    (session_id,event_type,sequence) B-tree for only user-message events.
    Unlike global FTS it never ranks or decodes the huge model/tool corpus.
    """
    result: dict[str, Any] = {
        'search_scope': 'same-session exact older user-message candidates',
        'selection_basis': 'lexical overlap, NOT semantic relevance or active policy',
        'complete_semantic_selection': False,
        'query_complete': False, 'status': 'not_attempted', 'candidates': [],
    }
    if (not session_id or not isinstance(request, str) or not isinstance(before_sequence, int)
            or before_sequence <= 1 or max_results <= 0 or max_chars <= 0):
        return result
    terms = history_query_terms(request)
    if not terms:
        result['status'] = 'no_discriminative_terms'
        return result
    # SQL computes relevance across all older user messages; LIMIT applies
    # only *after* relevance ranking, not to an arbitrary recent source window.
    content = "lower(json_extract(payload_json,'$.message.content'))"
    scores = [f"CASE WHEN instr({content}, ?) > 0 THEN {min(3,len(word)//5 + 1)} ELSE 0 END"
              for word in terms]
    relevance = ' + '.join(scores)
    sql = f'''
        SELECT sequence, event_hash, timestamp,
               CASE WHEN length(payload_json) <= 12000 THEN payload_json ELSE NULL END
                   AS payload_json,
               length(json_extract(payload_json,'$.message.content')) AS content_chars,
               ({relevance}) AS match_score
        FROM events INDEXED BY events_session_type_sequence
        WHERE session_id=? AND event_type='message_added' AND sequence<?
              AND json_extract(payload_json,'$.message.role')='user'
              AND ({relevance}) > 0
        ORDER BY match_score DESC, sequence DESC LIMIT ?
    '''
    start = time.monotonic()
    try:
        with history._sqlite_connect() as connection:
            # Never spend the user's foreground latency budget waiting for an
            # optional old-history hint while a writer holds a SQLite lock.
            connection.execute('PRAGMA busy_timeout=40')
            indexed = connection.execute(
                'SELECT complete_through FROM indexed_sessions WHERE session_id=?',
                (session_id,),
            ).fetchone()
            if indexed is None or int(indexed[0]) < indexed_through_required:
                result['status'] = 'index_not_current'
                return result
            connection.set_progress_handler(
                lambda: 1 if time.monotonic() - start > wall_budget_seconds else 0,
                300,
            )
            params = [*terms, session_id, before_sequence, *terms, max_results * 3 + 1]
            rows = connection.execute(sql, params).fetchall()
        if time.monotonic() - start > wall_budget_seconds:
            result['status'] = 'time_budget_exceeded'
            return result
        result['query_complete'] = True  # The SQL matched all older user rows.
        result['status'] = 'ok'
        spent = 0
        for row in rows:
            if len(result['candidates']) >= max_results:
                break
            payload_text = row['payload_json']
            payload = json.loads(str(payload_text)) if payload_text is not None else {}
            message = payload.get('message', {})
            exact = message.get('content')
            if payload_text is not None and (not isinstance(exact, str) or not exact):
                continue
            size = int(row['content_chars'] or 0)
            item = {
                'sequence': int(row['sequence']),
                'source_event_hash': str(row['event_hash']),
                'captured_at': str(row['timestamp']),
                'role': 'historical_user',
                'match_score': int(row['match_score']),
                'chars': size,
                'verbatim': isinstance(exact, str) and size <= max_chars - spent,
            }
            if item['verbatim']:
                item['content'] = exact
                spent += len(exact)
            else:
                item['lookup_instruction'] = 'Retrieve this exact event in full orchestrator; no source text is included.'
            result['candidates'].append(item)
        result['candidate_limit_reached'] = len(rows) > max_results
        return result
    except (sqlite3.Error, ValueError, TypeError, OSError):
        result['status'] = 'index_query_unavailable'
        return result
