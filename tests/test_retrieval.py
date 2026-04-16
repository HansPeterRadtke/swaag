from __future__ import annotations

from dataclasses import asdict
import json

from swaag.context_builder import build_context
from swaag.events import create_event
from swaag.retrieval.embeddings import EmbeddingBackend, LlmScoringBackend, SemanticBackendProtocolError
from swaag.retrieval import HybridRetriever
from swaag.retrieval.query_builder import RetrievalIntent
from swaag.retrieval.ranker import RetrievalCandidate, rank_candidates
from swaag.tokens import ConservativeEstimator
from swaag.types import Message, SessionState

import pytest
import requests


class _RetrievalSemanticBackend(EmbeddingBackend):
    mode = "llm_scoring"
    degraded = False

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        lowered_query = query.lower()
        scores: list[float] = []
        for text in texts:
            lowered = text.lower()
            if (
                ("configuration parsing" in lowered or "config.py" in lowered)
                and ("settings parsing" in lowered_query or "config parser" in lowered_query)
            ):
                scores.append(0.97)
            elif "repair the defect in the parser" in lowered and "fix the bug in the parser" in lowered_query:
                scores.append(0.96)
            elif "threshold must be 17" in lowered and "report threshold" in lowered_query:
                scores.append(0.93)
            elif "older summary says threshold=17" in lowered and "report threshold" in lowered_query:
                scores.append(0.90)
            elif "auth/settings.py" in lowered and "login repair" in lowered_query:
                scores.append(0.92)
            else:
                scores.append(0.02)
        return scores


@pytest.fixture()
def _semantic_retrieval_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_factory = lambda *args, **kwargs: _RetrievalSemanticBackend()
    monkeypatch.setattr("swaag.retrieval.retriever.build_backend", backend_factory)
    monkeypatch.setattr("swaag.guidance.resolver.build_backend", backend_factory)
    monkeypatch.setattr("swaag.skills.selector.build_backend", backend_factory)


def _state_with_messages(*messages: Message) -> SessionState:
    return SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        messages=list(messages),
    )


def test_semantically_relevant_old_item_outranks_recent_irrelevant_item(make_config, _semantic_retrieval_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring", context_builder__max_history_messages=1)
    state = _state_with_messages(
        Message(role="user", content="Older note: the defect is in configuration parsing and the fix belongs in config.py", created_at="t1"),
        Message(role="user", content="Recent chatter about lunch and weather", created_at="t2"),
    )
    retriever = HybridRetriever(config)

    result = retriever.retrieve(
        state,
        counter=ConservativeEstimator(),
        goal="Fix the bug in settings parsing",
        current_step_text="repair config parser",
        environment_summary="",
        guidance_summary="",
    )

    assert [item.content for item in result.history_messages] == [
        "Older note: the defect is in configuration parsing and the fix belongs in config.py"
    ]


def test_semantic_synonyms_allow_lexical_mismatch_selection(make_config, _semantic_retrieval_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring", context_builder__max_history_messages=1)
    state = _state_with_messages(
        Message(role="user", content="Repair the defect in the parser", created_at="t1"),
        Message(role="user", content="Discuss colors only", created_at="t2"),
    )
    retriever = HybridRetriever(config)

    result = retriever.retrieve(
        state,
        counter=ConservativeEstimator(),
        goal="Fix the bug in the parser",
        current_step_text="",
        environment_summary="",
        guidance_summary="",
    )

    assert result.history_messages[0].content == "Repair the defect in the parser"


def test_full_history_retrieval_uses_non_message_events(make_config, _semantic_retrieval_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring", context_builder__max_history_messages=2)
    state = _state_with_messages(Message(role="user", content="current task", created_at="t2"))
    history_events = [
        create_event(
            sequence=1,
            session_id="s1",
            event_type="tool_result",
            payload={
                "tool_name": "read_text",
                "raw_input": {"path": "spec.txt"},
                "validated_input": {"path": "spec.txt"},
                "output": {"text": "The report threshold must be 17."},
            },
            prev_hash=None,
        ),
        create_event(
            sequence=2,
            session_id="s1",
            event_type="history_compressed",
            payload={
                "source_message_count": 4,
                "summary_message": asdict(Message(role="summary", content="Older summary says threshold=17", created_at="t0")),
                "summary_budget_report": {},
            },
            prev_hash="dummy",
        ),
    ]

    bundle = build_context(
        config,
        state,
        ConservativeEstimator(),
        goal="What is the report threshold?",
        history_events=history_events,
    )

    assert any(item.item_type == "history_event" and item.selected for item in bundle.selection_trace)


def test_full_history_retrieval_can_use_arbitrary_event_types(make_config, _semantic_retrieval_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring", context_builder__max_history_messages=1)
    state = _state_with_messages(Message(role="user", content="Recent chatter about lunch only.", created_at="t2"))
    history_events = [
        create_event(
            sequence=1,
            session_id="s1",
            event_type="working_memory_updated",
            payload={
                "working_memory": {
                    "active_goal": "repair login flow",
                    "current_step_title": "inspect auth settings",
                    "recent_results": ["auth/settings.py is relevant"],
                    "active_entities": ["auth/settings.py"],
                    "last_updated": "t1",
                },
                "reason": "event replay test",
            },
            prev_hash=None,
        ),
    ]

    bundle = build_context(
        config,
        state,
        ConservativeEstimator(),
        goal="Where does the login repair belong?",
        history_events=history_events,
    )

    assert any(
        item.item_type == "history_event" and item.selected and item.item_id == "event:1"
        for item in bundle.selection_trace
    )


def test_degraded_retrieval_mode_is_visible(make_config) -> None:
    config = make_config(retrieval__backend="degraded_lexical")
    state = _state_with_messages(Message(role="user", content="hello", created_at="t1"))

    bundle = build_context(config, state, ConservativeEstimator(), goal="hello")

    assert bundle.retrieval_mode == "degraded_lexical"
    assert bundle.retrieval_degraded is True
    assert any(item.degraded for item in bundle.selection_trace if item.item_type == "history_message")


def test_llm_scoring_retriever_shortlists_candidates_before_backend_call(make_config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_batch_sizes: list[int] = []

    class RecordingBackend(EmbeddingBackend):
        mode = "llm_scoring"
        degraded = False

        def score_query(self, query: str, texts: list[str]) -> list[float]:
            del query
            seen_batch_sizes.append(len(texts))
            return [1.0 / (index + 1) for index, _text in enumerate(texts)]

    monkeypatch.setattr("swaag.retrieval.retriever.build_backend", lambda *args, **kwargs: RecordingBackend())
    config = make_config(retrieval__backend="llm_scoring", retrieval__max_candidates_per_source=3)
    state = _state_with_messages(
        *(Message(role="user", content=f"message {index}", created_at=f"t{index}") for index in range(8))
    )

    retriever = HybridRetriever(config)
    retriever.retrieve(
        state,
        counter=ConservativeEstimator(),
        goal="Find the relevant message",
        current_step_text="",
        environment_summary="",
        guidance_summary="",
    )

    assert seen_batch_sizes == [3]


def test_ranker_uses_semantic_signal_as_primary_decision(make_config) -> None:
    class DummyBackend(EmbeddingBackend):
        mode = "dummy_semantic"
        degraded = False

        def score_query(self, query: str, texts: list[str]) -> list[float]:
            assert query == "repair authentication pipeline"
            return [0.94, 0.08]

    config = make_config()
    intent = RetrievalIntent(
        query_text="repair authentication pipeline",
        goal="repair authentication pipeline",
        current_step_text="",
        active_entities=[],
        unresolved_failures=[],
        environment_summary="",
        guidance_summary="",
        role_name="primary",
        purpose="execution",
        dependency_terms=["auth"],
        terms=["repair", "authentication", "pipeline"],
    )
    candidates = [
        RetrievalCandidate(
            item_type="history_message",
            item_id="message:0",
            source="messages",
            text="stabilize the login workflow in credentials.py",
            token_cost=12,
            payload=None,
            metadata={"recency": 0.1, "structural": 0.0},
        ),
        RetrievalCandidate(
            item_type="history_message",
            item_id="message:1",
            source="messages",
            text="authentication pipeline authentication pipeline authentication pipeline",
            token_cost=12,
            payload=None,
            metadata={"recency": 1.0, "structural": 1.0},
        ),
    ]

    ranked = rank_candidates(config, intent, candidates, DummyBackend())

    assert ranked[0].candidate.item_id == "message:0"
    assert ranked[0].signals["semantic_similarity"] > ranked[1].signals["semantic_similarity"]


def test_llm_scoring_backend_retries_timeout_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"content": json.dumps({"scores": [0.73]})}

    def _fake_post(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("slow response")
        return _Response()

    monkeypatch.setattr("requests.post", _fake_post)
    backend = LlmScoringBackend(
        base_url="http://example.test",
        sleep_func=lambda seconds: sleeps.append(seconds),
        max_unavailable_attempts=2,
    )

    scores = backend.score_query("repair config parser", ["config.py handles settings parsing"])

    assert scores == [0.73]
    assert sleeps == [1.0]


def test_llm_scoring_backend_rejects_trailing_text_despite_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"content": '{"scores": [0.73]}\n\nextra'}

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _Response())
    backend = LlmScoringBackend(base_url="http://example.test")

    with pytest.raises(SemanticBackendProtocolError, match="violated the requested schema"):
        backend.score_query("repair config parser", ["config.py handles settings parsing"])
