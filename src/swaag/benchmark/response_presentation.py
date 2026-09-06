from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.grammar import audio_rendering_contract
from swaag.runtime import AgentRuntime, _validated_presentation_text
from swaag.types import PromptArtifact, PromptComponent
from swaag.utils import sha256_text, stable_json_dumps


ORIGINAL_REQUEST = (
    "Give me the meaningful final state. Include the test count, whether deployment "
    "actually completed, the exact blocker, and the localhost service port. Do not "
    "include routine commit hashes or command transcripts. Also provide a version that "
    "is understandable when heard."
)

RAW_WORKER_RESULT = """The implementation work is complete.

| Check | Result |
| --- | --- |
| Unit tests | 42 passed |
| Deployment | Not deployed: systemd start requires interactive polkit authentication |
| Listener | Intended localhost port 13401; not listening until deployment succeeds |

Operational bookkeeping: commit 0123456789abcdef was pushed. I ran git diff --check and /data/var/swaag/venv/bin/python -m pytest -q. Do not claim the service is deployed while polkit remains blocked.
"""

STRATEGIES = ("visual_only", "staged_audio", "single_call_audio")
REQUIRED_MARKERS = (("42", "forty-two"), ("polkit",), ("13401",))
FORBIDDEN_NOISE = ("0123456789abcdef", "git diff --check", "pytest -q")

_COMBINED_SYSTEM = """You produce one final answer that performs both semantic user-facing relevance selection and audio-style rendering.

Preserve every result, caveat, blocker, uncertainty, requested detail, and necessary evidence from the exact source. Remove internal operational narration only when it is neither requested nor needed. At the same time, turn tables, dense lists, symbols, and visual cross-references into coherent prose that is understandable through text-to-speech. Do not invent evidence, change status, hide failures, or omit meaningful information. Return only one JSON object matching the supplied schema.
"""


def select_strategies(names: Iterable[str] = ()) -> list[str]:
    requested = list(names)
    unknown = sorted(set(requested) - set(STRATEGIES))
    if unknown:
        raise ValueError(
            "Unknown response-presentation strategy: " + ", ".join(unknown)
        )
    return requested or list(STRATEGIES)


def verify_presentation(strategy: str, text: str, evaluation: dict[str, Any]) -> dict:
    lowered = text.casefold()
    checks = {
        "required_information": all(
            any(marker in lowered for marker in alternatives)
            for alternatives in REQUIRED_MARKERS
        ),
        "operational_noise_removed": all(
            marker.casefold() not in lowered for marker in FORBIDDEN_NOISE
        ),
        "independent_evaluation": evaluation.get("acceptable") is True,
        "audio_layout": (
            True
            if strategy == "visual_only"
            else "|" not in text and "---" not in text
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _model_identity(runtime: AgentRuntime) -> Any:
    provider = getattr(runtime.client, "cache_identity", None)
    identity = provider() if callable(provider) else type(runtime.client).__name__
    return identity


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        stable_json_dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_config(
    base: AgentConfig,
    *,
    sessions_root: Path,
    workspace: Path,
) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.tools.read_roots = [workspace]
    config.model.cache_enabled = False
    config.tools.enabled = []
    config.tools.allow_stateful_tools = False
    config.tools.allow_side_effect_tools = False
    config.runtime.completion_evaluation_enabled = False
    return config


def _single_call_audio(runtime: AgentRuntime, state) -> tuple[str, dict[str, Any]]:
    user_text = (
        "Original user request, verbatim:\n"
        + ORIGINAL_REQUEST
        + "\n\nVerified worker result, exact and authoritative:\n"
        + RAW_WORKER_RESULT
        + "\n\nProduce one listenable, user-relevant answer in audio_text."
    )
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="audio_rendering",
        system_instruction=_COMBINED_SYSTEM,
        components=[
            PromptComponent(
                name="combined_presentation_task",
                category="turn_context",
                text=user_text,
            )
        ],
    )
    assembly.prompt_artifacts.append(
        PromptArtifact(
            source="benchmark:combined_relevance_audio_v1",
            sha256=sha256_text(_COMBINED_SYSTEM),
        )
    )
    payload = runtime._execute_compiled_presentation_call(
        state,
        assembly,
        audio_rendering_contract(),
        validator=lambda value: _validated_presentation_text(
            value,
            field_name="audio_text",
        ),
    )
    text = str(payload["audio_text"]).strip()
    evaluation = runtime._evaluate_response_presentation(
        state,
        mode="combined_relevance_audio",
        original_request=ORIGINAL_REQUEST,
        source_answer=RAW_WORKER_RESULT,
        candidate_answer=text,
    )
    return text, evaluation


def _last_generated_evaluation(runtime: AgentRuntime, session_id: str, mode: str) -> dict:
    for event in reversed(runtime.history.read_history(session_id)):
        if (
            event.event_type == "response_presentation_generated"
            and event.payload.get("mode") == mode
        ):
            evaluation = event.payload.get("evaluation")
            return dict(evaluation) if isinstance(evaluation, dict) else {}
    return {}


def run_response_presentation_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    strategy_names: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identity: Any | None = None,
) -> dict[str, Any]:
    base = config or load_config()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "response_presentation_results.json"
    selected = select_strategies(strategy_names)
    previous: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("planned_strategies") != selected:
            raise ValueError(
                "Response-presentation checkpoint strategy list does not match this run"
            )
        stored_results = previous.get("results", [])
        if not isinstance(stored_results, list):
            raise ValueError("Response-presentation checkpoint results are invalid")
        results = [dict(item) for item in stored_results]
        checkpoint_identity = previous.get("model_identity")
        if model_identity is not None and model_identity != checkpoint_identity:
            raise ValueError(
                "Response-presentation checkpoint model identity does not match this run"
            )
        if model_identity is None and len(results) == len(selected):
            probe_workspace = output_dir / "identity-probe" / "workspace"
            probe_workspace.mkdir(parents=True, exist_ok=True)
            probe_config = _case_config(
                base,
                sessions_root=output_dir / "identity-probe" / "sessions",
                workspace=probe_workspace,
            )
            model_identity = _model_identity(runtime_factory(probe_config))
            if model_identity != checkpoint_identity:
                raise ValueError(
                    "Response-presentation checkpoint model identity changed"
                )
        elif model_identity is None:
            model_identity = checkpoint_identity

    for index, strategy in enumerate(selected, start=1):
        if any(item.get("strategy") == strategy for item in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{strategy}"
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=False)
        config_for_case = _case_config(
            base,
            sessions_root=case_root / "sessions",
            workspace=workspace,
        )
        runtime = runtime_factory(config_for_case)
        current_identity = _model_identity(runtime)
        if model_identity is None:
            model_identity = current_identity
        elif current_identity != model_identity:
            raise ValueError("Response-presentation runtime model identity changed")
        state = runtime.create_or_load_session()
        started = time.monotonic()
        if strategy == "single_call_audio":
            presentation, evaluation = _single_call_audio(runtime, state)
        else:
            modes = {"visual"} if strategy == "visual_only" else {"audio"}
            payload = runtime.generate_response_presentations(
                state,
                original_request=ORIGINAL_REQUEST,
                assistant_message=RAW_WORKER_RESULT,
                modes=modes,
            )
            key = "visual" if strategy == "visual_only" else "audio"
            presentation = str(payload.get(key) or "")
            event_mode = (
                "response_relevance"
                if strategy == "visual_only"
                else "audio_rendering"
            )
            evaluation = _last_generated_evaluation(
                runtime,
                state.session_id,
                event_mode,
            )
        verification = verify_presentation(strategy, presentation, evaluation)
        events = runtime.history.read_history(state.session_id)
        results.append(
            {
                "strategy": strategy,
                "session_id": state.session_id,
                "elapsed_seconds": time.monotonic() - started,
                "presentation": presentation,
                "evaluation": evaluation,
                "verification": verification,
                "model_call_kinds": [
                    event.payload.get("kind")
                    for event in events
                    if event.event_type == "model_request_sent"
                ],
                "context_compilations": [
                    {
                        "sequence": event.sequence,
                        **dict(event.payload),
                    }
                    for event in events
                    if event.event_type == "context_compiled"
                ],
            }
        )
        report = {
            "benchmark": "response_presentation_strategies",
            "status": "running",
            "model_identity": model_identity,
            "planned_strategies": selected,
            "results": results,
            "passed": sum(item["verification"]["passed"] for item in results),
            "total": len(results),
            "complete": len(results) == len(selected),
        }
        if report["complete"]:
            report["status"] = "completed"
        _write_report(report_path, report)

    return {
        **previous,
        "benchmark": "response_presentation_strategies",
        "status": "completed" if len(results) == len(selected) else "running",
        "model_identity": model_identity,
        "planned_strategies": selected,
        "results": results,
        "passed": sum(item["verification"]["passed"] for item in results),
        "total": len(results),
        "complete": len(results) == len(selected),
    }
