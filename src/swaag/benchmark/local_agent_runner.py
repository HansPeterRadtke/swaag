from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from swaag.config import load_config
from swaag.fsops import write_text
from swaag.model import ModelClientError
from swaag.model_cache import build_model_client
from swaag.types import ContractSpec
from swaag.utils import stable_json_dumps

_ALLOWED_TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".json",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
class LocalAgentRunnerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LocalRunnerPolicy:
    candidate_file_limit: int = 2
    file_excerpt_char_limit: int = 900
    issue_prompt_char_limit: int = 1200
    completion_max_tokens: int = 192
    solver_max_attempts: int = 2
    summary_max_chars: int = 120
    find_max_chars: int = 800
    replace_max_chars: int = 1600


@contextlib.contextmanager

def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m swaag.benchmark.local_agent_runner")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--sessions-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--read-root", required=True)
    return parser


def _policy_from_config() -> LocalRunnerPolicy:
    benchmark_policy = load_config().external_benchmarks.agent_generation
    return _policy_from_agent_generation(benchmark_policy)


def _policy_from_agent_generation(benchmark_policy: object) -> LocalRunnerPolicy:
    return LocalRunnerPolicy(
        candidate_file_limit=int(getattr(benchmark_policy, "candidate_file_limit")),
        file_excerpt_char_limit=int(getattr(benchmark_policy, "file_excerpt_char_limit")),
        issue_prompt_char_limit=int(getattr(benchmark_policy, "issue_prompt_char_limit")),
        completion_max_tokens=int(getattr(benchmark_policy, "completion_max_tokens")),
        solver_max_attempts=int(getattr(benchmark_policy, "solver_max_attempts")),
        summary_max_chars=int(getattr(benchmark_policy, "summary_max_chars")),
        find_max_chars=int(getattr(benchmark_policy, "find_max_chars")),
        replace_max_chars=int(getattr(benchmark_policy, "replace_max_chars")),
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _run_capture(command: list[str], *, cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _text_file_candidates(workspace: Path, *, limit: int) -> list[str]:
    completed = _run_capture(["rg", "--files", "--hidden", "-g", "!.git"], cwd=workspace)
    if completed.returncode != 0:
        return []
    candidates: list[str] = []
    for line in completed.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        suffix = Path(rel).suffix.lower()
        if suffix and suffix not in _ALLOWED_TEXT_EXTENSIONS:
            continue
        candidates.append(rel)
        if len(candidates) >= limit:
            break
    return candidates


def _build_file_selection_contract(candidate_paths: list[str], *, policy: LocalRunnerPolicy) -> ContractSpec:
    schema = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string", "enum": candidate_paths},
            },
            "reason": {"type": "string"},
        },
        "required": ["paths", "reason"],
        "additionalProperties": False,
    }
    return ContractSpec(name="local_benchmark_file_selection", mode="json_schema", json_schema=schema)


def _build_file_selection_prompt(base_prompt: str, candidate_paths: list[str], *, policy: LocalRunnerPolicy) -> str:
    trimmed_prompt = base_prompt.strip()
    if len(trimmed_prompt) > policy.issue_prompt_char_limit:
        trimmed_prompt = trimmed_prompt[: policy.issue_prompt_char_limit].rstrip() + "\n...[truncated]"
    return "\n".join(
        [
            "Return one JSON object with keys paths and reason.",
            f"paths must contain 1 to {policy.candidate_file_limit} entries chosen exactly from the listed repository paths.",
            "Choose the file or files that you need to inspect before proposing an edit.",
            "Do not return paths that are not listed.",
            "",
            "Task:",
            trimmed_prompt,
            "",
            "Repository paths:",
            *candidate_paths,
        ]
    ).strip() + "\n"


def _select_candidate_files(
    workspace: Path,
    prompt: str,
    *,
    client: LlamaCppClient,
    policy: LocalRunnerPolicy,
) -> list[str]:
    manifest_limit = max(policy.candidate_file_limit, 200)
    manifest = _text_file_candidates(workspace, limit=manifest_limit)
    if not manifest:
        return []
    contract = _build_file_selection_contract(manifest, policy=policy)
    completion = client.complete(
        _build_file_selection_prompt(prompt, manifest, policy=policy),
        max_tokens=policy.completion_max_tokens,
        contract=contract,
        kind="file_selection",
        live_mode=True,
    )
    payload = _parse_json(completion.text)
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths or not all(isinstance(item, str) for item in raw_paths):
        raise LocalAgentRunnerError("Model file-selection response missing paths")
    if len(raw_paths) > policy.candidate_file_limit:
        raise LocalAgentRunnerError("Model selected too many candidate files")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > policy.summary_max_chars:
        raise LocalAgentRunnerError("Model file-selection response has invalid reason")
    if any(path not in manifest for path in raw_paths):
        raise LocalAgentRunnerError("Model selected a path outside the listed repository paths")
    selected = _dedupe(raw_paths)
    if not selected:
        raise LocalAgentRunnerError("Model did not select any listed file paths")
    return selected[: policy.candidate_file_limit]


def _file_excerpt(path: Path, *, excerpt_char_limit: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= excerpt_char_limit:
        return text
    return text[:excerpt_char_limit]


def _build_edit_contract(candidate_paths: list[str], *, policy: LocalRunnerPolicy) -> ContractSpec:
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "path": {"type": "string", "enum": candidate_paths},
            "find": {"type": "string"},
            "replace": {"type": "string"},
        },
        "required": ["summary", "path", "find", "replace"],
        "additionalProperties": False,
    }
    return ContractSpec(name="local_benchmark_edit", mode="json_schema", json_schema=schema)


def _parse_json(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LocalAgentRunnerError(f"Model returned invalid JSON: {text!r}") from exc
    if not isinstance(payload, dict):
        raise LocalAgentRunnerError(f"Model returned non-object payload: {payload!r}")
    return {str(key): value for key, value in payload.items()}


def _apply_edit(workspace: Path, *, relative_path: str, find: str, replace: str) -> Path:
    candidate = (workspace / relative_path).resolve()
    workspace_resolved = workspace.resolve()
    if not str(candidate).startswith(str(workspace_resolved) + os.sep) and candidate != workspace_resolved:
        raise LocalAgentRunnerError(f"Refusing to edit path outside workspace: {relative_path}")
    if not candidate.is_file():
        raise LocalAgentRunnerError(f"Model selected missing file: {relative_path}")
    original = candidate.read_text(encoding="utf-8", errors="replace")
    if find in original:
        updated = original.replace(find, replace, 1)
    else:
        raise LocalAgentRunnerError("Model find text did not appear exactly in the selected file")
    if updated == original:
        raise LocalAgentRunnerError("Model edit would not change the selected file")
    write_text(candidate, updated, encoding="utf-8")
    return candidate


def _validate_edit_payload(payload: dict[str, object], candidate_paths: list[str], *, policy: LocalRunnerPolicy) -> dict[str, str]:
    summary = payload.get("summary")
    path = payload.get("path")
    find = payload.get("find")
    replace = payload.get("replace")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > policy.summary_max_chars:
        raise LocalAgentRunnerError("Model edit response has invalid summary")
    if not isinstance(path, str) or path not in candidate_paths:
        raise LocalAgentRunnerError("Model edit response selected an unlisted path")
    if not isinstance(find, str) or not find or len(find) > policy.find_max_chars:
        raise LocalAgentRunnerError("Model edit response has invalid find text")
    if not isinstance(replace, str) or not replace or len(replace) > policy.replace_max_chars:
        raise LocalAgentRunnerError("Model edit response has invalid replacement text")
    return {"summary": summary, "path": path, "find": find, "replace": replace}


def _build_solver_prompt(
    base_prompt: str,
    contexts: list[tuple[str, str]],
    *,
    policy: LocalRunnerPolicy,
    failure: str | None = None,
) -> str:
    trimmed_prompt = base_prompt.strip()
    if len(trimmed_prompt) > policy.issue_prompt_char_limit:
        trimmed_prompt = trimmed_prompt[: policy.issue_prompt_char_limit].rstrip() + "\n...[truncated]"
    sections = [
        "Return one JSON object with the keys summary, path, find, and replace.",
        "summary is one short description of the chosen edit.",
        "path is the single file to edit and must match one listed candidate path exactly.",
        "find is the exact text snippet to replace from the chosen file excerpt.",
        "replace is the exact replacement text that should be written in place of find.",
        "Make exactly one best-effort concrete code edit in one listed file.",
        "Choose the smallest plausible change that addresses the issue.",
        "The `find` text must appear exactly as written in the chosen file excerpt.",
        "Do not invent file paths. Do not return explanations outside JSON.",
        "",
        "Issue:",
        trimmed_prompt,
    ]
    if failure:
        sections.extend(["", "Previous attempt failed:", failure])
    sections.append("")
    sections.append("Candidate files and excerpts:")
    for relative_path, excerpt in contexts:
        sections.extend([
            "",
            f"FILE: {relative_path}",
            "```text",
            excerpt,
            "```",
        ])
    return "\n".join(sections).strip() + "\n"


def _solve_with_structured_edit(
    workspace: Path,
    prompt: str,
    *,
    client: LlamaCppClient,
    policy: LocalRunnerPolicy | None = None,
) -> dict[str, str]:
    effective_policy = policy or _policy_from_config()
    candidate_paths = _select_candidate_files(
        workspace,
        prompt,
        client=client,
        policy=effective_policy,
    )
    if not candidate_paths:
        raise LocalAgentRunnerError("Unable to identify any candidate files in the benchmark workspace")
    contexts = [
        (
            relative_path,
            _file_excerpt(
                workspace / relative_path,
                excerpt_char_limit=effective_policy.file_excerpt_char_limit,
            ),
        )
        for relative_path in candidate_paths
    ]
    contract = _build_edit_contract(candidate_paths, policy=effective_policy)
    failure: str | None = None
    for _attempt in range(effective_policy.solver_max_attempts):
        completion = client.complete(
            _build_solver_prompt(prompt, contexts, policy=effective_policy, failure=failure),
            max_tokens=effective_policy.completion_max_tokens,
            contract=contract,
            kind="answer",
            live_mode=True,
        )
        try:
            payload = _parse_json(completion.text)
        except LocalAgentRunnerError as exc:
            failure = f"{exc} Return a shorter valid JSON object that still makes one real file edit."
            continue
        try:
            validated = _validate_edit_payload(payload, candidate_paths, policy=effective_policy)
            changed_path = _apply_edit(
                workspace,
                relative_path=validated["path"],
                find=validated["find"],
                replace=validated["replace"],
            )
        except LocalAgentRunnerError as exc:
            failure = str(exc)
            continue
        validated["edited_path"] = str(changed_path.relative_to(workspace))
        return validated
    raise LocalAgentRunnerError(failure or "Structured local benchmark solver did not produce an applicable edit")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    sessions_root = Path(args.sessions_root).expanduser().resolve()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    env = {
        "SWAAG__SESSIONS__ROOT": str(sessions_root),
        "SWAAG__TOOLS__READ_ROOTS": f'["{Path(args.read_root).expanduser().resolve()}"]',
        "SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS": "true",
        "SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS": "true",
    }
    with _pushd(workspace):
        config = load_config(env=env)
        client = build_model_client(
            config,
            request_metadata={"cache_scope": "local_benchmark_solver"},
        )
        policy = _policy_from_agent_generation(config.external_benchmarks.agent_generation)
        try:
            result = _solve_with_structured_edit(workspace, prompt, client=client, policy=policy)
        except (LocalAgentRunnerError, ModelClientError) as exc:
            raise SystemExit(str(exc)) from exc
    print(
        stable_json_dumps(
            {
                "session_id": args.session,
                "assistant_text": result.get("summary", ""),
                "tool_results": 0,
                "edited_path": result.get("edited_path", ""),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
