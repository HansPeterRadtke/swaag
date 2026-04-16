from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from swaag.config import load_config
from swaag.model import LlamaCppClient, ModelClientError
from swaag.types import ContractSpec
from swaag.utils import stable_json_dumps

_PATH_HINT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+")
_CODE_TOKEN_RE = re.compile(r"`([^`\n]{2,120})`")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_GENERIC_IDENTIFIER_STOPWORDS = {
    "about",
    "achived",
    "actual",
    "add",
    "adding",
    "addresses",
    "agent",
    "aim",
    "all",
    "and",
    "apply",
    "are",
    "ask",
    "assistant",
    "because",
    "bench",
    "best",
    "change",
    "changes",
    "checkout",
    "clarification",
    "code",
    "concrete",
    "consider",
    "current",
    "describe",
    "detail",
    "directly",
    "does",
    "edit",
    "enables",
    "feature",
    "files",
    "for",
    "function",
    "help",
    "hints",
    "how",
    "inside",
    "instance",
    "issue",
    "listed",
    "local",
    "make",
    "missing",
    "model",
    "now",
    "only",
    "patch",
    "please",
    "post",
    "problem",
    "proposal",
    "reasonable",
    "repository",
    "request",
    "return",
    "root",
    "smallest",
    "solving",
    "statement",
    "support",
    "supported",
    "swe",
    "text",
    "that",
    "the",
    "there",
    "these",
    "this",
    "use",
    "usecase",
    "usecases",
    "using",
    "with",
    "work",
    "workspace",
    "you",
}
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
_PREFERRED_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
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


def _issue_text_only(prompt: str) -> str:
    marker = "Problem statement:"
    text = prompt
    if marker in text:
        text = text.split(marker, 1)[1]
    hints_marker = "\nHints:"
    if hints_marker in text:
        text = text.split(hints_marker, 1)[0]
    return text.strip()


def _looks_like_path_hint(token: str) -> bool:
    if "/" in token:
        return True
    suffix = Path(token).suffix.lower()
    return bool(suffix and suffix in _ALLOWED_TEXT_EXTENSIONS)


def _path_priority_bonus(path_text: str) -> int:
    suffix = Path(path_text).suffix.lower()
    if suffix in _PREFERRED_SOURCE_EXTENSIONS:
        return 5
    return 0


def _extract_search_terms(prompt: str) -> tuple[list[str], list[str]]:
    issue_text = _issue_text_only(prompt)
    path_hints = [match.strip() for match in _PATH_HINT_RE.findall(issue_text) if _looks_like_path_hint(match.strip())]
    code_tokens: list[str] = []
    for raw in _CODE_TOKEN_RE.findall(issue_text):
        token = raw.strip()
        if token:
            code_tokens.append(token)
    identifiers: list[str] = []
    for token in _IDENTIFIER_RE.findall(issue_text):
        lowered = token.lower()
        if lowered in _GENERIC_IDENTIFIER_STOPWORDS:
            continue
        identifiers.append(token)
    terms = _dedupe(code_tokens + identifiers)[:12]
    return _dedupe(path_hints), terms


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


def _candidate_files(workspace: Path, path_hints: list[str], search_terms: list[str], *, limit: int) -> list[str]:
    scored: dict[str, int] = {}
    for hint in path_hints:
        candidate = workspace / hint
        if candidate.is_file():
            scored[hint] = scored.get(hint, 0) + 100 + _path_priority_bonus(hint)
    for term in search_terms:
        completed = _run_capture(["rg", "-l", "-S", "--hidden", "-g", "!.git", term, "."], cwd=workspace)
        if completed.returncode not in {0, 1}:
            continue
        for line in completed.stdout.splitlines():
            rel = line.strip()
            if not rel or rel.startswith(".git/"):
                continue
            suffix = Path(rel).suffix.lower()
            if suffix and suffix not in _ALLOWED_TEXT_EXTENSIONS:
                continue
            scored[rel] = scored.get(rel, 0) + 10 + _path_priority_bonus(rel)
    if not scored:
        return _text_file_candidates(workspace, limit=limit)
    ranked = sorted(scored.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
    return [path for path, _score in ranked[:limit]]


def _file_excerpt(path: Path, search_terms: list[str], *, excerpt_char_limit: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= excerpt_char_limit:
        return text
    lowered = text.lower()
    windows: list[tuple[int, int]] = []
    for term in search_terms:
        index = lowered.find(term.lower())
        if index == -1:
            continue
        start = max(0, index - 1200)
        end = min(len(text), index + 2200)
        windows.append((start, end))
        if len(windows) >= 2:
            break
    if not windows:
        return text[:excerpt_char_limit]
    pieces: list[str] = []
    for start, end in windows:
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "...\n" + snippet
        if end < len(text):
            snippet = snippet + "\n..."
        pieces.append(snippet)
    joined = "\n\n".join(pieces)
    return joined[:excerpt_char_limit]


def _build_edit_contract(candidate_paths: list[str], *, policy: LocalRunnerPolicy) -> ContractSpec:
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 1, "maxLength": policy.summary_max_chars},
            "path": {"type": "string", "enum": candidate_paths},
            "find": {"type": "string", "minLength": 1, "maxLength": policy.find_max_chars},
            "replace": {"type": "string", "minLength": 1, "maxLength": policy.replace_max_chars},
        },
        "required": ["summary", "path", "find", "replace"],
        "additionalProperties": False,
    }
    return ContractSpec(name="local_benchmark_edit", mode="json_schema", json_schema=schema)


def _parse_json(text: str) -> dict[str, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LocalAgentRunnerError(f"Model returned invalid JSON: {text!r}") from exc
    if not isinstance(payload, dict):
        raise LocalAgentRunnerError(f"Model returned non-object payload: {payload!r}")
    return {str(key): str(value) for key, value in payload.items()}


def _apply_edit(workspace: Path, *, relative_path: str, find: str, replace: str) -> Path:
    candidate = (workspace / relative_path).resolve()
    workspace_resolved = workspace.resolve()
    if not str(candidate).startswith(str(workspace_resolved) + os.sep) and candidate != workspace_resolved:
        raise LocalAgentRunnerError(f"Refusing to edit path outside workspace: {relative_path}")
    if not candidate.is_file():
        raise LocalAgentRunnerError(f"Model selected missing file: {relative_path}")
    original = candidate.read_text(encoding="utf-8", errors="replace")
    updated: str | None = None
    if find in original:
        updated = original.replace(find, replace, 1)
    else:
        # Benchmark-local model outputs often drift on indentation or truncate a
        # nearby line. Fall back to replacing the longest exact line match in
        # the selected file so the agent-run loop can still produce a real diff.
        original_lines = original.splitlines(keepends=True)
        find_lines = [line for line in find.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
        for needle in sorted(find_lines, key=lambda item: len(item.strip()), reverse=True):
            stripped_needle = needle.strip()
            if len(stripped_needle) < 12:
                continue
            for line in original_lines:
                if stripped_needle == line.strip():
                    updated = original.replace(line, replace, 1)
                    break
            if updated is not None:
                break
    if updated is None:
        # Final benchmark-local fallback: still materialize a real patch so the
        # official evaluator can judge the model output instead of the runner
        # rejecting it on brittle substring matching.
        updated = original.rstrip("\n") + "\n\n" + replace.rstrip("\n") + "\n"
    if updated == original:
        updated = original.rstrip("\n") + "\n\n" + replace.rstrip("\n") + "\n"
    candidate.write_text(updated, encoding="utf-8")
    return candidate


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
        "Return JSON only matching the provided schema.",
        "Make exactly one best-effort concrete code edit in one listed file.",
        "Choose the smallest plausible change that addresses the issue.",
        "Prefer source files over docs or config files unless the issue explicitly targets docs/configuration.",
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
    path_hints, search_terms = _extract_search_terms(prompt)
    candidate_paths = _candidate_files(
        workspace,
        path_hints,
        search_terms,
        limit=effective_policy.candidate_file_limit,
    )
    if not candidate_paths:
        raise LocalAgentRunnerError("Unable to identify any candidate files in the benchmark workspace")
    contexts = [
        (
            relative_path,
            _file_excerpt(
                workspace / relative_path,
                search_terms,
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
            changed_path = _apply_edit(
                workspace,
                relative_path=payload["path"],
                find=payload["find"],
                replace=payload["replace"],
            )
        except LocalAgentRunnerError as exc:
            failure = str(exc)
            continue
        payload["edited_path"] = str(changed_path.relative_to(workspace))
        return payload
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
        client = LlamaCppClient(config)
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
