from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable

import requests

from swaag.utils import stable_json_dumps


_METRIC_RE = re.compile(
    r'^vllm:(num_requests_running|num_requests_waiting)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$'
)


def _parse_vllm_scheduler_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC_RE.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def _scheduler_snapshot(metrics_url: str, *, timeout_seconds: float) -> dict[str, Any]:
    if not metrics_url:
        return {"supported": False, "metrics": {}}
    started = time.monotonic()
    try:
        response = requests.get(metrics_url, timeout=min(timeout_seconds, 5.0))
        response.raise_for_status()
        values = _parse_vllm_scheduler_metrics(response.text)
        return {
            "supported": bool(values),
            "metrics": values,
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:  # noqa: BLE001 - benchmark retains transport evidence.
        return {
            "supported": False,
            "metrics": {},
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _stream_chat_call(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    origin: float,
    label: str,
) -> dict[str, Any]:
    submitted = time.monotonic()
    first_token_at: float | None = None
    first_visible_token_at: float | None = None
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    response = requests.post(
        endpoint,
        json=payload,
        stream=True,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = str(raw_line).strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            item = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(item.get("usage"), dict):
            usage = dict(item["usage"])
        choices = item.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        reasoning = delta.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = delta.get("reasoning") if isinstance(delta.get("reasoning"), str) else ""
        content = delta.get("content")
        if not isinstance(content, str):
            content = choice.get("text") if isinstance(choice.get("text"), str) else ""
        if reasoning or content:
            now = time.monotonic()
            if first_token_at is None:
                first_token_at = now
            if reasoning:
                reasoning_chunks.append(reasoning)
            if content:
                if first_visible_token_at is None:
                    first_visible_token_at = now
                chunks.append(content)
    finished = time.monotonic()
    text = "".join(chunks)
    reasoning_text = "".join(reasoning_chunks)
    return {
        "label": label,
        "submitted_offset_seconds": submitted - origin,
        "first_token_offset_seconds": None if first_token_at is None else first_token_at - origin,
        "first_visible_token_offset_seconds": (
            None if first_visible_token_at is None else first_visible_token_at - origin
        ),
        "finished_offset_seconds": finished - origin,
        "ttft_seconds": None if first_token_at is None else first_token_at - submitted,
        "visible_ttft_seconds": (
            None if first_visible_token_at is None else first_visible_token_at - submitted
        ),
        "elapsed_seconds": finished - submitted,
        "output_chars": len(text),
        "reasoning_chars": len(reasoning_text),
        "output_head": text[:240],
        "reasoning_head": reasoning_text[:240],
        "finish_reason": finish_reason,
        "usage": usage,
        "max_tokens": max_tokens,
    }


def _run_captured(target: Callable[[], dict[str, Any]], box: dict[str, Any], key: str) -> None:
    try:
        box[key] = target()
    except Exception as exc:  # noqa: BLE001 - benchmark retains exact failure class/text.
        box[key] = {
            "label": key,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_backend_scheduling_benchmark(
    *,
    output_dir: Path,
    endpoint: str,
    model: str,
    metrics_url: str = "",
    long_prompt_repetitions: int = 5000,
    long_max_tokens: int = 256,
    short_max_tokens: int = 32,
    short_submit_delay_seconds: float = 1.0,
    scheduler_sample_delay_seconds: float = 0.25,
    timeout_seconds: float = 300.0,
    clean: bool = False,
) -> dict[str, Any]:
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("endpoint must be an http(s) URL")
    if not model.strip():
        raise ValueError("model must be non-empty")
    if long_prompt_repetitions <= 0 or long_max_tokens <= 0 or short_max_tokens <= 0:
        raise ValueError("prompt repetitions and token limits must be positive")
    if short_submit_delay_seconds < 0 or scheduler_sample_delay_seconds < 0:
        raise ValueError("delays must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origin = time.monotonic()
    box: dict[str, Any] = {}
    long_prompt = (
        ("alpha " * long_prompt_repetitions)
        + "\nContinue with a detailed explanation of inference scheduling using many sentences. "
        + "Do not intentionally stop early."
    )
    short_prompt = "Reply briefly: interactive ping."

    idle_snapshot = _scheduler_snapshot(metrics_url, timeout_seconds=timeout_seconds)
    long_thread = threading.Thread(
        target=_run_captured,
        args=(
            lambda: _stream_chat_call(
                endpoint=endpoint,
                model=model,
                prompt=long_prompt,
                max_tokens=long_max_tokens,
                timeout_seconds=timeout_seconds,
                origin=origin,
                label="long",
            ),
            box,
            "long",
        ),
        name="swaag-backend-long",
    )
    long_thread.start()
    time.sleep(short_submit_delay_seconds)
    before_short = _scheduler_snapshot(metrics_url, timeout_seconds=timeout_seconds)

    short_thread = threading.Thread(
        target=_run_captured,
        args=(
            lambda: _stream_chat_call(
                endpoint=endpoint,
                model=model,
                prompt=short_prompt,
                max_tokens=short_max_tokens,
                timeout_seconds=timeout_seconds,
                origin=origin,
                label="short",
            ),
            box,
            "short",
        ),
        name="swaag-backend-short",
    )
    short_thread.start()
    time.sleep(scheduler_sample_delay_seconds)
    during_overlap = _scheduler_snapshot(metrics_url, timeout_seconds=timeout_seconds)

    short_thread.join(timeout=timeout_seconds + 5.0)
    long_thread.join(timeout=timeout_seconds + 5.0)
    if short_thread.is_alive() or long_thread.is_alive():
        raise TimeoutError("backend scheduling benchmark thread exceeded timeout")

    long_result = dict(box.get("long", {}))
    short_result = dict(box.get("short", {}))
    successful = "error" not in long_result and "error" not in short_result
    short_finished_before_long: bool | None = None
    overlap_seconds: float | None = None
    short_submitted_before_long_finished: bool | None = None
    short_first_token_before_long_finished: bool | None = None
    interactive_overlap_observed: bool | None = None
    short_serialized_behind_long: bool | None = None
    if successful:
        long_finished = float(long_result["finished_offset_seconds"])
        short_submitted = float(short_result["submitted_offset_seconds"])
        short_finished_before_long = (
            float(short_result["finished_offset_seconds"]) < long_finished
        )
        short_submitted_before_long_finished = short_submitted < long_finished
        short_first_token = short_result.get("first_token_offset_seconds")
        short_first_token_before_long_finished = (
            isinstance(short_first_token, (int, float))
            and float(short_first_token) < long_finished
        )
        interactive_overlap_observed = bool(
            short_submitted_before_long_finished
            and short_first_token_before_long_finished
        )
        short_serialized_behind_long = bool(
            short_submitted_before_long_finished
            and not short_first_token_before_long_finished
        )
        overlap_start = max(
            float(long_result["submitted_offset_seconds"]),
            short_submitted,
        )
        overlap_end = min(
            long_finished,
            float(short_result["finished_offset_seconds"]),
        )
        overlap_seconds = max(0.0, overlap_end - overlap_start)

    report = {
        "benchmark": "backend_scheduling_v1",
        "endpoint": endpoint,
        "model": model,
        "metrics_url": metrics_url,
        "settings": {
            "long_prompt_repetitions": long_prompt_repetitions,
            "long_max_tokens": long_max_tokens,
            "short_max_tokens": short_max_tokens,
            "short_submit_delay_seconds": short_submit_delay_seconds,
            "scheduler_sample_delay_seconds": scheduler_sample_delay_seconds,
            "timeout_seconds": timeout_seconds,
        },
        "idle_scheduler": idle_snapshot,
        "scheduler_before_short": before_short,
        "scheduler_during_overlap": during_overlap,
        "long": long_result,
        "short": short_result,
        "short_finished_before_long": short_finished_before_long,
        "short_submitted_before_long_finished": short_submitted_before_long_finished,
        "short_first_token_before_long_finished": short_first_token_before_long_finished,
        "interactive_overlap_observed": interactive_overlap_observed,
        "short_serialized_behind_long": short_serialized_behind_long,
        "overlap_seconds": overlap_seconds,
        "transport_succeeded": successful,
        "passed": successful,
    }
    (output_dir / "backend_scheduling_results.json").write_text(
        stable_json_dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
