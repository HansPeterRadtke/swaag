#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swaag.config import load_config
from swaag.live_runtime_profiles import discover_local_llama_profiles  # noqa: E402


LONG_PROMPT = (
    "You are evaluating local coding agents. Provide exactly 14 numbered bullets about why "
    "verification-first execution matters. Each bullet must be concise but specific."
)
SMALL_PROMPT = "Reply with exactly: ok"


@dataclass(slots=True)
class RequestMeasurement:
    name: str
    wall_seconds: float
    completion_tokens: int | None
    prompt_tokens: int | None
    finish_reason: str | None


def _request(
    *,
    base_url: str,
    prompt: str,
    n_predict: int,
    seed: int,
    temperature: float,
) -> RequestMeasurement:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": 0.95,
        "seed": seed,
        "stop": ["</s>"],
    }
    started = time.monotonic()
    response = requests.post(
        f"{base_url.rstrip('/')}/completion",
        json=payload,
        timeout=(10, 300),
    )
    response.raise_for_status()
    body = response.json()
    wall = time.monotonic() - started
    return RequestMeasurement(
        name="request",
        wall_seconds=round(wall, 3),
        completion_tokens=body.get("tokens_predicted"),
        prompt_tokens=body.get("tokens_evaluated"),
        finish_reason="stop" if body.get("stop") else None,
    )


def _background_shell_measurement(duration_seconds: float) -> subprocess.Popen[str]:
    command = [
        "python3",
        "-c",
        (
            "import hashlib, time; "
            "data = b'x' * 1048576; "
            f"deadline = time.time() + {duration_seconds}; "
            "count = 0; "
            "while time.time() < deadline: "
            "    hashlib.sha256(data).hexdigest(); "
            "    count += 1; "
            "print(count)"
        ),
    ]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def _with_tps(measurement: RequestMeasurement) -> dict[str, Any]:
    payload = asdict(measurement)
    tokens = measurement.completion_tokens or 0
    payload["tokens_per_second"] = round(tokens / measurement.wall_seconds, 3) if measurement.wall_seconds > 0 else None
    return payload


def benchmark(base_url: str, *, seed: int, temperature: float) -> dict[str, Any]:
    config = load_config()
    profiles = {name: asdict(item) for name, item in discover_local_llama_profiles().items()}
    profile_name = config.model.profile_name
    profile = profiles.get(profile_name, {})

    single_long = _request(
        base_url=base_url,
        prompt=LONG_PROMPT,
        n_predict=192,
        seed=seed,
        temperature=temperature,
    )

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            _request,
            base_url=base_url,
            prompt=LONG_PROMPT,
            n_predict=192,
            seed=seed,
            temperature=temperature,
        )
        future_b = executor.submit(
            _request,
            base_url=base_url,
            prompt=LONG_PROMPT,
            n_predict=192,
            seed=seed + 1,
            temperature=temperature,
        )
        two_long_a = future_a.result()
        two_long_b = future_b.result()
    two_long_wall = round(time.monotonic() - started, 3)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_long = executor.submit(
            _request,
            base_url=base_url,
            prompt=LONG_PROMPT,
            n_predict=192,
            seed=seed,
            temperature=temperature,
        )
        future_small = executor.submit(
            _request,
            base_url=base_url,
            prompt=SMALL_PROMPT,
            n_predict=8,
            seed=seed + 2,
            temperature=0.0,
        )
        long_plus_small_long = future_long.result()
        long_plus_small_small = future_small.result()
    long_plus_small_wall = round(time.monotonic() - started, 3)

    bg = _background_shell_measurement(duration_seconds=4.0)
    try:
        long_with_bg = _request(
            base_url=base_url,
            prompt=LONG_PROMPT,
            n_predict=192,
            seed=seed,
            temperature=temperature,
        )
    finally:
        bg.wait(timeout=10)

    total_two_long_tokens = (two_long_a.completion_tokens or 0) + (two_long_b.completion_tokens or 0)
    total_two_long_tps = round(total_two_long_tokens / two_long_wall, 3) if two_long_wall > 0 else None
    total_long_small_tokens = (long_plus_small_long.completion_tokens or 0) + (long_plus_small_small.completion_tokens or 0)
    total_long_small_tps = round(total_long_small_tokens / long_plus_small_wall, 3) if long_plus_small_wall > 0 else None

    recommendation = (
        "defer model-side concurrency"
        if not total_two_long_tps or total_two_long_tps <= ((single_long.completion_tokens or 0) / single_long.wall_seconds) * 1.05
        else "consider narrow concurrent request support"
    )

    return {
        "base_url": base_url,
        "seed": seed,
        "temperature": temperature,
        "profile_name": profile_name,
        "profile": profile,
        "single_long": _with_tps(single_long),
        "two_long_concurrent": {
            "combined_wall_seconds": two_long_wall,
            "combined_tokens_per_second": total_two_long_tps,
            "requests": [_with_tps(two_long_a), _with_tps(two_long_b)],
        },
        "long_plus_small_concurrent": {
            "combined_wall_seconds": long_plus_small_wall,
            "combined_tokens_per_second": total_long_small_tps,
            "requests": [_with_tps(long_plus_small_long), _with_tps(long_plus_small_small)],
        },
        "long_with_background_shell": _with_tps(long_with_bg),
        "decision": recommendation,
        "decision_reason": (
            "The current server profile advertises a single parallel slot. If concurrent requests do not materially "
            "improve aggregate useful throughput over the single long request baseline, keep scheduler work focused "
            "on background shell/process overlap instead of model-side concurrency."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark whether concurrent llama.cpp requests are worth implementing.")
    parser.add_argument("--base-url", default=os.environ.get("SWAAG_LIVE_BASE_URL", "http://127.0.0.1:14829"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results = benchmark(args.base_url, seed=args.seed, temperature=args.temperature)
    raw = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
