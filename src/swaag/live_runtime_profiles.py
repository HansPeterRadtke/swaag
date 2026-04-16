from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LiveRuntimeUseCase = Literal[
    "fast_live_tests",
    "final_live_benchmark",
    "heavy_structured",
    "slow_hardware_fallback",
]


@dataclass(frozen=True, slots=True)
class LlamaCppProfileDefinition:
    name: str
    ctx: int
    gpu_layers: int
    threads: int
    batch: int
    ubatch: int
    parallel: int


@dataclass(frozen=True, slots=True)
class LiveRuntimeRecommendation:
    use_case: LiveRuntimeUseCase
    model_profile: str
    structured_output_mode: str
    seeds: tuple[int, int, int]
    timeout_seconds: int
    connect_timeout_seconds: int
    progress_poll_seconds: float
    request_observability_mode: str
    rationale: str
    measured_task_count: int
    measured_success_rate: float
    measured_false_positive_rate: float
    measured_wall_clock_seconds: float


def _default_host_config_path(data_root: Path = Path("/data")) -> Path:
    hostname = socket.gethostname()
    short = hostname.split(".", 1)[0]
    short_path = data_root / "etc" / "hosts" / f"{short}.json"
    if short_path.exists():
        return short_path
    full_path = data_root / "etc" / "hosts" / f"{hostname}.json"
    return full_path


def discover_local_llama_profiles(*, host_config_path: Path | None = None) -> dict[str, LlamaCppProfileDefinition]:
    config_path = _default_host_config_path() if host_config_path is None else host_config_path
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = payload.get("llama_cpp", {}).get("profiles", {})
    discovered: dict[str, LlamaCppProfileDefinition] = {}
    for name, values in sorted(profiles.items()):
        if not isinstance(values, dict):
            continue
        discovered[str(name)] = LlamaCppProfileDefinition(
            name=str(name),
            ctx=int(values.get("ctx", 0)),
            gpu_layers=int(values.get("gpu_layers", 0)),
            threads=int(values.get("threads", 0)),
            batch=int(values.get("batch", 0)),
            ubatch=int(values.get("ubatch", 0)),
            parallel=int(values.get("parallel", 0)),
        )
    return discovered


_RECOMMENDATIONS: dict[LiveRuntimeUseCase, LiveRuntimeRecommendation] = {
    "fast_live_tests": LiveRuntimeRecommendation(
        use_case="fast_live_tests",
        model_profile="small_fast",
        structured_output_mode="post_validate",
        seeds=(11, 23, 37),
        timeout_seconds=120,
        connect_timeout_seconds=10,
        progress_poll_seconds=5.0,
        request_observability_mode="progress_polling",
        rationale=(
            "Use the smallest local context profile with generation-time schema enforcement plus local "
            "post-validation so short live tests stay observable without dropping hard output contracts."
        ),
        measured_task_count=8,
        measured_success_rate=1.0,
        measured_false_positive_rate=0.0,
        measured_wall_clock_seconds=251.77,
    ),
    "final_live_benchmark": LiveRuntimeRecommendation(
        use_case="final_live_benchmark",
        model_profile="small_fast",
        structured_output_mode="post_validate",
        seeds=(11, 23, 37),
        timeout_seconds=180,
        connect_timeout_seconds=10,
        progress_poll_seconds=5.0,
        request_observability_mode="progress_polling",
        rationale=(
            "The representative 30-task live subset is intentionally bounded to fit the local 2048-token "
            "profile. On current hardware this profile/mode completed the full live subset with zero false "
            "positives, so final proof uses the measured path instead of a larger slower context profile."
        ),
        measured_task_count=30,
        measured_success_rate=1.0,
        measured_false_positive_rate=0.0,
        measured_wall_clock_seconds=1022.282,
    ),
    "heavy_structured": LiveRuntimeRecommendation(
        use_case="heavy_structured",
        model_profile="mid_context",
        structured_output_mode="auto",
        seeds=(11, 23, 37),
        timeout_seconds=240,
        connect_timeout_seconds=10,
        progress_poll_seconds=5.0,
        request_observability_mode="progress_polling",
        rationale=(
            "Use the larger local context profile only for ad hoc human-directed runs when prompt assembly "
            "no longer fits the fixed final-proof profile. This is not part of the normal live/final-proof path."
        ),
        measured_task_count=0,
        measured_success_rate=0.0,
        measured_false_positive_rate=0.0,
        measured_wall_clock_seconds=0.0,
    ),
    "slow_hardware_fallback": LiveRuntimeRecommendation(
        use_case="slow_hardware_fallback",
        model_profile="small_fast",
        structured_output_mode="post_validate",
        seeds=(11, 23, 37),
        timeout_seconds=240,
        connect_timeout_seconds=15,
        progress_poll_seconds=10.0,
        request_observability_mode="progress_polling",
        rationale=(
            "On slower local hardware keep context small, retain generation-time schema enforcement, and "
            "increase timeouts so long constrained calls remain visible instead of looking dead."
        ),
        measured_task_count=0,
        measured_success_rate=0.0,
        measured_false_positive_rate=0.0,
        measured_wall_clock_seconds=0.0,
    ),
}


def get_live_runtime_recommendation(use_case: LiveRuntimeUseCase) -> LiveRuntimeRecommendation:
    return _RECOMMENDATIONS[use_case]


def get_documented_final_live_benchmark_recommendation() -> LiveRuntimeRecommendation:
    return get_live_runtime_recommendation("final_live_benchmark")
