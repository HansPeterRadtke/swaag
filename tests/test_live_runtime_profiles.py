from __future__ import annotations

import ast
from pathlib import Path

from swaag.config import load_config
from swaag.benchmark.benchmark_runner import _resolve_live_model_settings
from swaag.finalproof import build_finalproof_commands, build_finalproof_environment
from swaag.live_runtime_profiles import (
    discover_local_llama_profiles,
    get_documented_final_live_benchmark_recommendation,
    get_live_runtime_recommendation,
)


def test_discover_local_llama_profiles_reads_host_config(tmp_path: Path) -> None:
    host_config = tmp_path / "host.json"
    host_config.write_text(
        """
        {
          "llama_cpp": {
            "profiles": {
              "small_fast": {"ctx": 2048, "gpu_layers": 31, "threads": 8, "batch": 24, "ubatch": 24, "parallel": 1},
              "mid_context": {"ctx": 20000, "gpu_layers": 24, "threads": 8, "batch": 24, "ubatch": 24, "parallel": 1}
            }
          }
        }
        """,
        encoding="utf-8",
    )

    profiles = discover_local_llama_profiles(host_config_path=host_config)

    assert profiles["small_fast"].ctx == 2048
    assert profiles["mid_context"].gpu_layers == 24


def test_finalproof_and_benchmark_defaults_use_the_documented_final_live_profile() -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()

    commands = build_finalproof_commands(
        benchmark_output=Path("/tmp/bench"),
        live_benchmark_output=Path("/tmp/live"),
    )
    env = build_finalproof_environment()
    flattened = [" ".join(command) for command in commands]
    settings = _resolve_live_model_settings(
        use_live_model=True,
        model_base_url=None,
        timeout_seconds=None,
        connect_timeout_seconds=None,
        model_profile=None,
        structured_output_mode=None,
        progress_poll_seconds=None,
    )

    assert env["SWAAG_LIVE_MODEL_PROFILE"] == recommendation.model_profile
    assert env["SWAAG_LIVE_STRUCTURED_OUTPUT_MODE"] == recommendation.structured_output_mode
    assert env["SWAAG_LIVE_SEEDS"] == ",".join(str(seed) for seed in recommendation.seeds)
    assert env["SWAAG_LIVE_TIMEOUT_SECONDS"] == str(recommendation.timeout_seconds)
    assert settings["model_profile"] == recommendation.model_profile
    assert settings["structured_output_mode"] == recommendation.structured_output_mode
    assert settings["seeds"] == list(recommendation.seeds)
    assert settings["timeout_seconds"] == recommendation.timeout_seconds
    assert any(f"--model-profile {recommendation.model_profile}" in command for command in flattened)
    assert any(f"--structured-output-mode {recommendation.structured_output_mode}" in command for command in flattened)
    assert any(f"--seeds {','.join(str(seed) for seed in recommendation.seeds)}" in command for command in flattened)
    assert any(f"--timeout-seconds {recommendation.timeout_seconds}" in command for command in flattened)


def test_live_runtime_profiles_doc_matches_the_documented_final_recommendation() -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()
    fast = get_live_runtime_recommendation("fast_live_tests")
    doc_path = Path(__file__).resolve().parents[1] / "doc" / "live_runtime_profiles.md"
    text = doc_path.read_text(encoding="utf-8")

    assert f"profile: `{recommendation.model_profile}`" in text
    assert f"structured mode: `{recommendation.structured_output_mode}`" in text
    assert f"fixed seeds: `{','.join(str(seed) for seed in recommendation.seeds)}`" in text
    assert f"profile: `{fast.model_profile}`" in text
    assert "Measured live subset proof" in text
    assert "does not restart or switch llama.cpp profiles inside the runtime loop" in text


def test_default_config_matches_documented_final_live_profile(tmp_path: Path) -> None:
    recommendation = get_documented_final_live_benchmark_recommendation()
    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})

    assert config.model.profile_name == recommendation.model_profile
    assert config.model.structured_output_mode == recommendation.structured_output_mode


def test_runtime_does_not_assign_model_profile_or_mode() -> None:
    runtime_path = Path(__file__).resolve().parents[1] / "src" / "swaag" / "runtime.py"
    tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    forbidden = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr in {"profile_name", "structured_output_mode"}:
                forbidden.append((target.attr, getattr(node, "lineno", -1)))
    assert forbidden == []
