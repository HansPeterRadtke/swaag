from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_clean_install import _create_clean_room


@pytest.mark.agent_test
def test_package_runs_single_benchmark_task_from_clean_venv(tmp_path: Path) -> None:
    python, workspace, env, server, thread = _create_clean_room(tmp_path)
    try:
        single_task = subprocess.run(
            [
                str(python),
                "-m",
                "swaag.benchmark",
                "run",
                "--clean",
                "--task",
                "coding_generated_multifile_01",
                "--output",
                str(tmp_path / "single_benchmark_output"),
                "--json",
            ],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        single_payload = json.loads(single_task.stdout)
        assert single_payload["summary"]["failed_tasks"] == 0
        assert single_payload["summary"]["successful_tasks"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)

