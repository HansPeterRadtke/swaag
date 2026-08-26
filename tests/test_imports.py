from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import swaag

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "swaag"
MANDATORY_MODULES = {
    "swaag.action",
    "swaag.budgeting",
    "swaag.compression",
    "swaag.config",
    "swaag.history",
    "swaag.runtime",
    "swaag.scheduler",
    "swaag.telemetry",
    "swaag.tools.registry",
    "swaag.benchmark.benchmark_runner",
    "swaag.benchmark.failure_analyzer",
    "swaag.benchmark.metrics",
    "swaag.benchmark.report",
}


def test_all_swaag_modules_import_cleanly() -> None:
    imported: list[str] = []
    for module_info in pkgutil.walk_packages(swaag.__path__, prefix="swaag."):
        module = importlib.import_module(module_info.name)
        imported.append(module.__name__)

    assert "swaag.runtime" in imported
    assert "swaag.tools.registry" in imported
    assert MANDATORY_MODULES.issubset(set(imported))


def test_all_referenced_swaag_modules_exist() -> None:
    missing: list[str] = []
    referenced: set[str] = set()

    for path in PACKAGE_ROOT.rglob("*.py"):
        module_name = "swaag." + path.relative_to(PACKAGE_ROOT).with_suffix("").as_posix().replace("/", ".")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("swaag"):
                        referenced.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if not node.module or not node.module.startswith("swaag"):
                    continue
                referenced.add(node.module)
                if node.level != 0:
                    missing.append(f"{module_name}:relative import level {node.level} from {node.module}")

    for module_name in sorted(referenced | MANDATORY_MODULES):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    assert not missing, f"Missing or invalid module references: {missing}"
