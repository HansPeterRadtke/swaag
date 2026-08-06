from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def all_test_files(root: Path | None = None) -> tuple[str, ...]:
    base = project_root() if root is None else Path(root)
    return tuple(
        sorted(
            path.relative_to(base).as_posix()
            for path in (base / "tests").glob("test_*.py")
            if path.is_file()
        )
    )


CODE_CORRECTNESS_TEST_FILES = frozenset(all_test_files())
AGENT_TEST_FILES: frozenset[str] = frozenset()


def category_for_test_file(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if normalized in CODE_CORRECTNESS_TEST_FILES:
        return "code_correctness"
    return None


def validate_test_category_registry(root: Path | None = None) -> None:
    base = project_root() if root is None else Path(root)
    missing = sorted(path for path in CODE_CORRECTNESS_TEST_FILES if not (base / path).is_file())
    if missing:
        raise RuntimeError(f"Test registry references missing files: {missing}")
