from __future__ import annotations

import os
import shutil
from pathlib import Path


def _as_path(path: str | os.PathLike[str] | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(path: str | os.PathLike[str] | Path) -> Path:
    target = _as_path(path)
    os.makedirs(target, exist_ok=True)
    return target


def ensure_parent_dir(path: str | os.PathLike[str] | Path) -> Path:
    target = _as_path(path)
    parent = target.parent
    if parent != Path(""):
        os.makedirs(parent, exist_ok=True)
    return target


def write_bytes(path: str | os.PathLike[str] | Path, data: bytes) -> Path:
    target = ensure_parent_dir(path)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
    finally:
        os.close(fd)
    return target


def write_text(
    path: str | os.PathLike[str] | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    return write_bytes(path, content.encode(encoding))


def append_text(
    path: str | os.PathLike[str] | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Append text to a file using low-level OS calls (no Path.open)."""
    target = ensure_parent_dir(path)
    data = content.encode(encoding)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


def remove_file(path: str | os.PathLike[str] | Path, *, missing_ok: bool = False) -> None:
    try:
        os.remove(_as_path(path))
    except FileNotFoundError:
        if not missing_ok:
            raise


def atomic_replace(
    source: str | os.PathLike[str] | Path,
    target: str | os.PathLike[str] | Path,
) -> Path:
    destination = ensure_parent_dir(target)
    os.replace(_as_path(source), destination)
    return destination


def snapshot_tree(
    source_root: str | os.PathLike[str] | Path,
    destination_root: str | os.PathLike[str] | Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
    excluded_parts: frozenset[str] = frozenset({".git"}),
) -> list[str]:
    source = _as_path(source_root).resolve()
    destination = ensure_dir(destination_root).resolve()
    excluded = tuple(item.resolve() for item in excluded_roots)
    manifest: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(destination):
            continue
        if any(resolved.is_relative_to(root) for root in excluded):
            continue
        relative = path.relative_to(source)
        target = destination / relative
        ensure_parent_dir(target)
        shutil.copy2(path, target)
        manifest.append(str(relative))
    return manifest


def restore_tree(
    snapshot_root: str | os.PathLike[str] | Path,
    destination_root: str | os.PathLike[str] | Path,
    manifest: list[str] | set[str],
    *,
    excluded_roots: tuple[Path, ...] = (),
    excluded_parts: frozenset[str] = frozenset({".git"}),
) -> None:
    snapshot = _as_path(snapshot_root).resolve()
    destination = _as_path(destination_root).resolve()
    expected = {str(item) for item in manifest}
    excluded = tuple(item.resolve() for item in excluded_roots)
    for path in sorted(destination.rglob("*"), reverse=True):
        if not path.exists() or any(part in excluded_parts for part in path.parts):
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(root) for root in excluded):
            continue
        if path.is_file() and str(path.relative_to(destination)) not in expected:
            remove_file(path)
    for relative in sorted(expected):
        source = snapshot / relative
        target = destination / relative
        ensure_parent_dir(target)
        shutil.copy2(source, target)
