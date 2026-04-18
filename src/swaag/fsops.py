from __future__ import annotations

import os
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
