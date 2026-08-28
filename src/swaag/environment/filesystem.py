from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

from swaag.config import AgentConfig
from swaag.utils import sha256_text


class FilesystemError(RuntimeError):
    pass


class FilesystemManager:
    def __init__(self, config: AgentConfig, workspace_root: Path):
        self.config = config
        self.workspace_root = workspace_root.resolve()
        sessions_root = Path(config.sessions.root).expanduser()
        if not sessions_root.is_absolute():
            sessions_root = self.workspace_root / sessions_root
        self.sessions_root = sessions_root.resolve()
        configured_cache = str(config.model.cache_path).strip()
        if configured_cache:
            cache_path = Path(configured_cache).expanduser()
            if not cache_path.is_absolute():
                cache_path = self.workspace_root / cache_path
        else:
            cache_path = self.sessions_root.parent / "llm-model-cache.json"
        self.model_cache_path = cache_path.resolve()

    def _is_runtime_owned_snapshot_path(self, path: Path) -> bool:
        resolved = path.resolve()
        if self.sessions_root != self.workspace_root and resolved.is_relative_to(self.sessions_root):
            return True
        if resolved.parent != self.model_cache_path.parent:
            return False
        name = resolved.name
        cache_name = self.model_cache_path.name
        return (
            name == cache_name
            or name.startswith(cache_name + ".")
            or name.startswith("." + cache_name + ".")
        )

    def resolve_path(self, path_text: str, *, cwd: str | None = None) -> Path:
        path = Path(path_text).expanduser()
        base = Path(cwd).expanduser().resolve() if cwd else self.workspace_root
        resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
        if not self.is_within_workspace(resolved):
            raise FilesystemError(f"Path is outside workspace: {resolved}")
        return resolved

    def is_within_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    def relative_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.workspace_root))

    def list_files(self, path_text: str = ".", *, cwd: str | None = None) -> list[str]:
        root = self.resolve_path(path_text, cwd=cwd)
        if root.is_file():
            return [self.relative_path(root)]
        if not root.exists():
            raise FilesystemError(f"Path does not exist: {root}")
        items: list[str] = []
        for item in sorted(root.rglob("*")):
            if (
                item.is_file()
                and "__pycache__" not in item.parts
                and self.is_within_workspace(item)
                and not self._is_runtime_owned_snapshot_path(item)
            ):
                items.append(self.relative_path(item))
        return items


    def bounded_file_manifest(self, *, max_entries: int, path_text: str = ".", cwd: str | None = None) -> tuple[list[str], bool]:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        root = self.resolve_path(path_text, cwd=cwd)
        if root.is_file():
            return [self.relative_path(root)], False
        if not root.exists():
            raise FilesystemError(f"Path does not exist: {root}")
        entries: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
            for filename in sorted(filenames):
                item = Path(dirpath) / filename
                if self._is_runtime_owned_snapshot_path(item):
                    continue
                entries.append(self.relative_path(item))
                if len(entries) > max_entries:
                    return entries[:max_entries], True
        return entries, False

    def resolve_existing_file_path(self, path_text: str, *, cwd: str | None = None) -> Path:
        path = self.resolve_path(path_text, cwd=cwd)
        if path.exists() and path.is_file():
            return path
        raise FilesystemError(f"File does not exist: {path}")

    def read_text(self, path_text: str, *, cwd: str | None = None) -> tuple[Path, str]:
        path = self.resolve_existing_file_path(path_text, cwd=cwd)
        return path, path.read_text(encoding="utf-8")

    def search_in_file(
        self,
        path_text: str,
        *,
        pattern: str,
        cwd: str | None = None,
        regex: bool = False,
        ignore_case: bool = False,
        start_index: int = 0,
        max_matches: int = 50,
    ) -> tuple[Path, list[dict[str, object]], bool]:
        path, text = self.read_text(path_text, cwd=cwd)
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if max_matches <= 0:
            raise ValueError("max_matches must be positive")
        matches: list[dict[str, object]] = []
        for index, match in enumerate(
            self._iter_text_matches(
                text,
                pattern=pattern,
                regex=regex,
                ignore_case=ignore_case,
            )
        ):
            if index < start_index:
                continue
            if len(matches) >= max_matches:
                return path, matches, False
            matches.append(match)
        return path, matches, True

    @staticmethod
    def _iter_text_matches(
        text: str,
        *,
        pattern: str,
        regex: bool,
        ignore_case: bool,
    ) -> Iterator[dict[str, object]]:
        flags = re.IGNORECASE if ignore_case else 0
        if regex:
            compiled = re.compile(pattern, flags)
            for line_number, line in enumerate(text.splitlines(), start=1):
                for match in compiled.finditer(line):
                    yield {
                        "line_number": line_number,
                        "line_text": line,
                        "match_text": match.group(0),
                        "start_column": match.start() + 1,
                        "end_column": match.end(),
                    }
        else:
            haystack_pattern = pattern.lower() if ignore_case else pattern
            for line_number, line in enumerate(text.splitlines(), start=1):
                cursor = 0
                haystack = line.lower() if ignore_case else line
                while True:
                    index = haystack.find(haystack_pattern, cursor)
                    if index < 0:
                        break
                    yield {
                        "line_number": line_number,
                        "line_text": line,
                        "match_text": line[index:index + len(pattern)],
                        "start_column": index + 1,
                        "end_column": index + len(pattern),
                    }
                    cursor = index + max(len(pattern), 1)

    def search_repo(
        self,
        *,
        pattern: str,
        path_text: str = ".",
        cwd: str | None = None,
        regex: bool = False,
        ignore_case: bool = False,
        start_index: int = 0,
        max_matches: int = 100,
    ) -> tuple[list[dict[str, object]], bool]:
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if max_matches <= 0:
            raise ValueError("max_matches must be positive")
        results: list[dict[str, object]] = []
        match_index = 0
        for relative_path in self.list_files(path_text, cwd=cwd):
            path, text = self.read_text(relative_path, cwd=str(self.workspace_root))
            for match in self._iter_text_matches(
                text,
                pattern=pattern,
                regex=regex,
                ignore_case=ignore_case,
            ):
                if match_index < start_index:
                    match_index += 1
                    continue
                if len(results) >= max_matches:
                    return results, False
                results.append({"path": str(path), "relative_path": relative_path, **match})
                match_index += 1
        return results, True

    def snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        if not self.workspace_root.exists():
            return snapshot
        for item in sorted(self.workspace_root.rglob("*")):
            if not item.is_file() or "__pycache__" in item.parts:
                continue
            if not self.is_within_workspace(item):
                continue
            if self._is_runtime_owned_snapshot_path(item):
                continue
            raw = item.read_bytes()
            rel = self.relative_path(item)
            try:
                snapshot[rel] = raw.decode("utf-8")
            except UnicodeDecodeError:
                snapshot[rel] = "hex:" + raw.hex()
        return snapshot

    def compute_delta(self, before: dict[str, str], after: dict[str, str]) -> dict[str, object]:
        before_keys = set(before)
        after_keys = set(after)
        created = {key: after[key] for key in sorted(after_keys - before_keys)}
        deleted = sorted(before_keys - after_keys)
        modified = {key: after[key] for key in sorted(before_keys & after_keys) if before[key] != after[key]}
        return {
            "created": created,
            "deleted": deleted,
            "modified": modified,
            "created_files": sorted(created),
            "deleted_files": deleted,
            "modified_files": sorted(modified),
            "content_hash": sha256_text("\n".join(f"{key}:{after[key]}" for key in sorted(after))),
        }

    def stat(self, path_text: str, *, cwd: str | None = None) -> dict[str, object]:
        path = self.resolve_path(path_text, cwd=cwd)
        info = path.stat()
        return {
            "path": str(path),
            "relative_path": self.relative_path(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "size_bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        }
