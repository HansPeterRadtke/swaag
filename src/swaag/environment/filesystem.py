from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from swaag.config import AgentConfig
from swaag.utils import sha256_text


class FilesystemError(RuntimeError):
    pass


class FilesystemManager:
    def __init__(self, config: AgentConfig, workspace_root: Path):
        self.config = config
        self.workspace_root = workspace_root.resolve()

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
            if item.is_file() and "__pycache__" not in item.parts:
                items.append(self.relative_path(item))
        return items

    def read_text(self, path_text: str, *, cwd: str | None = None) -> tuple[Path, str]:
        path = self.resolve_path(path_text, cwd=cwd)
        if not path.exists() or not path.is_file():
            raise FilesystemError(f"File does not exist: {path}")
        return path, path.read_text(encoding="utf-8")

    def search_in_file(
        self,
        path_text: str,
        *,
        pattern: str,
        cwd: str | None = None,
        regex: bool = False,
        ignore_case: bool = False,
        max_matches: int = 50,
    ) -> tuple[Path, list[dict[str, object]]]:
        path, text = self.read_text(path_text, cwd=cwd)
        flags = re.IGNORECASE if ignore_case else 0
        matches: list[dict[str, object]] = []
        if regex:
            compiled = re.compile(pattern, flags)
            for line_number, line in enumerate(text.splitlines(), start=1):
                for match in compiled.finditer(line):
                    matches.append(
                        {
                            "line_number": line_number,
                            "line_text": line,
                            "match_text": match.group(0),
                            "start_column": match.start() + 1,
                            "end_column": match.end(),
                        }
                    )
                    if len(matches) >= max_matches:
                        return path, matches
        else:
            haystack_pattern = pattern.lower() if ignore_case else pattern
            for line_number, line in enumerate(text.splitlines(), start=1):
                cursor = 0
                haystack = line.lower() if ignore_case else line
                while True:
                    index = haystack.find(haystack_pattern, cursor)
                    if index < 0:
                        break
                    matches.append(
                        {
                            "line_number": line_number,
                            "line_text": line,
                            "match_text": line[index:index + len(pattern)],
                            "start_column": index + 1,
                            "end_column": index + len(pattern),
                        }
                    )
                    if len(matches) >= max_matches:
                        return path, matches
                    cursor = index + max(len(pattern), 1)
        return path, matches

    def search_repo(
        self,
        *,
        pattern: str,
        path_text: str = ".",
        cwd: str | None = None,
        regex: bool = False,
        ignore_case: bool = False,
        max_matches: int = 100,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for relative_path in self.list_files(path_text, cwd=cwd):
            path, matches = self.search_in_file(
                relative_path,
                cwd=str(self.workspace_root),
                pattern=pattern,
                regex=regex,
                ignore_case=ignore_case,
                max_matches=max_matches - len(results),
            )
            for match in matches:
                results.append({"path": str(path), "relative_path": relative_path, **match})
                if len(results) >= max_matches:
                    return results
        return results

    def snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        if not self.workspace_root.exists():
            return snapshot
        for item in sorted(self.workspace_root.rglob("*")):
            if not item.is_file() or "__pycache__" in item.parts:
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
