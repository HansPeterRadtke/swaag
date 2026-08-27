from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil

from swaag.fsops import ensure_dir, write_text
from swaag.utils import new_id, scoped_storage_path, sha256_text, utc_now_iso


@dataclass(slots=True, frozen=True)
class TextArtifact:
    artifact_id: str
    kind: str
    path: str
    metadata_path: str
    created_at: str
    size_chars: int
    sha256: str


class TextArtifactStore:
    """Durable exact text artifacts scoped to a single agent session."""

    def __init__(self, sessions_root: Path, session_id: str):
        self.sessions_root = Path(sessions_root).expanduser().resolve()
        session_root = scoped_storage_path(
            self.sessions_root,
            session_id,
            label="session_id",
        )
        self.session_id = session_root.name
        self.root = session_root / "artifacts"
        self.archive_root = (
            self.sessions_root / "archives" / "artifacts" / self.session_id
        )

    def _text_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.txt"

    def _metadata_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.json"

    def create(self, text: str, *, kind: str) -> TextArtifact:
        if self.archive_root.exists():
            raise RuntimeError(f"Cannot add an artifact to archived session: {self.session_id}")
        ensure_dir(self.root)
        artifact_id = new_id("artifact")
        text_path = self._text_path(artifact_id)
        metadata_path = self._metadata_path(artifact_id)
        write_text(text_path, text, encoding="utf-8")
        artifact = TextArtifact(
            artifact_id=artifact_id,
            kind=str(kind),
            path=str(text_path),
            metadata_path=str(metadata_path),
            created_at=utc_now_iso(),
            size_chars=len(text),
            sha256=sha256_text(text),
        )
        write_text(metadata_path, json.dumps(asdict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    def get(self, artifact_id: str) -> TextArtifact:
        if not artifact_id or "/" in artifact_id or "\\" in artifact_id or ".." in artifact_id:
            raise ValueError("invalid artifact_id")
        archived_metadata_path = self.archive_root / f"{artifact_id}.json"
        if archived_metadata_path.exists():
            metadata_path = archived_metadata_path
            artifact_root = self.archive_root
        else:
            metadata_path = self._metadata_path(artifact_id)
            artifact_root = self.root
        if not metadata_path.exists():
            raise FileNotFoundError(f"Unknown artifact: {artifact_id}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact = TextArtifact(**payload)
        expected = (artifact_root / f"{artifact_id}.txt").resolve()
        actual = Path(artifact.path).resolve()
        if (
            actual != expected
            or Path(artifact.metadata_path).resolve() != metadata_path.resolve()
            or not actual.is_file()
        ):
            raise ValueError(f"Invalid artifact metadata for {artifact_id}")
        text = actual.read_text(encoding="utf-8")
        if len(text) != artifact.size_chars or sha256_text(text) != artifact.sha256:
            raise ValueError(f"Artifact failed integrity verification: {artifact_id}")
        return artifact

    def read(self, artifact_id: str, *, start_offset: int = 0, max_chars: int = 4000) -> dict[str, object]:
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        artifact = self.get(artifact_id)
        text = Path(artifact.path).read_text(encoding="utf-8")
        start = min(start_offset, len(text))
        end = min(len(text), start + max_chars)
        return {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "start_offset": start,
            "end_offset": end,
            "next_offset": end,
            "finished": end >= len(text),
            "total_chars": len(text),
            "sha256": artifact.sha256,
            "text": text[start:end],
        }

    def archive(self) -> int:
        if not self.root.exists():
            return 0
        if self.archive_root.exists():
            active_ids = {path.stem for path in self.root.glob("*.json")}
            archived_ids = {path.stem for path in self.archive_root.glob("*.json")}
            archived_text_ids = {
                path.stem for path in self.archive_root.glob("*.txt")
            }
            if active_ids != archived_ids or archived_ids != archived_text_ids:
                raise ValueError(
                    f"Artifact archive differs from active session: {self.session_id}"
                )
            return len(archived_ids)
        ensure_dir(self.archive_root.parent)
        active_metadata_paths = sorted(self.root.glob("*.json"))
        active_ids = {path.stem for path in active_metadata_paths}
        active_text_ids = {path.stem for path in self.root.glob("*.txt")}
        if active_ids != active_text_ids:
            raise ValueError(
                f"Active artifact set is incomplete for session: {self.session_id}"
            )
        for metadata_path in active_metadata_paths:
            self.get(metadata_path.stem)
        temporary = self.archive_root.with_name(self.archive_root.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.root, temporary)
        metadata_paths = sorted(temporary.glob("*.json"))
        try:
            for metadata_path in metadata_paths:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                artifact_id = str(payload.get("artifact_id", ""))
                text_path = temporary / f"{artifact_id}.txt"
                if artifact_id != metadata_path.stem or not text_path.is_file():
                    raise ValueError(
                        f"Invalid artifact set while archiving session: {self.session_id}"
                    )
                final_text = self.archive_root / text_path.name
                final_metadata = self.archive_root / metadata_path.name
                payload["path"] = str(final_text)
                payload["metadata_path"] = str(final_metadata)
                write_text(
                    metadata_path,
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            os.replace(temporary, self.archive_root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        for path in self.archive_root.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o444)
        self.archive_root.chmod(0o755)
        return len(metadata_paths)
