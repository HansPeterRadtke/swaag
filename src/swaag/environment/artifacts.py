from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from swaag.fsops import ensure_dir, write_text
from swaag.utils import new_id, sha256_text, utc_now_iso


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
        self.root = Path(sessions_root).expanduser() / session_id / "artifacts"
        ensure_dir(self.root)

    def _text_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.txt"

    def _metadata_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.json"

    def create(self, text: str, *, kind: str) -> TextArtifact:
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
        metadata_path = self._metadata_path(artifact_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Unknown artifact: {artifact_id}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact = TextArtifact(**payload)
        expected = self._text_path(artifact_id).resolve()
        actual = Path(artifact.path).resolve()
        if actual != expected or not actual.is_file():
            raise ValueError(f"Invalid artifact metadata for {artifact_id}")
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
