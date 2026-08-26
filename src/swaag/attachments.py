from __future__ import annotations

from dataclasses import asdict
import hashlib
import mimetypes
import os
from pathlib import Path
import re

from swaag.fsops import ensure_dir
from swaag.types import AttachmentReference
from swaag.utils import new_id, utc_now_iso


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AttachmentStore:
    """Content-addressed raw attachment storage independent of session projections."""

    def __init__(self, sessions_root: Path, *, max_upload_bytes: int):
        self.root = Path(sessions_root).expanduser() / "_attachments"
        self.blobs = self.root / "blobs"
        self.max_upload_bytes = int(max_upload_bytes)
        ensure_dir(self.blobs)

    def add_bytes(
        self,
        data: bytes,
        *,
        original_name: str,
        media_type: str = "",
        source: str = "api",
    ) -> AttachmentReference:
        if not isinstance(data, bytes):
            raise TypeError("attachment data must be bytes")
        if len(data) > self.max_upload_bytes:
            raise ValueError(
                f"attachment exceeds max_upload_bytes: {len(data)} > {self.max_upload_bytes}"
            )
        name = str(original_name).strip()
        if not name:
            raise ValueError("attachment original_name must not be empty")
        digest = hashlib.sha256(data).hexdigest()
        target = self._blob_path(digest)
        ensure_dir(target.parent)
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            self._verify_blob(target, digest, len(data))
        else:
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                target.unlink(missing_ok=True)
                raise
        detected_type = str(media_type).strip() or mimetypes.guess_type(name)[0] or "application/octet-stream"
        return AttachmentReference(
            attachment_id=new_id("attachment"),
            original_name=name,
            media_type=detected_type,
            size_bytes=len(data),
            sha256=digest,
            storage_ref=f"sha256:{digest}",
            created_at=utc_now_iso(),
            source=str(source).strip() or "api",
            metadata={},
        )

    def path_for(self, reference: AttachmentReference) -> Path:
        digest = self._digest_from_reference(reference)
        path = self._blob_path(digest)
        self._verify_blob(path, digest, int(reference.size_bytes))
        return path

    def read_bytes(self, reference: AttachmentReference) -> bytes:
        return self.path_for(reference).read_bytes()

    def public_metadata(self, reference: AttachmentReference) -> dict:
        payload = asdict(reference)
        payload.pop("storage_ref", None)
        return payload

    def _blob_path(self, digest: str) -> Path:
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("invalid attachment sha256")
        return self.blobs / digest[:2] / digest

    @staticmethod
    def _digest_from_reference(reference: AttachmentReference) -> str:
        prefix, separator, digest = reference.storage_ref.partition(":")
        if prefix != "sha256" or separator != ":" or digest != reference.sha256:
            raise ValueError(f"invalid attachment storage reference: {reference.attachment_id}")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid attachment digest: {reference.attachment_id}")
        return digest

    @staticmethod
    def _verify_blob(path: Path, expected_sha256: str, expected_size: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"attachment blob is missing: sha256:{expected_sha256}")
        data = path.read_bytes()
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError(f"attachment blob failed integrity verification: sha256:{expected_sha256}")


def find_attachment(references: list[AttachmentReference], attachment_id: str) -> AttachmentReference:
    normalized = str(attachment_id).strip()
    for reference in references:
        if reference.attachment_id == normalized:
            return reference
    raise FileNotFoundError(f"Unknown attachment: {normalized}")
