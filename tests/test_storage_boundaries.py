from pathlib import Path

import pytest

from swaag.environment.artifacts import TextArtifactStore
from swaag.environment.terminal import TerminalStore
from swaag.history import HistoryStore
from swaag.history_archive import HistoryArchiveStore


@pytest.mark.parametrize(
    "session_id",
    ["../escape", "a/b", "a\\b", ".", "..", " spaced ", "x" * 129],
)
def test_session_scoped_stores_reject_unsafe_identifiers(
    tmp_path: Path,
    session_id: str,
) -> None:
    root = tmp_path / "sessions"
    history = HistoryStore(root)

    with pytest.raises(ValueError, match="session_id"):
        history.create(
            config_fingerprint="cfg",
            model_base_url="http://model",
            session_id=session_id,
        )
    with pytest.raises(ValueError, match="session_id"):
        TextArtifactStore(root, session_id)
    with pytest.raises(ValueError, match="session_id"):
        TerminalStore(root, session_id)
    with pytest.raises(ValueError, match="session_id"):
        HistoryArchiveStore(root).archive_events(session_id, "unsafe", [])


def test_session_storage_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "session_link").symlink_to(outside, target_is_directory=True)
    history = HistoryStore(root)

    assert history.list_sessions() == []
    with pytest.raises(ValueError, match="outside"):
        history.create(
            config_fingerprint="cfg",
            model_base_url="http://model",
            session_id="session_link",
        )


def test_user_session_name_is_not_treated_as_a_storage_path(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    history = HistoryStore(root)

    state = history.create_or_load_user_session(
        config_fingerprint="cfg",
        model_base_url="http://model",
        session_ref="../display name",
    )

    assert state.session_name == "../display name"
    assert state.session_id.startswith("session_")
    assert history.history_path(state.session_id).is_relative_to(root.resolve())
    assert not (tmp_path / "display name").exists()
