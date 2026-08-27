from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from swaag.fsops import atomic_replace, ensure_dir, write_text
from swaag.utils import new_id

_DURATION_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)\s*$")
_UNIT_SECONDS = {
    "ms": 0.001,
    "msec": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "w": 604800.0,
    "week": 604800.0,
    "weeks": 604800.0,
    "month": 2629800.0,
    "months": 2629800.0,
    "y": 31557600.0,
    "year": 31557600.0,
    "years": 31557600.0,
}


@dataclass(slots=True, frozen=True)
class Wakeup:
    wakeup_id: str
    session_id: str
    reason: str
    created_at: str
    wake_at: str
    status: str = "scheduled"
    cancelled_at: str | None = None
    claimed_at: str | None = None
    delivered_at: str | None = None


def parse_duration(text: str) -> timedelta:
    match = _DURATION_RE.match(text)
    if match is None:
        raise ValueError("duration must look like '250 ms', '30 seconds', '2 hours', '3 days', or '1 month'")
    unit = match.group("unit").lower()
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"unsupported duration unit: {unit}")
    seconds = float(match.group("value")) * _UNIT_SECONDS[unit]
    if seconds <= 0:
        raise ValueError("duration must be greater than zero")
    return timedelta(seconds=seconds)


def parse_utc_datetime(text: str) -> datetime:
    normalized = text.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("wake_at must be an ISO-8601 datetime with a timezone") from exc
    if value.tzinfo is None:
        raise ValueError("wake_at must include a timezone")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class WakeupStore:
    def __init__(self, sessions_root: Path):
        root = Path(sessions_root).expanduser()
        self.path = root / "scheduled_wakeups.json"
        self.lock_path = root / "scheduled_wakeups.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        ensure_dir(self.lock_path.parent)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load_unlocked(self) -> list[Wakeup]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("scheduled wakeup store must contain an array")
        normalized: list[Wakeup] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("scheduled wakeup entries must be objects")
            normalized.append(
                Wakeup(
                    wakeup_id=str(item["wakeup_id"]),
                    session_id=str(item["session_id"]),
                    reason=str(item["reason"]),
                    created_at=str(item["created_at"]),
                    wake_at=str(item["wake_at"]),
                    status=str(item.get("status", "scheduled")),
                    cancelled_at=item.get("cancelled_at"),
                    claimed_at=item.get("claimed_at"),
                    delivered_at=item.get("delivered_at"),
                )
            )
        return normalized

    def _save_unlocked(self, wakeups: list[Wakeup]) -> None:
        ensure_dir(self.path.parent)
        temp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        write_text(temp, json.dumps([asdict(item) for item in wakeups], indent=2, sort_keys=True) + "\n")
        atomic_replace(temp, self.path)

    def schedule(
        self,
        *,
        session_id: str,
        reason: str,
        duration: str | None = None,
        wake_at: str | None = None,
        now: datetime | None = None,
    ) -> Wakeup:
        if bool(duration) == bool(wake_at):
            raise ValueError("provide exactly one of duration or wake_at")
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        target = current + parse_duration(duration or "") if duration else parse_utc_datetime(wake_at or "")
        if target <= current:
            raise ValueError("wake time must be in the future")
        wakeup = Wakeup(
            wakeup_id=new_id("wakeup"),
            session_id=session_id,
            reason=reason.strip() or "scheduled wakeup",
            created_at=_iso(current),
            wake_at=_iso(target),
        )
        with self._locked():
            wakeups = self._load_unlocked()
            wakeups.append(wakeup)
            self._save_unlocked(wakeups)
        return wakeup

    def list(self, *, session_id: str, include_cancelled: bool = False) -> list[Wakeup]:
        with self._locked():
            result = [item for item in self._load_unlocked() if item.session_id == session_id]
        if not include_cancelled:
            result = [item for item in result if item.status != "cancelled"]
        return sorted(result, key=lambda item: (item.wake_at, item.wakeup_id))

    def list_all(self, *, include_cancelled: bool = False) -> list[Wakeup]:
        with self._locked():
            result = self._load_unlocked()
        if not include_cancelled:
            result = [item for item in result if item.status != "cancelled"]
        return sorted(result, key=lambda item: (item.wake_at, item.wakeup_id))

    def cancel(self, *, session_id: str, wakeup_id: str, now: datetime | None = None) -> Wakeup:
        with self._locked():
            wakeups = self._load_unlocked()
            for index, item in enumerate(wakeups):
                if item.session_id == session_id and item.wakeup_id == wakeup_id:
                    if item.status == "cancelled":
                        return item
                    if item.status == "delivered":
                        raise ValueError("cannot cancel an already delivered wakeup")
                    updated = Wakeup(
                        **{
                            **asdict(item),
                            "status": "cancelled",
                            "cancelled_at": _iso((now or datetime.now(timezone.utc)).astimezone(timezone.utc)),
                        }
                    )
                    wakeups[index] = updated
                    self._save_unlocked(wakeups)
                    return updated
        raise KeyError(f"unknown wakeup: {wakeup_id}")

    def cancel_pending(
        self, *, session_id: str, now: datetime | None = None
    ) -> list[Wakeup]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        cancelled: list[Wakeup] = []
        with self._locked():
            wakeups = self._load_unlocked()
            changed = False
            for index, item in enumerate(wakeups):
                if item.session_id != session_id or item.status not in {
                    "scheduled",
                    "claimed",
                }:
                    continue
                updated = Wakeup(
                    **{
                        **asdict(item),
                        "status": "cancelled",
                        "cancelled_at": _iso(current),
                    }
                )
                wakeups[index] = updated
                cancelled.append(updated)
                changed = True
            if changed:
                self._save_unlocked(wakeups)
        return sorted(cancelled, key=lambda item: (item.wake_at, item.wakeup_id))

    def claim_due(
        self,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
        claim_lease_seconds: float = 60.0,
    ) -> list[Wakeup]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        reclaim_before = current - timedelta(seconds=max(0.0, claim_lease_seconds))
        claimed: list[Wakeup] = []
        with self._locked():
            wakeups = self._load_unlocked()
            changed = False
            for index, item in enumerate(wakeups):
                if session_id is not None and item.session_id != session_id:
                    continue
                if parse_utc_datetime(item.wake_at) > current:
                    continue
                reclaimable = (
                    item.status == "claimed"
                    and item.claimed_at is not None
                    and parse_utc_datetime(item.claimed_at) <= reclaim_before
                )
                if item.status != "scheduled" and not reclaimable:
                    continue
                updated = Wakeup(**{**asdict(item), "status": "claimed", "claimed_at": _iso(current)})
                wakeups[index] = updated
                claimed.append(updated)
                changed = True
            if changed:
                self._save_unlocked(wakeups)
        return sorted(claimed, key=lambda item: (item.wake_at, item.wakeup_id))

    def mark_delivered(self, *, wakeup_id: str, now: datetime | None = None) -> Wakeup:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._locked():
            wakeups = self._load_unlocked()
            for index, item in enumerate(wakeups):
                if item.wakeup_id != wakeup_id:
                    continue
                if item.status == "delivered":
                    return item
                if item.status != "claimed":
                    raise ValueError(f"wakeup must be claimed before delivery: {wakeup_id}")
                updated = Wakeup(**{**asdict(item), "status": "delivered", "delivered_at": _iso(current)})
                wakeups[index] = updated
                self._save_unlocked(wakeups)
                return updated
        raise KeyError(f"unknown wakeup: {wakeup_id}")

    def due(self, *, session_id: str, now: datetime | None = None) -> list[Wakeup]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        return [
            item
            for item in self.list(session_id=session_id)
            if item.status in {"scheduled", "claimed"} and parse_utc_datetime(item.wake_at) <= current
        ]
