from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from swaag.fsops import ensure_dir, write_text
from swaag.utils import new_id

_DURATION_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)\s*$")
_UNIT_SECONDS = {
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


def parse_duration(text: str) -> timedelta:
    match = _DURATION_RE.match(text)
    if match is None:
        raise ValueError("duration must look like '30 seconds', '2 hours', '3 days', or '1 month'")
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
        self.path = Path(sessions_root).expanduser() / "scheduled_wakeups.json"

    def _load(self) -> list[Wakeup]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("scheduled wakeup store must contain an array")
        return [Wakeup(**item) for item in payload]

    def _save(self, wakeups: list[Wakeup]) -> None:
        ensure_dir(self.path.parent)
        write_text(self.path, json.dumps([asdict(item) for item in wakeups], indent=2, sort_keys=True) + "\n")

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
        wakeups = self._load()
        wakeups.append(wakeup)
        self._save(wakeups)
        return wakeup

    def list(self, *, session_id: str, include_cancelled: bool = False) -> list[Wakeup]:
        result = [item for item in self._load() if item.session_id == session_id]
        if not include_cancelled:
            result = [item for item in result if item.status != "cancelled"]
        return sorted(result, key=lambda item: (item.wake_at, item.wakeup_id))

    def cancel(self, *, session_id: str, wakeup_id: str, now: datetime | None = None) -> Wakeup:
        wakeups = self._load()
        for index, item in enumerate(wakeups):
            if item.session_id == session_id and item.wakeup_id == wakeup_id:
                if item.status == "cancelled":
                    return item
                updated = Wakeup(
                    **{**asdict(item), "status": "cancelled", "cancelled_at": _iso(now or datetime.now(timezone.utc))}
                )
                wakeups[index] = updated
                self._save(wakeups)
                return updated
        raise KeyError(f"unknown wakeup: {wakeup_id}")


    def claim_due(self, *, session_id: str, now: datetime | None = None) -> list[Wakeup]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        wakeups = self._load()
        claimed: list[Wakeup] = []
        changed = False
        for index, item in enumerate(wakeups):
            if item.session_id != session_id or item.status != "scheduled":
                continue
            if parse_utc_datetime(item.wake_at) > current:
                continue
            updated = Wakeup(**{**asdict(item), "status": "delivered"})
            wakeups[index] = updated
            claimed.append(updated)
            changed = True
        if changed:
            self._save(wakeups)
        return sorted(claimed, key=lambda item: (item.wake_at, item.wakeup_id))

    def due(self, *, session_id: str, now: datetime | None = None) -> list[Wakeup]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        return [
            item
            for item in self.list(session_id=session_id)
            if parse_utc_datetime(item.wake_at) <= current
        ]
