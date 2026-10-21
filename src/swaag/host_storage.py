"""Read-only host disk headroom observation for SWAAG's current world state.

Storage is shared with other projects. Neither this module nor the chat model
is authorized to delete protected files or claim a write can be completed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from swaag.utils import utc_now_iso


def capture_host_storage(root: Path | str) -> dict[str, Any]:
    """Observe filesystem capacity without scanning directories or mutating it.

    f_bavail excludes filesystem-reserved blocks. The percentage denominator
    matches a nonroot writer's usable capacity, rather than counting reserved
    blocks as available to a SWAAG process. Errors remain explicit unknowns.
    """
    recorded = {"observed_at": utc_now_iso(),
                "scope": "filesystem containing the SWAAG session store; shared with other projects"}
    try:
        stats = os.statvfs(Path(root))
        block = int(stats.f_frsize)
        total = int(stats.f_blocks) * block
        available = int(stats.f_bavail) * block
        used = (int(stats.f_blocks) - int(stats.f_bfree)) * block
        if block <= 0 or total <= 0 or available < 0 or used < 0 or used > total:
            raise ValueError("invalid filesystem capacity counters")
        accessible = used + available
        percent = round(100 * used / accessible, 1) if accessible else None
        return {**recorded, "observation_available": True,
                "available_gib": round(available / (1024 ** 3), 1),
                "used_percent_of_user_accessible_capacity": percent,
                "filesystem_total_gib": round(total / (1024 ** 3), 1),
                "evidence": "kernel statvfs counters; not project-specific quota or guaranteed write success"}
    except (OSError, ValueError, TypeError):
        return {**recorded, "observation_available": False,
                "reason": "filesystem capacity observation unavailable"}
