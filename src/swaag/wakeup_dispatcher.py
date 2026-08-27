from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.scheduler import WakeupStore, parse_utc_datetime

if TYPE_CHECKING:
    from swaag.workers import WorkerManager


def due_session_ids(config: AgentConfig, *, now: datetime | None = None) -> list[str]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    store = WakeupStore(config.sessions.root)
    session_ids = {
        item.session_id
        for item in store.list_all()
        if item.status in {"scheduled", "claimed"} and parse_utc_datetime(item.wake_at) <= current
    }
    return sorted(session_ids)


def dispatch_once(
    config: AgentConfig,
    *,
    runtime: AgentRuntime | None = None,
    workers: "WorkerManager | None" = None,
    now: datetime | None = None,
) -> list[str]:
    agent = runtime or AgentRuntime(config)
    resumed: list[str] = []
    for session_id in due_session_ids(config, now=now):
        state = agent.create_or_load_session(session_id)
        if now is not None:
            agent._deliver_due_wakeups(state, now=now)
        worker = (
            workers.dispatch_pending_controls_for_session(session_id)
            if workers is not None
            else None
        )
        if worker is not None:
            resumed.append(session_id)
            continue
        result = agent.run_pending_controls_in_session(state)
        if result is not None:
            resumed.append(session_id)
    return resumed


def run_forever(config: AgentConfig, *, poll_seconds: float = 1.0) -> None:
    from swaag.workers import WorkerManager

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    runtime = AgentRuntime(config)
    workers = WorkerManager(
        runtime,
        max_workers=config.communication.max_concurrent_requests,
    )
    workers.reconcile_orphans()
    try:
        while True:
            dispatch_once(config, runtime=runtime, workers=workers)
            time.sleep(poll_seconds)
    finally:
        workers.shutdown(wait=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch durable SWAAG wakeups and resume due sessions.")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    config = load_config(list(args.config))
    if args.once:
        from swaag.workers import WorkerManager

        runtime = AgentRuntime(config)
        workers = WorkerManager(
            runtime,
            max_workers=config.communication.max_concurrent_requests,
        )
        workers.reconcile_orphans()
        try:
            for session_id in dispatch_once(
                config, runtime=runtime, workers=workers
            ):
                print(session_id)
        finally:
            workers.shutdown()
        return 0
    run_forever(config, poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
