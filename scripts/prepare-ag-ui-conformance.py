#!/usr/bin/env python3

from __future__ import annotations

import argparse

from swaag.communication import CommunicationStore
from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.utils import stable_json_dumps
from swaag.workers import WorkerManager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a completed durable run for an AG-UI SDK replay probe."
    )
    parser.add_argument("thread_id")
    parser.add_argument("run_id")
    parser.add_argument("result")
    parser.add_argument("--objective", default="Exercise official AG-UI SDK replay")
    args = parser.parse_args()

    runtime = AgentRuntime(load_config())
    manager = WorkerManager(runtime)
    try:
        worker = manager.create(args.objective)
        manager.store.transition(
            worker.worker_id,
            "queued",
            expected={"created"},
            event_type="worker_queued",
        )
        manager.store.transition(
            worker.worker_id,
            "working",
            expected={"queued"},
            increment_run_count=True,
            event_type="worker_started",
        )
        completed = manager.store.transition(
            worker.worker_id,
            "completed",
            expected={"working"},
            result=args.result,
            event_type="worker_completed",
        )
        end_sequence = manager.events(worker.worker_id)[-1].sequence
        protocol_store = CommunicationStore(runtime.config.sessions.root)
        protocol_store.set_protocol_worker("ag_ui", args.thread_id, worker.worker_id)
        protocol_store.record_protocol_message(
            "ag_ui",
            args.run_id,
            args.thread_id,
            worker.worker_id,
            start_sequence=0,
        )
        protocol_store.finish_protocol_message(
            "ag_ui", args.run_id, end_sequence=end_sequence
        )
        print(
            stable_json_dumps(
                {
                    "worker_id": completed.worker_id,
                    "session_id": completed.session_id,
                    "thread_id": args.thread_id,
                    "run_id": args.run_id,
                    "end_sequence": end_sequence,
                },
                indent=2,
            )
        )
    finally:
        manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
