from __future__ import annotations

import os
import threading
import time

import pytest

from swaag.inference import InferenceRequestCoordinator
from swaag.preemption import ModelCallPreempted


def _coordinator(
    tmp_path, *, capacity: int = 1, aging: float = 1.0, max_running_seconds: float | None = None
):
    return InferenceRequestCoordinator(
        tmp_path,
        backend_key="backend",
        capacity_resolver=lambda: (capacity, "test"),
        poll_seconds=0.005,
        aging_seconds_per_priority=aging,
        max_running_seconds=max_running_seconds,
    )


def _enqueue(coordinator, call_id: str, priority: int = 0):
    return coordinator.enqueue(
        session_id=f"session-{call_id}",
        run_id=f"run-{call_id}",
        call_id=call_id,
        call_kind="agent_action",
        priority=priority,
        source="test",
    )


def test_durable_inference_lifecycle_records_admission_and_completion(tmp_path):
    coordinator = _coordinator(tmp_path, capacity=2)
    queued = _enqueue(coordinator, "call-1", priority=7)
    running = coordinator.acquire(queued.request_id)
    completed = coordinator.complete(queued.request_id)

    assert queued.status == "queued"
    assert running.status == "running"
    assert running.attempt_count == 1
    assert running.backend_capacity == 2
    assert running.capacity_source == "test"
    assert running.queue_wait_seconds is not None
    assert completed.status == "completed"
    assert completed.completed_at is not None
    assert coordinator.by_call_id("call-1") == completed


def test_higher_priority_request_runs_before_requeued_worker(tmp_path):
    coordinator = _coordinator(tmp_path)
    worker = _enqueue(coordinator, "worker", priority=0)
    coordinator.acquire(worker.request_id)
    communication = _enqueue(coordinator, "communication", priority=100)
    coordinator.requeue(worker.request_id, reason="communication preemption")

    admitted = coordinator.acquire(communication.request_id)
    assert admitted.status == "running"
    coordinator.complete(communication.request_id)
    replay = coordinator.acquire(worker.request_id)
    assert replay.attempt_count == 2
    coordinator.complete(worker.request_id)


def test_queue_aging_prevents_permanent_low_priority_starvation(tmp_path):
    coordinator = _coordinator(tmp_path, aging=0.001)
    blocker = _enqueue(coordinator, "blocker", priority=0)
    coordinator.acquire(blocker.request_id)
    old_worker = _enqueue(coordinator, "old-worker", priority=0)
    time.sleep(0.02)
    newer_control = _enqueue(coordinator, "new-control", priority=10)
    coordinator.complete(blocker.request_id)

    admitted = coordinator.acquire(old_worker.request_id, timeout_seconds=1)
    assert admitted.call_id == "old-worker"
    coordinator.complete(old_worker.request_id)
    coordinator.acquire(newer_control.request_id, timeout_seconds=1)
    coordinator.complete(newer_control.request_id)


def test_queued_request_observes_cancellation(tmp_path):
    coordinator = _coordinator(tmp_path)
    blocker = _enqueue(coordinator, "blocker")
    coordinator.acquire(blocker.request_id)
    waiting = _enqueue(coordinator, "waiting")

    with pytest.raises(ModelCallPreempted, match="while queued"):
        coordinator.acquire(waiting.request_id, cancel_check=lambda: True)

    assert coordinator.get(waiting.request_id).status == "queued"
    coordinator.cancel(waiting.request_id, reason="test cancellation")
    assert coordinator.get(waiting.request_id).status == "cancelled"
    coordinator.complete(blocker.request_id)


def test_orphaned_request_is_failed_durably(tmp_path):
    coordinator = _coordinator(tmp_path)
    queued = _enqueue(coordinator, "orphan")
    with coordinator._connect() as connection:
        connection.execute(
            "UPDATE inference_requests SET owner_pid=? WHERE request_id=?",
            (max(os.getpid() + 10_000_000, 99_999_999), queued.request_id),
        )

    reconciled = coordinator.reconcile_orphans()

    assert [item.request_id for item in reconciled] == [queued.request_id]
    assert coordinator.get(queued.request_id).status == "failed"


def test_waiting_request_is_admitted_when_capacity_releases(tmp_path):
    coordinator = _coordinator(tmp_path)
    first = _enqueue(coordinator, "first")
    second = _enqueue(coordinator, "second")
    coordinator.acquire(first.request_id)
    holder = {}

    thread = threading.Thread(
        target=lambda: holder.setdefault(
            "request", coordinator.acquire(second.request_id, timeout_seconds=2)
        )
    )
    thread.start()
    time.sleep(0.03)
    assert thread.is_alive()
    coordinator.complete(first.request_id)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert holder["request"].status == "running"
    coordinator.complete(second.request_id)


def test_stale_running_request_is_failed_even_when_owner_pid_is_alive(tmp_path):
    coordinator = _coordinator(tmp_path, max_running_seconds=1.0)
    first = _enqueue(coordinator, "stale-running")
    running = coordinator.acquire(first.request_id)
    assert running.status == "running"
    with coordinator._connect() as connection:
        connection.execute(
            "UPDATE inference_requests SET started_at=?, updated_at=? WHERE request_id=?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", first.request_id),
        )

    reconciled = coordinator.reconcile_orphans()

    assert [item.request_id for item in reconciled] == [first.request_id]
    failed = coordinator.get(first.request_id)
    assert failed is not None
    assert failed.status == "failed"
    assert "running lease exceeded" in (failed.error or "")


def test_stale_running_request_releases_capacity_for_same_process(tmp_path):
    coordinator = _coordinator(tmp_path, capacity=1, max_running_seconds=1.0)
    first = _enqueue(coordinator, "stale-running")
    coordinator.acquire(first.request_id)
    with coordinator._connect() as connection:
        connection.execute(
            "UPDATE inference_requests SET started_at=?, updated_at=? WHERE request_id=?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", first.request_id),
        )
    second = _enqueue(coordinator, "next")

    admitted = coordinator.acquire(second.request_id, timeout_seconds=1.0)

    assert admitted.status == "running"
    assert coordinator.get(first.request_id).status == "failed"
    coordinator.complete(second.request_id)


def test_suspended_preempted_request_does_not_block_control_inference(tmp_path):
    coordinator = _coordinator(tmp_path, aging=0.001)
    worker = _enqueue(coordinator, "worker", priority=0)
    coordinator.acquire(worker.request_id)
    suspended = coordinator.suspend(worker.request_id, reason="communication preemption")
    assert suspended.status == "suspended"
    control = _enqueue(coordinator, "control", priority=100)
    admitted = coordinator.acquire(control.request_id, timeout_seconds=1.0)
    assert admitted.request_id == control.request_id
    coordinator.complete(control.request_id)
    resumed = coordinator.resume(worker.request_id, reason="communication resolved")
    assert resumed.status == "queued"
    assert resumed.started_at is None
    replay = coordinator.acquire(worker.request_id, timeout_seconds=1.0)
    assert replay.status == "running"
    assert replay.attempt_count == 2
    coordinator.complete(worker.request_id)


def test_suspended_request_can_finish_terminally(tmp_path):
    coordinator = _coordinator(tmp_path)
    worker = _enqueue(coordinator, "worker")
    coordinator.acquire(worker.request_id)
    coordinator.suspend(worker.request_id, reason="preempted")
    failed = coordinator.fail(worker.request_id, error="communication failed")
    assert failed.status == "failed"
