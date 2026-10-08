"""Explicitly authorized idle work on the existing durable orchestration graph."""
from __future__ import annotations

import time

from swaag.history import HistoryStore
from swaag.utils import utc_now_iso


class BackgroundWork:
    def __init__(self, manager, *, mode="finish_only", max_pending=128, foreground_busy=None,
                 idea_generator=None, autonomous_idea_interval_seconds=300.0):
        if mode not in {"finish_only", "authorized_backlog", "autonomous_continue"}:
            raise ValueError("background work mode must be finish_only, authorized_backlog, or autonomous_continue")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("background work max_pending must be positive")
        self.manager = manager
        self.store = manager.store
        self.mode = mode
        self.max_pending = max_pending
        self.foreground_busy = foreground_busy or self._workers_busy
        self.idea_generator = idea_generator
        self.autonomous_idea_interval_seconds = float(autonomous_idea_interval_seconds)
        if self.autonomous_idea_interval_seconds <= 0:
            raise ValueError("autonomous idea interval must be positive")
        self._next_autonomous_idea_time = 0.0

    def _workers_busy(self):
        return any(manager.store.list(statuses={"queued", "working", "cancellation_requested", "input_required"})
                   for manager in self.manager.worker_managers.values())

    def list(self):
        with self.store._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM orchestration_backlog WHERE state IN ('pending','held') ORDER BY created_at,plan_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def enqueue(self, plan_id, *, authorization_session_id, authorization_event_sequence):
        """The caller/model interprets authorization; code verifies its exact provenance.

        No natural-language phrase is treated as automatic permission. The complete
        plan revision is frozen; subsequent plan changes require renewed authorization.
        """
        if (isinstance(authorization_event_sequence, bool)
                or not isinstance(authorization_event_sequence, int) or authorization_event_sequence < 1):
            raise ValueError("authorization_event_sequence must be a positive integer")
        history = HistoryStore(self.store.path.parent, write_projections=False)
        events = list(history.iter_history(authorization_session_id,
            start_sequence=authorization_event_sequence, end_sequence=authorization_event_sequence))
        if len(events) != 1 or events[0].event_type != "message_added":
            raise ValueError("background authorization must cite a recorded user message")
        message = events[0].payload.get("message", {})
        if message.get("role") != "user" or not str(message.get("content", "")).strip():
            raise ValueError("background authorization must cite a nonempty user message")
        validation = self.manager.validate_plan(plan_id)
        if not validation["valid"] or not validation["node_count"]:
            raise ValueError("background plan must be nonempty and valid before enqueue")
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            plan = connection.execute("SELECT * FROM orchestration_plans WHERE plan_id=?",(plan_id,)).fetchone()
            if plan is None or plan["status"] != "draft":
                raise ValueError("only an unstarted draft plan can enter the background backlog")
            if plan["revision"] != validation["revision"]:
                raise ValueError("plan changed during background authorization; review the current plan")
            if connection.execute("SELECT 1 FROM orchestration_nodes WHERE plan_id=? AND worker_id IS NOT NULL",(plan_id,)).fetchone():
                raise ValueError("background plans must not already have bound workers")
            previous = connection.execute("SELECT * FROM orchestration_backlog WHERE plan_id=?",(plan_id,)).fetchone()
            if previous is not None and previous["state"] == "pending":
                if (previous["authorized_revision"] == plan["revision"]
                    and previous["authorization_session_id"] == authorization_session_id
                    and previous["authorization_event_sequence"] == authorization_event_sequence):
                    return dict(previous)  # exact retry, no duplicate work or event
                raise ValueError("cancel the queued authorization before replacing it")
            count = connection.execute("SELECT COUNT(*) FROM orchestration_backlog WHERE state IN ('pending','held')").fetchone()[0]
            if count >= self.max_pending and (previous is None or previous["state"] != "held"):
                raise ValueError("background backlog capacity reached; no work was discarded")
            now = utc_now_iso()
            connection.execute("""INSERT INTO orchestration_backlog
                (plan_id,state,authorized_revision,authorization_session_id,authorization_event_sequence,
                 authorization_event_hash,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(plan_id) DO UPDATE SET
                state=excluded.state,authorized_revision=excluded.authorized_revision,
                authorization_session_id=excluded.authorization_session_id,
                authorization_event_sequence=excluded.authorization_event_sequence,
                authorization_event_hash=excluded.authorization_event_hash,
                created_at=excluded.created_at,updated_at=excluded.updated_at""",
                (plan_id,"pending",plan["revision"],authorization_session_id,authorization_event_sequence,
                 events[0].hash,now,now))
            self.store._event(connection,plan_id,"background_work_enqueued",{
                "authorized_revision":plan["revision"],"authorization_session_id":authorization_session_id,
                "authorization_event_sequence":authorization_event_sequence,"authorization_event_hash":events[0].hash})
            row=connection.execute("SELECT * FROM orchestration_backlog WHERE plan_id=?",(plan_id,)).fetchone()
        return dict(row)

    def cancel(self, plan_id):
        """Remove only queued work. Running plans use the ordinary cancel operation."""
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed=connection.execute("UPDATE orchestration_backlog SET state='canceled',updated_at=? WHERE plan_id=? AND state IN ('pending','held')",
                                       (utc_now_iso(),plan_id)).rowcount
            if changed:
                self.store._event(connection,plan_id,"background_work_canceled",{})
        return bool(changed)

    def dispatch_once(self):
        pending=self.list()
        if self.mode == "finish_only":
            return None
        # Blocked foreground work still belongs to the foreground objective.
        if self.store.active_plans() or self.foreground_busy():
            return None
        for item in pending:
            if item["state"] != "pending":
                continue
            plan_id=item["plan_id"]
            with self.store._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current=connection.execute("SELECT * FROM orchestration_backlog WHERE plan_id=?",(plan_id,)).fetchone()
                plan=connection.execute("SELECT * FROM orchestration_plans WHERE plan_id=?",(plan_id,)).fetchone()
                if current["state"] != "pending":
                    continue
                # Recheck after obtaining the write lock, so concurrent dispatchers
                # cannot claim a second plan before the first is recorded active.
                if connection.execute("SELECT 1 FROM orchestration_plans WHERE status='active' LIMIT 1").fetchone():
                    return None
                if plan["status"] != "draft" or plan["revision"] != current["authorized_revision"]:
                    connection.execute("UPDATE orchestration_backlog SET state='held',updated_at=? WHERE plan_id=?",(utc_now_iso(),plan_id))
                    self.store._event(connection,plan_id,"background_work_held",{"reason":"plan changed since authorization"})
                    continue
                connection.execute("UPDATE orchestration_backlog SET state='dispatched',updated_at=? WHERE plan_id=?",(utc_now_iso(),plan_id))
                connection.execute("UPDATE orchestration_plans SET status='active',revision=revision+1,updated_at=? WHERE plan_id=?",(utc_now_iso(),plan_id))
                self.store._event(connection,plan_id,"background_work_dispatched",{"authorized_revision":current["authorized_revision"]})
            # A crash after this commit is recovered by advance_active_plans. The
            # backlog never creates a second plan or repeats a completed plan.
            self.manager.advance(plan_id)
            return plan_id
        if self.mode != "autonomous_continue" or self.idea_generator is None:
            return None
        now = time.monotonic()
        if now < self._next_autonomous_idea_time:
            return None
        self._next_autonomous_idea_time = now + self.autonomous_idea_interval_seconds
        idea = self.idea_generator()
        if not isinstance(idea, dict) or not idea.get("create"):
            return None
        objective = str(idea.get("objective") or "").strip()
        worker_objective = str(idea.get("worker_objective") or "").strip()
        reason = str(idea.get("reason") or "").strip()
        if not objective or not worker_objective or not reason:
            raise ValueError("autonomous idea generator returned an incomplete plan candidate")
        plan = self.manager.create_plan(objective, reporting_mode="important")
        self.manager.add_worker(plan.plan_id, worker_objective)
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.store._event(connection, plan.plan_id, "autonomous_work_generated", {
                "reason": reason, "objective": objective, "worker_objective": worker_objective,
            })
        self.store.set_plan_status(plan.plan_id, "active", reason="autonomous keep-working mode")
        self.manager.advance(plan.plan_id)
        return plan.plan_id
