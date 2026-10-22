# SWAAG nightly checkpoint — 2026-10-22, midnight Europe/Berlin

Scope: Calendar event `0d6ulkihfueaiulaco4ek2rd90`, development of `/data/src/github/swaag` on Nitro. This is a verified checkpoint, not a claim that the guideline implementation or live deployment is complete.

## Repository / concurrency

- Nitro main checkout started clean at `c4476a6`, matching `origin/main`; no code was changed or merged.
- Separate acceptance worktree `/data/src/worktrees/swaag-realistic-acceptance-20261018` on `test/nitro-browser-acceptance-20261018` exists and was clean at inspection. Do not overwrite or reset it; preserve external voice-agent work.
- No SWAAG-named user/system service was returned by a Nitro unit-list attempt; `systemctl --user` could not connect to its bus. This does not establish that SWAAG is absent on other hosts or ports. A real end-to-end runtime test is pending.

## Guidelines consulted

Read infra `AGENTS.md`, `docs/guidlines/README.md`, programming, testing, performance and audio-style guidance, and agent architecture, fast dispatch/preemption, answer-first, reusable workspace, logging, configuration, networking, security, retrieval and progress requirements. Domain-specific recordings and the entire source-preservation appendix were not independently audited in this run; do not treat this as a complete requirement-to-source audit.

## Reproducible test environment blockers

1. `python3 -m pytest` in the shared `/data/venv` fails before collecting tests: missing `opentelemetry.context` (the global installation has namespace files but lacks the core API). Do not alter global site-packages to mask this.
2. Successfully created a project-local `.venv` using `uv sync --locked --extra test --no-dev` without changing tracked sources or the lockfile. This environment loads pytest, OpenTelemetry and SWAAG.
3. In the isolated `.venv`, the command below collects tests but the first selected status-freshness test fails during `HistoryStore` initialization: SWAAG requires `sqlite3.sqlite_version >= 3.51.3`; the environment has `3.50.4`. The guard resides in `src/swaag/history.py::_init_sqlite_history`. Do not remove the guard without verifying database schema and concurrency guarantees. Exact command:

   `.venv/bin/python -m pytest -q tests/test_status_freshness.py tests/test_communication_status.py tests/test_foreground_world_view.py tests/test_status_admission.py --disable-warnings --maxfail=1`

4. The runtime environment advertised in the repository README (`/data/var/swaag/venv`) is absent on Nitro. Deployment documents / actual live hosts and Python build must be reconciled before installation or production claims.

## Follow-up

- Establish a project-isolated Python with SQLite at or above the required version; rerun the selected tests and whole deterministic suite, investigate any actual failures, and only then patch defects with regression coverage.
- Inspect live host deployment, request/status worker snapshots and voice-agent process state without touching active voice-agent changes; run end-to-end model/transport latency tests against real deployed instances.
- Continue the infra guideline coverage map and update this document with results; do not claim that percentage estimates or context snapshots are correct without execution evidence.

Additional isolated test-environment probe: `pysqlite3-binary==0.5.4.post2` was installed only into the ignored `.venv`; its bundled SQLite is `3.51.1`, still below the application's `3.51.3` floor, so it is not an acceptable substitute. No runtime database files were migrated and the safeguard was not bypassed.
