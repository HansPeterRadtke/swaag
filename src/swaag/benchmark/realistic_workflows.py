"""Independent, model-driven acceptance scenarios written in ordinary user language.

Only the fixture and scoring contract are deterministic. These definitions never
manufacture model actions or classify a real language-model run as passing. The
existing benchmark runner supplies the real production AgentRuntime and model.
"""

from __future__ import annotations

from pathlib import Path

from swaag.benchmark.task_definitions import (
    BenchmarkTaskDefinition,
    BenchmarkVerificationContract,
    TaskScenario,
)
from swaag.fsops import write_text
from swaag.types import Message


def _file(root: Path, relative: str, content: str) -> Path:
    return Path(write_text(root / relative, content, encoding="utf-8"))


def _history(role: str, content: str, minute: int) -> Message:
    return Message(role=role, content=content, created_at=f"2026-10-01T08:{minute:02d}:00+00:00")  # type: ignore[arg-type]


def _spoken_quality(
    workspace: Path, prompt: str, *, necessary: list[str], alternatives: list[list[str]] | None = None,
    prior: list[Message] | None = None, min_calls: int = 0, max_calls: int | None = None,
    required_events: list[str] | None = None,
    observed: list[str] | None = None,
) -> TaskScenario:
    return TaskScenario(
        prompt=prompt,
        workspace=workspace,
        model_client=None,
        history_messages=list(prior or []),
        verification_contract=BenchmarkVerificationContract(
            task_type="quality", expected_answer_contains=necessary,
            expected_answer_any_of=list(alternatives or []),
            required_history_events=["agent_action_selected", *(required_events or [])],
            forbidden_tools_used=["write_file", "edit_text", "run_tests"],
            required_tool_observations=list(observed or []),
            min_tool_calls=min_calls,
            max_tool_calls=max_calls,
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _identity(workspace: Path) -> TaskScenario:
    _file(workspace, "README.md", "An intentionally irrelevant local workspace. Identity comes from the runtime.\n")
    return _spoken_quality(
        workspace,
        "Uh, hey, what are you actually? An agent or just a chat thing?",
        necessary=["SWAAG"], max_calls=0,
    )


def _project_recon(workspace: Path) -> TaskScenario:
    _file(workspace, "apps/harborboard/README.md", "Harborboard: inventory dashboard, Python API and browser UI. Jobs live under jobs/.\n")
    _file(workspace, "apps/harborboard/api/storage.py", "import sqlite3\nDATABASE_URL = 'sqlite:///data/harborboard.sqlite'\n")
    _file(workspace, "apps/harborboard/api/server.py", "from .storage import DATABASE_URL\n\ndef health():\n    return {'application': 'Harborboard', 'db': DATABASE_URL}\n")
    _file(workspace, "apps/harborboard/jobs/importer.py", "def admit_import(status):\n    return status == 'ready'\n")
    _file(workspace, "apps/harborboard/jobs/last_run.log", "2026-10-17 batch 091: rejected; state=SCHEMA_HOLD_42; required column location_id missing\n")
    _file(workspace, "apps/harborboard/web/index.html", "<title>Harborboard stock overview</title>\n")
    _file(workspace, "apps/old-harbor-dashboard/README.md", "Archived prototype. MySQL was used. Old imports once failed with timeout.\n")
    _file(workspace, "notes/old-onboarding.txt", "This note predates Harborboard. It refers to the retired MySQL prototype.\n")
    return _spoken_quality(
        workspace,
        "Could you find that Harborboard stock dashboard thing, think it's under apps? "
        "Just get me an overview, what it does, where it stores data, and why imports aren't going through. "
        "Don't change anything yet.",
        necessary=["Harborboard", "SQLite", "SCHEMA_HOLD_42"], min_calls=2,
        observed=["sqlite:///data/harborboard.sqlite", "SCHEMA_HOLD_42"],
        required_events=["tool_called"],
    )


def _project_recon_typos(workspace: Path) -> TaskScenario:
    scenario = _project_recon(workspace)
    scenario.prompt = (
        "Um, can u find my harbrbord stock dashbord? I can't remembr which folder, "
        "think it was an app. Just check what it does, how the data is stored and "
        "why the import is stcuk. Don't chnge anything."
    )
    return scenario


def _larger_project_recon(workspace: Path) -> TaskScenario:
    _file(workspace, "projects/workshopops/README.md", (
        "WorkshopOps is the production workshop operations application. Inspect source "
        "and deployment data for current architecture; notes/archive-2019 is obsolete. "
        "The main application includes an API, web client, workers, and release tooling.\n"
    ))
    _file(workspace, "projects/workshopops/server/pyproject.toml", (
        "[project]\nname = 'workshopops-api'\nversion = '1.12'\n"
        "dependencies = ['fastapi', 'psycopg']\n"
    ))
    _file(workspace, "projects/workshopops/server/app.py", (
        "from fastapi import FastAPI\nfrom api.routes import router\n"
        "app = FastAPI(title='WorkshopOps')\napp.include_router(router)\n"
    ))
    _file(workspace, "projects/workshopops/server/api/routes.py", (
        "from fastapi import APIRouter\nrouter = APIRouter()\n"
        "@router.get('/jobs')\ndef jobs():\n    return {'status': 'available'}\n"
    ))
    _file(workspace, "projects/workshopops/server/storage.py", (
        "DRIVER = 'PostgreSQL'\nDSN_KEY = 'WORKSHOPOPS_DB_URL'\n"
    ))
    _file(workspace, "projects/workshopops/worker/consumer.py", (
        "QUEUE_BACKEND = 'Redis'\nJOB_TOPIC = 'parts-import'\n"
    ))
    _file(workspace, "projects/workshopops/web/package.json", (
        '{"name":"workshopops-frontend","version":"1.0.0","dependencies":{"svelte":"^4"}}\n'
    ))
    _file(workspace, "projects/workshopops/web/src/App.svelte", (
        "<h1>WorkshopOps</h1>\n<main>Active operations</main>\n"
    ))
    _file(workspace, "projects/workshopops/deploy/release/current.json", (
        '{"revision":"twelve","published":false,"state":"held","reason":"ARTIFACT_MISSING_29"}\n'
    ))
    _file(workspace, "projects/workshopops/deploy/scripts/check-release.sh", (
        "#!/bin/sh\n# Required built web artifact must exist before publication.\n"
    ))
    _file(workspace, "projects/workshopops/notes/archive-2019/old-stack.txt", (
        "2019 prototype: Django and MySQL; do not use for current deployment.\n"
    ))
    _file(workspace, "projects/obsolete-workshopops/README.md", (
        "Older Django/MySQL application retired. No longer deployed.\n"
    ))
    for idx in range(12):
        _file(workspace, f"projects/workshopops/server/api/handlers/handler_{idx:02d}.py", (
            f"def handle_{idx:02d}(record):\n    return {{'record': record, 'source': 'handler-{idx:02d}'}}\n"
        ))
    return _spoken_quality(
        workspace,
        "Take a look at WorkshopOps, wherever it ended up under projects. Need a proper "
        "overview: what's the backend, storage, job queue, and browser frontend built "
        "with? Also why isn't the current release published? Don't make changes, "
        "I just need to understand what is actually there, not the old docs.",
        necessary=["WorkshopOps", "FastAPI", "PostgreSQL", "Redis", "Svelte", "ARTIFACT_MISSING_29"],
        min_calls=1, observed=["ARTIFACT_MISSING_29", "PostgreSQL", "Redis", "svelte"], required_events=["tool_called"],
    )


def _project_followup(workspace: Path) -> TaskScenario:
    _file(workspace, "work/old-orion/README.md", "ORION-LITE: discontinued mobile prototype. File storage only.\n")
    _file(workspace, "work/orionstack/README.md", "OrionStack: current sensor monitoring service.\n")
    _file(workspace, "work/orionstack/server/db.py", "DATABASE_ENGINE = 'PostgreSQL'\n")
    _file(workspace, "work/orionstack/queue/consumer.py", "# queue processing is paused until vendor signature check succeeds\n")
    _file(workspace, "work/orionstack/queue/last_state.json", '{"status":"paused","reason":"QUEUE_PAUSED_17","checked":"2026-10-17"}\n')
    return _spoken_quality(
        workspace,
        "Yeah, that *newer* Orion one, not the old tiny app. Check what database it uses and "
        "what's holding up the sensor feed. Just look, don't fix it.",
        necessary=["OrionStack", "PostgreSQL", "QUEUE_PAUSED_17"], min_calls=2,
        observed=["QUEUE_PAUSED_17", "PostgreSQL"],
        required_events=["tool_called"],
        prior=[
            _history("user", "I have an old project called Orion Lite, and a current one called OrionStack. I only want read-only inspection for now.", 1),
            _history("assistant", "Understood, I will distinguish the active OrionStack repository from Orion Lite.", 2),
        ],
    )


def _source_freshness(workspace: Path) -> TaskScenario:
    _file(workspace, "runbooks/current-status.json", '{"active_revision":"release-12","verification":"queued","result":"unknown"}\n')
    _file(workspace, "archive/previous-revision.log", "release-11 tests complete, all green; previous release only\n")
    _file(workspace, "README.md", "Current status is in runbooks/current-status.json. Archived logs refer to previous revisions.\n")
    return _spoken_quality(
        workspace,
        "So, are the checks already green now? I saw a passing log yesterday but I don't "
        "remember if it was for this build. Have a look and tell me. No changes.",
        necessary=["queued", "release-12"], alternatives=[["unknown", "not finished", "not yet", "pending"]],
        observed=["release-12", "queued"],
        min_calls=1, required_events=["tool_called"],
    )


def _corrected_instruction(workspace: Path) -> TaskScenario:
    _file(workspace, "configs/retention.ini", "retention_days=7\n")
    return _spoken_quality(
        workspace,
        "Actually, no, don't change it yet. How many days is that retention value right now? "
        "Just tell me what's in the file, that's all.",
        necessary=["7"], min_calls=1, required_events=["tool_called"], observed=["retention_days=7"],
        prior=[
            _history("user", "Later I want to set retention_days to 30 in configs/retention.ini.", 1),
            _history("assistant", "That proposed retention change has not been made.", 2),
        ],
    )


def _invoice_project(workspace: Path) -> TaskScenario:
    _file(workspace, "projects/invoicehub/README.md", (
        "InvoiceHub internal invoice preview API. backend.pricing.amount_due receives subtotal, "
        "discount fraction, VAT fraction. Apply discount before VAT and round final amount to cents. "
        "The API preview's amount_due must display the computed amount, not the raw subtotal. "
        "Tests in tests/ describe the public contract. Do not edit tests or migrations.\n"
    ))
    pricing = _file(workspace, "projects/invoicehub/backend/pricing.py", (
        "from decimal import Decimal, ROUND_HALF_UP\n\n"
        "def amount_due(subtotal, discount_rate, tax_rate):\n"
        "    gross = Decimal(str(subtotal))\n"
        "    discount = Decimal(str(discount_rate))\n"
        "    tax = Decimal(str(tax_rate))\n"
        "    discounted = gross * (1 - discount)\n"
        "    due = discounted + gross * tax  # BUG: VAT on undiscounted subtotal\n"
        "    return due.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\n"
    ))
    api = _file(workspace, "projects/invoicehub/backend/api.py", (
        "from backend.pricing import amount_due\n\n"
        "def invoice_preview(invoice):\n"
        "    due = amount_due(invoice['subtotal'], invoice['discount_rate'], invoice['tax_rate'])\n"
        "    return {'invoice_id': invoice['invoice_id'], 'amount_due': str(invoice['subtotal'])}  # BUG: ignored due\n"
    ))
    _file(workspace, "projects/invoicehub/backend/__init__.py", "")
    _file(workspace, "projects/invoicehub/migrations/001_initial.sql", "-- Protected migration, not part of this issue.\n")
    _file(workspace, "projects/invoicehub/tests/test_invoice.py", (
        "import unittest\n"
        "from backend.pricing import amount_due\n"
        "from backend.api import invoice_preview\n\n"
        "class InvoiceTests(unittest.TestCase):\n"
        "    def test_vat_after_discount(self):\n"
        "        self.assertEqual(str(amount_due('100', '0.10', '0.20')), '108.00')\n"
        "    def test_no_discount(self):\n"
        "        self.assertEqual(str(amount_due('42.05', '0', '0.20')), '50.46')\n"
        "    def test_full_discount(self):\n"
        "        self.assertEqual(str(amount_due('100', '1', '0.20')), '0.00')\n"
        "    def test_preview_uses_computed_price(self):\n"
        "        doc = {'invoice_id': 'inv-17', 'subtotal': '100', 'discount_rate': '0.10', 'tax_rate': '0.20'}\n"
        "        self.assertEqual(invoice_preview(doc), {'invoice_id': 'inv-17', 'amount_due': '108.00'})\n"
    ))
    _file(workspace, "projects/invoicehub/frontend/index.html", "<h1>InvoiceHub</h1><div id='invoice-preview'></div>\n")
    _file(workspace, "projects/retired-invoice-portal/README.md", "Legacy archived checkout. This is not InvoiceHub.\n")
    return TaskScenario(
        prompt=(
            "Could you find the InvoiceHub thing? I think the total is wrong when discounts are "
            "involved, and the preview is showing the wrong amount as well. Please sort out both "
            "problems, look at the existing code first and run the project's tests. "
            "Don't change migrations, tests or the old archived portal."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding", expected_file_patterns={
                str(pricing): ["def amount_due", "ROUND_HALF_UP"],
                str(api): ["def invoice_preview", "amount_due"],
            },
            command=["python3", "-m", "unittest", "discover", "-s", "tests", "-v"],
            command_cwd=str(workspace / "projects/invoicehub"), command_framework="unittest",
            required_history_events=["agent_action_selected", "tool_called"],
            min_tool_calls=3,
            allowed_modified_files=[str(pricing), str(api)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def _ui_project(workspace: Path) -> TaskScenario:
    _file(workspace, "work/operator-portal/README.md", (
        "Operator Portal: small offline status editor, CommonJS modules and Node tests. "
        "Status changes must be reported through an aria-live region and Save must commit "
        "the currently selected status. No dependency installation is required.\n"
    ))
    ui = _file(workspace, "work/operator-portal/src/status-editor.js", (
        "function mountStatusEditor({ picker, currentLabel, saveButton, persist }) {\n"
        "  let selected = picker.value;\n"
        "  picker.addEventListener('change', () => {\n"
        "    currentLabel.textContent = picker.value;\n"
        "    // BUG: selected remains the initial value, although the display changes.\n"
        "  });\n"
        "  saveButton.addEventListener('click', () => persist(selected));\n"
        "}\nif (typeof module !== 'undefined' && module.exports) { module.exports = { mountStatusEditor }; }\n"
    ))
    _file(workspace, "work/operator-portal/src/bootstrap.js", (
        "document.addEventListener('DOMContentLoaded', () => {\n"
        "  const picker = document.getElementById('jobStatus');\n"
        "  const savedStatus = document.getElementById('savedStatus');\n"
        "  mountStatusEditor({ picker, currentLabel: document.getElementById('statusLabel'),\n"
        "    saveButton: document.getElementById('saveButton'),\n"
        "    persist: value => { localStorage.setItem('operatorStatus', value);\n"
        "      savedStatus.textContent = 'Saved: ' + value; } });\n"
        "});\n"
    ))
    _file(workspace, "work/operator-portal/index.html", (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Operator Portal</title><style>'
        'body{font:18px system-ui,sans-serif;margin:32px;max-width:600px}'
        'label,select,button,output{display:block;margin:12px 0}'
        'button{min-height:44px;min-width:110px}'
        '</style></head><body><main><h1>Operator Portal</h1>'
        '<label for="jobStatus">Job status</label>'
        '<select id="jobStatus"><option>queued</option><option>running</option><option>finished</option></select>'
        '<output id="statusLabel" role="status" aria-live="polite"></output>'
        '<button id="saveButton" type="button">Save</button>'
        '<output id="savedStatus" role="status" aria-live="polite"></output>'
        '</main><script src="src/status-editor.js"></script>'
        '<script src="src/bootstrap.js"></script></body></html>\n'
    ))
    _file(workspace, "work/operator-portal/tests/status-editor.test.cjs", (
        "const assert = require('node:assert/strict');\n"
        "const { mountStatusEditor } = require('../src/status-editor.js');\n"
        "function element(value) {\n"
        "  const listeners = {};\n"
        "  return { value, textContent: '', addEventListener(name, fn) { listeners[name] = fn; },\n"
        "    fire(name) { if (!listeners[name]) throw new Error('missing event listener ' + name); listeners[name](); } };\n"
        "}\n"
        "function scenario(nextStatus) {\n"
        "  const picker = element('queued'), currentLabel = element(), saveButton = element();\n"
        "  const persisted = [];\n"
        "  mountStatusEditor({ picker, currentLabel, saveButton, persist: value => persisted.push(value) });\n"
        "  picker.value = nextStatus; picker.fire('change');\n"
        "  assert.equal(currentLabel.textContent, nextStatus, 'visible label');\n"
        "  saveButton.fire('click');\n"
        "  assert.deepEqual(persisted, [nextStatus], 'Save must commit the current selection');\n"
        "}\n"
        "scenario('running'); scenario('finished');\n"
        "console.log('UI interaction and save tests passed');\n"
    ))
    _file(workspace, "work/operator-portal/tests/markup-check.py", (
        "from html.parser import HTMLParser\nfrom pathlib import Path\n"
        "class Checker(HTMLParser):\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.by_id = {}\n"
        "    def handle_starttag(self, tag, attrs):\n"
        "        a = dict(attrs)\n"
        "        if 'id' in a: self.by_id[a['id']] = (tag, a)\n"
        "p = Checker(); p.feed(Path('index.html').read_text())\n"
        "assert p.by_id['jobStatus'][0] == 'select'\n"
        "assert p.by_id['statusLabel'][1].get('aria-live') == 'polite'\n"
        "assert p.by_id['saveButton'][0] == 'button'\n"
    ))
    _file(workspace, "work/operator-portal/tests/verify.py", (
        "import subprocess,sys\n"
        "for command in ([sys.executable, 'tests/markup-check.py'], ['node', 'tests/status-editor.test.cjs']):\n"
        "    result = subprocess.run(command, check=False)\n"
        "    if result.returncode: raise SystemExit(result.returncode)\n"
    ))
    _file(workspace, "work/operator-portal/deploy/production.conf", "# untouched production configuration\n")
    _file(workspace, "work/deprecated-portal/README.md", "Old discontinued operator interface.\n")
    return TaskScenario(
        prompt=(
            "Could you check the operator portal? The status picker looks like it changes the job "
            "but when I hit save it keeps the old one. It's annoying. Find the UI code, fix it, "
            "run the tests, and keep the accessible status announcement working. "
            "Leave deployment settings and tests alone."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding", expected_file_patterns={str(ui): ["mountStatusEditor", "persist"]},
            command=["python3", "tests/verify.py"],
            command_cwd=str(workspace / "work/operator-portal"), command_framework="unittest",
            required_history_events=["agent_action_selected", "tool_called"],
            min_tool_calls=3, allowed_modified_files=[str(ui)],
            forbid_unexpected_workspace_changes=True,
        ),
    )


def realistic_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    """Benchmarks require a real model later; fixture generation never needs one."""
    specs = (
        ("realistic_identity_casual", "quality", "easy", ("conversation", "natural-language", "identity"), _identity,
         "Answer a casual identity question without fictional tools or workers."),
        ("realistic_find_project_overview", "quality", "hard", ("project-discovery", "messy-language", "read-only", "repository"), _project_recon,
         "Find an unfamiliar app among a decoy and ground an overview in multiple files."),
        ("realistic_find_project_with_typos", "quality", "hard", ("project-discovery", "typos", "messy-language", "read-only"), _project_recon_typos,
         "Resolve an incorrectly spelled application name and ground the answer in actual files."),
        ("realistic_complex_project_overview", "quality", "extremely_hard", ("project-discovery", "large-source", "architecture", "stale-evidence", "read-only"), _larger_project_recon,
         "Discover a deeper web/API/worker application and reject misleading archived stack information."),
        ("realistic_followup_project_referent", "quality", "hard", ("history", "coreference", "project-discovery", "read-only"), _project_followup,
         "Resolve a casual follow-up to the right project instead of an older namesake."),
        ("realistic_old_log_is_not_current_status", "quality", "normal", ("status", "stale-evidence", "natural-language"), _source_freshness,
         "Avoid promoting an earlier successful build log into current completion."),
        ("realistic_changed_mind_no_edit", "quality", "normal", ("conversation", "correction", "read-only"), _corrected_instruction,
         "Obey the user's correction and inspect current configuration without making the earlier proposed edit."),
        ("realistic_invoicehub_multifile", "coding", "extremely_hard", ("programming", "realistic-code", "multifile", "repository", "recovery"), _invoice_project,
         "Find an existing invoice service, fix two related financial calculation defects, and run regression tests."),
        ("realistic_operator_portal_ui", "coding", "hard", ("programming", "realistic-code", "multifile", "ui", "repository"), _ui_project,
         "Find an existing browser UI, fix a state/save bug, and preserve accessibility checks."),
    )
    definitions = []
    for task_id, family, difficulty, tags, builder, description in specs:
        definitions.append(BenchmarkTaskDefinition(
            task_id=task_id,
            task_type=family,  # type: ignore[arg-type]
            description=description,
            build=builder,
            build_live=builder,
            difficulty=difficulty,
            tags=list(tags),
            setup_instructions=[
                "Create the isolated synthetic project fixture. Do not expose hidden oracle values to the model as instructions.",
                "Run the unchanged production agent loop against the natural-language user prompt.",
                "Score only observed source-backed facts, executed verification, and actual workspace changes.",
            ],
            config_overrides={"tools_allow_side_effect_tools": family == "coding"},
            benchmark_max_total_actions=32 if family == "coding" else 12,
            benchmark_tool_call_budget=50 if family == "coding" else 20,
            benchmark_max_consecutive_identical_tool_calls=4,
        ))
    return definitions
