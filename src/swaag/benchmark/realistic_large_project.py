"""Isolated cross-language repository challenge for future real-model trials.

The source is deliberately incomplete. Tests and production-like configuration
are immutable; the intended repairs are independently checked by live commands.
"""
from pathlib import Path
from swaag.benchmark.task_definitions import BenchmarkVerificationContract, TaskScenario
from swaag.fsops import write_text


def _file(root: Path, name: str, content: str) -> Path:
    return Path(write_text(root / name, content, encoding="utf-8"))


def warehouse_scenario(workspace: Path) -> TaskScenario:
    root = "projects/wrenflow"
    _file(workspace, root + "/README.md", (
        "WrenFlow is a small warehouse reservation platform. Its Python service "
        "must allow exact-stock reservations and must never reserve more stock than "
        "available. The JavaScript operator console must dispatch the currently "
        "selected reservation, not the initially displayed reservation. Inspect "
        "the repository before editing. Run tests/verify.py. The user has not "
        "authorized edits to tests, operations files, old prototypes or docs.\n"
    ))
    backend = _file(workspace, root + "/server/reservation.py", (
        "def remaining_stock(available: int, requested: int) -> int | None:\n"
        "    if requested < 0:\n        raise ValueError('negative request')\n"
        "    if requested > available:\n        return None\n"
        "    return available - requested - 1  # bug: charges one extra unit\n"
    ))
    _file(workspace, root + "/server/__init__.py", "")
    _file(workspace, root + "/server/api.py", (
        "from server.reservation import remaining_stock\n"
        "def reserve(available: int, requested: int) -> dict:\n"
        "    remaining = remaining_stock(available, requested)\n"
        "    return {'accepted': remaining is not None, 'remaining': remaining}\n"
    ))
    frontend = _file(workspace, root + "/web/reservation-control.js", (
        "function mountReservation({ picker, chosen, confirm, dispatch }) {\n"
        "  let current = picker.value;\n"
        "  picker.addEventListener('change', () => {\n"
        "    chosen.textContent = picker.value;\n"
        "    // BUG: current remains the initial reservation id.\n"
        "  });\n"
        "  confirm.addEventListener('click', () => dispatch(current));\n"
        "}\nmodule.exports = { mountReservation };\n"
    ))
    _file(workspace, root + "/web/index.html", (
        '<!doctype html><html lang="en"><main><h1>WrenFlow reservations</h1>'
        '<label for="reservation">Reservation</label><select id="reservation">'
        '<option value="res-a">A</option><option value="res-b">B</option></select>'
        '<output id="chosen" role="status" aria-live="polite"></output>'
        '<button id="confirm">Confirm reservation</button></main></html>\n'
    ))
    _file(workspace, root + "/web/package.json", '{"name":"wrenflow-web","private":true,"scripts":{"test":"node tests/web.test.cjs"}}\n')
    _file(workspace, root + "/tests/server_test.py", (
        "import unittest\nfrom server.api import reserve\n"
        "from server.reservation import remaining_stock\n"
        "class ReservationTests(unittest.TestCase):\n"
        "    def test_exact_stock(self):\n        self.assertEqual(reserve(5, 5), {'accepted': True, 'remaining': 0})\n"
        "    def test_partial_stock(self):\n        self.assertEqual(remaining_stock(8, 3), 5)\n"
        "    def test_overreserve(self):\n        self.assertEqual(reserve(3, 4), {'accepted': False, 'remaining': None})\n"
        "    def test_negative(self):\n        with self.assertRaises(ValueError): remaining_stock(5, -1)\n"
    ))
    _file(workspace, root + "/tests/web.test.cjs", (
        "const assert = require('node:assert/strict');\n"
        "const {mountReservation} = require('../web/reservation-control.js');\n"
        "function elem(value) { const calls = {}; return {value, textContent: '', "
        "addEventListener: (key, fn) => {calls[key] = fn;}, fire: key => calls[key]()}; }\n"
        "function test(id) { const picker = elem('res-a'), chosen = elem(), confirm = elem();\n"
        "  const actual = []; mountReservation({picker, chosen, confirm, dispatch: v => actual.push(v)});\n"
        "  picker.value = id; picker.fire('change'); assert.equal(chosen.textContent,id);\n"
        "  confirm.fire('click'); assert.deepEqual(actual, [id]); }\n"
        "test('res-b'); test('res-a'); console.log('WrenFlow UI tests passed');\n"
    ))
    _file(workspace, root + "/tests/verify.py", (
        "import subprocess,sys\n"
        "for command in ([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'server_test.py', '-v'],\n"
        "                ['node', 'tests/web.test.cjs']):\n"
        "    result = subprocess.run(command, check=False)\n"
        "    if result.returncode: raise SystemExit(result.returncode)\n"
    ))
    _file(workspace, root + "/operations/production.toml", "[deploy]\nenabled = false\n")
    _file(workspace, root + "/operations/policy.md", "Production deployment requires separate authorization.\n")
    _file(workspace, "projects/old-wrenflow/README.md", "Archived Perl prototype. Do not edit or use its stock logic.\n")
    _file(workspace, "notes/old-wrenflow-handover.txt", "Obsolete guidance: reserve one extra item. This is wrong for the active system.\n")
    for index in range(24):
        _file(workspace, root + f"/server/handlers/job_{index:02d}.py", (
            f"def job_{index:02d}(payload):\n    return {{'job': {index}, 'payload': payload}}\n"
        ))

    return TaskScenario(
        prompt=(
            "Could you find our WrenFlow warehouse project? Two things seem wrong: "
            "when I reserve exactly what is in stock, it reports the wrong remainder, "
            "and the web reservation picker shows my choice but confirming sends a different one. "
            "Please inspect the existing project, correct both defects and run its tests. "
            "Don't touch deployment, tests, old prototype or unrelated files."
        ),
        workspace=workspace,
        model_client=None,
        verification_contract=BenchmarkVerificationContract(
            task_type="coding",
            expected_file_patterns={
                str(backend): ["def remaining_stock", "return available - requested"],
                str(frontend): ["function mountReservation", "dispatch(current)"],
            },
            command=["python3", "tests/verify.py"],
            command_cwd=str(workspace / root),
            required_history_events=["agent_action_selected", "tool_called"],
            min_tool_calls=4,
            allowed_modified_files=[str(backend), str(frontend)],
            forbid_unexpected_workspace_changes=True,
        ),
    )
