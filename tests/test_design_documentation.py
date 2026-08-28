from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hmi_contract_preserves_research_backed_requirements() -> None:
    source = (ROOT / "docs" / "hmi-design-requirements.md").read_text(
        encoding="utf-8"
    )

    required_contracts = (
        "Never hide meaningful information with blind clipping",
        "Show complete result",
        "200 percent text resize",
        "320 CSS-pixel viewport",
        "24 by 24 CSS pixels",
        "48 by 48 dp",
        "7 to 10 mm",
        "action-oriented labels",
        "hover the only route",
        'role="tooltip"',
        "reversible, checked, or explicitly confirmed",
        "representative physical hardware",
    )
    for contract in required_contracts:
        assert contract in source

    primary_sources = (
        "https://www.w3.org/TR/WCAG22/",
        "https://support.google.com/accessibility/android/answer/7101858",
        "https://developer.apple.com/design/human-interface-guidelines/typography",
        "https://learn.microsoft.com/en-us/windows/apps/develop/input/guidelines-for-targeting",
        "https://design-system.service.gov.uk/components/button/",
    )
    for url in primary_sources:
        assert url in source


def test_engineering_contract_preserves_recording_driven_risk_workflow() -> None:
    source = (ROOT / "docs" / "engineering-workflow.md").read_text(
        encoding="utf-8"
    )

    required_contracts = (
        "Scale the work semantically",
        "purpose, inputs, outputs, interfaces, invariants",
        "Risk priority follows consequences",
        "exact installed versions",
        "Treat documentation as a hypothesis",
        "Simulate critical conditions",
        "integration or whole-system simulators",
        "Parameterize ranges and hold out variants",
        "higher-fidelity acceptance",
        "Choose structure by responsibility and likely change",
        "repeated behavior into one named, reusable unit",
        "Bundle data and behavior in an object when instances own meaningful",
        "state, identity, lifecycle, or an invariant",
        "Rename a stale `counter` instead of adding",
        "simplest adequate design",
        "does not substitute for behavior",
    )
    for contract in required_contracts:
        assert contract in source

    primary_sources = (
        "https://csrc.nist.gov/pubs/sp/800/218/final",
        "https://developer.android.com/training/testing/fundamentals/strategies",
        "https://developer.android.com/guide/components/activities/testing",
        "https://developer.android.com/training/monitoring-device-state/doze-standby",
        "https://man7.org/linux/man-pages/man8/tc-netem.8.html",
        "https://docs.pytest.org/en/stable/how-to/monkeypatch.html",
        "https://docs.python.org/3/tutorial/classes.html",
        "https://peps.python.org/pep-0008/",
        "https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines",
    )
    for url in primary_sources:
        assert url in source
