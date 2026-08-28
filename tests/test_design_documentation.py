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
