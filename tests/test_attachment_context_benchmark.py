from __future__ import annotations

import pytest

from swaag.benchmark.attachment_context import (
    CASES,
    select_cases,
    verify_attachment_case,
)
from swaag.benchmark.benchmark_runner import _build_parser


def _read_output(*, text: str, finished: bool) -> dict:
    return {
        "text": text,
        "finished": finished,
        "source_event_references": [
            {
                "event_type": "attachment_added",
                "hash": "source-event-hash",
            }
        ],
    }


def _case(case_id: str):
    return next(case for case in CASES if case.case_id == case_id)


def test_attachment_context_case_selection_is_explicit() -> None:
    selected = select_cases(["sequential_raw_reexpansion"])
    assert [case.case_id for case in selected] == ["sequential_raw_reexpansion"]
    with pytest.raises(ValueError, match="Unknown attachment-context case"):
        select_cases(["missing"])


def test_no_inspection_case_rejects_extra_reads_and_inexact_answer() -> None:
    case = _case("no_unnecessary_inspection")
    passed = verify_attachment_case(
        case,
        assistant_text="READY",
        read_outputs=[],
        projection_events=[],
    )
    extra_read = verify_attachment_case(
        case,
        assistant_text="READY",
        read_outputs=[_read_output(text="irrelevant", finished=True)],
        projection_events=[],
    )
    padded_answer = verify_attachment_case(
        case,
        assistant_text="The answer is READY",
        read_outputs=[],
        projection_events=[],
    )
    assert passed["passed"] is True
    assert extra_read["passed"] is False
    assert padded_answer["passed"] is False


def test_reexpansion_requires_grounded_complete_evidence() -> None:
    case = _case("sequential_raw_reexpansion")
    incomplete = verify_attachment_case(
        case,
        assistant_text="K7M-2041-ZETA",
        read_outputs=[_read_output(text="routine prefix", finished=False)],
        projection_events=[],
    )
    grounded = verify_attachment_case(
        case,
        assistant_text="K7M-2041-ZETA",
        read_outputs=[
            _read_output(text="routine prefix", finished=False),
            _read_output(text="attestation-code=K7M-2041-ZETA", finished=True),
        ],
        projection_events=[],
    )
    assert incomplete["passed"] is False
    assert grounded["passed"] is True


def test_partial_inspection_stops_after_grounded_answer() -> None:
    case = _case("partial_raw_inspection")
    partial = verify_attachment_case(
        case,
        assistant_text="DEC-482-NORTH",
        read_outputs=[
            _read_output(text="routine prefix", finished=False),
            _read_output(text="decision-code=DEC-482-NORTH", finished=False),
        ],
        projection_events=[],
    )
    unnecessary_completion = verify_attachment_case(
        case,
        assistant_text="DEC-482-NORTH",
        read_outputs=[
            _read_output(text="decision-code=DEC-482-NORTH", finished=False),
            _read_output(text="unrelated appendix", finished=True),
        ],
        projection_events=[],
    )
    assert partial["passed"] is True
    assert unnecessary_completion["passed"] is False


def test_projection_case_requires_projection_with_exact_source_identity() -> None:
    case = _case("oversized_result_projection")
    read_outputs = [
        _read_output(text="projection-code=PX9-771-OMEGA", finished=True)
    ]
    missing = verify_attachment_case(
        case,
        assistant_text="PX9-771-OMEGA",
        read_outputs=read_outputs,
        projection_events=[],
    )
    projected = verify_attachment_case(
        case,
        assistant_text="PX9-771-OMEGA",
        read_outputs=read_outputs,
        projection_events=[
            {"source_event_sequence": 7, "source_event_hash": "tool-result-hash"}
        ],
    )
    assert missing["passed"] is False
    assert projected["passed"] is True


def test_specialist_case_requires_grounded_ocr_extraction() -> None:
    case = _case("specialist_ocr_inspection")
    source_references = _read_output(text="", finished=True)[
        "source_event_references"
    ]
    base_output = {
        "text": "SWAAG OCR MARKER 48291",
        "source_event_references": source_references,
        "artifact_id": "artifact_exact",
        "artifact_sha256": "sha256",
    }
    missing_specialist = verify_attachment_case(
        case,
        assistant_text="SWAAG OCR MARKER 48291",
        read_outputs=[],
        projection_events=[],
        extraction_outputs=[
            {**base_output, "manifest": {"entry": {"ocr_used": False}}}
        ],
    )
    grounded = verify_attachment_case(
        case,
        assistant_text="SWAAG OCR MARKER 48291",
        read_outputs=[],
        projection_events=[],
        extraction_outputs=[
            {**base_output, "manifest": {"entry": {"ocr_used": True}}}
        ],
    )
    unnecessary_raw_read = verify_attachment_case(
        case,
        assistant_text="SWAAG OCR MARKER 48291",
        read_outputs=[_read_output(text="", finished=True)],
        projection_events=[],
        extraction_outputs=[
            {**base_output, "manifest": {"entry": {"ocr_used": True}}}
        ],
    )

    assert missing_specialist["passed"] is False
    assert grounded["passed"] is True
    assert unnecessary_raw_read["passed"] is False


def test_attachment_context_cli_accepts_case_filter() -> None:
    args = _build_parser().parse_args(
        ["attachment-context", "--case", "sequential_raw_reexpansion"]
    )
    assert args.command == "attachment-context"
    assert args.case == ["sequential_raw_reexpansion"]
