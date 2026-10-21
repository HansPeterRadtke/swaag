"""Read-only, source-linked coverage ledger for the recording-derived Infra rules.

This is a coverage/verification aid, NEVER an auto-compliance classifier. A
lexical match in source, a passing test and a checked requirement are distinct
forms of evidence. The status of every new source block defaults to unverified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


OTHER = frozenset({"README.md", "RESEARCH_2026-08-30.md", "RESEARCH_AUDIT.md",
                   "OPEN_QUESTIONS_AUDIT_2026-10-09.md"})
PREFIXES = (
    ("agents/", "SWAAG agent/runtime"),
    ("gui/", "Client UI / Infra web / Android"),
    ("kmd/", "KnowMoreDiRT / SWAAG integration"),
    ("robotics/", "Robot controller project"),
    ("games/", "Games project"),
    ("trading/", "Trading project"),
    ("projects/", "Infra project/workflow"),
    ("appendix/", "Evidence appendix"),
)
CROSS_CUTTING = frozenset({
    "agents.md", "configuration.md", "constraint_decoding.txt", "logging.md",
    "networking.md", "network_transport_and_testing.md", "performance_reliability.md",
    "programming/README.md", "security_privacy.md", "testing.md",
    "knowledge_retrieval.md", "doc/audio_style.md",
})


def owner(path: str) -> str:
    if path in CROSS_CUTTING:
        return "SWAAG + every applicable project"
    return next((group for prefix, group in PREFIXES if path.startswith(prefix)),
                "Infra policy, research or documentation")


def category(headings: list[str], path: str) -> str:
    descriptor = " ".join(headings).casefold()
    if (path in OTHER or path.startswith("appendix/")
            or "/templates/" in path or path.endswith("INDEX_FOR_BOTS.md")
            or path.endswith("REFERENCES.md")):
        return "supporting evidence / templates / past audit"
    if any(word in descriptor for word in (
        "open questions", "research and", "researched answer", "research status",
        "research references", "research and tradeoffs", "unresolved engineering",
    )):
        return "research or qualification; not automatically a binding requirement"
    if any(word in descriptor for word in (
        "application-dependent possibilities", "alternatives", "brainstorm",
        "optional", "possible implementation", "possibilities",
    )):
        return "application-dependent possibility, not universal policy"
    return "source proposition requiring semantic classification and verification"


def blocks(text: str, path: str) -> list[dict]:
    """Losslessly reference all non-heading, nonblank content in source order."""
    out: list[dict] = []
    headings: list[str] = []
    lines = text.splitlines()
    buf: list[str] = []
    start: int | None = None
    fence = False
    first_header = ""

    def flush(end: int) -> None:
        nonlocal buf, start
        if start is None:
            return
        content = "\n".join(buf)
        if content.strip():
            fingerprint = sha256((path + "\n" + str(start) + "\n" + content).encode()).hexdigest()
            out.append({
                "source_file": path,
                "source_start_line": start,
                "source_end_line": end,
                "source_block_sha256": fingerprint,
                "section": list(headings),
                "source_text": content,
                "semantic_kind": category(headings, path),
                "responsible_repositories": owner(path),
                "acceptance": "NOT_VERIFIED",
                "proof": [],
                "limitations": "source inventory only; requirement meaning and actual effect require independent examination",
            })
        buf = []
        start = None

    for idx, line in enumerate(lines, 1):
        if re.match(r'^#{1,6} ', line):
            flush(idx - 1)
            level = len(line) - len(line.lstrip('#'))
            while len(headings) >= level:
                headings.pop()
            headings.append(line[level:].strip())
            first_header = headings[0] if headings else ""
        elif not line.strip():
            flush(idx - 1)
        elif line.startswith(("- ", "* ")) or re.match(r'^\d+[.)] ', line):
            flush(idx - 1)
            start = idx
            buf.append(line)
        else:
            if start is None:
                start = idx
            buf.append(line)
    flush(len(lines))
    return out


def generate(guidelines: Path, target: Path) -> dict:
    if not guidelines.is_dir():
        raise ValueError("guideline directory is missing")
    norm = sorted(p for p in guidelines.rglob('*.md') if 'original-recordings' not in p.parts)
    original = sorted((guidelines / 'original-recordings').rglob('*.txt'))
    if not norm or not original:
        raise ValueError('guideline tree is incomplete')
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    files = []
    for source in norm:
        raw = source.read_bytes()
        relative = source.relative_to(guidelines).as_posix()
        text = raw.decode('utf-8-sig')
        entries = blocks(text, relative)
        files.append({"path": relative, "sha256": sha256(raw).hexdigest(),
                      "bytes": len(raw), "lines": len(text.splitlines()),
                      "trace_blocks": len(entries), "owner": owner(relative)})
        records.extend(entries)
    preserved = [{"path": p.relative_to(guidelines).as_posix(),
                  "sha256": sha256(p.read_bytes()).hexdigest(),
                  "bytes": p.stat().st_size} for p in original]
    assert len({(r['source_file'],r['source_start_line']) for r in records}) == len(records)
    with (target / 'requirement_source_trace.jsonl').open('w') as stream:
        for entry in records:
            stream.write(json.dumps(entry,ensure_ascii=False,sort_keys=True) + '\n')
    manifest = {
        'guideline_files': files,
        'original_recordings': preserved,
        'source_blocks': len(records),
        'source_directory': str(guidelines),
        'verification_policy': 'ALL source claims unverified until linked to exact code, tests and observed effects',
    }
    (target / 'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2) + '\n')
    return {
        'documents':len(norm), 'recordings':len(original),
        'source_blocks':len(records),
        'owners':dict(Counter(r['responsible_repositories'] for r in records)),
        'semantics':dict(Counter(r['semantic_kind'] for r in records)),
        'accepted_complete':0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--guidelines', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(generate(args.guidelines,args.output),sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
