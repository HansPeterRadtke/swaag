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
                   "OPEN_QUESTIONS_AUDIT_2026-10-09.md", "infra/README.md"})
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
    "tests.txt", "constraint_decoding.txt", "infra/AGENTS.md", "infra/README.md",
})


def owner(path: str) -> str:
    if path.startswith('infra/'):
        return 'Infra repository/workflow'
    if path == 'constraint_decoding.txt':
        return 'OpenAI-compatible structured-output provider integration'
    if path in CROSS_CUTTING:
        return "SWAAG + every applicable project"
    return next((group for prefix, group in PREFIXES if path.startswith(prefix)),
                "Infra policy, research or documentation")


def source_scope(path: str) -> str:
    """Scope/precedence metadata prevents older guides and repo notes leaking."""
    return {
        'tests.txt': 'Older supplementary testing guidance; current testing.md and recorded rules take precedence on conflict.',
        'constraint_decoding.txt': 'OpenRouter/OpenAI-compatible structured-output providers; verify support before applying to local llama.cpp.',
        'infra/AGENTS.md': 'Binding only for operations inside the Infra repository and owned infrastructure services.',
        'infra/README.md': 'Repository overview and navigation, not a general-purpose agent policy.',
    }.get(path, 'Classify source rule, hypothesis, example and applicability from original context.')


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
    fence_kind: str | None = None
    fence_width = 0

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
                "guideline_scope": source_scope(path),
                "acceptance": "NOT_VERIFIED",
                "proof": [],
                "limitations": "source inventory only; requirement meaning and actual effect require independent examination",
            })
        buf = []
        start = None

    for idx, line in enumerate(lines, 1):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence_kind is not None:
            if start is None:
                start = idx
            buf.append(line)
            if (marker is not None and marker.group(1)[0] == fence_kind
                    and len(marker.group(1)) >= fence_width
                    and not marker.group(2).strip()):
                fence_kind = None
            continue
        if marker is not None:
            flush(idx - 1)
            start = idx
            buf = [line]
            fence_kind = marker.group(1)[0]
            fence_width = len(marker.group(1))
            continue
        if re.match(r'^#{1,6} ', line):
            flush(idx - 1)
            level = len(line) - len(line.lstrip('#'))
            while len(headings) >= level:
                headings.pop()
            headings.append(line[level:].strip())
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


def heading_records(text: str, path: str) -> list[dict]:
    """Preserve real Markdown heading lines, excluding headings in code blocks.

    Headings can contain binding words such as MUST/NEVER. They must remain
    retrievable by source position, but neither a heading nor its position
    establishes proof of an implemented feature.
    """
    out = []
    sections: list[str] = []
    fence_kind = None
    fence_width = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence_kind is not None:
            if (marker is not None and marker.group(1)[0] == fence_kind
                    and len(marker.group(1)) >= fence_width
                    and not marker.group(2).strip()):
                fence_kind = None
            continue
        if marker is not None:
            fence_kind = marker.group(1)[0]
            fence_width = len(marker.group(1))
            continue
        heading = re.match(r'^(#{1,6}) (.*)$', line)
        if heading is None:
            continue
        level = len(heading.group(1))
        sections = sections[:level-1]
        sections.append(heading.group(2).strip())
        out.append({
            'source_file': path, 'source_start_line': line_number,
            'source_end_line': line_number,
            'source_block_sha256': sha256((path + '\n' + str(line_number) + '\n' + line).encode()).hexdigest(),
            'section': list(sections), 'source_text': line,
            'semantic_kind': 'section heading, not independent feature proof',
            'responsible_repositories': owner(path), 'guideline_scope': source_scope(path),
            'acceptance': 'NOT_VERIFIED',
            'proof': [],
            'limitations': 'section header only; may carry policy wording, not proof of compliance',
        })
    return out


def generate(guidelines: Path, target: Path, *, infra_root: Path | None = None) -> dict:
    if not guidelines.is_dir():
        raise ValueError("guideline directory is missing")
    norm = sorted(p for p in guidelines.rglob('*.md') if 'original-recordings' not in p.parts)
    source_files = [(p, p.relative_to(guidelines).as_posix()) for p in norm]
    # Two older but still meaningful companion policy documents are plain
    # text. They are NOT original voice transcripts. Read only these
    # explicitly named policy files; never sweep arbitrary adjacent text.
    for name in ('constraint_decoding.txt', 'tests.txt'):
        source = guidelines / name
        if source.is_file():
            source_files.append((source, name))
    # Infra repository ownership rules sit outside docs/guidlines; they must
    # be explicitly supplied by the caller, not inferred from arbitrary parent
    # directories or an untrusted symlink that merely resembles an Infra root.
    if infra_root is not None:
        for name in ('AGENTS.md', 'README.md'):
            source = infra_root / name
            if source.is_file():
                source_files.append((source, 'infra/' + name))
    if infra_root is not None:
        required_files = (guidelines/'constraint_decoding.txt', guidelines/'tests.txt',
                          infra_root/'AGENTS.md', infra_root/'README.md')
        for path in required_files:
            if path.is_symlink():
                raise ValueError('symlinked authoritative guideline source: ' + str(path))
            if not path.is_file():
                raise ValueError('missing required Infra guideline source: ' + str(path))
    source_files.sort(key=lambda row: row[1])
    if len({alias for _,alias in source_files}) != len(source_files):
        raise ValueError('duplicate guideline source aliases')
    original = sorted((guidelines / 'original-recordings').rglob('*.txt'))
    if not norm or not original:
        raise ValueError('guideline tree is incomplete')
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    headings: list[dict] = []
    files = []
    for source, relative in source_files:
        raw = source.read_bytes()
        text = raw.decode('utf-8-sig')
        entries = blocks(text, relative)
        page_headings = heading_records(text, relative)
        # Every nonempty original source line must be present in exactly one
        # body passage or real heading record. Fenced code examples are kept
        # intact and never mistaken for actual Markdown headings.
        coverage: dict[int, int] = {}
        for entry in entries + page_headings:
            for number in range(entry['source_start_line'], entry['source_end_line'] + 1):
                if text.splitlines()[number-1].strip():
                    coverage[number] = coverage.get(number, 0) + 1
        substantial = {index for index, line in enumerate(text.splitlines(), 1)
                       if line.strip()}
        if set(coverage) != substantial or any(count != 1 for count in coverage.values()):
            raise ValueError('nonblank source lines lost or double-counted: ' + relative)
        files.append({"path": relative, "sha256": sha256(raw).hexdigest(),
                      "bytes": len(raw), "lines": len(text.splitlines()),
                      "trace_blocks": len(entries), "heading_records": len(page_headings),
                      "nonblank_lines_covered": len(substantial), "owner": owner(relative),
                      "guideline_scope": source_scope(relative)})
        records.extend(entries)
        headings.extend(page_headings)
    preserved = [{"path": p.relative_to(guidelines).as_posix(),
                  "sha256": sha256(p.read_bytes()).hexdigest(),
                  "bytes": p.stat().st_size} for p in original]
    assert len({(r['source_file'],r['source_start_line']) for r in records}) == len(records)
    with (target / 'requirement_source_trace.jsonl').open('w') as stream:
        for entry in records:
            stream.write(json.dumps(entry,ensure_ascii=False,sort_keys=True) + '\n')
    with (target / 'source_headings.jsonl').open('w') as stream:
        for heading in headings:
            stream.write(json.dumps(heading,ensure_ascii=False,sort_keys=True) + '\n')
    manifest = {
        'guideline_files': files,
        'original_recordings': preserved,
        'source_blocks': len(records), 'heading_records': len(headings),
        'all_nonblank_lines_covered': True,
        'source_directory': str(guidelines),
        'owner_root': str(infra_root) if infra_root is not None else None,
        'verification_policy': 'ALL source claims unverified until linked to exact code, tests and observed effects',
    }
    (target / 'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2) + '\n')
    return {
        'documents':len(source_files), 'recordings':len(original),
        'source_blocks':len(records), 'heading_records':len(headings),
        'all_nonblank_lines_covered': True,
        'owners':dict(Counter(r['responsible_repositories'] for r in records)),
        'semantics':dict(Counter(r['semantic_kind'] for r in records)),
        'accepted_complete':0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--guidelines', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--infra-root', type=Path,
                   help='Explicit Infra repository root, for root AGENTS.md and README.md')
    args = p.parse_args()
    print(json.dumps(generate(args.guidelines,args.output,infra_root=args.infra_root),
                     sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
