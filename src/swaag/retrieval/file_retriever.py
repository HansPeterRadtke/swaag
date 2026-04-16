from __future__ import annotations

from swaag.config import AgentConfig
from swaag.retrieval.ranker import RetrievalCandidate
from swaag.types import SessionState


def file_candidates(state: SessionState, *, config: AgentConfig, counter) -> list[RetrievalCandidate]:
    pol = config.selection_policy
    candidates: list[RetrievalCandidate] = []
    for relative_path, content in sorted(state.environment.workspace.known_files.items()):
        ranking_text = f"{relative_path}\n{content[: pol.retrieval_file_ranking_chars]}"
        structural = (
            pol.retrieval_structural_modified_file
            if relative_path in state.project_state.files_modified
            else 0.0
        )
        candidates.append(
            RetrievalCandidate(
                item_type="environment_file",
                item_id=relative_path,
                source="workspace",
                text=ranking_text,
                token_cost=counter.count_text(content).tokens,
                payload=(relative_path, content),
                metadata={"structural": structural},
            )
        )
    return candidates
