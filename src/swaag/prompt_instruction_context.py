from __future__ import annotations

import copy
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from swaag.context_compiler import ContextCompilation
from swaag.grammar import prompt_instruction_projection_contract, prompt_instruction_selection_contract
from swaag.model import ModelClientError
from swaag.preemption import ModelCallStateChanged, RunCancellationRequested
from swaag.prompt_instruction_store import PromptInstructionStore
from swaag.prompt_instructions import (
    MAX_PROMPT_INSTRUCTION_CATEGORIES,
    MAX_PROMPT_INSTRUCTION_CATEGORY_CHARS,
    is_trusted_prompt_instruction,
    prompt_instructions_for_kind,
    sort_prompt_instructions_by_authority,
)
from swaag.tools.base import SemanticCallRequest
from swaag.types import (
    ContractSpec,
    ModelCallKind,
    PromptArtifact,
    PromptAssembly,
    PromptComponent,
    PromptMessageRange,
    SessionState,
)
from swaag.utils import sha256_text, stable_json_dumps

if TYPE_CHECKING:
    from swaag.runtime import AgentRuntime


class PromptInstructionContextManager:
    def __init__(
        self, runtime: "AgentRuntime", store: PromptInstructionStore
    ) -> None:
        self.runtime = runtime
        self.store = store

    def inject(
        self,
        state: SessionState | None,
        assembly: PromptAssembly,
    ) -> None:
        runtime = self.runtime
        if state is None or any(
            component.name
            in {
                "durable_prompt_instructions",
                "durable_prompt_instruction_projection",
            }
            for component in assembly.components
        ):
            return
        scoped_sources = self._prompt_instruction_sources(state, assembly.kind)
        if not scoped_sources:
            return
        selected_sources, selection = self._select_prompt_instruction_sources(
            state,
            assembly,
            scoped_sources,
        )
        if not selected_sources:
            runtime.history.record_event(
                state,
                "prompt_instructions_selected",
                {
                    "kind": assembly.kind,
                    "instruction_ids": [],
                    "instruction_sources": [],
                    "instruction_hashes": [],
                    "exact": True,
                    **selection,
                },
            )
            return
        selected_references = [
            {
                "instruction_store": instruction_store,
                "instruction_id": item.instruction_id,
            }
            for instruction_store, item in selected_sources
        ]
        assembly.metadata["prompt_instruction_sources"] = selected_references
        rendered_rows = [
            {"instruction_store": instruction_store, **asdict(item)}
            for instruction_store, item in selected_sources
        ]
        rendered = stable_json_dumps(rendered_rows, indent=2)
        component = PromptComponent(
            name="durable_prompt_instructions",
            category="system_prompt_instruction",
            text=(
                "\n\n[DURABLE INSTRUCTIONS SELECTED FOR THIS CALL]\n"
                "Apply every instruction below. The current user request remains the highest "
                "authority for this turn. Within durable instructions, higher authority wins "
                "on conflict; for equal authority, higher specificity then newer updated_at wins. "
                "Trusted recording/user/project instructions are never semantically deselected by a model. "
                "Learned-model instructions are lower-authority operating preferences.\n"
                + rendered
            ),
        )
        insert_at = next(
            (
                index
                for index, existing in enumerate(assembly.components)
                if existing.name == "fallback_message_separator"
            ),
            None,
        )
        if insert_at is None:
            raise ModelClientError(
                "Prompt assembly is missing the system/user message separator"
            )
        assembly.components.insert(insert_at, component)
        ranges: list[PromptMessageRange] = []
        for message_range in assembly.message_ranges:
            start = message_range.component_start
            end = message_range.component_end
            if message_range.role == "system" and end == insert_at:
                end += 1
            else:
                if start >= insert_at:
                    start += 1
                if end >= insert_at:
                    end += 1
            ranges.append(
                PromptMessageRange(
                    role=message_range.role,
                    component_start=start,
                    component_end=end,
                )
            )
        assembly.message_ranges = ranges
        assembly.prompt_text = "".join(item.text for item in assembly.components)
        instruction_hashes = [
            {
                "instruction_id": item.instruction_id,
                "instruction_store": instruction_store,
                "sha256": sha256_text(
                    stable_json_dumps(
                        {
                            "instruction_store": instruction_store,
                            **asdict(item),
                        },
                        indent=None,
                    )
                ),
            }
            for instruction_store, item in selected_sources
        ]
        combined_hash = sha256_text(rendered)
        assembly.prompt_artifacts.append(
            PromptArtifact(
                source=f"durable_prompt_instructions:{assembly.kind}",
                sha256=combined_hash,
            )
        )
        runtime.history.record_event(
            state,
            "prompt_instructions_selected",
            {
                "kind": assembly.kind,
                "instruction_ids": [
                    item.instruction_id for _, item in selected_sources
                ],
                "instruction_sources": selected_references,
                "instruction_hashes": instruction_hashes,
                "exact": True,
                **selection,
            },
        )

    def _prompt_instruction_sources(
        self,
        state: SessionState,
        kind: ModelCallKind,
    ) -> list[tuple[str, Any]]:
        runtime = self.runtime
        rows = [
            ("user", item)
            for item in prompt_instructions_for_kind(
                self.store.list(),
                kind,
            )
        ] + [
            ("session", item)
            for item in prompt_instructions_for_kind(
                state.prompt_instructions,
                kind,
            )
        ]
        ordered = sort_prompt_instructions_by_authority([item for _, item in rows])
        source_by_id = {item.instruction_id: store for store, item in rows}
        return [(source_by_id[item.instruction_id], item) for item in ordered]

    def _select_prompt_instruction_sources(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        scoped_sources: list[tuple[str, Any]],
    ) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
        runtime = self.runtime
        candidates = [
            source
            for source in scoped_sources
            if source[1].categories and not is_trusted_prompt_instruction(source[1])
        ]
        if not candidates:
            return scoped_sources, {
                "semantic_selection": False,
                "selection_fallback": False,
                "operation_categories": [],
                "selection_reason": "No fine-grained categorized candidates.",
            }
        target_rows = [
            asdict(component)
            for component in assembly.components
            if component.category != "wrapper" and component.include_in_context
        ]
        target_context = stable_json_dumps(
            {"call_kind": assembly.kind, "components": target_rows},
            indent=2,
        )
        target_context_sha256 = sha256_text(target_context)
        candidate_rows = [
            {"instruction_store": instruction_store, **asdict(instruction)}
            for instruction_store, instruction in candidates
        ]
        candidate_references = [
            {
                "instruction_store": instruction_store,
                "instruction_id": instruction.instruction_id,
                "sha256": sha256_text(
                    stable_json_dumps(
                        {
                            "instruction_store": instruction_store,
                            **asdict(instruction),
                        },
                        indent=None,
                    )
                ),
            }
            for instruction_store, instruction in candidates
        ]
        system_template = (
            runtime.config.prompts.prompt_instruction_selection_system_template
        )
        user_template = runtime.config.prompts.prompt_instruction_selection_template
        user_text = runtime.prompts.template_text(user_template).format(
            call_kind=assembly.kind,
            target_context_sha256=target_context_sha256,
            target_context=target_context,
            candidate_instructions=stable_json_dumps(candidate_rows, indent=2),
        )
        request = SemanticCallRequest(
            kind="prompt_instruction_selection",
            system_instruction=runtime.prompts.template_text(system_template),
            components=[
                PromptComponent(
                    name="prompt_instruction_selection_task",
                    category="system_prompt_instruction",
                    text=user_text,
                )
            ],
            contract=prompt_instruction_selection_contract(
                (
                    instruction_store,
                    instruction.instruction_id,
                )
                for instruction_store, instruction in candidates
            ),
            minimum_output_tokens=128,
            desired_output_tokens=384,
            include_prompt_instructions=False,
            prompt_template_names=(system_template, user_template),
        )
        try:
            payload = runtime._execute_tool_semantic_call(state, request)
            operation_categories: list[str] = []
            for raw_category in payload["operation_categories"]:
                category = str(raw_category).strip()
                if (
                    not category
                    or len(category) > MAX_PROMPT_INSTRUCTION_CATEGORY_CHARS
                ):
                    raise ValueError(
                        "prompt instruction selector returned an invalid operation category"
                    )
                if category not in operation_categories:
                    operation_categories.append(category)
            if len(operation_categories) > MAX_PROMPT_INSTRUCTION_CATEGORIES:
                raise ValueError(
                    "prompt instruction selector returned too many operation categories"
                )
            selected_keys = {
                (
                    str(reference["instruction_store"]),
                    str(reference["instruction_id"]),
                )
                for reference in payload["selected_instructions"]
            }
            selected_sources = [
                source
                for source in scoped_sources
                if is_trusted_prompt_instruction(source[1])
                or not source[1].categories
                or (source[0], source[1].instruction_id) in selected_keys
            ]
            return selected_sources, {
                "semantic_selection": True,
                "selection_fallback": False,
                "operation_categories": operation_categories,
                "selection_reason": str(payload["reason"]),
                "selection_target_context_sha256": target_context_sha256,
                "selection_candidate_references": candidate_references,
            }
        except (ModelCallStateChanged, RunCancellationRequested):
            raise
        except Exception as exc:
            runtime.history.record_event(
                state,
                "prompt_instruction_selection_failed",
                {
                    "kind": assembly.kind,
                    "target_context_sha256": target_context_sha256,
                    "candidate_instruction_references": candidate_references,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "fallback": "include_all_scoped_candidates",
                },
            )
            return scoped_sources, {
                "semantic_selection": False,
                "selection_fallback": True,
                "operation_categories": [],
                "selection_reason": (
                    "Semantic selector failed; every broad-scope candidate was included."
                ),
                "selection_target_context_sha256": target_context_sha256,
                "selection_candidate_references": candidate_references,
            }

    def _selected_prompt_instruction_rows(
        self,
        state: SessionState,
        assembly: PromptAssembly,
    ) -> list[dict[str, Any]]:
        runtime = self.runtime
        rows = [
            {"instruction_store": instruction_store, **asdict(item)}
            for instruction_store, item in self._prompt_instruction_sources(
                state,
                assembly.kind,
            )
        ]
        references = assembly.metadata.get("prompt_instruction_sources")
        if not isinstance(references, list):
            return rows
        selected = {
            (
                str(reference.get("instruction_store", "")),
                str(reference.get("instruction_id", "")),
            )
            for reference in references
            if isinstance(reference, dict)
        }
        return [
            row
            for row in rows
            if (str(row["instruction_store"]), str(row["instruction_id"]))
            in selected
        ]

    def recover_overflow(
        self,
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        failed: ContextCompilation,
        *,
        minimum_output_tokens: int,
        desired_output_tokens: int | None = None,
        context_limit_resolution: tuple[int, str] | None = None,
    ) -> ContextCompilation | None:
        runtime = self.runtime
        if (
            failed.report.fits
            or not runtime.config.context.compact_on_overflow
            or assembly.kind == "prompt_instruction_projection"
        ):
            return None
        source_component = next(
            (
                component
                for component in assembly.components
                if component.name == "durable_prompt_instructions"
            ),
            None,
        )
        source_report = next(
            (
                component
                for component in failed.report.breakdown
                if component.name == "durable_prompt_instructions"
            ),
            None,
        )
        if source_component is None or source_report is None:
            return None
        source_tokens = int(source_report.tokens)
        overflow_tokens = max(1, int(failed.overflow_tokens))
        if source_tokens <= overflow_tokens + 32:
            return None

        source_rows = self._selected_prompt_instruction_rows(
            state,
            assembly,
        )
        if not source_rows:
            return None
        exact_source = stable_json_dumps(source_rows, indent=2)
        source_sha256 = sha256_text(exact_source)
        references = [
            {
                "instruction_store": str(row["instruction_store"]),
                "instruction_id": str(row["instruction_id"]),
                "sha256": sha256_text(stable_json_dumps(row, indent=None)),
            }
            for row in source_rows
        ]
        projection_header = (
            "\n\n[DURABLE MODEL-AUTHORED INSTRUCTION PROJECTION FOR THIS CALL KIND]\n"
            "Measured context overflow required this model-authored derived view. "
            "Apply every operative rule below. Exact source instructions remain "
            "authoritative and recoverable through the prompt_instructions capability.\n"
        )
        counter = runtime._counter(state)
        header_tokens = counter.count_text(projection_header).tokens
        target_tokens = max(
            32,
            source_tokens - overflow_tokens - header_tokens - 16,
        )
        if target_tokens >= source_tokens:
            return None
        remaining_calls = [max(8, int(runtime.config.context.max_compaction_rounds) * 8)]
        maximum_rounds = max(1, int(runtime.config.context.max_compaction_rounds) + 1)
        for round_index in range(maximum_rounds):
            projection, projection_report = runtime._reduce_text_hierarchically(
                state,
                source_text=exact_source,
                source_label=(
                    f"exact durable instructions for {assembly.kind} calls"
                ),
                target_tokens=target_tokens,
                contract=prompt_instruction_projection_contract(),
                output_key="projection",
                build_assembly=lambda text, _label, target: (
                    runtime.prompts.build_prompt_instruction_projection_prompt(
                        call_kind=assembly.kind,
                        source_instructions=text,
                        source_sha256=sha256_text(text),
                        source_tokens=counter.count_text(text).tokens,
                        overflow_tokens=overflow_tokens,
                        target_tokens=target,
                    )
                ),
                remaining_calls=remaining_calls,
                context_limit_resolution=context_limit_resolution,
                include_prompt_instructions=False,
            )
            projected_tokens = counter.count_text(projection).tokens
            candidate = copy.deepcopy(assembly)
            replacement_index = next(
                index
                for index, component in enumerate(candidate.components)
                if component.name == "durable_prompt_instructions"
            )
            candidate.components[replacement_index] = PromptComponent(
                name="durable_prompt_instruction_projection",
                category="system_prompt_instruction",
                text=projection_header + projection,
            )
            candidate.prompt_text = "".join(
                component.text for component in candidate.components
            )
            candidate.prompt_artifacts = [
                artifact
                for artifact in candidate.prompt_artifacts
                if artifact.source != "prompt_protocol:server_chat_template"
                and not artifact.source.startswith("durable_prompt_instructions:")
                and not artifact.source.startswith(
                    "durable_prompt_instruction_projection:"
                )
            ] + [
                PromptArtifact(
                    source=(
                        f"durable_prompt_instruction_projection:{assembly.kind}:"
                        f"{source_sha256}"
                    ),
                    sha256=sha256_text(projection),
                )
            ]
            recovered = runtime._compile_context(
                state,
                candidate,
                contract,
                minimum_output_tokens=minimum_output_tokens,
                desired_output_tokens=desired_output_tokens,
                context_limit_resolution=context_limit_resolution,
                include_prompt_instructions=False,
            )
            if recovered.report.fits:
                assembly.components = candidate.components
                assembly.message_ranges = candidate.message_ranges
                assembly.prompt_text = candidate.prompt_text
                assembly.prompt_artifacts = candidate.prompt_artifacts
                runtime.history.record_event(
                    state,
                    "prompt_instruction_projection_created",
                    {
                        "kind": assembly.kind,
                        "source_instruction_references": references,
                        "source_sha256": source_sha256,
                        "source_tokens": source_tokens,
                        "overflow_tokens": overflow_tokens,
                        "target_tokens": target_tokens,
                        "projected_tokens": projected_tokens,
                        "projection": projection,
                        "projection_sha256": sha256_text(projection),
                        "projection_budget_report": asdict(projection_report),
                        "reduction_round": round_index,
                        "exact_source_recovery": {
                            "session_id": state.session_id,
                            "capability": "prompt_instructions",
                            "instruction_references": references,
                        },
                    },
                )
                return recovered
            reduction = max(
                16,
                int(recovered.overflow_tokens) + 16,
                projected_tokens - target_tokens,
            )
            next_target = max(32, target_tokens - reduction)
            if next_target >= target_tokens:
                break
            target_tokens = next_target
        return None
