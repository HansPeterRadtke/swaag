from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from swaag.context_compiler import ContextCompilation
from swaag.prompt_instruction_context import PromptInstructionContextManager
from swaag.prompt_instruction_store import PromptInstructionStore
from swaag.types import ContractSpec, PromptAssembly, SessionState

if TYPE_CHECKING:
    from swaag.runtime import AgentRuntime


class SystemPromptContributor(Protocol):
    name: str

    def enabled(self, runtime: "AgentRuntime") -> bool: ...

    def inject(
        self, runtime: "AgentRuntime", state: SessionState | None, assembly: PromptAssembly
    ) -> None: ...

    def recover_overflow(
        self,
        runtime: "AgentRuntime",
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        failed: ContextCompilation,
        *,
        minimum_output_tokens: int,
        desired_output_tokens: int | None = None,
        context_limit_resolution: tuple[int, str] | None = None,
    ) -> ContextCompilation | None: ...


@dataclass(frozen=True)
class PromptInstructionContributor:
    store: PromptInstructionStore
    name: str = "prompt_instructions"

    def enabled(self, runtime: "AgentRuntime") -> bool:
        return self.name in set(runtime.config.tools.enabled)

    def inject(
        self, runtime: "AgentRuntime", state: SessionState | None, assembly: PromptAssembly
    ) -> None:
        if self.enabled(runtime):
            PromptInstructionContextManager(runtime, self.store).inject(state, assembly)

    def recover_overflow(
        self,
        runtime: "AgentRuntime",
        state: SessionState,
        assembly: PromptAssembly,
        contract: ContractSpec,
        failed: ContextCompilation,
        *,
        minimum_output_tokens: int,
        desired_output_tokens: int | None = None,
        context_limit_resolution: tuple[int, str] | None = None,
    ) -> ContextCompilation | None:
        if not self.enabled(runtime):
            return None
        return PromptInstructionContextManager(runtime, self.store).recover_overflow(
            state,
            assembly,
            contract,
            failed,
            minimum_output_tokens=minimum_output_tokens,
            desired_output_tokens=desired_output_tokens,
            context_limit_resolution=context_limit_resolution,
        )


def default_system_prompt_contributors(
    config,
) -> tuple[SystemPromptContributor, ...]:
    return (
        PromptInstructionContributor(
            PromptInstructionStore(config.sessions.root, config)
        ),
    )
