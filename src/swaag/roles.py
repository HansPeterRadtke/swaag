from __future__ import annotations

from swaag.types import AgentEnvelope, AgentRoleDefinition
from swaag.utils import utc_now_iso


def default_agent_roles() -> list[AgentRoleDefinition]:
    return [
        AgentRoleDefinition(role_name="primary", responsibilities=["coordinate", "answer"], can_plan=True, can_execute_tools=True, can_verify=True),
        AgentRoleDefinition(role_name="planner", responsibilities=["plan", "replan"], can_plan=True, can_execute_tools=False, can_verify=False),
        AgentRoleDefinition(role_name="executor", responsibilities=["execute_subsystems"], can_plan=False, can_execute_tools=True, can_verify=False),
        AgentRoleDefinition(role_name="verifier", responsibilities=["verify_results"], can_plan=False, can_execute_tools=False, can_verify=True),
    ]


def build_agent_envelope(sender_role: str, recipient_role: str, purpose: str, content: str) -> AgentEnvelope:
    return AgentEnvelope(
        sender_role=sender_role,
        recipient_role=recipient_role,
        purpose=purpose,
        content=content,
        created_at=utc_now_iso(),
    )
