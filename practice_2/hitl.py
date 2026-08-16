"""Human-in-the-Loop для ризикового estimation submission."""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Literal,
    Self,
    TypedDict,
)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from guardrails import (
    requires_human_approval,
    validate_tool_call,
)
from mcp_client import call_mcp_tool


ToolExecutor = Callable[
    [str, dict[str, Any]],
    Awaitable[Any],
]


class HumanDecision(BaseModel):
    """Структуроване рішення людини."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    action: Literal[
        "approve",
        "reject",
        "edit",
    ]
    reason: str | None = Field(
        default=None,
        max_length=500,
    )
    edited_arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_edit_arguments(self) -> Self:
        """Edit має містити хоча б один змінений аргумент."""

        if (
            self.action == "edit"
            and not self.edited_arguments
        ):
            raise ValueError(
                "edited_arguments required for edit"
            )

        return self


class HITLState(TypedDict, total=False):
    """Стан процесу погодження."""

    tool_name: str
    tool_arguments: dict[str, Any]
    validated_arguments: dict[str, Any]
    human_decision: dict[str, Any]
    status: Literal[
        "pending",
        "approved",
        "rejected",
        "blocked",
        "executed",
    ]
    status_reason: str
    tool_result: Any
    audit_log: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


def audit_event(
    event: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Створити JSON-safe audit event."""

    return {
        "event": event,
        "details": details,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def create_submission_state(
    arguments: dict[str, Any],
) -> HITLState:
    """Підготувати початковий state для risky tool."""

    return {
        "tool_name": "submit_estimation_request",
        "tool_arguments": dict(arguments),
        "status": "pending",
        "status_reason": (
            "Очікується Human-in-the-Loop approval."
        ),
        "audit_log": [],
    }


def approval_gate(state: HITLState) -> dict[str, Any]:
    """Зупинити graph перед виконанням risky tool."""

    tool_name = state["tool_name"]
    original_arguments = state["tool_arguments"]

    validation = validate_tool_call(
        agent_name="estimation_agent",
        tool_name=tool_name,
        arguments=original_arguments,
    )

    if not validation.allowed:
        return {
            "status": "blocked",
            "status_reason": validation.reason,
            "audit_log": [
                audit_event(
                    "tool_call_blocked",
                    {
                        "tool_name": tool_name,
                        "reason": validation.reason,
                    },
                )
            ],
        }

    if not requires_human_approval(tool_name):
        return {
            "status": "blocked",
            "status_reason": (
                "Workflow приймає лише risky tools."
            ),
            "audit_log": [
                audit_event(
                    "non_risky_tool_blocked",
                    {
                        "tool_name": tool_name,
                    },
                )
            ],
        }

    # До interrupt немає жодного side effect.
    review = interrupt(
        {
            "type": "tool_approval",
            "question": (
                "Підтвердити передачу initiative "
                "на estimation?"
            ),
            "tool_name": tool_name,
            "arguments": (
                validation.validated_arguments
            ),
            "allowed_actions": [
                "approve",
                "reject",
                "edit",
            ],
        }
    )

    try:
        decision = HumanDecision.model_validate(
            review
        )

    except ValidationError as exception:
        return {
            "status": "blocked",
            "status_reason": (
                "Некоректний формат human decision."
            ),
            "audit_log": [
                audit_event(
                    "invalid_human_decision",
                    {
                        "errors": exception.errors(
                            include_url=False,
                            include_input=False,
                        )
                    },
                )
            ],
        }

    if decision.action == "reject":
        return {
            "human_decision": decision.model_dump(
                mode="json"
            ),
            "status": "rejected",
            "status_reason": (
                decision.reason
                or "Операцію відхилено людиною."
            ),
            "audit_log": [
                audit_event(
                    "tool_call_rejected",
                    {
                        "tool_name": tool_name,
                        "reason": decision.reason,
                    },
                )
            ],
        }

    final_arguments = dict(
        validation.validated_arguments or {}
    )

    if decision.action == "edit":
        final_arguments.update(
            decision.edited_arguments or {}
        )

        edited_validation = validate_tool_call(
            agent_name="estimation_agent",
            tool_name=tool_name,
            arguments=final_arguments,
        )

        if not edited_validation.allowed:
            return {
                "human_decision": (
                    decision.model_dump(mode="json")
                ),
                "status": "blocked",
                "status_reason": (
                    "Edited arguments не пройшли "
                    "повторну валідацію."
                ),
                "audit_log": [
                    audit_event(
                        "edited_arguments_blocked",
                        {
                            "tool_name": tool_name,
                            "errors": (
                                edited_validation
                                .validation_errors
                            ),
                        },
                    )
                ],
            }

        final_arguments = dict(
            edited_validation.validated_arguments
            or {}
        )

    return {
        "validated_arguments": final_arguments,
        "human_decision": decision.model_dump(
            mode="json"
        ),
        "status": "approved",
        "status_reason": (
            "Tool call підтверджено людиною."
        ),
        "audit_log": [
            audit_event(
                "tool_call_approved",
                {
                    "tool_name": tool_name,
                    "action": decision.action,
                },
            )
        ],
    }


def route_after_approval(
    state: HITLState,
) -> Literal[
    "execute_submission",
    "reject_submission",
]:
    """Маршрутизувати workflow після approval gate."""

    if state.get("status") == "approved":
        return "execute_submission"

    return "reject_submission"


async def execute_submission(
    state: HITLState,
    tool_executor: ToolExecutor,
) -> dict[str, Any]:
    """Виконати MCP tool лише після approval."""

    result = await tool_executor(
        state["tool_name"],
        state["validated_arguments"],
    )

    return {
        "status": "executed",
        "status_reason": (
            "Risky MCP tool виконано після approval."
        ),
        "tool_result": result,
        "audit_log": [
            audit_event(
                "tool_call_executed",
                {
                    "tool_name": state["tool_name"],
                },
            )
        ],
    }


def reject_submission(
    state: HITLState,
) -> dict[str, Any]:
    """Завершити workflow без side effect."""

    status = state.get("status", "blocked")

    return {
        "status": (
            "rejected"
            if status == "rejected"
            else "blocked"
        ),
        "tool_result": {
            "status": "not_executed",
            "reason": state.get(
                "status_reason",
                "Tool call blocked.",
            ),
        },
        "audit_log": [
            audit_event(
                "workflow_finished_without_execution",
                {
                    "tool_name": state.get(
                        "tool_name"
                    ),
                },
            )
        ],
    }


def build_hitl_graph(
    tool_executor: ToolExecutor = call_mcp_tool,
    checkpointer: Any | None = None,
) -> Any:
    """Побудувати approval workflow."""

    builder = StateGraph(HITLState)

    async def execute_node(
        state: HITLState,
    ) -> dict[str, Any]:
        return await execute_submission(
            state,
            tool_executor,
        )

    builder.add_node(
        "approval_gate",
        approval_gate,
    )
    builder.add_node(
        "execute_submission",
        execute_node,
    )
    builder.add_node(
        "reject_submission",
        reject_submission,
    )

    builder.add_edge(
        START,
        "approval_gate",
    )
    builder.add_conditional_edges(
        "approval_gate",
        route_after_approval,
    )
    builder.add_edge(
        "execute_submission",
        END,
    )
    builder.add_edge(
        "reject_submission",
        END,
    )

    return builder.compile(
        checkpointer=(
            checkpointer
            if checkpointer is not None
            else InMemorySaver()
        )
    )