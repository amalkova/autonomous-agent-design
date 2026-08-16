"""Human-in-the-Loop для ризикових tools."""

from __future__ import annotations

from typing import Any, Literal, Self

from langgraph.types import interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class HumanDecision(BaseModel):
    """Рішення людини щодо ризикової операції."""

    model_config = ConfigDict(
        extra="forbid",
    )

    decision: Literal[
        "approve",
        "reject",
        "edit",
    ] = Field(
        description=(
            "approve=виконати, reject=відхилити, "
            "edit=змінити аргументи та виконати"
        ),
    )
    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Причина рішення людини",
    )
    edited_args: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Змінені аргументи для decision=edit"
        ),
    )

    @model_validator(mode="after")
    def validate_decision_payload(self) -> Self:
        """Перевіряє узгодженість decision та edited_args."""

        if (
            self.decision == "edit"
            and not self.edited_args
        ):
            raise ValueError(
                "Для decision=edit потрібно "
                "передати edited_args."
            )

        if (
            self.decision != "edit"
            and self.edited_args is not None
        ):
            raise ValueError(
                "edited_args дозволено лише "
                "для decision=edit."
            )

        if self.reason is not None:
            normalized_reason = self.reason.strip()
            self.reason = normalized_reason or None

        return self


class HITLResolution(BaseModel):
    """Нормалізований результат HITL-рішення."""

    decision: Literal[
        "approve",
        "reject",
        "edit",
    ]
    approved: bool
    tool_name: str
    arguments: dict[str, Any]
    reason: str | None = None


def resolve_human_decision(
    tool_name: str,
    original_arguments: dict[str, Any],
    raw_decision: dict[str, Any],
) -> HITLResolution:
    """Обробляє approve, reject або edit."""

    decision = HumanDecision.model_validate(
        raw_decision
    )

    if decision.decision == "reject":
        return HITLResolution(
            decision="reject",
            approved=False,
            tool_name=tool_name,
            arguments=original_arguments,
            reason=(
                decision.reason
                or "Операцію відхилено людиною."
            ),
        )

    if decision.decision == "edit":
        edited_arguments = (
            decision.edited_args or {}
        )
        unknown_arguments = (
            set(edited_arguments)
            - set(original_arguments)
        )

        if unknown_arguments:
            unknown_list = ", ".join(
                sorted(unknown_arguments)
            )
            raise ValueError(
                "Не можна додавати невідомі "
                f"аргументи: {unknown_list}."
            )

        merged_arguments = {
            **original_arguments,
            **edited_arguments,
        }

        return HITLResolution(
            decision="edit",
            approved=True,
            tool_name=tool_name,
            arguments=merged_arguments,
            reason=decision.reason,
        )

    return HITLResolution(
        decision="approve",
        approved=True,
        tool_name=tool_name,
        arguments=original_arguments,
        reason=decision.reason,
    )


def request_tool_approval(
    tool_name: str,
    arguments: dict[str, Any],
) -> HITLResolution:
    """Зупиняє graph до рішення людини."""

    human_response = interrupt(
        {
            "type": "tool_approval",
            "risk_level": "high",
            "action": tool_name,
            "arguments": arguments,
            "message": (
                "Підтвердіть ризикову операцію. "
                "Перевірте tool та всі аргументи."
            ),
            "allowed_decisions": [
                "approve",
                "reject",
                "edit",
            ],
        }
    )

    return resolve_human_decision(
        tool_name=tool_name,
        original_arguments=arguments,
        raw_decision=human_response,
    )