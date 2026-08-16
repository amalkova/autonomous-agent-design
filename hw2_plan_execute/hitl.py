"""Human-in-the-Loop approval для ризикових tools."""

from typing import Any, Literal, Self

from langgraph.types import interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class HumanDecision(BaseModel):
    """Рішення людини після зупинки графа."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal[
        "approve",
        "reject",
        "edit",
    ] = Field(
        description=(
            "approve=виконати дію, reject=відхилити, "
            "edit=змінити параметри та виконати"
        )
    )
    reason: str | None = Field(
        default=None,
        description="Причина рішення людини",
    )
    edited_args: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Змінені параметри tool для decision=edit"
        ),
    )

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(
        cls,
        value: object,
    ) -> str | None:
        """Нормалізує необов'язкову причину рішення."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                "reason має бути текстовим значенням."
            )

        normalized_reason = value.strip()
        return normalized_reason or None

    @model_validator(mode="after")
    def validate_edit_decision(self) -> Self:
        """Для edit вимагає змінені параметри."""

        if (
            self.decision == "edit"
            and not self.edited_args
        ):
            raise ValueError(
                "Для decision=edit потрібно передати edited_args."
            )

        if (
            self.decision != "edit"
            and self.edited_args is not None
        ):
            raise ValueError(
                "edited_args дозволено лише для decision=edit."
            )

        return self


class HITLResolution(BaseModel):
    """Результат обробки рішення людини."""

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
    response: object,
) -> HITLResolution:
    """Перетворює відповідь людини на рішення для executor."""

    decision = HumanDecision.model_validate(response)

    if decision.decision == "reject":
        return HITLResolution(
            decision="reject",
            approved=False,
            tool_name=tool_name,
            arguments=original_arguments,
            reason=decision.reason,
        )

    if decision.decision == "edit":
        edited_arguments = decision.edited_args or {}

        unknown_arguments = (
            set(edited_arguments)
            - set(original_arguments)
        )

        if unknown_arguments:
            unknown_names = ", ".join(
                sorted(unknown_arguments)
            )
            raise ValueError(
                "Не можна додавати невідомі параметри: "
                f"{unknown_names}."
            )

        final_arguments = {
            **original_arguments,
            **edited_arguments,
        }

        return HITLResolution(
            decision="edit",
            approved=True,
            tool_name=tool_name,
            arguments=final_arguments,
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
    """Зупиняє граф і запитує рішення для ризикової дії."""

    human_response = interrupt(
        {
            "type": "tool_approval",
            "risk_level": "high",
            "action": tool_name,
            "arguments": arguments,
            "message": (
                "Підтвердіть ризикову дію. "
                "Перевірте tool та всі параметри перед виконанням."
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
        response=human_response,
    )