"""Guardrails для Requirements & Estimation MAS."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from tools_legacy import (
    CheckRequirementsReadinessInput,
    ClassifyEstimationComplexityInput,
    IdentifyHandoverGapsInput,
    SubmitEstimationRequestInput,
)


InputSource = Literal[
    "user",
    "external_tool",
    "agent_message",
]


INJECTION_RULES = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"\bignore\b.{0,80}"
            r"\b(previous|prior|all|system|developer)\b"
            r".{0,40}\b(instruction|instructions|prompt|messages)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "ukrainian_ignore_instructions",
        re.compile(
            r"\b(ігноруй|ігнорувати)\b.{0,80}"
            r"\b(попередні|системні)\b.{0,40}"
            r"\b(інструкції|промпт|повідомлення)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "russian_ignore_instructions",
        re.compile(
            r"\b(игнорируй|игнорировать)\b.{0,80}"
            r"\b(предыдущие|системные)\b.{0,40}"
            r"\b(инструкции|промпт|сообщения)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(reveal|show|print|expose|return)\b.{0,80}"
            r"\b(system|developer)\b.{0,30}"
            r"\b(prompt|message|instructions)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "bypass_guardrails",
        re.compile(
            r"\b(bypass|disable|remove|evade)\b.{0,60}"
            r"\b(guardrail|security|policy|restriction)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "jailbreak_marker",
        re.compile(
            r"\b(jailbreak|developer mode|do anything now|DAN)\b",
            re.IGNORECASE,
        ),
    ),
)


TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "demand_supervisor": frozenset(),
    "requirements_agent": frozenset(
        {
            "check_requirements_readiness",
            "identify_handover_gaps",
        }
    ),
    "solution_security_agent": frozenset(),
    "estimation_agent": frozenset(
        {
            "classify_estimation_complexity",
            "submit_estimation_request",
        }
    ),
}


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "check_requirements_readiness": (
        CheckRequirementsReadinessInput
    ),
    "classify_estimation_complexity": (
        ClassifyEstimationComplexityInput
    ),
    "identify_handover_gaps": (
        IdentifyHandoverGapsInput
    ),
    "submit_estimation_request": (
        SubmitEstimationRequestInput
    ),
}


RISKY_TOOLS = frozenset(
    {
        "submit_estimation_request",
    }
)


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_CANDIDATE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
)


class InputGuardrailResult(BaseModel):
    """Результат перевірки вхідного контенту."""

    allowed: bool
    source: InputSource
    category: Literal[
        "safe",
        "invalid_input",
        "prompt_injection",
    ]
    reason: str
    matched_rule: str | None = None


class ToolGuardrailResult(BaseModel):
    """Результат перевірки tool call."""

    allowed: bool
    agent_name: str
    tool_name: str
    reason: str
    validated_arguments: dict[str, Any] | None = None
    validation_errors: list[dict[str, Any]] = Field(
        default_factory=list
    )


def inspect_input(
    text: str,
    source: InputSource = "user",
) -> InputGuardrailResult:
    """Виявити direct або indirect prompt injection."""

    if not isinstance(text, str):
        return InputGuardrailResult(
            allowed=False,
            source=source,
            category="invalid_input",
            reason="Input має бути текстом.",
        )

    normalized = text.replace("\x00", "").strip()

    if not normalized:
        return InputGuardrailResult(
            allowed=False,
            source=source,
            category="invalid_input",
            reason="Порожній input заборонено.",
        )

    if len(normalized) > 8000:
        return InputGuardrailResult(
            allowed=False,
            source=source,
            category="invalid_input",
            reason="Input перевищує дозволений розмір.",
        )

    for rule_name, pattern in INJECTION_RULES:
        if pattern.search(normalized):
            return InputGuardrailResult(
                allowed=False,
                source=source,
                category="prompt_injection",
                reason=(
                    "Виявлено потенційну prompt injection."
                ),
                matched_rule=rule_name,
            )

    return InputGuardrailResult(
        allowed=True,
        source=source,
        category="safe",
        reason="Input пройшов перевірку.",
    )


def validate_tool_call(
    agent_name: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolGuardrailResult:
    """Перевірити allowlist та Pydantic-схему аргументів."""

    if agent_name not in TOOL_ALLOWLIST:
        return ToolGuardrailResult(
            allowed=False,
            agent_name=agent_name,
            tool_name=tool_name,
            reason="Невідомий agent.",
        )

    schema = TOOL_SCHEMAS.get(tool_name)

    if schema is None:
        return ToolGuardrailResult(
            allowed=False,
            agent_name=agent_name,
            tool_name=tool_name,
            reason="Tool відсутній у загальному registry.",
        )

    if tool_name not in TOOL_ALLOWLIST[agent_name]:
        return ToolGuardrailResult(
            allowed=False,
            agent_name=agent_name,
            tool_name=tool_name,
            reason="Tool не дозволений для цього agent.",
        )

    try:
        validated = schema.model_validate(arguments)

    except ValidationError as exception:
        return ToolGuardrailResult(
            allowed=False,
            agent_name=agent_name,
            tool_name=tool_name,
            reason="Аргументи tool не пройшли валідацію.",
            validation_errors=exception.errors(
                include_url=False,
                include_input=False,
            ),
        )

    return ToolGuardrailResult(
        allowed=True,
        agent_name=agent_name,
        tool_name=tool_name,
        reason="Tool call дозволено.",
        validated_arguments=validated.model_dump(
            mode="json"
        ),
    )


def requires_human_approval(tool_name: str) -> bool:
    """Визначити, чи потребує tool HITL approval."""

    return tool_name in RISKY_TOOLS


def redact_pii_text(text: str) -> str:
    """Замаскувати email та phone у текстовій відповіді."""

    redacted = EMAIL_PATTERN.sub(
        "[EMAIL_REDACTED]",
        text,
    )

    def replace_phone(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digit_count = len(
            re.sub(r"\D", "", candidate)
        )

        if 10 <= digit_count <= 15:
            return "[PHONE_REDACTED]"

        return candidate

    return PHONE_CANDIDATE_PATTERN.sub(
        replace_phone,
        redacted,
    )


def redact_output(value: Any) -> Any:
    """Рекурсивно застосувати PII redaction до output."""

    if isinstance(value, str):
        return redact_pii_text(value)

    if isinstance(value, dict):
        return {
            key: redact_output(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            redact_output(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            redact_output(item)
            for item in value
        )

    return value

# Advanced guardrails реекспортуються через основний модуль.
from advanced_guardrails import (  # noqa: E402
    RateLimitResult,
    RollingWindowRateLimiter,
    redact_sensitive_text,
)

# Старий public API збережено для MAS та існуючих тестів.
redact_pii_text = redact_sensitive_text
