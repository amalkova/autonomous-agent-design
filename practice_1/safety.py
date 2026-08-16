"""Захисні механізми для автономних агентів."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


class SafetyLimitExceeded(RuntimeError):
    """Агент зупинений через перевищення safety-ліміту."""


@dataclass(frozen=True)
class SafetyConfig:
    """Конфігурація захисних обмежень агента."""

    max_steps: int = 10
    timeout_seconds: float = 120.0
    max_identical_tool_calls: int = 2

    def __post_init__(self) -> None:
        """Перевіряє коректність safety-конфігурації."""

        if self.max_steps < 1:
            raise ValueError(
                "max_steps має бути не меншим за 1."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds має бути більшим за 0."
            )

        if self.max_identical_tool_calls < 1:
            raise ValueError(
                "max_identical_tool_calls має бути "
                "не меншим за 1."
            )


@dataclass
class SafetyController:
    """Контролює кроки, timeout і повторні tool calls."""

    config: SafetyConfig | None = None
    started_at: float = field(
        default_factory=time.monotonic,
        init=False,
    )
    step_count: int = field(default=0, init=False)
    registered_tool_calls: int = field(
        default=0,
        init=False,
    )
    _tool_call_counts: dict[str, int] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        """Встановлює стандартну конфігурацію."""

        if self.config is None:
            self.config = SafetyConfig()

    def elapsed_seconds(self) -> float:
        """Повертає час роботи контролера."""

        return time.monotonic() - self.started_at

    def check_timeout(self) -> None:
        """Зупиняє агент після перевищення timeout."""

        elapsed = self.elapsed_seconds()

        if elapsed > self.config.timeout_seconds:
            raise SafetyLimitExceeded(
                "Перевищено загальний timeout агента: "
                f"{self.config.timeout_seconds} секунд."
            )

    def check_before_step(self) -> None:
        """Перевіряє timeout і реєструє новий крок."""

        self.check_timeout()

        if self.step_count >= self.config.max_steps:
            raise SafetyLimitExceeded(
                "Перевищено максимальну кількість кроків: "
                f"{self.config.max_steps}."
            )

        self.step_count += 1

    def register_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Реєструє tool call і виявляє повторний цикл."""

        self.check_timeout()

        normalized_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        signature = (
            f"{tool_name}:{normalized_arguments}"
        )

        current_count = (
            self._tool_call_counts.get(signature, 0) + 1
        )
        self._tool_call_counts[signature] = current_count
        self.registered_tool_calls += 1

        if (
            current_count
            > self.config.max_identical_tool_calls
        ):
            raise SafetyLimitExceeded(
                "Виявлено повторний tool call: "
                f"{tool_name} з однаковими аргументами "
                f"викликано {current_count} рази."
            )

    def snapshot(self) -> dict[str, Any]:
        """Повертає поточний стан safety-контролера."""

        return {
            "step_count": self.step_count,
            "max_steps": self.config.max_steps,
            "elapsed_seconds": round(
                self.elapsed_seconds(),
                3,
            ),
            "timeout_seconds": (
                self.config.timeout_seconds
            ),
            "registered_tool_calls": (
                self.registered_tool_calls
            ),
            "max_identical_tool_calls": (
                self.config.max_identical_tool_calls
            ),
            "unique_tool_calls": len(
                self._tool_call_counts
            ),
        }