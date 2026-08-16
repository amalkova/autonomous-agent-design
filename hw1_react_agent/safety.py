"""Safety-обмеження для ReAct-агента."""

import json
import time
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field


class SafetyConfig(BaseModel):
    """Конфігурація обмежень агента."""

    max_steps: int = Field(default=8, ge=1, le=50)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    repeated_call_limit: int = Field(default=2, ge=1, le=10)


class SafetyLimitExceeded(RuntimeError):
    """Агент зупинений через порушення safety-обмеження."""


class SafetyController:
    """Контролює кількість кроків, час виконання та повтори tool calls."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.started_at = time.monotonic()
        self.step_count = 0
        self._tool_calls: Counter[str] = Counter()

    @property
    def elapsed_seconds(self) -> float:
        """Повертає тривалість поточного запуску."""

        return time.monotonic() - self.started_at

    def check_timeout(self) -> None:
        """Зупиняє виконання після перевищення timeout."""

        if self.elapsed_seconds > self.config.timeout_seconds:
            raise SafetyLimitExceeded(
                f"Перевищено timeout {self.config.timeout_seconds} секунд."
            )

    def check_before_step(self) -> None:
        """Перевіряє обмеження перед наступним кроком агента."""

        self.check_timeout()

        if self.step_count >= self.config.max_steps:
            raise SafetyLimitExceeded(
                f"Досягнуто максимальну кількість кроків: "
                f"{self.config.max_steps}."
            )

        self.step_count += 1

    def register_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Реєструє tool call і виявляє повторюваний цикл."""

        self.check_timeout()

        signature = json.dumps(
            {
                "tool_name": tool_name,
                "arguments": arguments,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        self._tool_calls[signature] += 1
        repeat_count = self._tool_calls[signature]

        if repeat_count > self.config.repeated_call_limit:
            raise SafetyLimitExceeded(
                f"Виявлено цикл: інструмент {tool_name} викликано "
                f"з однаковими аргументами {repeat_count} рази."
            )

    def snapshot(self) -> dict[str, Any]:
        """Повертає поточний стан safety-контролера."""

        return {
            "step_count": self.step_count,
            "max_steps": self.config.max_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "timeout_seconds": self.config.timeout_seconds,
            "registered_tool_calls": sum(self._tool_calls.values()),
        }