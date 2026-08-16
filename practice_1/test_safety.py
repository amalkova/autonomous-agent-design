"""Тести safety-механізмів агента."""

from __future__ import annotations

import pytest

import safety as safety_module
from safety import (
    SafetyConfig,
    SafetyController,
    SafetyLimitExceeded,
)


def test_default_safety_config_matches_requirements() -> None:
    """Стандартні ліміти відповідають умовам завдання."""

    config = SafetyConfig()

    assert config.max_steps == 10
    assert config.timeout_seconds == 120.0
    assert config.max_identical_tool_calls == 2


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_steps", 0),
        ("timeout_seconds", 0),
        ("max_identical_tool_calls", 0),
    ],
)
def test_invalid_safety_config_is_rejected(
    field_name: str,
    invalid_value: int,
) -> None:
    """Некоректні safety-ліміти не приймаються."""

    arguments = {
        "max_steps": 10,
        "timeout_seconds": 120.0,
        "max_identical_tool_calls": 2,
    }
    arguments[field_name] = invalid_value

    with pytest.raises(ValueError):
        SafetyConfig(**arguments)


def test_max_steps_stops_agent() -> None:
    """Агент зупиняється після досягнення max_steps."""

    safety = SafetyController(
        SafetyConfig(max_steps=3)
    )

    safety.check_before_step()
    safety.check_before_step()
    safety.check_before_step()

    with pytest.raises(
        SafetyLimitExceeded,
        match="кількість кроків",
    ):
        safety.check_before_step()

    assert safety.step_count == 3


def test_timeout_stops_agent(monkeypatch) -> None:
    """Агент зупиняється після загального timeout."""

    safety = SafetyController(
        SafetyConfig(timeout_seconds=120.0)
    )
    safety.started_at = 100.0

    monkeypatch.setattr(
        safety_module.time,
        "monotonic",
        lambda: 221.0,
    )

    with pytest.raises(
        SafetyLimitExceeded,
        match="timeout",
    ):
        safety.check_timeout()


def test_repeated_identical_tool_call_is_detected() -> None:
    """Третій однаковий tool call блокується."""

    safety = SafetyController(
        SafetyConfig(
            max_identical_tool_calls=2,
        )
    )

    safety.register_tool_call(
        "check_requirements_readiness",
        {
            "initiative_id": "DEM-001",
            "business_objective": "Goal",
        },
    )
    safety.register_tool_call(
        "check_requirements_readiness",
        {
            "business_objective": "Goal",
            "initiative_id": "DEM-001",
        },
    )

    with pytest.raises(
        SafetyLimitExceeded,
        match="повторний tool call",
    ):
        safety.register_tool_call(
            "check_requirements_readiness",
            {
                "initiative_id": "DEM-001",
                "business_objective": "Goal",
            },
        )

    assert safety.registered_tool_calls == 3
    assert safety.snapshot()["unique_tool_calls"] == 1


def test_different_tool_arguments_do_not_trigger_loop() -> None:
    """Різні аргументи вважаються різними tool calls."""

    safety = SafetyController(
        SafetyConfig(
            max_identical_tool_calls=1,
        )
    )

    safety.register_tool_call(
        "demo_tool",
        {"initiative_id": "DEM-001"},
    )
    safety.register_tool_call(
        "demo_tool",
        {"initiative_id": "DEM-002"},
    )

    snapshot = safety.snapshot()

    assert snapshot["registered_tool_calls"] == 2
    assert snapshot["unique_tool_calls"] == 2