"""Тести safety-обмежень ReAct-агента."""

import pytest

from safety import SafetyConfig, SafetyController, SafetyLimitExceeded


def test_max_steps_limit() -> None:
    safety = SafetyController(SafetyConfig(max_steps=2))

    safety.check_before_step()
    safety.check_before_step()

    with pytest.raises(SafetyLimitExceeded, match="максимальну кількість"):
        safety.check_before_step()


def test_repeated_tool_call_limit() -> None:
    safety = SafetyController(
        SafetyConfig(repeated_call_limit=2)
    )

    arguments = {"initiative_id": "DEM-001"}

    safety.register_tool_call("get_initiative_status", arguments)
    safety.register_tool_call("get_initiative_status", arguments)

    with pytest.raises(SafetyLimitExceeded, match="Виявлено цикл"):
        safety.register_tool_call("get_initiative_status", arguments)


def test_different_tool_arguments_are_not_a_loop() -> None:
    safety = SafetyController(
        SafetyConfig(repeated_call_limit=1)
    )

    safety.register_tool_call(
        "get_initiative_status",
        {"initiative_id": "DEM-001"},
    )
    safety.register_tool_call(
        "get_initiative_status",
        {"initiative_id": "DEM-002"},
    )

    assert safety.snapshot()["registered_tool_calls"] == 2


def test_timeout_limit() -> None:
    safety = SafetyController(
        SafetyConfig(timeout_seconds=1)
    )

    # Імітуємо, що запуск почався дві секунди тому.
    safety.started_at -= 2

    with pytest.raises(SafetyLimitExceeded, match="timeout"):
        safety.check_timeout()