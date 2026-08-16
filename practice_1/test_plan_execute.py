"""Тести Plan-and-Execute графа."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import ValidationError

from plan_execute import (
    Plan,
    append_unique,
    build_graph,
    create_initial_state,
)


class StaticModel:
    """Fake structured model із постійною відповіддю."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = 0

    def invoke(self, messages: Any) -> dict[str, Any]:
        """Повертає підготовлену відповідь."""

        self.calls += 1
        return self.response


class SequenceModel:
    """Fake structured model із послідовністю відповідей."""

    def __init__(
        self,
        responses: list[dict[str, Any]],
    ) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, messages: Any) -> dict[str, Any]:
        """Повертає наступну відповідь."""

        if self.calls >= len(self.responses):
            raise AssertionError(
                "Fake model отримала забагато викликів."
            )

        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_graph_contains_plan_execute_nodes() -> None:
    """Граф містить planner, executor, approval та replanner."""

    graph = build_graph()
    nodes = set(graph.get_graph().nodes)

    assert {
        "__start__",
        "planner",
        "executor",
        "approval",
        "replanner",
        "__end__",
    }.issubset(nodes)


def test_plan_rejects_unsupported_tool() -> None:
    """Plan не дозволяє невідомі tools."""

    with pytest.raises(ValidationError):
        Plan(
            goal="Підготувати estimation request",
            steps=["delete_project: initiative_id=DEM-001"],
        )


def test_initial_state_rejects_blank_request() -> None:
    """Порожній запит не запускає граф."""

    with pytest.raises(
        ValueError,
        match="не може бути порожнім",
    ):
        create_initial_state("   ")


def test_append_unique_removes_duplicates() -> None:
    """Список використаних tools не містить дублікатів."""

    result = append_unique(
        ["check_requirements_readiness"],
        [
            "check_requirements_readiness",
            "identify_handover_gaps",
        ],
    )

    assert result == [
        "check_requirements_readiness",
        "identify_handover_gaps",
    ]


def test_executes_steps_sequentially_with_nested_react() -> None:
    """Executor послідовно запускає plan steps через ReAct."""

    planner = StaticModel(
        {
            "goal": (
                "Перевірити готовність вимог та знайти "
                "handover gaps"
            ),
            "steps": [
                (
                    "check_requirements_readiness: "
                    "initiative_id=DEM-020"
                ),
                (
                    "identify_handover_gaps: "
                    "initiative_id=DEM-020"
                ),
            ],
        }
    )

    replanner = SequenceModel(
        [
            {
                "action": "continue",
                "reasoning": (
                    "Перший крок завершено, виконуємо наступний."
                ),
                "updated_steps": [],
                "final_answer": None,
            },
            {
                "action": "finish",
                "reasoning": "Усі заплановані кроки завершено.",
                "updated_steps": [],
                "final_answer": (
                    "Готовність вимог перевірено, handover gaps "
                    "визначено."
                ),
            },
        ]
    )

    executed_prompts: list[str] = []

    def fake_step_runner(prompt: str) -> dict[str, Any]:
        executed_prompts.append(prompt)

        if "check_requirements_readiness:" in prompt:
            return {
                "status": "completed",
                "answer": "Requirements readiness становить 100%.",
                "used_tools": [
                    "check_requirements_readiness",
                ],
            }

        return {
            "status": "completed",
            "answer": "Критичні handover gaps відсутні.",
            "used_tools": [
                "identify_handover_gaps",
            ],
        }

    graph = build_graph(
        planner_model=planner,
        replanner_model=replanner,
        step_runner=fake_step_runner,
    )

    result = graph.invoke(
        create_initial_state(
            "Перевір готовність DEM-020 та handover gaps."
        )
    )

    assert result["completed"] is True
    assert result["status"] == "completed"
    assert result["current_step"] == 2
    assert len(executed_prompts) == 2
    assert result["used_tools"] == [
        "check_requirements_readiness",
        "identify_handover_gaps",
    ]
    assert "Готовність вимог перевірено" in result["final_answer"]


def test_replanner_replaces_remaining_steps() -> None:
    """Replanner може замінити ще не виконані кроки."""

    planner = StaticModel(
        {
            "goal": "Перевірити readiness та політики estimation",
            "steps": [
                (
                    "check_requirements_readiness: "
                    "initiative_id=DEM-021"
                ),
                (
                    "identify_handover_gaps: "
                    "initiative_id=DEM-021"
                ),
            ],
        }
    )

    replanner = SequenceModel(
        [
            {
                "action": "replan",
                "reasoning": (
                    "Після readiness потрібен пошук політик, "
                    "а не handover analysis."
                ),
                "updated_steps": [
                    (
                        "search_delivery_knowledge: "
                        "query=estimation readiness policy"
                    ),
                ],
                "final_answer": None,
            },
            {
                "action": "finish",
                "reasoning": "Оновлений план виконано.",
                "updated_steps": [],
                "final_answer": (
                    "Readiness перевірено, релевантні політики "
                    "знайдено."
                ),
            },
        ]
    )

    executed_tools: list[str] = []

    def fake_step_runner(prompt: str) -> dict[str, Any]:
        if "search_delivery_knowledge:" in prompt:
            tool_name = "search_delivery_knowledge"
        else:
            tool_name = "check_requirements_readiness"

        executed_tools.append(tool_name)

        return {
            "status": "completed",
            "answer": f"Виконано {tool_name}.",
            "used_tools": [tool_name],
        }

    graph = build_graph(
        planner_model=planner,
        replanner_model=replanner,
        step_runner=fake_step_runner,
    )

    result = graph.invoke(
        create_initial_state(
            "Перевір DEM-021 та знайди estimation policy."
        )
    )

    assert result["completed"] is True
    assert result["replan_count"] == 1
    assert executed_tools == [
        "check_requirements_readiness",
        "search_delivery_knowledge",
    ]
    assert all(
        "identify_handover_gaps" not in prompt
        for prompt in result["results"]
    )


def test_continue_after_last_step_uses_fallback_answer() -> None:
    """Граф безпечно завершується після останнього кроку."""

    planner = StaticModel(
        {
            "goal": "Перевірити readiness DEM-022",
            "steps": [
                (
                    "check_requirements_readiness: "
                    "initiative_id=DEM-022"
                ),
            ],
        }
    )

    replanner = StaticModel(
        {
            "action": "continue",
            "reasoning": "Крок виконано.",
            "updated_steps": [],
            "final_answer": None,
        }
    )

    def fake_step_runner(prompt: str) -> dict[str, Any]:
        return {
            "status": "completed",
            "answer": "Readiness перевірено.",
            "used_tools": [
                "check_requirements_readiness",
            ],
        }

    graph = build_graph(
        planner_model=planner,
        replanner_model=replanner,
        step_runner=fake_step_runner,
    )

    result = graph.invoke(
        create_initial_state("Перевір readiness DEM-022.")
    )

    assert result["completed"] is True
    assert result["status"] == "completed"
    assert "Readiness перевірено" in result["final_answer"]


def test_risky_step_interrupts_and_can_be_rejected(
    tmp_path,
) -> None:
    """Risky submit зупиняється перед виконанням."""

    planner = StaticModel(
        {
            "goal": "Відправити estimation request DEM-023",
            "steps": [
                (
                    "submit_estimation_request: "
                    "initiative_id=DEM-023, "
                    "target_team=Core Banking Delivery, "
                    "requested_by=Lead Business Analyst, "
                    "estimation_complexity=High, "
                    "estimation_points=8"
                ),
            ],
        }
    )

    submit_arguments_model = StaticModel(
    {
        "initiative_id": "DEM-023",
        "target_team": "Core Banking Delivery",
        "requested_by": "Lead Business Analyst",
        "estimation_complexity": "High",
        "estimation_points": 8,
        "estimation_summary": (
            "Вимоги готові для передачі команді estimation."
        ),
    }
    )

    replanner = StaticModel(
        {
            "action": "finish",
            "reasoning": (
                "Людина відхилила ризикову операцію."
            ),
            "updated_steps": [],
            "final_answer": (
                "Estimation request не відправлено через "
                "відхилення людиною."
            ),
        }
    )

    database_path = tmp_path / "plan_state.db"

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
            planner_model=planner,
            replanner_model=replanner,
            submit_arguments_model=submit_arguments_model,
        )

        config = {
            "configurable": {
                "thread_id": "test-risky-reject",
            }
        }

        interrupted_result = graph.invoke(
            create_initial_state(
                "Відправ estimation request для DEM-023."
            ),
            config=config,
        )

        assert interrupted_result["status"] == "waiting_human"
        assert "__interrupt__" in interrupted_result
        assert interrupted_result["current_step"] == 0
        assert interrupted_result["used_tools"] == []

        interrupt_data = interrupted_result[
            "__interrupt__"
        ][0].value

        assert interrupt_data["type"] == "tool_approval"
        assert interrupt_data["action"] == (
            "submit_estimation_request"
        )
        assert interrupt_data["arguments"]["initiative_id"] == (
            "DEM-023"
        )

        final_result = graph.invoke(
            Command(
                resume={
                    "decision": "reject",
                    "reason": "Потрібне погодження Delivery Lead.",
                }
            ),
            config=config,
        )

    assert final_result["completed"] is True
    assert final_result["status"] == "rejected"
    assert final_result["current_step"] == 1
    assert final_result["used_tools"] == []
    assert "не відправлено" in final_result["final_answer"]