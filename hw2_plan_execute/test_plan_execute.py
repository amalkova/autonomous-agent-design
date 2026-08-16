"""Unit-тести Plan-and-Execute графа без зовнішньої LLM."""

from typing import Any

import pytest
from langchain.messages import AIMessage
from pydantic import ValidationError

from plan_execute import (
    Plan,
    ReplanDecision,
    build_graph,
    initial_state,
    replanner_node_factory,
    thread_config,
)


class FakePlannerModel:
    """Повертає детермінований structured plan."""

    def invoke(self, messages) -> dict[str, Any]:
        return {
            "goal": (
                "Перевірити intake та визначити "
                "Discovery scope для DEM-004"
            ),
            "steps": [
                (
                    "check_intake_completeness: "
                    "перевірити повноту Gate 0"
                ),
                (
                    "classify_discovery_scope: "
                    "розрахувати Discovery Points"
                ),
            ],
        }


class FakeExecutorModel:
    """Вибирає tool відповідно до поточного кроку."""

    def invoke(self, messages) -> AIMessage:
        prompt = messages[-1].content

        if "classify_discovery_scope:" in prompt:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "classify_discovery_scope",
                        "args": {
                            "initiative_id": "DEM-004",
                            "systems_count": 4,
                            "ownership_clarity": "partial",
                            "technical_uncertainty": "high",
                            "dependency_count": 3,
                            "regulatory_impact": "possible",
                            "data_readiness": "partial",
                        },
                        "id": "call-classify",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_intake_completeness",
                    "args": {
                        "initiative_id": "DEM-004",
                        "business_owner": "Retail Director",
                        "business_driver": (
                            "Reduce onboarding time"
                        ),
                        "success_metrics": (
                            "Onboarding time reduced by 30%"
                        ),
                        "financial_effect": (
                            "Lower operational costs"
                        ),
                        "constraints": (
                            "Legacy CRM integration"
                        ),
                    },
                    "id": "call-intake",
                    "type": "tool_call",
                }
            ],
        )


class FakeReplannerModel:
    """Після першого кроку продовжує, після другого завершує."""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, messages) -> dict[str, Any]:
        self.call_count += 1

        if self.call_count == 1:
            return {
                "action": "continue",
                "updated_steps": None,
                "reasoning": (
                    "Intake повний, наступний крок актуальний."
                ),
                "final_answer": None,
            }

        return {
            "action": "finish",
            "updated_steps": None,
            "reasoning": (
                "Усі заплановані розрахунки завершено."
            ),
            "final_answer": (
                "Intake повний. Ініціатива DEM-004 "
                "отримала 8 Discovery Points та Deep scope."
            ),
        }


class FakeReplanModel:
    """Повертає рішення про заміну невиконаних кроків."""

    def invoke(self, messages) -> dict[str, Any]:
        return {
            "action": "replan",
            "updated_steps": [
                (
                    "search_knowledge: знайти правила "
                    "для Deep discovery"
                )
            ],
            "reasoning": (
                "Залишковий план потрібно адаптувати."
            ),
            "final_answer": None,
        }


def test_plan_model_accepts_goal_and_steps() -> None:
    """Plan містить goal та послідовність steps."""

    plan = Plan(
        goal="Визначити Discovery scope",
        steps=[
            "check_intake_completeness: перевірити intake",
            "classify_discovery_scope: визначити scope",
        ],
    )

    assert plan.goal == "Визначити Discovery scope"
    assert len(plan.steps) == 2


def test_plan_rejects_duplicate_steps() -> None:
    """Plan не приймає дублікати кроків."""

    with pytest.raises(
        ValidationError,
        match="однакові кроки",
    ):
        Plan(
            goal="Перевірити ініціативу",
            steps=[
                "search_knowledge: знайти правило",
                "search_knowledge: знайти правило",
            ],
        )


def test_replan_requires_updated_steps() -> None:
    """Action replan без updated_steps відхиляється."""

    with pytest.raises(
        ValidationError,
        match="updated_steps",
    ):
        ReplanDecision(
            action="replan",
            reasoning="Потрібна зміна плану.",
        )


def test_graph_has_required_three_nodes() -> None:
    """Граф містить planner, executor та replanner."""

    graph = build_graph(
        planner_model=FakePlannerModel(),
        executor_model=FakeExecutorModel(),
        replanner_model=FakeReplannerModel(),
    )

    assert list(graph.get_graph().nodes) == [
        "__start__",
        "planner",
        "executor",
        "replanner",
        "__end__",
    ]


def test_plan_execute_runs_tools_sequentially() -> None:
    """Executor виконує два кроки через відповідні tools."""

    graph = build_graph(
        planner_model=FakePlannerModel(),
        executor_model=FakeExecutorModel(),
        replanner_model=FakeReplannerModel(),
    )

    result = graph.invoke(
        initial_state(
            (
                "Перевір intake DEM-004 та визнач "
                "Discovery scope за наданими параметрами."
            )
        )
    )

    assert result["completed"] is True
    assert result["status"] == "completed"
    assert result["current_step"] == 2
    assert result["used_tools"] == [
        "check_intake_completeness",
        "classify_discovery_scope",
    ]
    assert len(result["results"]) == 2
    assert "8 Discovery Points" in result["final_answer"]
    assert "Deep" in result["final_answer"]


def test_replanner_replaces_only_remaining_steps() -> None:
    """Replanner зберігає виконані кроки та змінює залишок."""

    replanner_node = replanner_node_factory(
        FakeReplanModel()
    )

    update = replanner_node(
        {
            "messages": [],
            "user_request": "Перевірити DEM-004",
            "goal": "Визначити scope",
            "plan": [
                (
                    "check_intake_completeness: "
                    "перевірити intake"
                ),
                (
                    "classify_discovery_scope: "
                    "визначити scope"
                ),
            ],
            "current_step": 1,
            "results": [
                "Крок 1 виконано успішно."
            ],
            "completed": False,
            "status": "running",
            "used_tools": [
                "check_intake_completeness"
            ],
            "replan_count": 0,
        }
    )

    assert update["plan"] == [
        (
            "check_intake_completeness: "
            "перевірити intake"
        ),
        (
            "search_knowledge: знайти правила "
            "для Deep discovery"
        ),
    ]
    assert update["replan_count"] == 1


def test_empty_thread_id_is_rejected() -> None:
    """Порожній thread_id не дозволяється."""

    with pytest.raises(
        ValueError,
        match="thread_id",
    ):
        thread_config("   ")