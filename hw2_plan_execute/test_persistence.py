"""Тести SqliteSaver persistence та незалежності thread_id."""

import sqlite3
from pathlib import Path
from typing import Any

from langchain.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from plan_execute import (
    build_graph,
    initial_state,
    thread_config,
)


class PersistencePlannerModel:
    """Створює детермінований план із двох кроків."""

    def invoke(self, messages) -> dict[str, Any]:
        return {
            "goal": (
                "Перевірити intake та визначити "
                "Discovery scope"
            ),
            "steps": [
                (
                    "check_intake_completeness: "
                    "перевірити Gate 0"
                ),
                (
                    "classify_discovery_scope: "
                    "визначити Discovery scope"
                ),
            ],
        }


class PersistenceExecutorModel:
    """Детерміновано викликає tool поточного кроку."""

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
                        "id": "persistence-classify",
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
                            "Reduce time by 30%"
                        ),
                        "financial_effect": (
                            "Lower operational costs"
                        ),
                        "constraints": (
                            "Legacy CRM integration"
                        ),
                    },
                    "id": "persistence-intake",
                    "type": "tool_call",
                }
            ],
        )


class PersistenceReplannerModel:
    """Продовжує до виконання всіх кроків."""

    def invoke(self, messages) -> dict[str, Any]:
        prompt = messages[-1].content

        if "Виконано кроків: 2/2" in prompt:
            return {
                "action": "finish",
                "updated_steps": None,
                "reasoning": (
                    "Усі кроки виконано."
                ),
                "final_answer": (
                    "Intake перевірено. "
                    "Discovery scope визначено."
                ),
            }

        return {
            "action": "continue",
            "updated_steps": None,
            "reasoning": (
                "Наступний крок залишається актуальним."
            ),
            "final_answer": None,
        }


def create_test_app(
    database_path: Path,
):
    """Створює persistent test app."""

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(connection)

    app = build_graph(
        checkpointer=checkpointer,
        planner_model=PersistencePlannerModel(),
        executor_model=PersistenceExecutorModel(),
        replanner_model=PersistenceReplannerModel(),
        interrupt_after=["executor"],
    )

    return app, connection


def test_state_is_restored_after_process_restart(
    tmp_path,
) -> None:
    """Стан відновлюється після закриття і нового підключення."""

    database_path = tmp_path / "agent_state.db"
    config = thread_config("restart-session")

    first_app, first_connection = create_test_app(
        database_path
    )

    first_app.invoke(
        initial_state(
            "Перевірити DEM-004 та визначити scope."
        ),
        config=config,
    )

    first_snapshot = first_app.get_state(config)

    assert first_snapshot.values["current_step"] == 1
    assert first_snapshot.next == ("replanner",)
    assert len(first_snapshot.values["results"]) == 1

    first_connection.close()

    assert database_path.exists()
    assert database_path.stat().st_size > 0

    second_app, second_connection = create_test_app(
        database_path
    )

    restored_snapshot = second_app.get_state(config)

    assert restored_snapshot.values["current_step"] == 1
    assert restored_snapshot.next == ("replanner",)
    assert (
        restored_snapshot.values["user_request"]
        == "Перевірити DEM-004 та визначити scope."
    )

    second_app.invoke(
        None,
        config=config,
    )

    second_snapshot = second_app.get_state(config)

    assert second_snapshot.values["current_step"] == 2
    assert second_snapshot.next == ("replanner",)
    assert len(second_snapshot.values["results"]) == 2

    second_connection.close()

    third_app, third_connection = create_test_app(
        database_path
    )

    third_app.invoke(
        None,
        config=config,
    )

    completed_snapshot = third_app.get_state(config)

    assert completed_snapshot.values["completed"] is True
    assert completed_snapshot.values["status"] == "completed"
    assert completed_snapshot.next == ()
    assert (
        completed_snapshot.values["used_tools"]
        == [
            "check_intake_completeness",
            "classify_discovery_scope",
        ]
    )

    third_connection.close()


def test_different_thread_ids_have_independent_state(
    tmp_path,
) -> None:
    """Два thread_id не змішують user request та результати."""

    database_path = tmp_path / "independent.db"
    app, connection = create_test_app(database_path)

    first_config = thread_config("session-A")
    second_config = thread_config("session-B")

    try:
        app.invoke(
            initial_state(
                "Перший незалежний запит DEM-004."
            ),
            config=first_config,
        )
        app.invoke(
            initial_state(
                "Другий незалежний запит DEM-005."
            ),
            config=second_config,
        )

        first_snapshot = app.get_state(first_config)
        second_snapshot = app.get_state(second_config)

        assert (
            first_snapshot.values["user_request"]
            == "Перший незалежний запит DEM-004."
        )
        assert (
            second_snapshot.values["user_request"]
            == "Другий незалежний запит DEM-005."
        )
        assert (
            first_snapshot.config["configurable"]["thread_id"]
            != second_snapshot.config["configurable"]["thread_id"]
        )
        assert first_snapshot.next == ("replanner",)
        assert second_snapshot.next == ("replanner",)
    finally:
        connection.close()


def test_completed_thread_remains_available(
    tmp_path,
) -> None:
    """Checkpoint завершеного thread доступний після restart."""

    database_path = tmp_path / "completed.db"
    config = thread_config("completed-session")

    app, connection = create_test_app(database_path)

    app.invoke(
        initial_state("Перевірити ініціативу."),
        config=config,
    )
    app.invoke(None, config=config)
    app.invoke(None, config=config)

    connection.close()

    restored_app, restored_connection = create_test_app(
        database_path
    )

    try:
        snapshot = restored_app.get_state(config)

        assert snapshot.values["completed"] is True
        assert snapshot.next == ()
        assert snapshot.values["final_answer"]
    finally:
        restored_connection.close()