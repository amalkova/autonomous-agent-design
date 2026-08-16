"""Тести SQLite persistence для Plan-and-Execute графа."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from persistence_demo import snapshot_report, thread_config
from plan_execute import build_graph, create_initial_state


class StaticModel:
    """Fake structured model із постійною відповіддю."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = 0

    def invoke(self, messages: Any) -> dict[str, Any]:
        """Повертає підготовлену structured response."""

        self.calls += 1
        return self.response


def make_planner() -> StaticModel:
    """Створює planner для одного безпечного кроку."""

    return StaticModel(
        {
            "goal": "Перевірити requirements readiness",
            "steps": [
                (
                    "check_requirements_readiness: "
                    "initiative_id=DEM-040"
                ),
            ],
        }
    )


def make_replanner() -> StaticModel:
    """Створює replanner, який завершує роботу."""

    return StaticModel(
        {
            "action": "finish",
            "reasoning": "Запланований крок виконано.",
            "updated_steps": [],
            "final_answer": (
                "Requirements readiness успішно перевірено."
            ),
        }
    )


def fake_step_runner(prompt: str) -> dict[str, Any]:
    """Імітує вкладений ReAct без LLM-виклику."""

    return {
        "status": "completed",
        "answer": "Requirements readiness становить 100%.",
        "used_tools": [
            "check_requirements_readiness",
        ],
    }


def test_thread_config_rejects_blank_id() -> None:
    """Порожній thread_id заборонений."""

    with pytest.raises(
        ValueError,
        match="thread_id не може бути порожнім",
    ):
        thread_config("   ")


def test_thread_config_contains_thread_id() -> None:
    """thread_id передається через configurable config."""

    config = thread_config("persistence-test-001")

    assert config["configurable"]["thread_id"] == (
        "persistence-test-001"
    )
    assert config["recursion_limit"] == 50


def test_state_survives_database_reconnection(
    tmp_path,
) -> None:
    """Checkpoint відновлюється після закриття SQLite connection."""

    database_path = tmp_path / "persistent_state.db"
    config = thread_config("restart-test")

    first_planner = make_planner()

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as first_checkpointer:
        first_graph = build_graph(
            checkpointer=first_checkpointer,
            planner_model=first_planner,
            replanner_model=make_replanner(),
            step_runner=fake_step_runner,
            interrupt_after=["planner"],
        )

        first_graph.invoke(
            create_initial_state(
                "Перевір readiness для DEM-040."
            ),
            config=config,
        )

        saved_snapshot = first_graph.get_state(config)

        assert saved_snapshot.values["status"] == "running"
        assert saved_snapshot.values["current_step"] == 0
        assert saved_snapshot.next == ("executor",)
        assert first_planner.calls == 1

    assert database_path.exists()
    assert database_path.stat().st_size > 0

    second_planner = make_planner()

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as second_checkpointer:
        second_graph = build_graph(
            checkpointer=second_checkpointer,
            planner_model=second_planner,
            replanner_model=make_replanner(),
            step_runner=fake_step_runner,
            interrupt_after=["planner"],
        )

        restored_snapshot = second_graph.get_state(config)

        assert restored_snapshot.values["goal"] == (
            "Перевірити requirements readiness"
        )
        assert restored_snapshot.next == ("executor",)

        final_result = second_graph.invoke(
            None,
            config=config,
        )

        final_snapshot = second_graph.get_state(config)

    assert second_planner.calls == 0
    assert final_result["completed"] is True
    assert final_result["status"] == "completed"
    assert final_snapshot.next == ()
    assert final_snapshot.values["current_step"] == 1
    assert final_snapshot.values["used_tools"] == [
        "check_requirements_readiness",
    ]


def test_different_thread_ids_have_independent_state(
    tmp_path,
) -> None:
    """Два thread_id не змішують свої checkpoints."""

    database_path = tmp_path / "independent_threads.db"
    planner = make_planner()

    first_config = thread_config("thread-alpha")
    second_config = thread_config("thread-beta")

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
            planner_model=planner,
            replanner_model=make_replanner(),
            step_runner=fake_step_runner,
            interrupt_after=["planner"],
        )

        graph.invoke(
            create_initial_state(
                "Перший незалежний запит для DEM-040."
            ),
            config=first_config,
        )

        graph.invoke(
            create_initial_state(
                "Другий незалежний запит для DEM-041."
            ),
            config=second_config,
        )

        first_snapshot = graph.get_state(first_config)
        second_snapshot = graph.get_state(second_config)

    assert first_snapshot.values["user_request"] == (
        "Перший незалежний запит для DEM-040."
    )
    assert second_snapshot.values["user_request"] == (
        "Другий незалежний запит для DEM-041."
    )
    assert (
        first_snapshot.values["user_request"]
        != second_snapshot.values["user_request"]
    )
    assert first_snapshot.config != second_snapshot.config
    assert planner.calls == 2


def test_snapshot_report_reads_checkpoint(
    tmp_path,
) -> None:
    """snapshot_report повертає persisted state та next nodes."""

    database_path = tmp_path / "snapshot_report.db"
    config = thread_config("report-thread")

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
            planner_model=make_planner(),
            replanner_model=make_replanner(),
            step_runner=fake_step_runner,
            interrupt_after=["planner"],
        )

        graph.invoke(
            create_initial_state(
                "Перевір persisted report для DEM-040."
            ),
            config=config,
        )

        report = snapshot_report(
            graph,
            "report-thread",
            database_path,
        )

    assert report["thread_id"] == "report-thread"
    assert report["checkpoint_exists"] is True
    assert report["next_nodes"] == ["executor"]
    assert report["state"]["current_step"] == 0
    assert report["state"]["completed"] is False
    assert report["interrupts"] == []
    assert report["database"] == str(database_path.resolve())