"""Демонстрація SQLite persistence та static breakpoints."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from plan_execute_agent import (
    DEFAULT_DATABASE_PATH,
    build_graph,
    create_initial_state,
    public_result,
)
from trajectory_logger import make_json_safe


DATABASE_PATH = Path(DEFAULT_DATABASE_PATH)


class StaticModel:
    """Детермінована заміна LLM для persistence demo."""

    def __init__(
        self,
        response: dict[str, Any],
    ) -> None:
        self.response = response

    def invoke(
        self,
        _messages: Any,
    ) -> dict[str, Any]:
        """Повертає наперед визначену structured response."""

        return self.response


PLANNER_MODEL = StaticModel(
    {
        "goal": (
            "Перевірити готовність DEM-901 "
            "до estimation."
        ),
        "steps": [
            (
                "check_requirements_readiness: "
                "перевірити обов'язкові requirements "
                "для DEM-901"
            ),
        ],
    }
)

REPLANNER_MODEL = StaticModel(
    {
        "action": "finish",
        "reasoning": (
            "Детермінований demo step виконано."
        ),
        "updated_steps": [],
        "final_answer": (
            "DEM-901 перевірено після відновлення "
            "з SQLite checkpoint."
        ),
    }
)

RISKY_PLANNER_MODEL = StaticModel(
    {
        "goal": (
            "Підготувати DEM-902 до submit "
            "після людського погодження."
        ),
        "steps": [
            (
                "submit_estimation_request: "
                "відправити DEM-902 після approval"
            ),
        ],
    }
)

SUBMIT_ARGUMENTS_MODEL = StaticModel(
    {
        "initiative_id": "DEM-902",
        "estimation_complexity": "Medium",
        "estimation_points": 5,
        "target_team": "Delivery Team Alpha",
        "requested_by": "Persistence Demo",
        "estimation_summary": (
            "Детермінований запит для перевірки "
            "static interrupt_before."
        ),
    }
)


def deterministic_step_runner(
    _step_prompt: str,
) -> dict[str, Any]:
    """Імітує успішний tool step без LLM та мережі."""

    return {
        "status": "success",
        "answer": (
            "Requirements readiness перевірено."
        ),
        "used_tools": [
            "check_requirements_readiness",
        ],
        "requires_human_confirmation": False,
    }


def build_config(
    thread_id: str,
) -> dict[str, Any]:
    """Створює LangGraph config зі стабільним thread_id."""

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 30,
    }


@contextmanager
def open_demo_graph(
    *,
    interrupt_before: list[str],
    planner_model: Any = PLANNER_MODEL,
    submit_arguments_model: Any = (
        SUBMIT_ARGUMENTS_MODEL
    ),
) -> Iterator[Any]:
    """Відкриває graph поверх persistent SQLite saver."""

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    saver = SqliteSaver(connection)
    saver.setup()

    graph = build_graph(
        checkpointer=saver,
        planner_model=planner_model,
        replanner_model=REPLANNER_MODEL,
        submit_arguments_model=(
            submit_arguments_model
        ),
        step_runner=deterministic_step_runner,
        interrupt_before=interrupt_before,
    )

    try:
        yield graph
    finally:
        connection.close()


def snapshot_summary(
    snapshot: Any,
) -> dict[str, Any]:
    """Серіалізує ключовий стан checkpoint."""

    values = dict(snapshot.values or {})

    return {
        "next": list(snapshot.next),
        "status": values.get("status"),
        "current_step": values.get(
            "current_step"
        ),
        "completed": values.get("completed"),
        "results": make_json_safe(
            values.get("results", [])
        ),
    }


def start(
    thread_id: str,
) -> dict[str, Any]:
    """Запускає graph і зупиняє його перед executor."""

    config = build_config(thread_id)

    with open_demo_graph(
        interrupt_before=["executor"],
    ) as graph:
        graph.invoke(
            create_initial_state(
                "Перевір готовність DEM-901."
            ),
            config=config,
        )

        snapshot = graph.get_state(config)
        summary = snapshot_summary(snapshot)

    if "executor" not in summary["next"]:
        raise RuntimeError(
            "Graph не зупинився перед executor."
        )

    return summary


def inspect(
    thread_id: str,
) -> dict[str, Any]:
    """Читає checkpoint після повторного відкриття БД."""

    config = build_config(thread_id)

    with open_demo_graph(
        interrupt_before=["executor"],
    ) as graph:
        snapshot = graph.get_state(config)
        return snapshot_summary(snapshot)


def resume(
    thread_id: str,
) -> dict[str, Any]:
    """Відновлює graph з тим самим thread_id."""

    config = build_config(thread_id)

    with open_demo_graph(
        interrupt_before=["executor"],
    ) as graph:
        result = graph.invoke(
            None,
            config=config,
        )

    return public_result(result)


def compare(
    before: dict[str, Any],
    inspected: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Порівнює стан до crash, після reopen і resume."""

    return {
        "checkpoint_survived_restart": (
            before == inspected
        ),
        "paused_before_executor": (
            "executor" in before["next"]
        ),
        "resumed_with_same_thread": (
            after.get("completed") is True
        ),
        "final_answer": after.get(
            "final_answer"
        ),
    }


def demonstrate_approval_breakpoint() -> dict[str, Any]:
    """Зупиняє risky flow перед входом в approval node."""

    thread_id = (
        "approval-"
        + uuid.uuid4().hex
    )
    config = build_config(thread_id)

    with open_demo_graph(
        interrupt_before=["approval"],
        planner_model=RISKY_PLANNER_MODEL,
    ) as graph:
        graph.invoke(
            create_initial_state(
                "Відправ DEM-902 на estimation."
            ),
            config=config,
        )

        snapshot = graph.get_state(config)
        summary = snapshot_summary(snapshot)

    if "approval" not in summary["next"]:
        raise RuntimeError(
            "Graph не зупинився перед approval."
        )

    return {
        "thread_id": thread_id,
        "paused_before": "approval",
        "state": summary,
        "dynamic_interrupt_entered": False,
    }


def main() -> None:
    """Виконує повний persistence demonstration."""

    thread_id = (
        "persistence-"
        + uuid.uuid4().hex
    )

    before = start(thread_id)

    # Закриття SQLite connection вище імітує crash.
    inspected = inspect(thread_id)
    after = resume(thread_id)

    artifact = {
        "database": str(DATABASE_PATH),
        "thread_id": thread_id,
        "before_crash": before,
        "after_restart": inspected,
        "after_resume": make_json_safe(after),
        "comparison": compare(
            before,
            inspected,
            after,
        ),
        "approval_breakpoint": (
            demonstrate_approval_breakpoint()
        ),
    }

    print(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
