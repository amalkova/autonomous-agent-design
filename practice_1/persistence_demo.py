"""CLI-демонстрація SQLite persistence для Plan-and-Execute агента."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from plan_execute import (
    DEFAULT_DATABASE_PATH,
    build_graph,
    create_initial_state,
)


DEMO_REQUEST = """
Перевір готовність вимог ініціативи DEM-030 до estimation.

Відомі дані:
business_objective=Скоротити час погодження кредитної заявки;
functional_requirements=Описані основні користувацькі сценарії;
non_functional_requirements=Визначені performance та availability;
acceptance_criteria=Підготовлені Given-When-Then критерії;
integration_scope=CRM та Core Banking;
data_requirements=Описані джерела й правила міграції.

Після перевірки визнач estimation complexity:
systems_count=3;
integration_count=2;
nfr_complexity=high;
data_migration=required;
security_review=required;
dependency_count=3;
requirements_stability=partial.

Також знайди у knowledge base правила підготовки до estimation.
""".strip()


def thread_config(thread_id: str) -> dict[str, Any]:
    """Створює LangGraph config для окремого thread."""

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise ValueError("thread_id не може бути порожнім.")

    return {
        "configurable": {
            "thread_id": normalized_thread_id,
        },
        "recursion_limit": 50,
    }


def serialize_interrupts(snapshot: Any) -> list[dict[str, Any]]:
    """Збирає dynamic interrupts зі snapshot tasks."""

    serialized: list[dict[str, Any]] = []

    for task in getattr(snapshot, "tasks", ()):
        for interrupt_value in getattr(task, "interrupts", ()):
            value = getattr(interrupt_value, "value", interrupt_value)

            if isinstance(value, dict):
                serialized.append(value)
            else:
                serialized.append({"message": str(value)})

    return serialized


def snapshot_report(
    graph: Any,
    thread_id: str,
    database_path: Path,
) -> dict[str, Any]:
    """Повертає JSON-сумісний звіт про checkpoint."""

    config = thread_config(thread_id)
    snapshot = graph.get_state(config)
    state = dict(snapshot.values or {})

    return {
        "thread_id": thread_id,
        "database": str(database_path.resolve()),
        "checkpoint_exists": bool(snapshot.values),
        "created_at": getattr(snapshot, "created_at", None),
        "next_nodes": list(snapshot.next),
        "state": state,
        "interrupts": serialize_interrupts(snapshot),
    }


def start_thread(
    thread_id: str,
    request: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """Створює thread і зупиняється після planner."""

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
            interrupt_after=[
                "planner",
                "executor",
                "replanner",
            ],
        )

        existing_snapshot = graph.get_state(
            thread_config(thread_id)
        )

        if existing_snapshot.values:
            raise ValueError(
                f"Thread {thread_id} вже існує. "
                "Використайте інший thread_id або команду resume."
            )

        graph.invoke(
            create_initial_state(request),
            config=thread_config(thread_id),
        )

        return snapshot_report(
            graph,
            thread_id,
            database_path,
        )


def inspect_thread(
    thread_id: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """Читає checkpoint без продовження виконання."""

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
            interrupt_after=[
                "planner",
                "executor",
                "replanner",
            ],
        )

        report = snapshot_report(
            graph,
            thread_id,
            database_path,
        )

        if not report["checkpoint_exists"]:
            raise ValueError(
                f"Checkpoint для thread {thread_id} не знайдено."
            )

        return report


def resume_thread(
    thread_id: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """Відновлює thread із наступного збереженого вузла."""

    with SqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        graph = build_graph(
            checkpointer=checkpointer,
            interrupt_after=[
                "planner",
                "executor",
                "replanner",
            ],
        )

        before_resume = graph.get_state(
            thread_config(thread_id)
        )

        if not before_resume.values:
            raise ValueError(
                f"Checkpoint для thread {thread_id} не знайдено."
            )

        if not before_resume.next:
            return snapshot_report(
                graph,
                thread_id,
                database_path,
            )

        if serialize_interrupts(before_resume):
            raise ValueError(
                "Thread очікує HITL-рішення. "
                "Використайте інтерактивний plan_execute.py."
            )

        graph.invoke(
            None,
            config=thread_config(thread_id),
        )

        return snapshot_report(
            graph,
            thread_id,
            database_path,
        )


def compare_threads(
    first_thread_id: str,
    second_thread_id: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """Порівнює два незалежні persisted threads."""

    first_report = inspect_thread(
        first_thread_id,
        database_path,
    )
    second_report = inspect_thread(
        second_thread_id,
        database_path,
    )

    first_state = first_report["state"]
    second_state = second_report["state"]

    return {
    "threads_are_independent": (
        first_thread_id != second_thread_id
        and first_report["checkpoint_exists"]
        and second_report["checkpoint_exists"]
    ),
    "state_values_differ": first_state != second_state,
    "progress_is_independent": (
        first_state.get("current_step"),
        first_state.get("status"),
        first_state.get("completed"),
    )
    != (
        second_state.get("current_step"),
        second_state.get("status"),
        second_state.get("completed"),
    ),
    "first_thread": first_report,
    "second_thread": second_report,
}


def print_json(value: dict[str, Any]) -> None:
    """Друкує відформатований JSON."""

    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Створює CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Демонстрація SQLite persistence для "
            "Plan-and-Execute агента."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="Шлях до SQLite database.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    start_parser = subparsers.add_parser(
        "start",
        help="Створити новий persisted thread.",
    )
    start_parser.add_argument("thread_id")
    start_parser.add_argument(
        "--request",
        default=DEMO_REQUEST,
        help="Користувацький запит для planner.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Переглянути checkpoint без виконання.",
    )
    inspect_parser.add_argument("thread_id")

    resume_parser = subparsers.add_parser(
        "resume",
        help="Продовжити виконання з checkpoint.",
    )
    resume_parser.add_argument("thread_id")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Порівняти два persisted threads.",
    )
    compare_parser.add_argument("first_thread_id")
    compare_parser.add_argument("second_thread_id")

    return parser


def main() -> None:
    """Виконує CLI-команду."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    try:
        if arguments.command == "start":
            result = start_thread(
                arguments.thread_id,
                arguments.request,
                arguments.database,
            )

        elif arguments.command == "inspect":
            result = inspect_thread(
                arguments.thread_id,
                arguments.database,
            )

        elif arguments.command == "resume":
            result = resume_thread(
                arguments.thread_id,
                arguments.database,
            )

        else:
            result = compare_threads(
                arguments.first_thread_id,
                arguments.second_thread_id,
                arguments.database,
            )

        print_json(result)

    except (ValueError, RuntimeError) as error:
        parser.exit(
            status=1,
            message=f"{error}\n",
        )


if __name__ == "__main__":
    main()