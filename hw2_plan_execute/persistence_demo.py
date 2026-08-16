"""Демонстрація SQLite persistence між окремими процесами."""

import argparse
import json
from typing import Any

from plan_execute import (
    AGENT_STATE_DB,
    create_persistent_app,
    public_result,
    start_run,
    thread_config,
)


DEFAULT_REQUEST = """
Для ініціативи DEM-004:

1. Перевір повноту intake:
business_owner=Retail Director;
business_driver=Reduce onboarding time;
success_metrics=Reduce onboarding time by 30%;
financial_effect=Lower operational costs;
constraints=Legacy CRM integration.

2. Розрахуй Discovery Points:
systems_count=4;
ownership_clarity=partial;
technical_uncertainty=high;
dependency_count=3;
regulatory_impact=possible;
data_readiness=partial.

3. Знайди у knowledge base правила та timebox
для рекомендованого Discovery scope.

Не відправляй фінальний assessment.
""".strip()


def snapshot_result(
    app: Any,
    thread_id: str,
) -> dict[str, Any]:
    """Повертає JSON-сумісний checkpoint thread."""

    snapshot = app.get_state(
        thread_config(thread_id)
    )
    state_values = dict(snapshot.values)

    return {
        "thread_id": thread_id,
        "database": str(AGENT_STATE_DB),
        "checkpoint_exists": bool(state_values),
        "next_nodes": list(snapshot.next),
        "state": (
            public_result(state_values)
            if state_values
            else None
        ),
    }


def start_demo(
    app: Any,
    thread_id: str,
    request: str,
) -> dict[str, Any]:
    """Запускає новий thread і зупиняє після executor."""

    existing_snapshot = snapshot_result(
        app,
        thread_id,
    )

    if existing_snapshot["checkpoint_exists"]:
        raise ValueError(
            f"Thread {thread_id} вже існує. "
            "Використайте інший thread_id або команду resume."
        )

    start_run(
        app=app,
        user_request=request,
        thread_id=thread_id,
    )

    return snapshot_result(
        app,
        thread_id,
    )


def resume_demo(
    app: Any,
    thread_id: str,
) -> dict[str, Any]:
    """Продовжує виконання із збереженого checkpoint."""

    before_resume = snapshot_result(
        app,
        thread_id,
    )

    if not before_resume["checkpoint_exists"]:
        raise ValueError(
            f"Checkpoint для thread {thread_id} не знайдено."
        )

    if not before_resume["next_nodes"]:
        return {
            **before_resume,
            "message": "Thread уже завершено.",
        }

    app.invoke(
        None,
        config=thread_config(thread_id),
    )

    return snapshot_result(
        app,
        thread_id,
    )


def compare_threads(
    app: Any,
    first_thread_id: str,
    second_thread_id: str,
) -> dict[str, Any]:
    """Показує незалежний стан двох thread_id."""

    return {
        "threads_are_independent": (
            first_thread_id != second_thread_id
        ),
        "first_thread": snapshot_result(
            app,
            first_thread_id,
        ),
        "second_thread": snapshot_result(
            app,
            second_thread_id,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Створює CLI parser для persistence demo."""

    parser = argparse.ArgumentParser(
        description=(
            "Демонстрація SqliteSaver persistence "
            "для Plan-and-Execute агента."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    start_parser = subparsers.add_parser(
        "start",
        help="Створити новий thread і виконати один крок",
    )
    start_parser.add_argument(
        "thread_id",
        help="Унікальний thread_id",
    )
    start_parser.add_argument(
        "--request",
        default=DEFAULT_REQUEST,
        help="Користувацький запит",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Показати збережений стан thread",
    )
    inspect_parser.add_argument(
        "thread_id",
        help="Thread для перевірки",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="Продовжити thread з останнього checkpoint",
    )
    resume_parser.add_argument(
        "thread_id",
        help="Thread для відновлення",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Порівняти незалежний стан двох threads",
    )
    compare_parser.add_argument("first_thread_id")
    compare_parser.add_argument("second_thread_id")

    return parser


def main() -> None:
    """Виконує вибрану persistence-команду."""

    arguments = build_parser().parse_args()

    app, connection = create_persistent_app(
        database_path=AGENT_STATE_DB,
        interrupt_after=["executor"],
    )

    try:
        if arguments.command == "start":
            result = start_demo(
                app=app,
                thread_id=arguments.thread_id,
                request=arguments.request,
            )
        elif arguments.command == "resume":
            result = resume_demo(
                app=app,
                thread_id=arguments.thread_id,
            )
        elif arguments.command == "inspect":
            result = snapshot_result(
                app=app,
                thread_id=arguments.thread_id,
            )
        else:
            result = compare_threads(
                app=app,
                first_thread_id=(
                    arguments.first_thread_id
                ),
                second_thread_id=(
                    arguments.second_thread_id
                ),
            )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    finally:
        connection.close()


if __name__ == "__main__":
    main()