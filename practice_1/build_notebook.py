"""Генерує notebook для Практичного завдання №1."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent

NOTEBOOK_PATH = (
    PROJECT_DIR
    / "Task_001_Malkova_Requirements_Estimation.ipynb"
)


def load_json(
    filename: str,
    default: dict[str, Any],
) -> dict[str, Any]:
    """Читає локальний JSON artifact."""

    path = PROJECT_DIR / filename

    if not path.exists():
        return default

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def markdown(source: str) -> dict[str, Any]:
    """Створює Markdown cell."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip(),
    }


def code(source: str) -> dict[str, Any]:
    """Створює code cell."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip(),
    }


def build_notebook() -> dict[str, Any]:
    """Формує фінальний notebook."""

    react_mermaid = (
        PROJECT_DIR / "react_graph.mmd"
    ).read_text(encoding="utf-8")

    plan_mermaid = (
        PROJECT_DIR / "plan_execute_graph.mmd"
    ).read_text(encoding="utf-8")

    comparison = load_json(
        "comparison_results.json",
        {"runs": [], "comparison": {}},
    )

    trajectory = load_json(
        "trajectory.json",
        {"runs": []},
    )

    comparison_runs = comparison.get("runs", [])

    react_result = next(
        (
            item
            for item in comparison_runs
            if item.get("architecture") == "ReAct"
        ),
        {},
    )

    plan_result = next(
        (
            item
            for item in comparison_runs
            if item.get("architecture")
            == "Plan-and-Execute"
        ),
        {},
    )

    react_time = react_result.get(
        "execution_time_seconds",
        "n/a",
    )
    plan_time = plan_result.get(
        "execution_time_seconds",
        "n/a",
    )

    react_quality = (
        react_result
        .get("quality", {})
        .get("quality_score_percent", "n/a")
    )
    plan_quality = (
        plan_result
        .get("quality", {})
        .get("quality_score_percent", "n/a")
    )

    trajectory_count = len(
        trajectory.get("runs", [])
    )

    cells = [
        markdown(
            """
            # Практичне завдання №1

            ## Requirements & Estimation Readiness Agent

            **Авторка:** Anna Malkova
            **Модель:** `gemini-3.1-flash-lite`
            **Стек:** LangGraph, LangChain, ChromaDB,
            Pydantic v2, SQLite
            **Архітектури:** ReAct та Plan-and-Execute
            """
        ),
        markdown(
            """
            ## 1. Мета та бізнес-контекст

            Агент підтримує перехід від discovery до delivery
            estimation:

            `Demand → Discovery → Requirements Readiness
            → Estimation → Delivery`

            Основні функції:

            - перевірка requirements readiness;
            - класифікація estimation complexity;
            - пошук handover gaps;
            - Agentic RAG;
            - SQLite persistence;
            - HITL перед ризиковою відправкою;
            - JSON trajectory logging.
            """
        ),
        markdown(
            """
            ## 2. Acceptance criteria

            | Критерій | Реалізація |
            |---|---|
            | Domain tools | 4 Pydantic tools |
            | ReAct | LLM–tools–LLM LangGraph |
            | Safety | steps, timeout, repeated calls |
            | Plan-and-Execute | planner/executor/replanner |
            | Persistence | SqliteSaver та thread_id |
            | RAG | 12 документів у ChromaDB |
            | HITL | approve/reject/edit |
            | Tests | 59 unit/integration tests |
            | Trajectory | обидві архітектури |
            | Bonus | comparison, visualization, fallback |
            """
        ),
        markdown(
            f"""
            ## 3. ReAct architecture

            ```mermaid
            {react_mermaid}
            ```

            ReAct самостійно обирає tools і продовжує цикл
            reasoning після кожного observation.
            """
        ),
        code(
            """
            from safety import SafetyController
            from react_agent import (
                REACT_TOOLS,
                build_react_graph,
                run_react_agent,
            )

            react_graph = build_react_graph(
                SafetyController()
            )

            print(list(react_graph.get_graph().nodes))
            print([tool.name for tool in REACT_TOOLS])
            """
        ),
        markdown(
            """
            ### ReAct safety

            - `max_steps = 10`;
            - `timeout = 120` секунд;
            - максимум два однакові tool calls;
            - structured final response;
            - статус `safety_stop`.
            """
        ),
        code(
            """
            RUN_LIVE = False

            request = (
                "Для DEM-050 перевір requirements readiness, "
                "визнач estimation complexity та знайди "
                "правила у knowledge base."
            )

            if RUN_LIVE:
                response = run_react_agent(request)
                print(response)
            else:
                print(
                    "Live call вимкнений. "
                    "Результат є у trajectory.json."
                )
            """
        ),
        markdown(
            f"""
            ## 4. Plan-and-Execute architecture

            ```mermaid
            {plan_mermaid}
            ```

            Planner формує structured plan. Executor виконує
            кожен step через вкладений ReAct. Replanner обирає
            `continue`, `replan` або `finish`. Approval node
            зупиняє risky operation до рішення людини.
            """
        ),
        code(
            """
            from plan_execute import (
                build_graph,
                create_initial_state,
            )

            plan_graph = build_graph()

            print(list(plan_graph.get_graph().nodes))

            create_initial_state(
                "Перевір readiness DEM-050."
            )
            """
        ),
        markdown(
            """
            ## 5. Domain tools

            1. `check_requirements_readiness`
            2. `classify_estimation_complexity`
            3. `identify_handover_gaps`
            4. `submit_estimation_request`

            Окремий RAG tool:

            5. `search_delivery_knowledge`

            Усі domain tools мають Pydantic v2 schemas,
            validators та стандартну JSON-відповідь
            `{status, data, error}`.
            """
        ),
        code(
            """
            from tools import DOMAIN_TOOLS

            for tool in DOMAIN_TOOLS:
                print(
                    tool.name,
                    tool.args_schema.__name__,
                )
            """
        ),
        markdown(
            """
            ## 6. Agentic RAG

            Persistent ChromaDB collection містить 12
            документів про Definition of Ready, FR/NFR,
            acceptance criteria, integrations, data migration,
            security, dependencies, sizing та handover.

            Агент сам вирішує, коли викликати semantic search.
            """
        ),
        code(
            """
            from knowledge import (
                initialize_knowledge_base,
                search_delivery_knowledge,
            )

            print(initialize_knowledge_base())

            result = search_delivery_knowledge.invoke(
                {
                    "query": (
                        "requirements readiness "
                        "before estimation"
                    )
                }
            )

            print(result)
            """
        ),
        markdown(
            """
            ## 7. Human-in-the-Loop

            `submit_estimation_request` є high-risk operation.

            Перед виконанням людина бачить tool та всі
            arguments і може обрати:

            - `approve`;
            - `reject`;
            - `edit`.

            Live demo:

            - DEM-060 схвалено, створено `EST-001`;
            - DEM-061 відхилено;
            - для DEM-061 tool не виконувався.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            submission_path = Path(
                "submitted_estimation_requests.json"
            )

            if submission_path.exists():
                submissions = json.loads(
                    submission_path.read_text(
                        encoding="utf-8"
                    )
                )

                print(
                    [
                        (
                            item["request_id"],
                            item["initiative_id"],
                            item["status"],
                        )
                        for item in submissions
                    ]
                )
            """
        ),
        markdown(
            """
            ## 8. SQLite persistence

            LangGraph state зберігається через `SqliteSaver`
            у `agent_state.db`.

            Live demo показав:

            - окремі checkpoints для двох thread_id;
            - відновлення з `executor`, а не з початку;
            - незалежний progress;
            - завершення після кількох restart/resume.
            """
        ),
        code(
            """
            from persistence_demo import (
                compare_threads,
                inspect_thread,
            )

            completed = inspect_thread(
                "practice-persistence-002"
            )

            print(
                completed["state"]["status"],
                completed["state"]["current_step"],
                completed["state"]["used_tools"],
            )
            """
        ),
        markdown(
            f"""
            ## 9. Trajectory logging

            Кількість збережених runs: **{trajectory_count}**.

            `trajectory.json` містить ReAct та
            Plan-and-Execute runs із messages, tools,
            final response, safety та metadata.
            """
        ),
        code(
            """
            trajectory = json.loads(
                Path("trajectory.json").read_text(
                    encoding="utf-8"
                )
            )

            for run in trajectory["runs"]:
                print(
                    run["agent_type"],
                    len(run["messages"]),
                    run["final_response"].get(
                        "used_tools",
                        [],
                    ),
                )
            """
        ),
        markdown(
            f"""
            ## 10. Числове порівняння

            | Метрика | ReAct | Plan-and-Execute |
            |---|---:|---:|
            | Execution time | {react_time} s | {plan_time} s |
            | Quality score | {react_quality}% | {plan_quality}% |
            | Tool coverage | 100% | 100% |

            ReAct був швидшим для короткого сценарію.
            Plan-and-Execute надав явний plan, persistence,
            replanning та HITL.
            """
        ),
        code(
            """
            comparison = json.loads(
                Path("comparison_results.json").read_text(
                    encoding="utf-8"
                )
            )

            comparison["comparison"]
            """
        ),
        markdown(
            """
            ## 11. Автоматичні тести

            Тести покривають schemas, tools, safety, RAG,
            ReAct, Plan-and-Execute, replanning, HITL,
            persistence, thread independence та comparison.
            """
        ),
        code(
            """
            import subprocess
            import sys

            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                capture_output=True,
                text=True,
                check=False,
            )

            print(result.stdout)
            assert result.returncode == 0
            """
        ),
        markdown(
            """
            ## 12. Висновки

            ReAct доцільний для коротких динамічних задач із
            мінімальною latency.

            Plan-and-Execute краще підходить для довгих
            workflow, де потрібні планування, progress tracking,
            persistence, replanning, HITL та auditability.

            ### Обмеження

            - локальний mock storage замість Jira;
            - навчальна knowledge base;
            - залежність від Gemini quota;
            - SQLite не є distributed production storage;
            - quality score не замінює LLM-as-a-judge.
            """
        ),
        markdown(
            """
            ## 13. Основні артефакти

            - `README.md`
            - `trajectory.json`
            - `agent_state.db`
            - `comparison_results.json`
            - `test_results.json`
            - `react_graph.mmd`
            - `plan_execute_graph.mmd`
            - `Task_001_Malkova_Requirements_Estimation.ipynb`
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.2",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Записує notebook."""

    notebook = build_notebook()

    NOTEBOOK_PATH.write_text(
        json.dumps(
            notebook,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(NOTEBOOK_PATH)
    print(f"cells: {len(notebook['cells'])}")
    print(f"size: {NOTEBOOK_PATH.stat().st_size}")


if __name__ == "__main__":
    main()