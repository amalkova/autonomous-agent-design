"""Запуск trace-демонстрації LangGraph MAS."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


def validate_tracing_environment() -> str:
    """Перевірити налаштування LangSmith і повернути назву проєкту."""

    required_variables = (
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    )

    missing = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    os.environ["LANGSMITH_TRACING"] = "true"

    return os.environ["LANGSMITH_PROJECT"]


async def resolve_awaitable(value: Any) -> Any:
    """Дочекатися async-значень із підтримкою синхронних фабрик."""

    if inspect.isawaitable(value):
        return await value

    return value


async def run_traced_demo() -> dict[str, Any]:
    """Запустити безпечний estimation request із LangSmith tracing."""

    project_name = validate_tracing_environment()

    import langsmith as ls

    from mas_langgraph import (
        build_production_mas,
        run_mas_query,
    )

    request = (
        "Estimate demand DEM-021. "
        "systems_count=3; integration_count=2; "
        "nfr_criticality=high; "
        "data_migration_required=false; "
        "security_review_required=true; "
        "dependency_count=1; "
        "requirements_stability=partial. "
        "Return complexity, Fibonacci points and main drivers. "
        "Do not submit the estimation request."
    )

    with ls.tracing_context(
        enabled=True,
        project_name=project_name,
        tags=[
            "practice-2",
            "langgraph",
            "requirements-estimation",
        ],
        metadata={
            "assignment": "Practical Assignment 2",
            "framework": "LangGraph",
            "case": "Requirements & Estimation",
            "contains_real_pii": False,
        },
    ):
        graph = await resolve_awaitable(
            build_production_mas()
        )

        result = await resolve_awaitable(
            run_mas_query(
                graph,
                request,
                thread_id="practice-2-trace-001",
            )
        )

    public_result = {
        key: result.get(key)
        for key in (
            "current_agent",
            "route_reasoning",
            "final_answer",
            "blocked",
            "completed",
            "handoff_count",
            "trajectory",
        )
        if key in result
    }

    print(
        json.dumps(
            public_result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print(f"\nLangSmith project: {project_name}")

    return result


def main() -> None:
    """Запустити trace-демонстрацію."""

    asyncio.run(run_traced_demo())


if __name__ == "__main__":
    main()