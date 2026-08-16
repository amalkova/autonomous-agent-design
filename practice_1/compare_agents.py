"""Числове порівняння ReAct та Plan-and-Execute агентів."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from plan_execute import (
    MODEL_NAME,
    build_graph,
    create_initial_state,
)
from react_agent import run_react_agent


COMPARISON_RESULTS_PATH = Path(__file__).with_name(
    "comparison_results.json"
)

COMPARISON_REQUEST = """
Для ініціативи DEM-050 виконай три завдання.

1. Перевір requirements readiness:
business_objective=Скоротити час погодження кредитної заявки;
functional_requirements=Описані користувацькі сценарії;
non_functional_requirements=Визначені performance та availability;
acceptance_criteria=Підготовлені Given-When-Then критерії;
integration_scope=CRM та Core Banking;
data_requirements=Описані джерела даних і правила міграції.

2. Визнач estimation complexity:
systems_count=4;
integration_count=3;
nfr_complexity=high;
data_migration=required;
security_review=required;
dependency_count=4;
requirements_stability=partial.

3. Знайди у knowledge base правила та рекомендації для
передачі вимог на estimation.
""".strip()

EXPECTED_TOOLS = [
    "check_requirements_readiness",
    "classify_estimation_complexity",
    "search_delivery_knowledge",
]

KEYWORD_GROUPS = [
    ["100", "complete", "готов"],
    ["high", "висок"],
    ["estimation", "оцін"],
]


def calculate_quality(
    status: str,
    answer: str,
    used_tools: list[str],
) -> dict[str, Any]:
    """Обчислює просту детерміновану quality score."""

    normalized_answer = answer.lower()

    matched_tools = [
        tool_name
        for tool_name in EXPECTED_TOOLS
        if tool_name in used_tools
    ]

    matched_keyword_groups = sum(
        1
        for alternatives in KEYWORD_GROUPS
        if any(
            keyword.lower() in normalized_answer
            for keyword in alternatives
        )
    )

    tool_coverage_percent = (
        len(matched_tools)
        / len(EXPECTED_TOOLS)
        * 100
    )

    keyword_coverage_percent = (
        matched_keyword_groups
        / len(KEYWORD_GROUPS)
        * 100
    )

    completion_percent = (
        100.0
        if status == "completed"
        else 0.0
    )

    quality_score = (
        tool_coverage_percent * 0.5
        + keyword_coverage_percent * 0.3
        + completion_percent * 0.2
    )

    return {
        "quality_score_percent": round(quality_score, 2),
        "tool_coverage_percent": round(
            tool_coverage_percent,
            2,
        ),
        "keyword_coverage_percent": round(
            keyword_coverage_percent,
            2,
        ),
        "completion_percent": completion_percent,
        "matched_tools": matched_tools,
        "missing_tools": [
            tool_name
            for tool_name in EXPECTED_TOOLS
            if tool_name not in used_tools
        ],
        "matched_keyword_groups": matched_keyword_groups,
        "total_keyword_groups": len(KEYWORD_GROUPS),
    }


def run_react_comparison(
    request: str,
    runner: Callable[[str], dict[str, Any]] = run_react_agent,
) -> dict[str, Any]:
    """Вимірює ReAct-агента."""

    started_at = time.perf_counter()
    response = runner(request)
    elapsed_seconds = time.perf_counter() - started_at

    status = str(response.get("status", "error"))
    answer = str(response.get("answer", ""))
    used_tools = [
        str(tool)
        for tool in response.get("used_tools", [])
    ]
    safety = response.get("safety", {})

    return {
        "architecture": "ReAct",
        "status": status,
        "completed": status == "completed",
        "execution_time_seconds": round(
            elapsed_seconds,
            3,
        ),
        "reasoning_steps": safety.get("step_count"),
        "executed_plan_steps": None,
        "tool_calls": used_tools,
        "tool_calls_count": len(used_tools),
        "answer_length": len(answer),
        "answer": answer,
        "quality": calculate_quality(
            status,
            answer,
            used_tools,
        ),
        "raw_response": response,
    }


def run_plan_execute_comparison(
    request: str,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Вимірює Plan-and-Execute агента."""

    active_graph = graph or build_graph()

    started_at = time.perf_counter()

    response = active_graph.invoke(
        create_initial_state(request),
        config={
            "recursion_limit": 50,
        },
    )

    elapsed_seconds = time.perf_counter() - started_at

    status = str(response.get("status", "error"))
    answer = str(response.get("final_answer") or "")
    used_tools = [
        str(tool)
        for tool in response.get("used_tools", [])
    ]
    plan = [
        str(step)
        for step in response.get("plan", [])
    ]
    executed_steps = int(
        response.get("current_step", 0)
    )

    return {
        "architecture": "Plan-and-Execute",
        "status": status,
        "completed": bool(response.get("completed")),
        "execution_time_seconds": round(
            elapsed_seconds,
            3,
        ),
        "reasoning_steps": None,
        "planned_steps": len(plan),
        "executed_plan_steps": executed_steps,
        "replan_count": int(
            response.get("replan_count", 0)
        ),
        "tool_calls": used_tools,
        "tool_calls_count": len(used_tools),
        "answer_length": len(answer),
        "answer": answer,
        "quality": calculate_quality(
            status,
            answer,
            used_tools,
        ),
        "plan": plan,
        "raw_response": response,
    }


def build_comparison_summary(
    react_result: dict[str, Any],
    plan_result: dict[str, Any],
) -> dict[str, Any]:
    """Порівнює числові метрики двох запусків."""

    react_time = float(
        react_result["execution_time_seconds"]
    )
    plan_time = float(
        plan_result["execution_time_seconds"]
    )

    react_quality = float(
        react_result["quality"]["quality_score_percent"]
    )
    plan_quality = float(
        plan_result["quality"]["quality_score_percent"]
    )

    if react_time < plan_time:
        faster_architecture = "ReAct"
    elif plan_time < react_time:
        faster_architecture = "Plan-and-Execute"
    else:
        faster_architecture = "tie"

    if react_quality > plan_quality:
        higher_quality_architecture = "ReAct"
    elif plan_quality > react_quality:
        higher_quality_architecture = "Plan-and-Execute"
    else:
        higher_quality_architecture = "tie"

    latency_difference = abs(plan_time - react_time)

    conclusions = [
        (
            f"Швидша архітектура: {faster_architecture}; "
            f"різниця latency — "
            f"{round(latency_difference, 3)} секунди."
        ),
        (
            "Вища детермінована quality score: "
            f"{higher_quality_architecture}."
        ),
        (
            "ReAct є компактнішим для коротких запитів, "
            "а Plan-and-Execute явно фіксує план, прогрес "
            "та можливість replanning."
        ),
        (
            "Latency залежить від зовнішньої LLM, тому один "
            "запуск не є статистично репрезентативним benchmark."
        ),
    ]

    return {
        "faster_architecture": faster_architecture,
        "higher_quality_architecture": (
            higher_quality_architecture
        ),
        "latency_difference_seconds": round(
            latency_difference,
            3,
        ),
        "react_quality_score_percent": react_quality,
        "plan_execute_quality_score_percent": plan_quality,
        "react_tool_coverage_percent": (
            react_result["quality"][
                "tool_coverage_percent"
            ]
        ),
        "plan_execute_tool_coverage_percent": (
            plan_result["quality"][
                "tool_coverage_percent"
            ]
        ),
        "conclusions": conclusions,
    }


def run_comparison(
    request: str = COMPARISON_REQUEST,
) -> dict[str, Any]:
    """Запускає обидві архітектури на одному сценарії."""

    react_result = run_react_comparison(request)
    plan_result = run_plan_execute_comparison(request)

    report = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": MODEL_NAME,
        "scenario": {
            "input": request,
            "expected_tools": EXPECTED_TOOLS,
            "keyword_groups": KEYWORD_GROUPS,
        },
        "runs": [
            react_result,
            plan_result,
        ],
        "comparison": build_comparison_summary(
            react_result,
            plan_result,
        ),
    }

    COMPARISON_RESULTS_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


def main() -> None:
    """Запускає порівняння та зберігає JSON report."""

    report = run_comparison()

    print(
        json.dumps(
            {
                "model": report["model"],
                "react": {
                    "status": report["runs"][0]["status"],
                    "execution_time_seconds": (
                        report["runs"][0][
                            "execution_time_seconds"
                        ]
                    ),
                    "quality_score_percent": (
                        report["runs"][0]["quality"][
                            "quality_score_percent"
                        ]
                    ),
                    "tool_calls": report["runs"][0][
                        "tool_calls"
                    ],
                },
                "plan_execute": {
                    "status": report["runs"][1]["status"],
                    "execution_time_seconds": (
                        report["runs"][1][
                            "execution_time_seconds"
                        ]
                    ),
                    "quality_score_percent": (
                        report["runs"][1]["quality"][
                            "quality_score_percent"
                        ]
                    ),
                    "tool_calls": report["runs"][1][
                        "tool_calls"
                    ],
                },
                "comparison": report["comparison"],
                "saved_to": str(
                    COMPARISON_RESULTS_PATH
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()