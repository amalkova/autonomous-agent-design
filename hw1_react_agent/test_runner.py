"""End-to-end перевірка ReAct-агента на п'яти сценаріях."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import MODEL_NAME, run_agent


TEST_DELAY_SECONDS = 65

TEST_CASES = [
    {
        "test_id": "TC-001",
        "complexity": "simple",
        "name": "Статус існуючої ініціативи",
        "input": "Який статус ініціативи DEM-001?",
        "expected_tool": "get_initiative_status",
        "allowed_statuses": ["completed"],
        "keyword_groups": [["discovery"]],
    },
    {
        "test_id": "TC-002",
        "complexity": "medium",
        "name": "Перевірка неповного intake",
        "input": (
            "Перевір повноту intake DEM-004. "
            "business_owner=Retail Director, "
            "business_driver=Reduce onboarding time. "
            "Інші поля не надані."
        ),
        "expected_tool": "check_intake_completeness",
        "allowed_statuses": ["completed", "needs_input"],
        "keyword_groups": [
            ["40"],
            ["success_metrics", "метрик"],
        ],
    },
    {
        "test_id": "TC-003",
        "complexity": "complex",
        "name": "Класифікація Deep discovery",
        "input": (
            "Розрахуй Discovery Points для DEM-004: "
            "systems_count=4, ownership_clarity=partial, "
            "technical_uncertainty=high, dependency_count=3, "
            "regulatory_impact=possible, data_readiness=partial."
        ),
        "expected_tool": "classify_discovery_scope",
        "allowed_statuses": ["completed"],
        "keyword_groups": [
            ["8"],
            ["deep"],
        ],
    },
    {
        "test_id": "TC-004",
        "complexity": "complex",
        "name": "Розрахунок високого пріоритету",
        "input": (
            "Розрахуй пріоритет DEM-004 за підтвердженими "
            "оцінками: strategic_alignment=5, customer_impact=4, "
            "financial_impact=3, regulatory_urgency=5, "
            "implementation_feasibility=2."
        ),
        "expected_tool": "calculate_priority_score",
        "allowed_statuses": ["completed"],
        "keyword_groups": [
            ["4.05", "4,05"],
            ["high", "висок"],
        ],
    },
    {
        "test_id": "TC-005",
        "complexity": "medium",
        "name": "Відсутні параметри discovery",
        "input": (
            "Визнач Discovery scope для DEM-006. "
            "Відомо лише, що systems_count=2."
        ),
        "expected_tool": None,
        "allowed_statuses": ["needs_input"],
        "keyword_groups": [
            ["ownership", "влас"],
            ["technical", "техніч"],
        ],
    },
]


def evaluate_case(
    test_case: dict[str, Any],
    response: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Перевіряє tool calls, статус і зміст відповіді."""

    failures: list[str] = []
    used_tools = response.get("used_tools", [])
    expected_tool = test_case["expected_tool"]

    if expected_tool is None:
        if used_tools:
            failures.append(
                f"Очікувалось 0 tools, отримано: {used_tools}"
            )
    elif expected_tool not in used_tools:
        failures.append(
            f"Не викликано очікуваний tool: {expected_tool}"
        )

    actual_status = response.get("status")

    if actual_status not in test_case["allowed_statuses"]:
        failures.append(
            f"Неочікуваний status: {actual_status}"
        )

    answer = str(response.get("answer", "")).lower()

    for alternatives in test_case["keyword_groups"]:
        if not any(
            str(keyword).lower() in answer
            for keyword in alternatives
        ):
            failures.append(
                f"У відповіді немає жодного з: {alternatives}"
            )

    return not failures, failures


def build_test_result(
    test_case: dict[str, Any],
    response: dict[str, Any],
    passed: bool,
    failures: list[str],
) -> dict[str, Any]:
    """Формує повний запис результату відповідно до рубрики."""

    safety = response.get("safety", {})
    expected_tool = test_case["expected_tool"]

    expected_tool_calls = (
        [] if expected_tool is None else [expected_tool]
    )

    return {
        "test_id": test_case["test_id"],
        "name": test_case["name"],
        "complexity": test_case["complexity"],
        "input_query": test_case["input"],
        "expected_result": {
            "allowed_statuses": test_case["allowed_statuses"],
            "expected_tool_calls": expected_tool_calls,
            "required_keyword_groups": test_case["keyword_groups"],
        },
        "actual_result": response,
        "steps": safety.get("step_count"),
        "tool_calls": response.get("used_tools", []),
        "execution_time_seconds": safety.get("elapsed_seconds"),
        "passed": passed,
        "failures": failures,
    }


def main() -> None:
    """Запускає сценарії та зберігає test_results.json."""

    results: list[dict[str, Any]] = []

    for index, test_case in enumerate(TEST_CASES, start=1):
        print(
            f"[{index}/{len(TEST_CASES)}] "
            f"{test_case['test_id']} — {test_case['name']}"
        )

        response = run_agent(test_case["input"])
        passed, failures = evaluate_case(test_case, response)

        results.append(
            build_test_result(
                test_case=test_case,
                response=response,
                passed=passed,
                failures=failures,
            )
        )

        print("PASS" if passed else f"FAIL: {failures}")

        if index < len(TEST_CASES):
            print(
                f"Waiting {TEST_DELAY_SECONDS} seconds "
                "for Gemini quota reset..."
            )
            time.sleep(TEST_DELAY_SECONDS)

    passed_count = sum(
        result["passed"]
        for result in results
    )
    failed_count = len(results) - passed_count

    execution_times = [
        result["execution_time_seconds"]
        for result in results
        if result["execution_time_seconds"] is not None
    ]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "success_rate_percent": round(
                passed_count / len(results) * 100,
                2,
            ),
            "maximum_steps": max(
                result["steps"] or 0
                for result in results
            ),
            "maximum_execution_time_seconds": (
                max(execution_times)
                if execution_times
                else None
            ),
        },
        "test_cases": results,
    }

    Path("test_results.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nResult: {passed_count}/{len(results)} passed. "
        "Saved to test_results.json"
    )

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()