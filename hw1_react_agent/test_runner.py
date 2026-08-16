"""End-to-end перевірка ReAct-агента на п'яти сценаріях."""

import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import MODEL_NAME, run_agent


TEST_CASES = [
    {
        "name": "Existing initiative status",
        "input": "Какой статус у инициативы DEM-001?",
        "expected_tool": "get_initiative_status",
        "allowed_statuses": ["completed"],
        "keyword_groups": [["discovery"]],
    },
    {
        "name": "Incomplete intake",
        "input": (
            "Проверь полноту intake DEM-004. "
            "business_owner=Retail Director, "
            "business_driver=Reduce onboarding time. "
            "Остальные поля не предоставлены."
        ),
        "expected_tool": "check_intake_completeness",
        "allowed_statuses": ["completed", "needs_input"],
        "keyword_groups": [
            ["40"],
            ["success_metrics", "метрик"],
        ],
    },
    {
        "name": "Deep discovery classification",
        "input": (
            "Рассчитай Discovery Points для DEM-004: "
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
        "name": "High priority calculation",
        "input": (
            "Рассчитай приоритет DEM-004 по подтверждённым оценкам: "
            "strategic_alignment=5, customer_impact=4, "
            "financial_impact=3, regulatory_urgency=5, "
            "implementation_feasibility=2."
        ),
        "expected_tool": "calculate_priority_score",
        "allowed_statuses": ["completed"],
        "keyword_groups": [
            ["4.05", "4,05"],
            ["high", "высок", "висок"],
        ],
    },
    {
        "name": "Missing discovery inputs",
        "input": (
            "Определи Discovery scope для DEM-006. "
            "Известно только, что systems_count=2."
        ),
        "expected_tool": None,
        "allowed_statuses": ["needs_input"],
        "keyword_groups": [
            ["ownership", "влад", "влас"],
            ["technical", "технич", "техніч"],
        ],
    },
]


def evaluate_case(
    test_case: dict[str, Any],
    response: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Перевіряє інструмент, статус та зміст відповіді."""

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

    if response.get("status") not in test_case["allowed_statuses"]:
        failures.append(
            f"Неочікуваний status: {response.get('status')}"
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


def main() -> None:
    """Запускає сценарії та зберігає test_results.json."""

    results = []

    for index, test_case in enumerate(TEST_CASES, start=1):
        print(f"[{index}/{len(TEST_CASES)}] {test_case['name']}")

        response = run_agent(test_case["input"])
        passed, failures = evaluate_case(test_case, response)

        results.append(
            {
                "name": test_case["name"],
                "input": test_case["input"],
                "passed": passed,
                "failures": failures,
                "response": response,
            }
        )

        print("PASS" if passed else f"FAIL: {failures}")
        if index < len(TEST_CASES):
            print("Waiting 65 seconds for Gemini quota reset...")
            time.sleep(65)

    passed_count = sum(result["passed"] for result in results)
    failed_count = len(results) - passed_count

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
        },
        "test_cases": results,
    }

    Path("test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
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