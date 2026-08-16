"""FastMCP-сервер для Requirements & Estimation MAS.

Сервер перевикористовує доменні tools із Практичного завдання №1
та публікує їх через Model Context Protocol.

Запуск:
    python mcp_server.py
"""

from __future__ import annotations

import json
from typing import Any, Literal

from fastmcp import FastMCP

from tools_legacy import (
    check_requirements_readiness as readiness_tool,
    classify_estimation_complexity as complexity_tool,
    identify_handover_gaps as handover_tool,
    submit_estimation_request as submission_tool,
)


mcp = FastMCP(
    name="requirements_estimation",
    instructions=(
        "MCP-сервер Demand & Discovery платформи. "
        "Надає інструменти для перевірки requirements, "
        "класифікації estimation complexity, аналізу "
        "handover gaps та передачі запиту на estimation. "
        "submit_estimation_request є ризиковим tool і може "
        "викликатися лише після Human-in-the-Loop approval."
    ),
)


def invoke_domain_tool(
    tool: Any,
    arguments: dict[str, Any],
) -> str:
    """Безпечно виконати LangChain tool та повернути JSON.

    Pydantic-схема відповідного domain tool повторно перевіряє
    аргументи після MCP-валідації.
    """

    try:
        result = tool.invoke(arguments)

    except Exception as exception:
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exception).__name__,
                "message": str(exception),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    if isinstance(result, str):
        return result

    return json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


@mcp.tool
def check_requirements_readiness(
    initiative_id: str,
    business_objective: str | None = None,
    functional_requirements: str | None = None,
    non_functional_requirements: str | None = None,
    acceptance_criteria: str | None = None,
    integration_scope: str | None = None,
    data_requirements: str | None = None,
) -> str:
    """Перевірити готовність requirements до estimation handover.

    Args:
        initiative_id: ID ініціативи у форматі DEM-XXX.
        business_objective: Бізнес-мета та очікувана цінність.
        functional_requirements: Функціональні вимоги.
        non_functional_requirements: NFR та quality attributes.
        acceptance_criteria: Тестовані критерії приймання.
        integration_scope: Системи та інтеграції у scope.
        data_requirements: Джерела, обсяг і правила обробки даних.

    Returns:
        JSON з readiness percentage, missing fields і рекомендаціями.
    """

    return invoke_domain_tool(
        readiness_tool,
        {
            "initiative_id": initiative_id,
            "business_objective": business_objective,
            "functional_requirements": (
                functional_requirements
            ),
            "non_functional_requirements": (
                non_functional_requirements
            ),
            "acceptance_criteria": acceptance_criteria,
            "integration_scope": integration_scope,
            "data_requirements": data_requirements,
        },
    )


@mcp.tool
def classify_estimation_complexity(
    initiative_id: str,
    systems_count: int,
    integration_count: int,
    nfr_criticality: Literal[
        "low",
        "medium",
        "high",
    ],
    data_migration_required: bool,
    security_review_required: bool,
    dependency_count: int,
    requirements_stability: Literal[
        "high",
        "partial",
        "low",
    ],
) -> str:
    """Класифікувати estimation complexity та Fibonacci points.

    Args:
        initiative_id: Унікальний ID ініціативи.
        systems_count: Кількість систем у solution scope.
        integration_count: Кількість інтеграцій.
        nfr_criticality: Критичність NFR.
        data_migration_required: Чи потрібна міграція даних.
        security_review_required: Чи потрібен security review.
        dependency_count: Кількість зовнішніх залежностей.
        requirements_stability: Стабільність requirements.

    Returns:
        JSON з рівнем Low/Medium/High, points і поясненням.
    """

    return invoke_domain_tool(
        complexity_tool,
        {
            "initiative_id": initiative_id,
            "systems_count": systems_count,
            "integration_count": integration_count,
            "nfr_criticality": nfr_criticality,
            "data_migration_required": (
                data_migration_required
            ),
            "security_review_required": (
                security_review_required
            ),
            "dependency_count": dependency_count,
            "requirements_stability": (
                requirements_stability
            ),
        },
    )


@mcp.tool
def identify_handover_gaps(
    initiative_id: str,
    solution_scope_defined: bool,
    dependencies_confirmed: bool,
    nfr_reviewed: bool,
    acceptance_criteria_testable: bool,
    data_owners_confirmed: bool,
    security_classification_completed: bool,
    known_blockers: list[str] | None = None,
) -> str:
    """Знайти прогалини, що блокують estimation handover.

    Args:
        initiative_id: Унікальний ID ініціативи.
        solution_scope_defined: Чи визначено solution scope.
        dependencies_confirmed: Чи підтверджено dependencies.
        nfr_reviewed: Чи пройшли NFR review.
        acceptance_criteria_testable: Чи є критерії тестованими.
        data_owners_confirmed: Чи підтверджено data owners.
        security_classification_completed: Чи виконано classification.
        known_blockers: Відомі блокери або unresolved questions.

    Returns:
        JSON зі списком gaps, severity та readiness decision.
    """

    return invoke_domain_tool(
        handover_tool,
        {
            "initiative_id": initiative_id,
            "solution_scope_defined": (
                solution_scope_defined
            ),
            "dependencies_confirmed": (
                dependencies_confirmed
            ),
            "nfr_reviewed": nfr_reviewed,
            "acceptance_criteria_testable": (
                acceptance_criteria_testable
            ),
            "data_owners_confirmed": (
                data_owners_confirmed
            ),
            "security_classification_completed": (
                security_classification_completed
            ),
            "known_blockers": known_blockers or [],
        },
    )


@mcp.tool
def submit_estimation_request(
    initiative_id: str,
    estimation_complexity: Literal[
        "Low",
        "Medium",
        "High",
    ],
    estimation_points: int,
    target_team: str,
    requested_by: str,
    estimation_summary: str,
) -> str:
    """Передати запит на estimation після HITL approval.

    Увага:
        Це ризиковий tool із side effect. Supervisor та інші
        агенти не повинні викликати його напряму. Виклик дозволено
        лише estimation_agent після LangGraph interrupt approval.

    Args:
        initiative_id: Унікальний ID ініціативи.
        estimation_complexity: Рівень Low, Medium або High.
        estimation_points: Узгоджені Fibonacci points.
        target_team: Команда, що має виконати estimation.
        requested_by: Ідентифікатор requester.
        estimation_summary: Підсумок scope і assumptions.

    Returns:
        JSON зі статусом submission або поясненням помилки.
    """

    return invoke_domain_tool(
        submission_tool,
        {
            "initiative_id": initiative_id,
            "estimation_complexity": (
                estimation_complexity
            ),
            "estimation_points": estimation_points,
            "target_team": target_team,
            "requested_by": requested_by,
            "estimation_summary": estimation_summary,
        },
    )


@mcp.resource(
    "demand://standards/readiness",
    name="requirements_readiness_standard",
    description=(
        "Definition of Ready для передачі demand "
        "на estimation."
    ),
    mime_type="application/json",
)
def get_requirements_readiness_standard() -> str:
    """Повернути read-only стандарт готовності requirements.

    Resource надає агентам і MCP-клієнтам перелік
    обов'язкових полів та правило readiness без виконання
    side effects.
    """

    return json.dumps(
        {
            "standard": (
                "Requirements Estimation "
                "Definition of Ready"
            ),
            "required_fields": [
                "business_objective",
                "functional_requirements",
                "non_functional_requirements",
                "acceptance_criteria",
                "integration_scope",
                "data_requirements",
            ],
            "ready_when": (
                "Усі шість полів заповнені "
                "змістовними значеннями."
            ),
            "incomplete_action": (
                "Повернути demand requester на "
                "уточнення відсутніх полів."
            ),
            "side_effects": False,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


@mcp.resource(
    "demand://standards/complexity",
    name="estimation_complexity_standard",
    description=(
        "Scoring model для complexity та "
        "Fibonacci estimation points."
    ),
    mime_type="application/json",
)
def get_estimation_complexity_standard() -> str:
    """Повернути read-only правила complexity scoring.

    Resource документує фактори та межі, які застосовує
    classify_estimation_complexity.
    """

    return json.dumps(
        {
            "factors": [
                "systems_count",
                "integration_count",
                "nfr_criticality",
                "data_migration_required",
                "security_review_required",
                "dependency_count",
                "requirements_stability",
            ],
            "raw_score_to_points": {
                "0-1": 1,
                "2-3": 2,
                "4-5": 3,
                "6-7": 5,
                "8-9": 8,
                "10+": 13,
            },
            "points_to_complexity": {
                "1-2": "Low",
                "3-5": "Medium",
                "8-13": "High",
            },
            "human_confirmation_required": True,
            "side_effects": False,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


@mcp.prompt(
    name="prepare_estimation_handover",
    description=(
        "Підготувати структурований prompt для "
        "requirements-to-estimation handover."
    ),
)
def prepare_estimation_handover_prompt(
    initiative_id: str,
    business_objective: str,
    known_gaps: str = "Не вказано",
) -> str:
    """Сформувати reusable prompt для estimation handover.

    Args:
        initiative_id: ID demand у форматі DEM-XXX.
        business_objective: Бізнес-мета ініціативи.
        known_gaps: Відомі прогалини або unresolved questions.

    Returns:
        Інструкція для агента з обов'язковою перевіркою
        Resource, MCP tools та HITL перед risky submit.
    """

    return (
        "Підготуй demand до estimation handover.\n\n"
        f"Initiative ID: {initiative_id.strip()}\n"
        f"Business objective: {business_objective.strip()}\n"
        f"Known gaps: {known_gaps.strip()}\n\n"
        "Послідовність:\n"
        "1. Прочитай Resource "
        "demand://standards/readiness.\n"
        "2. Перевір requirements readiness через "
        "check_requirements_readiness.\n"
        "3. Перевір handover gaps через "
        "identify_handover_gaps.\n"
        "4. Розрахуй complexity через "
        "classify_estimation_complexity.\n"
        "5. Не викликай submit_estimation_request "
        "без окремого Human-in-the-Loop approval.\n"
        "6. Не вигадуй відсутні значення."
    )


if __name__ == "__main__":
    mcp.run()