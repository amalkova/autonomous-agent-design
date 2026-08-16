"""Мультиагентна система Requirements & Estimation на AG2 v1."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal

from anyio import Path

from ag2 import Agent
from ag2.config import GeminiConfig
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from mcp_client import call_mcp_tool


AgentName = Literal[
    "requirements_agent",
    "solution_security_agent",
    "estimation_agent",
]


class RouteDecision(BaseModel):
    """Рішення supervisor щодо маршрутизації."""

    action: AgentName
    reasoning: str = Field(min_length=1)


class AG2MASResult(BaseModel):
    """Публічний результат реалізації AG2."""

    framework: Literal["AG2"] = "AG2"
    selected_agent: AgentName
    route_reasoning: str
    final_answer: str
    handoff_count: int = 1


async def check_requirements_readiness(
    initiative_id: str,
    business_objective: str | None = None,
    functional_requirements: str | None = None,
    non_functional_requirements: str | None = None,
    acceptance_criteria: str | None = None,
    integration_scope: str | None = None,
    data_requirements: str | None = None,
) -> Any:
    """Перевірити готовність demand requirements до estimation."""

    return await call_mcp_tool(
        "check_requirements_readiness",
        {
            "initiative_id": initiative_id,
            "business_objective": business_objective,
            "functional_requirements": functional_requirements,
            "non_functional_requirements": non_functional_requirements,
            "acceptance_criteria": acceptance_criteria,
            "integration_scope": integration_scope,
            "data_requirements": data_requirements,
        },
    )


async def identify_handover_gaps(
    initiative_id: str,
    solution_scope_defined: bool,
    dependencies_confirmed: bool,
    nfr_reviewed: bool,
    acceptance_criteria_testable: bool,
    data_owners_confirmed: bool,
    security_classification_completed: bool,
    known_blockers: list[str],
) -> Any:
    """Виявити прогалини, що блокують estimation handover."""

    return await call_mcp_tool(
        "identify_handover_gaps",
        {
            "initiative_id": initiative_id,
            "solution_scope_defined": solution_scope_defined,
            "dependencies_confirmed": dependencies_confirmed,
            "nfr_reviewed": nfr_reviewed,
            "acceptance_criteria_testable": (
                acceptance_criteria_testable
            ),
            "data_owners_confirmed": data_owners_confirmed,
            "security_classification_completed": (
                security_classification_completed
            ),
            "known_blockers": known_blockers,
        },
    )


async def classify_estimation_complexity(
    initiative_id: str,
    systems_count: int,
    integration_count: int,
    nfr_criticality: Literal["low", "medium", "high"],
    data_migration_required: bool,
    security_review_required: bool,
    dependency_count: int,
    requirements_stability: Literal["high", "partial", "low"],
) -> Any:
    """Розрахувати складність demand і Fibonacci estimation points."""

    return await call_mcp_tool(
        "classify_estimation_complexity",
        {
            "initiative_id": initiative_id,
            "systems_count": systems_count,
            "integration_count": integration_count,
            "nfr_criticality": nfr_criticality,
            "data_migration_required": data_migration_required,
            "security_review_required": security_review_required,
            "dependency_count": dependency_count,
            "requirements_stability": requirements_stability,
        },
    )


def build_model_config() -> GeminiConfig:
    """Створити спільну конфігурацію Gemini."""

    load_dotenv(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".env",
    )
)

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to the local .env file."
        )

    return GeminiConfig(
        model=os.getenv(
            "MODEL_NAME",
            "gemini-3.1-flash-lite",
        ),
        api_key=api_key,
        temperature=0.0,
        streaming=False,
    )


def build_ag2_agents(
    config: GeminiConfig | None = None,
) -> dict[str, Agent]:
    """Побудувати supervisor і трьох доменних агентів."""

    model_config = config or build_model_config()

    supervisor = Agent(
        name="demand_supervisor",
        prompt=(
            "You coordinate a Requirements & Estimation multi-agent "
            "system. Select exactly one specialist.\n\n"
            "Route requirements completeness, readiness and handover "
            "gaps to requirements_agent.\n"
            "Route architecture, integrations, security, data ownership "
            "and NFR analysis to solution_security_agent.\n"
            "Route complexity, estimation points and effort assessment "
            "to estimation_agent.\n"
            "Return only the structured routing decision."
        ),
        config=model_config,
        response_schema=RouteDecision,
    )

    requirements_agent = Agent(
        name="requirements_agent",
        prompt=(
            "You are a Requirements Readiness specialist for demand "
            "management. Analyse completeness and estimation handover "
            "gaps. Use the available MCP-backed tools when sufficient "
            "structured arguments are present. Never invent missing "
            "requirements. Clearly state what requires human input."
        ),
        config=model_config,
        tools=[
            check_requirements_readiness,
            identify_handover_gaps,
        ],
    )

    solution_security_agent = Agent(
        name="solution_security_agent",
        prompt=(
            "You are a Solution and Security specialist. Analyse system "
            "scope, integrations, NFRs, data ownership, migration and "
            "security-review needs. Do not estimate effort and do not "
            "submit estimation requests. Identify assumptions and "
            "questions requiring human confirmation."
        ),
        config=model_config,
    )

    estimation_agent = Agent(
        name="estimation_agent",
        prompt=(
            "You are an Estimation specialist. Classify demand "
            "complexity and calculate Fibonacci estimation points using "
            "the available MCP-backed tool. Explain the important score "
            "drivers. You cannot submit an estimation request because "
            "submission requires a separate human-approval workflow."
        ),
        config=model_config,
        tools=[classify_estimation_complexity],
    )

    return {
        "demand_supervisor": supervisor,
        "requirements_agent": requirements_agent,
        "solution_security_agent": solution_security_agent,
        "estimation_agent": estimation_agent,
    }


async def extract_reply_text(reply: Any) -> str:
    """Отримати текст або structured content з AG2 AgentReply."""

    if reply.body:
        return str(reply.body)

    content = await reply.content()

    if isinstance(content, BaseModel):
        return content.model_dump_json(indent=2)

    if isinstance(content, str):
        return content

    return json.dumps(
        content,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


async def run_ag2_mas(
    user_request: str,
    agents: dict[str, Agent] | None = None,
) -> AG2MASResult:
    """Маршрутизувати запит і виконати один handoff до спеціаліста."""

    agent_registry = agents or build_ag2_agents()
    supervisor = agent_registry["demand_supervisor"]

    route_reply = await supervisor.ask(user_request)
    route_content = await route_reply.content()

    if isinstance(route_content, RouteDecision):
        decision = route_content
    else:
        decision = RouteDecision.model_validate(route_content)

    specialist = agent_registry[decision.action]

    specialist_reply = await specialist.ask(
        "User request:\n"
        f"{user_request}\n\n"
        "Supervisor routing reason:\n"
        f"{decision.reasoning}"
    )

    final_answer = await extract_reply_text(specialist_reply)

    return AG2MASResult(
        selected_agent=decision.action,
        route_reasoning=decision.reasoning,
        final_answer=final_answer,
    )


async def main() -> None:
    """Запустити демонстрацію AG2 MAS."""

    result = await run_ag2_mas(
        "For DEM-020, estimate a demand involving three systems, "
        "two integrations, high NFR criticality, security review, "
        "one dependency, no migration and partially stable requirements."
    )

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())