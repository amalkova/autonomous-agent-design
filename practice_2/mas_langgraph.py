"""LangGraph MAS для Requirements & Estimation.

Архітектура:
    input guardrail
        -> demand supervisor
        -> requirements / solution-security / estimation agent
        -> output PII guardrail
        -> END
"""

from __future__ import annotations

import asyncio
import operator
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Literal,
    TypedDict,
)

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from guardrails import (
    TOOL_ALLOWLIST,
    inspect_input,
    redact_pii_text,
    requires_human_approval,
)
from mcp_client import (
    index_tools,
    load_all_mcp_tools,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


AgentName = Literal[
    "requirements_agent",
    "solution_security_agent",
    "estimation_agent",
]

RouteSelector = Callable[
    [str],
    Awaitable["RouteDecision"],
]

AgentRunner = Callable[
    [str],
    Awaitable[str],
]


class RouteDecision(BaseModel):
    """Структурований результат demand supervisor."""

    action: AgentName = Field(
        description=(
            "Agent, який має обробити demand request."
        )
    )
    reasoning: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "Коротке пояснення routing decision."
        ),
    )


class MASState(TypedDict, total=False):
    """Shared state мультиагентної системи."""

    user_request: str
    current_agent: AgentName
    route_reasoning: str
    agent_output: str
    final_answer: str
    blocked: bool
    block_reason: str
    completed: bool
    handoff_count: int
    trajectory: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


def log_event(
    agent_name: str,
    event: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Створити JSON-safe trajectory event."""

    return {
        "agent_name": agent_name,
        "event": event,
        "details": details,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def create_initial_state(
    user_request: str,
) -> MASState:
    """Підготувати початковий MAS state."""

    return {
        "user_request": user_request,
        "blocked": False,
        "completed": False,
        "handoff_count": 0,
        "trajectory": [],
    }


def extract_message_text(message: Any) -> str:
    """Отримати plain text з останнього agent message."""

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts)

    return str(content)


def build_mas_graph(
    route_selector: RouteSelector,
    agent_runners: dict[AgentName, AgentRunner],
    checkpointer: Any | None = None,
) -> Any:
    """Побудувати supervisor MAS graph."""

    required_agents: set[AgentName] = {
        "requirements_agent",
        "solution_security_agent",
        "estimation_agent",
    }
    missing_agents = (
        required_agents - set(agent_runners)
    )

    if missing_agents:
        raise ValueError(
            "Missing agent runners: "
            + ", ".join(sorted(missing_agents))
        )

    def input_guardrail_node(
        state: MASState,
    ) -> dict[str, Any]:
        """Перевірити direct prompt injection."""

        inspection = inspect_input(
            state["user_request"],
            source="user",
        )

        if not inspection.allowed:
            return {
                "blocked": True,
                "block_reason": inspection.reason,
                "agent_output": (
                    "Запит заблоковано input guardrail."
                ),
                "trajectory": [
                    log_event(
                        "input_guardrail",
                        "input_blocked",
                        {
                            "category": (
                                inspection.category
                            ),
                            "matched_rule": (
                                inspection.matched_rule
                            ),
                        },
                    )
                ],
            }

        return {
            "blocked": False,
            "trajectory": [
                log_event(
                    "input_guardrail",
                    "input_allowed",
                    {
                        "category": inspection.category,
                    },
                )
            ],
        }

    async def supervisor_node(
        state: MASState,
    ) -> dict[str, Any]:
        """Вибрати спеціалізованого agent."""

        decision = await route_selector(
            state["user_request"]
        )

        return {
            "current_agent": decision.action,
            "route_reasoning": (
                decision.reasoning
            ),
            "handoff_count": (
                state.get("handoff_count", 0) + 1
            ),
            "trajectory": [
                log_event(
                    "demand_supervisor",
                    "handoff",
                    {
                        "target_agent": (
                            decision.action
                        ),
                        "reasoning": (
                            decision.reasoning
                        ),
                    },
                )
            ],
        }

    async def run_specialist(
        state: MASState,
        agent_name: AgentName,
    ) -> dict[str, Any]:
        """Запустити specialist agent."""

        runner = agent_runners[agent_name]
        output = await runner(
            state["user_request"]
        )

        return {
            "agent_output": output,
            "trajectory": [
                log_event(
                    agent_name,
                    "agent_completed",
                    {
                        "output_length": len(output),
                    },
                )
            ],
        }

    async def requirements_node(
        state: MASState,
    ) -> dict[str, Any]:
        return await run_specialist(
            state,
            "requirements_agent",
        )

    async def solution_security_node(
        state: MASState,
    ) -> dict[str, Any]:
        return await run_specialist(
            state,
            "solution_security_agent",
        )

    async def estimation_node(
        state: MASState,
    ) -> dict[str, Any]:
        return await run_specialist(
            state,
            "estimation_agent",
        )

    def output_guardrail_node(
        state: MASState,
    ) -> dict[str, Any]:
        """Застосувати PII redaction до final output."""

        raw_output = state.get(
            "agent_output",
            "Не вдалося сформувати відповідь.",
        )
        redacted_output = redact_pii_text(
            raw_output
        )

        return {
            "final_answer": redacted_output,
            "completed": True,
            "trajectory": [
                log_event(
                    "output_guardrail",
                    "output_processed",
                    {
                        "pii_redacted": (
                            raw_output != redacted_output
                        ),
                    },
                )
            ],
        }

    def route_after_input(
        state: MASState,
    ) -> Literal[
        "supervisor",
        "output_guardrail",
    ]:
        if state.get("blocked"):
            return "output_guardrail"

        return "supervisor"

    def route_to_specialist(
        state: MASState,
    ) -> AgentName:
        return state["current_agent"]

    builder = StateGraph(MASState)

    builder.add_node(
        "input_guardrail",
        input_guardrail_node,
    )
    builder.add_node(
        "supervisor",
        supervisor_node,
    )
    builder.add_node(
        "requirements_agent",
        requirements_node,
    )
    builder.add_node(
        "solution_security_agent",
        solution_security_node,
    )
    builder.add_node(
        "estimation_agent",
        estimation_node,
    )
    builder.add_node(
        "output_guardrail",
        output_guardrail_node,
    )

    builder.add_edge(
        START,
        "input_guardrail",
    )
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_input,
    )
    builder.add_conditional_edges(
        "supervisor",
        route_to_specialist,
    )
    builder.add_edge(
        "requirements_agent",
        "output_guardrail",
    )
    builder.add_edge(
        "solution_security_agent",
        "output_guardrail",
    )
    builder.add_edge(
        "estimation_agent",
        "output_guardrail",
    )
    builder.add_edge(
        "output_guardrail",
        END,
    )

    return builder.compile(
        checkpointer=(
            checkpointer
            if checkpointer is not None
            else InMemorySaver()
        )
    )


def build_model() -> BaseChatModel:
    """Створити Gemini model для production demo."""

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured."
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv(
            "MODEL_NAME",
            "gemini-3.1-flash-lite",
        ),
        temperature=0,
        max_retries=2,
    )


def make_agent_runner(
    agent: Any,
) -> AgentRunner:
    """Огорнути LangChain agent як MAS runner."""

    async def runner(
        user_request: str,
    ) -> str:
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_request,
                    }
                ]
            },
            config={
                "recursion_limit": 12,
            },
        )

        messages = result.get(
            "messages",
            [],
        )

        if not messages:
            return (
                "Agent не повернув повідомлення."
            )

        return extract_message_text(
            messages[-1]
        )

    return runner


async def build_production_mas(
    model: BaseChatModel | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Побудувати MAS з Gemini та MCP tools."""

    llm = model or build_model()
    structured_router = (
        llm.with_structured_output(
            RouteDecision
        )
    )

    all_mcp_tools = await load_all_mcp_tools()
    tool_registry = index_tools(
        all_mcp_tools
    )

    requirements_tools = [
        tool_registry[name]
        for name in TOOL_ALLOWLIST[
            "requirements_agent"
        ]
    ]

    estimation_tools = [
        tool_registry[name]
        for name in TOOL_ALLOWLIST[
            "estimation_agent"
        ]
        if not requires_human_approval(name)
    ]

    requirements_agent = create_agent(
        model=llm,
        tools=requirements_tools,
        system_prompt=(
            "Ти Requirements Agent Demand платформи. "
            "Перевіряй completeness, readiness, acceptance "
            "criteria та handover gaps. Використовуй MCP "
            "tools, коли в запиті достатньо аргументів. "
            "Не вигадуй відсутні дані."
        ),
    )

    solution_security_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=(
            "Ти Solution & Security Agent. Аналізуй "
            "integration scope, NFR, data ownership, "
            "security classification, dependencies і "
            "solution risks. Не роби estimation і не "
            "відправляй initiative назовні."
        ),
    )

    estimation_agent = create_agent(
        model=llm,
        tools=estimation_tools,
        system_prompt=(
            "Ти Estimation Agent. Класифікуй complexity "
            "та Fibonacci points через MCP tool. "
            "Не викликай submit_estimation_request: "
            "цей risky tool виконується лише окремим "
            "HITL approval workflow."
        ),
    )

    async def production_route_selector(
        user_request: str,
    ) -> RouteDecision:
        decision = await structured_router.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Ти Demand Supervisor. "
                        "Маршрутизуй запит:\n"
                        "- requirements_agent: requirements, "
                        "readiness, gaps, acceptance criteria;\n"
                        "- solution_security_agent: solution, "
                        "integrations, NFR, data, security;\n"
                        "- estimation_agent: complexity, "
                        "points, estimation.\n"
                        "Обери рівно одного agent."
                    ),
                },
                {
                    "role": "user",
                    "content": user_request,
                },
            ]
        )

        if isinstance(
            decision,
            RouteDecision,
        ):
            return decision

        return RouteDecision.model_validate(
            decision
        )

    runners: dict[AgentName, AgentRunner] = {
        "requirements_agent": make_agent_runner(
            requirements_agent
        ),
        "solution_security_agent": make_agent_runner(
            solution_security_agent
        ),
        "estimation_agent": make_agent_runner(
            estimation_agent
        ),
    }

    return build_mas_graph(
        route_selector=production_route_selector,
        agent_runners=runners,
        checkpointer=checkpointer,
    )


async def run_mas_query(
    graph: Any,
    user_request: str,
    thread_id: str,
) -> MASState:
    """Запустити MAS query з thread persistence."""

    return await graph.ainvoke(
        create_initial_state(user_request),
        config={
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": 20,
        },
    )


async def main() -> None:
    """Запустити три routing demo cases."""

    graph = await build_production_mas()

    demo_cases = [
        (
            "mas-requirements",
            "Перевір готовність requirements DEM-101. "
            "Business objective визначено, але бракує "
            "acceptance criteria та NFR.",
        ),
        (
            "mas-security",
            "Проаналізуй integration, data та security "
            "risks для initiative DEM-102.",
        ),
        (
            "mas-estimation",
            "Оціни complexity DEM-103: 3 systems, "
            "2 integrations, high NFR criticality, "
            "data migration required, security review "
            "required, 2 dependencies, partial stability.",
        ),
    ]

    for thread_id, request in demo_cases:
        result = await run_mas_query(
            graph,
            request,
            thread_id,
        )

        print(
            f"\n[{thread_id}] "
            f"agent={result.get('current_agent')}"
        )
        print(result["final_answer"])


if __name__ == "__main__":
    asyncio.run(main())