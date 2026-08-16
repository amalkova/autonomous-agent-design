"""ReAct-агент для requirements та estimation readiness."""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from knowledge import search_delivery_knowledge
from safety import (
    SafetyConfig,
    SafetyController,
    SafetyLimitExceeded,
)
from tools_legacy import (
    build_tool_response,
    check_requirements_readiness,
    classify_estimation_complexity,
    identify_handover_gaps,
)
from trajectory_logger import save_trajectory


load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"


REACT_TOOLS = [
    check_requirements_readiness,
    classify_estimation_complexity,
    identify_handover_gaps,
    search_delivery_knowledge,
]

REACT_TOOLS_BY_NAME = {
    tool.name: tool
    for tool in REACT_TOOLS
}


class ModelAnswer(BaseModel):
    """Структурована фінальна відповідь моделі."""

    status: Literal[
        "completed",
        "needs_input",
    ]
    answer: str = Field(
        min_length=1,
        description="Зрозуміла відповідь користувачу",
    )
    requires_human_confirmation: bool = Field(
        default=False,
        description=(
            "Чи потрібне підтвердження людини"
        ),
    )


class ReActResponse(BaseModel):
    """Публічний результат запуску ReAct-агента."""

    status: Literal[
        "completed",
        "needs_input",
        "safety_stop",
        "error",
    ]
    answer: str
    requires_human_confirmation: bool = False
    used_tools: list[str] = Field(
        default_factory=list,
    )
    safety: dict[str, Any] = Field(
        default_factory=dict,
    )


class ReActState(TypedDict, total=False):
    """Стан ReAct-графа."""

    messages: Annotated[
        list[AnyMessage],
        operator.add,
    ]
    final_response: dict[str, Any]


SYSTEM_PROMPT = """
Ти — Requirements & Estimation Readiness Assistant.

Твої завдання:
1. Перевіряти готовність requirements до estimation.
2. Розраховувати estimation complexity та Fibonacci points.
3. Виявляти handover gaps і blockers.
4. Шукати правила у delivery knowledge base.

Правила:
- Використовуй tool, якщо запит потребує перевірки,
  розрахунку або пошуку фактичних правил.
- Не вигадуй відсутні аргументи.
- Якщо обов'язкових даних недостатньо, конкретно переліч,
  які поля має надати користувач.
- Estimation complexity та points є рекомендацією і
  потребують підтвердження людини.
- Фінальна відправка estimation request є ризиковою дією.
  Вона виконується окремим Plan-and-Execute workflow з HITL.
- Не стверджуй, що request відправлено, якщо відповідний
  tool фактично не виконувався.
- Відповідай мовою користувача.
""".strip()


BASE_MODEL = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    timeout=120,
    max_retries=2,
)

MODEL_WITH_TOOLS = BASE_MODEL.bind_tools(
    REACT_TOOLS
)

STRUCTURED_MODEL = BASE_MODEL.with_structured_output(
    schema=ModelAnswer.model_json_schema(),
    method="json_schema",
)


def collect_used_tools(
    messages: list[AnyMessage],
) -> list[str]:
    """Збирає назви фактично викликаних tools."""

    used_tools: list[str] = []

    for message in messages:
        tool_calls = getattr(
            message,
            "tool_calls",
            [],
        )

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")

            if (
                tool_name
                and tool_name not in used_tools
            ):
                used_tools.append(tool_name)

    return used_tools


def normalize_structured_response(
    response: Any,
) -> dict[str, Any]:
    """Нормалізує dict або Pydantic response."""

    if isinstance(response, dict):
        return response

    if hasattr(response, "model_dump"):
        return response.model_dump()

    raise TypeError(
        "Structured model повернула "
        "непідтримуваний формат."
    )


def build_react_graph(
    safety: SafetyController,
    model_with_tools: Any = MODEL_WITH_TOOLS,
    structured_model: Any = STRUCTURED_MODEL,
):
    """Створює LangGraph ReAct-агента для одного запуску."""

    def agent_node(
        state: ReActState,
    ) -> dict[str, Any]:
        """LLM обирає tool або формує попередню відповідь."""

        safety.check_before_step()

        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                *state["messages"],
            ]
        )

        safety.check_timeout()

        return {
            "messages": [response],
        }

    def tool_node(
        state: ReActState,
    ) -> dict[str, Any]:
        """Виконує tool calls, обрані моделлю."""

        safety.check_before_step()

        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for tool_call in getattr(
            last_message,
            "tool_calls",
            [],
        ):
            tool_name = tool_call.get(
                "name",
                "",
            )
            arguments = tool_call.get(
                "args",
                {},
            )

            safety.register_tool_call(
                tool_name,
                arguments,
            )

            selected_tool = (
                REACT_TOOLS_BY_NAME.get(tool_name)
            )

            if selected_tool is None:
                observation = build_tool_response(
                    status="error",
                    error=(
                        f"Невідомий tool: {tool_name}"
                    ),
                )
            else:
                try:
                    observation = (
                        selected_tool.invoke(arguments)
                    )
                except Exception as error:
                    observation = build_tool_response(
                        status="error",
                        error=(
                            "Помилка виконання tool "
                            f"{tool_name}: {error}"
                        ),
                    )

            tool_messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call.get(
                        "id",
                        f"call-{len(tool_messages) + 1}",
                    ),
                    name=tool_name,
                )
            )

        safety.check_timeout()

        return {
            "messages": tool_messages,
        }

    def route_after_agent(
        state: ReActState,
    ) -> Literal[
        "tools",
        "finalize",
    ]:
        """Маршрутизує стан після LLM-вузла."""

        last_message = state["messages"][-1]

        if getattr(
            last_message,
            "tool_calls",
            [],
        ):
            return "tools"

        return "finalize"

    def finalize_node(
        state: ReActState,
    ) -> dict[str, Any]:
        """Формує структуровану фінальну відповідь."""

        safety.check_before_step()

        formatting_prompt = SystemMessage(
            content=(
                SYSTEM_PROMPT
                + "\n\nСформуй фінальну відповідь "
                "за наданою JSON-схемою. "
                "Якщо бракує даних, встанови "
                "status='needs_input'."
            )
        )

        structured_response = (
            structured_model.invoke(
                [
                    formatting_prompt,
                    *state["messages"],
                ]
            )
        )

        safety.check_timeout()

        return {
            "final_response": (
                normalize_structured_response(
                    structured_response
                )
            ),
        }

    builder = StateGraph(ReActState)

    builder.add_node(
        "agent",
        agent_node,
    )
    builder.add_node(
        "tools",
        tool_node,
    )
    builder.add_node(
        "finalize",
        finalize_node,
    )

    builder.add_edge(
        START,
        "agent",
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )
    builder.add_edge(
        "tools",
        "agent",
    )
    builder.add_edge(
        "finalize",
        END,
    )

    return builder.compile()


def log_run_safely(
    user_input: str,
    messages: list[AnyMessage],
    response: dict[str, Any],
    safety: SafetyController,
) -> None:
    """Логує trajectory, не ламаючи відповідь агента."""

    try:
        save_trajectory(
            agent_type="react",
            agent_name="solution_security_agent",
            user_input=user_input,
            messages=messages,
            final_response=response,
            safety=safety.snapshot(),
            metadata={
                "model": MODEL_NAME,
                "graph": (
                    "agent -> tools -> agent "
                    "-> finalize"
                ),
            },
        )
    except OSError:
        pass


def run_react_agent(
    user_input: str,
    safety_config: SafetyConfig | None = None,
    model_with_tools: Any = MODEL_WITH_TOOLS,
    structured_model: Any = STRUCTURED_MODEL,
    log_trajectory: bool = True,
) -> dict[str, Any]:
    """Запускає ReAct-агента та повертає JSON-сумісний результат."""

    normalized_input = user_input.strip()

    if not normalized_input:
        return ReActResponse(
            status="error",
            answer=(
                "Запит користувача не може бути порожнім."
            ),
            requires_human_confirmation=False,
        ).model_dump()

    safety = SafetyController(safety_config)
    graph = build_react_graph(
        safety=safety,
        model_with_tools=model_with_tools,
        structured_model=structured_model,
    )
    graph_result: ReActState = {
        "messages": [],
    }

    try:
        graph_result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=normalized_input
                    ),
                ],
            },
            config={
                "recursion_limit": (
                    safety.config.max_steps * 3 + 5
                ),
            },
        )

        final_response = graph_result.get(
            "final_response"
        )

        if final_response is None:
            raise RuntimeError(
                "ReAct-агент не сформував "
                "фінальну відповідь."
            )

        response = ReActResponse(
            **final_response,
            used_tools=collect_used_tools(
                graph_result["messages"]
            ),
            safety=safety.snapshot(),
        ).model_dump()

    except SafetyLimitExceeded as error:
        response = ReActResponse(
            status="safety_stop",
            answer=str(error),
            requires_human_confirmation=True,
            used_tools=collect_used_tools(
                graph_result.get(
                    "messages",
                    [],
                )
            ),
            safety=safety.snapshot(),
        ).model_dump()

    except Exception as error:
        response = ReActResponse(
            status="error",
            answer=(
                f"Помилка запуску ReAct-агента: {error}"
            ),
            requires_human_confirmation=True,
            used_tools=collect_used_tools(
                graph_result.get(
                    "messages",
                    [],
                )
            ),
            safety=safety.snapshot(),
        ).model_dump()

    if log_trajectory:
        log_run_safely(
            user_input=normalized_input,
            messages=graph_result.get(
                "messages",
                [],
            ),
            response=response,
            safety=safety,
        )

    return response


if __name__ == "__main__":
    print("Requirements Readiness ReAct Agent")
    print("Для завершення введіть: exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {
            "exit",
            "quit",
            "выход",
            "вихід",
        }:
            print("Agent stopped.")
            break

        if not user_input:
            continue

        result = run_react_agent(user_input)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )