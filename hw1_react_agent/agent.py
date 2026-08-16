"""ReAct-агент для супроводу demand та discovery ініціатив."""

import operator
from typing import Annotated, Any, Literal
from trajectory_logger import save_trajectory

from dotenv import load_dotenv
from langchain.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from safety import SafetyConfig, SafetyController, SafetyLimitExceeded
from tools import (
    calculate_priority_score,
    check_intake_completeness,
    classify_discovery_scope,
    get_initiative_status,
)


load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"

TOOLS = [
    get_initiative_status,
    check_intake_completeness,
    classify_discovery_scope,
    calculate_priority_score,
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


class ModelAnswer(BaseModel):
    """Структурована відповідь, яку формує модель."""

    status: Literal["completed", "needs_input"]
    answer: str = Field(description="Зрозуміла відповідь користувачу")
    requires_human_confirmation: bool = False


class AgentResponse(BaseModel):
    """Фінальний результат запуску ReAct-агента."""

    status: Literal["completed", "needs_input", "safety_stop", "error"]
    answer: str
    requires_human_confirmation: bool = False
    used_tools: list[str] = Field(default_factory=list)
    safety: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """Стан, який передається між вузлами LangGraph."""

    messages: Annotated[list[AnyMessage], operator.add]
    final_response: dict[str, Any]


SYSTEM_PROMPT = """
Ти — Demand & Discovery Assistant.

Твої завдання:
1. Перевіряти статус demand-ініціатив.
2. Перевіряти повноту intake.
3. Розраховувати Discovery Points і рекомендувати Light, Standard або Deep.
4. Розраховувати priority score лише за оцінками, наданими людиною.

Правила:
- Використовуй інструмент, коли запит потребує даних або розрахунку.
- Не вигадуй відсутні аргументи для інструментів.
- Якщо даних недостатньо, попроси користувача надати конкретні поля.
- Не визначай оцінки пріоритету самостійно.
- Discovery scope та priority score потребують підтвердження людиною.
- Відповідай мовою користувача.
""".strip()


BASE_MODEL = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    timeout=60,
    max_retries=2,
)

MODEL_WITH_TOOLS = BASE_MODEL.bind_tools(TOOLS)

STRUCTURED_MODEL = BASE_MODEL.with_structured_output(
    schema=ModelAnswer.model_json_schema(),
    method="json_schema",
)

def collect_used_tools(messages: list[AnyMessage]) -> list[str]:
    """Збирає назви фактично викликаних інструментів."""

    used_tools: list[str] = []

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            tool_name = tool_call["name"]

            if tool_name not in used_tools:
                used_tools.append(tool_name)

    return used_tools


def build_agent_graph(safety: SafetyController):
    """Створює окремий LangGraph для одного запуску агента."""

    def llm_node(state: AgentState) -> dict[str, Any]:
        """Модель обирає інструмент або формує попередню відповідь."""

        safety.check_before_step()

        response = MODEL_WITH_TOOLS.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )

        safety.check_timeout()
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict[str, Any]:
        """Виконує tool calls, обрані моделлю."""

        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            arguments = tool_call.get("args", {})

            safety.register_tool_call(tool_name, arguments)
            selected_tool = TOOLS_BY_NAME.get(tool_name)

            if selected_tool is None:
                observation = (
                    '{"status":"error","data":null,'
                    f'"error":"Невідомий інструмент: {tool_name}"'
                    "}"
                )
            else:
                try:
                    observation = selected_tool.invoke(arguments)
                except Exception as error:
                    observation = (
                        '{"status":"error","data":null,'
                        f'"error":"Помилка виконання: {str(error)}"'
                        "}"
                    )

            tool_messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        return {"messages": tool_messages}

    def route_after_llm(
        state: AgentState,
    ) -> Literal["tools", "finalize"]:
        """Визначає, чи треба виконувати інструмент."""

        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", []):
            return "tools"

        return "finalize"

    def finalize_node(state: AgentState) -> dict[str, Any]:
        """Перетворює фінальну відповідь на задану структуру."""

        safety.check_before_step()

        formatting_prompt = SystemMessage(
            content=(
                SYSTEM_PROMPT
                + "\n\nСформуй фінальну відповідь за заданою JSON-схемою. "
                "Якщо для виконання запиту бракує даних, встанови "
                "status='needs_input'."
            )
        )

        structured_response = STRUCTURED_MODEL.invoke(
            [
                formatting_prompt,
                *state["messages"],
            ]
        )

        safety.check_timeout()
        return {"final_response": structured_response}

    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )
    builder.add_edge("tools", "llm")
    builder.add_edge("finalize", END)

    return builder.compile()


def run_agent(
    user_input: str,
    safety_config: SafetyConfig | None = None,
) -> dict[str, Any]:
    """Запускає ReAct-агента та повертає структурований результат."""

    safety = SafetyController(safety_config)
    graph = build_agent_graph(safety)
    graph_result: AgentState = {"messages": []}

    try:
        graph_result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input),
                ]
            },
            config={
                "recursion_limit": safety.config.max_steps * 3 + 5,
            },
        )

        final_response = graph_result.get("final_response")

        if final_response is None:
            raise RuntimeError("Агент не сформував фінальну відповідь.")

        response = AgentResponse(
            **final_response,
            used_tools=collect_used_tools(graph_result["messages"]),
            safety=safety.snapshot(),
        )

        response_data = response.model_dump()

        save_trajectory(
            user_input=user_input,
            messages=graph_result["messages"],
            final_response=response_data,
        )

        return response_data

    except SafetyLimitExceeded as error:
        return AgentResponse(
            status="safety_stop",
            answer=str(error),
            requires_human_confirmation=True,
            used_tools=collect_used_tools(
                graph_result.get("messages", [])
            ),
            safety=safety.snapshot(),
        ).model_dump()

    except Exception as error:
        return AgentResponse(
            status="error",
            answer=f"Помилка запуску агента: {error}",
            requires_human_confirmation=True,
            used_tools=collect_used_tools(
                graph_result.get("messages", [])
            ),
            safety=safety.snapshot(),
        ).model_dump()

if __name__ == "__main__":
    import json

    print("Demand & Discovery Assistant")
    print("Для завершення введіть: exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit", "выход", "вихід"}:
            print("Agent stopped.")
            break

        if not user_input:
            continue

        result = run_agent(user_input)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )