"""Plan-and-Execute агент для Demand and Discovery."""

import json
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from dotenv import load_dotenv
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import TypedDict

from hitl import request_tool_approval
from knowledge import search_knowledge
from tools import (
    check_intake_completeness,
    classify_discovery_scope,
    submit_discovery_assessment,
)


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
AGENT_STATE_DB = BASE_DIR / "agent_state.db"

MODEL_NAME = "gemini-3.1-flash-lite"

RISKY_TOOLS = {
    "submit_discovery_assessment",
}

TOOLS = [
    check_intake_completeness,
    classify_discovery_scope,
    search_knowledge,
    submit_discovery_assessment,
]

TOOLS_BY_NAME = {
    tool.name: tool
    for tool in TOOLS
}


class Plan(BaseModel):
    """Структурований план виконання задачі."""

    goal: str = Field(
        min_length=5,
        max_length=500,
        description="Головна ціль запиту користувача",
    )
    steps: list[str] = Field(
        min_length=1,
        max_length=6,
        description=(
            "Послідовний список із 2-6 виконуваних кроків. "
            "Кожен крок починається з назви очікуваного tool."
        ),
    )

    @field_validator("goal", mode="before")
    @classmethod
    def normalize_goal(cls, value: object) -> str:
        """Перевіряє та нормалізує ціль."""

        if not isinstance(value, str):
            raise ValueError(
                "goal має бути текстовим значенням."
            )

        normalized_goal = value.strip()

        if not normalized_goal:
            raise ValueError("goal не може бути порожнім.")

        return normalized_goal

    @field_validator("steps", mode="after")
    @classmethod
    def normalize_steps(
        cls,
        steps: list[str],
    ) -> list[str]:
        """Очищує кроки та забороняє дублікати."""

        normalized_steps = [
            step.strip()
            for step in steps
            if step.strip()
        ]

        if not normalized_steps:
            raise ValueError(
                "План має містити хоча б один крок."
            )

        if len(normalized_steps) != len(set(normalized_steps)):
            raise ValueError(
                "План не повинен містити однакові кроки."
            )

        return normalized_steps


class ReplanDecision(BaseModel):
    """Структуроване рішення replanner."""

    action: Literal[
        "continue",
        "replan",
        "finish",
    ] = Field(
        description=(
            "continue=виконати наступний крок, "
            "replan=замінити невиконані кроки, "
            "finish=завершити виконання"
        )
    )
    updated_steps: list[str] | None = Field(
        default=None,
        description=(
            "Нові невиконані кроки для action=replan"
        ),
    )
    reasoning: str = Field(
        min_length=3,
        max_length=1000,
        description="Пояснення рішення replanner",
    )
    final_answer: str | None = Field(
        default=None,
        description=(
            "Фінальна відповідь користувачу для action=finish"
        ),
    )

    @model_validator(mode="after")
    def validate_replan_decision(self) -> Self:
        """Перевіряє узгодженість action та updated_steps."""

        if (
            self.action == "replan"
            and not self.updated_steps
        ):
            raise ValueError(
                "Для action=replan потрібні updated_steps."
            )

        if (
            self.action != "replan"
            and self.updated_steps
        ):
            raise ValueError(
                "updated_steps дозволені лише для action=replan."
            )

        return self


class PlanExecuteState(TypedDict, total=False):
    """Стан Plan-and-Execute графа."""

    messages: Annotated[
        list[AnyMessage],
        operator.add,
    ]
    user_request: str
    goal: str
    plan: list[str]
    current_step: int
    results: Annotated[
        list[str],
        operator.add,
    ]
    completed: bool
    final_answer: str | None
    status: str
    used_tools: Annotated[
        list[str],
        operator.add,
    ]
    replan_count: int


PLANNER_SYSTEM_PROMPT = """
Ти — planner для Demand and Discovery агента.

Створи повний послідовний план ДО початку виконання.

Доступні tools:
1. check_intake_completeness — перевірка Gate 0 intake.
2. classify_discovery_scope — розрахунок Discovery Points і scope.
3. search_knowledge — пошук правил, timebox та політик у ChromaDB.
4. submit_discovery_assessment — фінальна ризикова відправка assessment.

Правила:
- План має містити від 2 до 6 конкретних кроків.
- Один крок має передбачати рівно один tool.
- Кожен крок починай із точної назви tool та символу ":".
- Не додавай submit_discovery_assessment, якщо користувач явно
  не попросив фінально відправити або зафіксувати assessment.
- Не вигадуй відсутні бізнес-параметри.
- Кроки мають іти у логічному порядку.
""".strip()


EXECUTOR_SYSTEM_PROMPT = """
Ти — executor Plan-and-Execute агента.

Виконай рівно один поточний крок плану.

Правила:
- Якщо крок починається з назви tool, виклич саме цей tool.
- Не виконуй наступні кроки наперед.
- Не вигадуй аргументи, яких немає у запиті або результатах.
- search_knowledge використовуй лише для довідкових правил.
- submit_discovery_assessment є ризиковою дією і буде
  додатково зупинена Human-in-the-Loop механізмом.
""".strip()


REPLANNER_SYSTEM_PROMPT = """
Ти — replanner Plan-and-Execute агента.

Після кожного виконаного кроку оціни прогрес.

Обери:
- continue — якщо наступний запланований крок залишається актуальним;
- replan — якщо невиконані кроки потрібно змінити;
- finish — якщо ціль досягнута, потрібні додаткові дані від людини
  або ризикова дія була відхилена.

Правила:
- Не повторюй уже успішно виконані кроки.
- Для replan поверни лише нові НЕВИКОНАНІ кроки.
- Для finish сформуй зрозумілу final_answer.
- Якщо ще залишилися коректні кроки, обирай continue.
""".strip()


BASE_MODEL = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    timeout=60,
    max_retries=2,
)

PLANNER_MODEL = BASE_MODEL.with_structured_output(
    schema=Plan.model_json_schema(),
    method="json_schema",
)

EXECUTOR_MODEL = BASE_MODEL.bind_tools(TOOLS)

REPLANNER_MODEL = BASE_MODEL.with_structured_output(
    schema=ReplanDecision.model_json_schema(),
    method="json_schema",
)


def planner_node_factory(
    planner_model: Any,
):
    """Створює planner node із переданою моделлю."""

    def planner_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Генерує повний структурований план."""

        raw_plan = planner_model.invoke(
            [
                SystemMessage(
                    content=PLANNER_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=state["user_request"]
                ),
            ]
        )

        plan = Plan.model_validate(raw_plan)

        return {
            "goal": plan.goal,
            "plan": plan.steps,
            "current_step": 0,
            "results": [],
            "completed": False,
            "final_answer": None,
            "status": "running",
            "used_tools": [],
            "replan_count": 0,
            "messages": [
                AIMessage(
                    content=json.dumps(
                        {
                            "goal": plan.goal,
                            "steps": plan.steps,
                        },
                        ensure_ascii=False,
                    )
                )
            ],
        }

    return planner_node


def executor_node_factory(
    executor_model: Any,
):
    """Створює executor node із переданою моделлю."""

    def executor_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Виконує один поточний крок через tool."""

        step_index = state["current_step"]
        plan = state["plan"]

        if step_index >= len(plan):
            return {
                "completed": True,
                "status": "completed",
                "final_answer": (
                    "Усі заплановані кроки виконано."
                ),
            }

        current_step = plan[step_index]
        previous_results = (
            "\n".join(state.get("results", []))
            or "Попередніх результатів немає."
        )

        response = executor_model.invoke(
            [
                SystemMessage(
                    content=EXECUTOR_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=(
                        f"Початковий запит:\n"
                        f"{state['user_request']}\n\n"
                        f"Поточний крок {step_index + 1}:\n"
                        f"{current_step}\n\n"
                        f"Попередні результати:\n"
                        f"{previous_results}"
                    )
                ),
            ]
        )

        tool_calls = getattr(
            response,
            "tool_calls",
            [],
        )

        if not tool_calls:
            text_result = str(response.content)

            return {
                "current_step": step_index + 1,
                "results": [
                    (
                        f"Крок {step_index + 1} "
                        f"({current_step}): {text_result}"
                    )
                ],
                "messages": [response],
            }

        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        arguments = tool_call.get("args", {})
        selected_tool = TOOLS_BY_NAME.get(tool_name)

        if selected_tool is None:
            observation = json.dumps(
                {
                    "status": "error",
                    "data": None,
                    "error": (
                        f"Невідомий tool: {tool_name}"
                    ),
                },
                ensure_ascii=False,
            )
            tool_executed = False

        elif tool_name in RISKY_TOOLS:
            resolution = request_tool_approval(
                tool_name=tool_name,
                arguments=arguments,
            )

            if not resolution.approved:
                observation = json.dumps(
                    {
                        "status": "rejected",
                        "data": {
                            "tool": tool_name,
                            "decision": resolution.decision,
                        },
                        "error": resolution.reason,
                    },
                    ensure_ascii=False,
                )
                tool_executed = False
            else:
                try:
                    observation = selected_tool.invoke(
                        resolution.arguments
                    )
                    tool_executed = True
                except Exception as error:
                    observation = json.dumps(
                        {
                            "status": "error",
                            "data": None,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    tool_executed = True

        else:
            try:
                observation = selected_tool.invoke(
                    arguments
                )
                tool_executed = True
            except Exception as error:
                observation = json.dumps(
                    {
                        "status": "error",
                        "data": None,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
                tool_executed = True

        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_name,
        )

        state_update: dict[str, Any] = {
            "current_step": step_index + 1,
            "results": [
                (
                    f"Крок {step_index + 1} "
                    f"({current_step}): "
                    f"{tool_name}: {observation}"
                )
            ],
            "messages": [
                response,
                tool_message,
            ],
        }

        if tool_executed:
            state_update["used_tools"] = [tool_name]

        return state_update

    return executor_node


def replanner_node_factory(
    replanner_model: Any,
):
    """Створює replanner node із переданою моделлю."""

    def replanner_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Продовжує, змінює або завершує план."""

        if state.get("completed"):
            return {}

        step_index = state["current_step"]
        plan = state["plan"]
        results = state.get("results", [])
        remaining_steps = plan[step_index:]

        raw_decision = replanner_model.invoke(
            [
                SystemMessage(
                    content=REPLANNER_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=(
                        f"Ціль:\n{state['goal']}\n\n"
                        f"Початковий запит:\n"
                        f"{state['user_request']}\n\n"
                        f"План:\n"
                        f"{json.dumps(plan, ensure_ascii=False)}\n\n"
                        f"Виконано кроків: "
                        f"{step_index}/{len(plan)}\n\n"
                        f"Результати:\n"
                        f"{json.dumps(results, ensure_ascii=False)}\n\n"
                        f"Залишилися кроки:\n"
                        f"{json.dumps(remaining_steps, ensure_ascii=False)}"
                    )
                ),
            ]
        )

        decision = ReplanDecision.model_validate(
            raw_decision
        )

        decision_message = AIMessage(
            content=json.dumps(
                decision.model_dump(),
                ensure_ascii=False,
            )
        )

        if decision.action == "finish":
            last_result = (
                results[-1]
                if results
                else ""
            )
            normalized_result = last_result.lower()

            if "requires_human_input" in normalized_result:
                final_status = "needs_input"
            elif '"status": "rejected"' in normalized_result:
                final_status = "rejected"
            else:
                final_status = "completed"

            return {
                "completed": True,
                "status": final_status,
                "final_answer": (
                    decision.final_answer
                    or decision.reasoning
                ),
                "messages": [decision_message],
            }

        if decision.action == "replan":
            replan_count = state.get(
                "replan_count",
                0,
            )

            if replan_count >= 2:
                return {
                    "completed": True,
                    "status": "safety_stop",
                    "final_answer": (
                        "Виконання зупинено після двох "
                        "перепланувань."
                    ),
                    "messages": [decision_message],
                }

            completed_steps = plan[:step_index]
            updated_steps = (
                decision.updated_steps
                or []
            )

            return {
                "plan": [
                    *completed_steps,
                    *updated_steps,
                ],
                "replan_count": replan_count + 1,
                "messages": [decision_message],
            }

        if step_index >= len(plan):
            return {
                "completed": True,
                "status": "completed",
                "final_answer": (
                    decision.final_answer
                    or decision.reasoning
                ),
                "messages": [decision_message],
            }

        return {
            "messages": [decision_message],
        }

    return replanner_node


def route_after_replanner(
    state: PlanExecuteState,
) -> Literal["executor", "__end__"]:
    """Маршрутизує граф після replanner."""

    if state.get("completed"):
        return "__end__"

    return "executor"


def build_graph(
    checkpointer: SqliteSaver | None = None,
    planner_model: Any = PLANNER_MODEL,
    executor_model: Any = EXECUTOR_MODEL,
    replanner_model: Any = REPLANNER_MODEL,
    interrupt_after: list[str] | None = None,
):
    """Будує Plan-and-Execute LangGraph."""

    builder = StateGraph(PlanExecuteState)

    builder.add_node(
        "planner",
        planner_node_factory(planner_model),
    )
    builder.add_node(
        "executor",
        executor_node_factory(executor_model),
    )
    builder.add_node(
        "replanner",
        replanner_node_factory(replanner_model),
    )

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "replanner")
    builder.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor": "executor",
            "__end__": END,
        },
    )

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def create_persistent_app(
    database_path: Path = AGENT_STATE_DB,
    interrupt_after: list[str] | None = None,
):
    """Створює граф із файловим SQLite checkpointer."""

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(connection)

    app = build_graph(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )

    return app, connection


def thread_config(
    thread_id: str,
) -> dict[str, dict[str, str]]:
    """Формує config із thread_id."""

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise ValueError(
            "thread_id не може бути порожнім."
        )

    return {
        "configurable": {
            "thread_id": normalized_thread_id,
        }
    }


def initial_state(
    user_request: str,
) -> PlanExecuteState:
    """Формує початковий стан нового запуску."""

    normalized_request = user_request.strip()

    if not normalized_request:
        raise ValueError(
            "user_request не може бути порожнім."
        )

    return {
        "messages": [
            HumanMessage(content=normalized_request)
        ],
        "user_request": normalized_request,
        "goal": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False,
        "final_answer": None,
        "status": "new",
        "used_tools": [],
        "replan_count": 0,
    }


def start_run(
    app: Any,
    user_request: str,
    thread_id: str,
) -> dict[str, Any]:
    """Запускає новий Plan-and-Execute flow."""

    return app.invoke(
        initial_state(user_request),
        config=thread_config(thread_id),
    )


def resume_run(
    app: Any,
    thread_id: str,
    human_decision: dict[str, Any],
) -> dict[str, Any]:
    """Відновлює граф після interrupt."""

    return app.invoke(
        Command(resume=human_decision),
        config=thread_config(thread_id),
    )


def interrupt_payloads(
    state: dict[str, Any],
) -> list[Any]:
    """Витягує payload усіх активних interrupts."""

    return [
        interrupt_item.value
        for interrupt_item in state.get(
            "__interrupt__",
            [],
        )
    ]


def public_result(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Формує JSON-сумісний результат без message objects."""

    return {
        "status": state.get("status"),
        "goal": state.get("goal"),
        "plan": state.get("plan", []),
        "current_step": state.get(
            "current_step",
            0,
        ),
        "results": state.get("results", []),
        "completed": state.get(
            "completed",
            False,
        ),
        "final_answer": state.get(
            "final_answer"
        ),
        "used_tools": state.get(
            "used_tools",
            [],
        ),
        "interrupts": interrupt_payloads(state),
    }


if __name__ == "__main__":
    app, database_connection = create_persistent_app()

    try:
        print("Demand & Discovery Plan-and-Execute Agent")
        print(f"Model: {MODEL_NAME}")
        print("Для завершення введіть: exit\n")

        while True:
            request = input("You: ").strip()

            if request.lower() in {
                "exit",
                "quit",
                "вихід",
                "выход",
            }:
                break

            if not request:
                continue

            session_id = input(
                "Thread ID [demo-session]: "
            ).strip() or "demo-session"

            result = start_run(
                app=app,
                user_request=request,
                thread_id=session_id,
            )

            print(
                json.dumps(
                    public_result(result),
                    ensure_ascii=False,
                    indent=2,
                )
            )

            if interrupt_payloads(result):
                decision = input(
                    "Decision [approve/reject/edit]: "
                ).strip().lower()

                human_response: dict[str, Any] = {
                    "decision": decision,
                }

                if decision == "reject":
                    human_response["reason"] = input(
                        "Reason: "
                    ).strip()

                if decision == "edit":
                    edited_json = input(
                        "Edited args as JSON: "
                    ).strip()
                    human_response["edited_args"] = (
                        json.loads(edited_json)
                    )

                resumed_result = resume_run(
                    app=app,
                    thread_id=session_id,
                    human_decision=human_response,
                )

                print(
                    json.dumps(
                        public_result(resumed_result),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
    finally:
        database_connection.close()