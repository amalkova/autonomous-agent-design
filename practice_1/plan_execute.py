"""Plan-and-Execute агент для підготовки estimation request."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict
from trajectory_logger import save_trajectory

from hitl import request_tool_approval
from react_agent import MODEL_NAME, run_react_agent
from tools import (
    SubmitEstimationRequestInput,
    submit_estimation_request,
)


load_dotenv()

DEFAULT_DATABASE_PATH = Path(__file__).with_name("agent_state.db")

SUPPORTED_PLAN_TOOLS = {
    "check_requirements_readiness",
    "classify_estimation_complexity",
    "identify_handover_gaps",
    "search_delivery_knowledge",
    "submit_estimation_request",
}

RISKY_TOOL_NAME = "submit_estimation_request"


class Plan(BaseModel):
    """Структурований план виконання користувацької цілі."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=5,
        max_length=1000,
        description="Уточнена ціль користувача",
    )
    steps: list[str] = Field(
        min_length=1,
        max_length=6,
        description=(
            "Послідовні кроки. Кожен крок починається з назви tool"
        ),
    )

    @field_validator("goal", mode="before")
    @classmethod
    def normalize_goal(cls, value: object) -> str:
        """Нормалізує ціль."""

        if not isinstance(value, str):
            raise ValueError("goal має бути текстовим значенням.")

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("goal не може бути порожнім.")

        return normalized_value

    @field_validator("steps", mode="before")
    @classmethod
    def validate_steps(cls, value: object) -> list[str]:
        """Перевіряє формат і підтримувані tools у плані."""

        if not isinstance(value, list):
            raise ValueError("steps має бути списком.")

        normalized_steps: list[str] = []

        for raw_step in value:
            if not isinstance(raw_step, str):
                raise ValueError("Кожен крок має бути текстом.")

            step = raw_step.strip()

            if not step:
                raise ValueError("Крок плану не може бути порожнім.")

            tool_name = step.split(":", maxsplit=1)[0].strip()

            if tool_name not in SUPPORTED_PLAN_TOOLS:
                raise ValueError(
                    f"Непідтримуваний tool у плані: {tool_name}"
                )

            normalized_steps.append(step)

        return normalized_steps


class ReplanDecision(BaseModel):
    """Рішення replanner після виконання чергового кроку."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["continue", "replan", "finish"]
    reasoning: str = Field(
        min_length=3,
        max_length=1000,
        description="Пояснення рішення replanner",
    )
    updated_steps: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Оновлені невиконані кроки для action=replan",
    )
    final_answer: str | None = Field(
        default=None,
        max_length=3000,
        description="Фінальна відповідь для action=finish",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def normalize_reasoning(cls, value: object) -> str:
        """Нормалізує пояснення."""

        if not isinstance(value, str):
            raise ValueError("reasoning має бути текстом.")

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("reasoning не може бути порожнім.")

        return normalized_value

    @field_validator("updated_steps", mode="before")
    @classmethod
    def validate_updated_steps(cls, value: object) -> list[str]:
        """Перевіряє оновлені кроки."""

        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError("updated_steps має бути списком.")

        normalized_steps: list[str] = []

        for raw_step in value:
            if not isinstance(raw_step, str):
                raise ValueError("Кожен оновлений крок має бути текстом.")

            step = raw_step.strip()

            if not step:
                raise ValueError(
                    "Оновлений крок не може бути порожнім."
                )

            tool_name = step.split(":", maxsplit=1)[0].strip()

            if tool_name not in SUPPORTED_PLAN_TOOLS:
                raise ValueError(
                    f"Непідтримуваний tool у replanning: {tool_name}"
                )

            normalized_steps.append(step)

        return normalized_steps


class PlanExecuteState(TypedDict, total=False):
    """Стан Plan-and-Execute графа."""

    user_request: str
    goal: str
    plan: list[str]
    current_step: int
    results: list[str]
    status: Literal[
        "planning",
        "running",
        "waiting_human",
        "completed",
        "rejected",
        "error",
    ]
    completed: bool
    final_answer: str | None
    used_tools: list[str]
    replan_count: int
    pending_action: dict[str, Any] | None


BASE_MODEL = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    timeout=120,
    max_retries=2,
)

PLANNER_MODEL = BASE_MODEL.with_structured_output(
    Plan,
    method="json_schema",
)

REPLANNER_MODEL = BASE_MODEL.with_structured_output(
    ReplanDecision,
    method="json_schema",
)

SUBMIT_ARGUMENTS_MODEL = BASE_MODEL.with_structured_output(
    SubmitEstimationRequestInput,
    method="json_schema",
)


PLANNER_PROMPT = """
Ти — planner для Requirements & Estimation Readiness Agent.

Побудуй короткий послідовний план виконання запиту.

Доступні tools та їхні параметри:

1. check_requirements_readiness:
initiative_id, business_objective, functional_requirements,
non_functional_requirements, acceptance_criteria,
integration_scope, data_requirements.

2. classify_estimation_complexity:
initiative_id, systems_count, integration_count,
nfr_complexity, data_migration, security_review,
dependency_count, requirements_stability.

3. identify_handover_gaps:
використовуй точні назви параметрів із запиту користувача
та не вигадуй відсутні значення.

4. search_delivery_knowledge:
query.

5. submit_estimation_request:
initiative_id, target_team, requested_by,
estimation_complexity, estimation_points,
estimation_summary.

Правила:
- Кожен крок повинен починатися з точної назви tool.
- Після назви tool постав двокрапку та переліч параметри.
- Зберігай точні назви аргументів.
- Зберігай текстові та категоріальні значення без змін.
- Не перетворюй required, partial, high та інші категорії
  на true або false.
- Додавай initiative_id у кожен domain tool, який його приймає.
- Не скорочуй functional_requirements до functional.
- Не додавай кроки, яких користувач не просив.
- Не вигадуй відсутні бізнес-дані.
- submit_estimation_request є ризиковою дією та потребує HITL.
- План має містити від одного до шести кроків.
""".strip()


REPLANNER_PROMPT = """
Ти — replanner для Requirements & Estimation Readiness Agent.

Проаналізуй початковий запит, план та результати виконаних кроків.

Обери одну дію:
- continue — якщо наступний крок чинного плану можна виконувати;
- replan — якщо залишок плану треба замінити;
- finish — якщо ціль досягнута або виконання треба завершити.

Правила:
- Не повторюй успішно виконані кроки.
- Якщо бракує обов'язкових даних, заверши роботу та чітко назви їх.
- Якщо людина відхилила ризикову дію, не намагайся виконати її повторно.
- Для finish обов'язково сформуй final_answer.
- Для replan заповни updated_steps.
""".strip()


SUBMIT_ARGUMENTS_PROMPT = """
Витягни параметри для submit_estimation_request.

Використовуй лише значення, присутні у запиті користувача або кроці плану.
Не вигадуй initiative_id, estimation_complexity, estimation_points
чи estimation_summary.
""".strip()


def model_to_dict(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Перетворює structured output на словник."""

    if isinstance(value, BaseModel):
        return value.model_dump()

    if isinstance(value, dict):
        return value

    raise TypeError(
        "Structured model повернула непідтримуваний тип відповіді."
    )


def append_unique(
    current_values: list[str],
    new_values: list[str],
) -> list[str]:
    """Додає нові значення без дублікатів."""

    result = list(current_values)

    for value in new_values:
        if value not in result:
            result.append(value)

    return result


def default_step_runner(step_prompt: str) -> dict[str, Any]:
    """Виконує один plan step через вкладений ReAct-агент."""

    return run_react_agent(
        step_prompt,
        log_trajectory=False,
    )


def create_initial_state(user_request: str) -> PlanExecuteState:
    """Створює початковий стан графа."""

    normalized_request = user_request.strip()

    if not normalized_request:
        raise ValueError("Запит користувача не може бути порожнім.")

    return {
        "user_request": normalized_request,
        "goal": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "status": "planning",
        "completed": False,
        "final_answer": None,
        "used_tools": [],
        "replan_count": 0,
        "pending_action": None,
    }


def fallback_final_answer(state: PlanExecuteState) -> str:
    """Формує резервну фінальну відповідь без додаткового LLM-виклику."""

    results = state.get("results", [])

    if not results:
        return "Виконання завершено без результатів."

    return (
        "Виконання завершено.\n\n"
        + "\n\n".join(results)
    )


def build_graph(
    checkpointer: Any | None = None,
    *,
    planner_model: Any | None = None,
    replanner_model: Any | None = None,
    submit_arguments_model: Any | None = None,
    step_runner: Callable[[str], dict[str, Any]] | None = None,
    interrupt_after: list[str] | None = None,
):
    """Створює Plan-and-Execute LangGraph."""

    active_planner = planner_model or PLANNER_MODEL
    active_replanner = replanner_model or REPLANNER_MODEL
    active_submit_model = (
        submit_arguments_model or SUBMIT_ARGUMENTS_MODEL
    )
    active_step_runner = step_runner or default_step_runner

    def planner_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Створює початковий план."""

        raw_plan = active_planner.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=state["user_request"]),
            ]
        )

        plan = Plan.model_validate(model_to_dict(raw_plan))

        return {
            "goal": plan.goal,
            "plan": plan.steps,
            "current_step": 0,
            "results": [],
            "status": "running",
            "completed": False,
            "final_answer": None,
            "used_tools": [],
            "replan_count": 0,
            "pending_action": None,
        }

    def executor_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Виконує один крок плану через вкладений ReAct."""

        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])

        if current_step >= len(plan):
            return {
                "status": "running",
                "pending_action": None,
            }

        step = plan[current_step]
        tool_name = step.split(":", maxsplit=1)[0].strip()

        if tool_name == RISKY_TOOL_NAME:
            raw_arguments = active_submit_model.invoke(
                [
                    SystemMessage(content=SUBMIT_ARGUMENTS_PROMPT),
                    HumanMessage(
                        content=(
                            f"Запит користувача:\n"
                            f"{state['user_request']}\n\n"
                            f"Поточний крок:\n{step}"
                        )
                    ),
                ]
            )

            arguments = SubmitEstimationRequestInput.model_validate(
                model_to_dict(raw_arguments)
            ).model_dump()

            return {
                "status": "waiting_human",
                "pending_action": {
                    "tool_name": RISKY_TOOL_NAME,
                    "arguments": arguments,
                    "step": step,
                },
            }

        nested_prompt = (
            "Виконай рівно один крок плану через відповідний tool.\n"
            "Не виконуй інші кроки та не додавай нові цілі.\n\n"
            f"Загальна ціль:\n{state['goal']}\n\n"
            f"Запит користувача:\n{state['user_request']}\n\n"
            f"Поточний крок:\n{step}"
        )

        nested_response = active_step_runner(nested_prompt)
        nested_status = str(
            nested_response.get("status", "error")
        )
        nested_answer = str(
            nested_response.get(
                "answer",
                "Вкладений ReAct-агент не повернув відповідь.",
            )
        )
        nested_tools = [
            str(tool)
            for tool in nested_response.get("used_tools", [])
        ]

        step_result = (
            f"Крок {current_step + 1} ({step}): "
            f"status={nested_status}. {nested_answer}"
        )

        return {
            "current_step": current_step + 1,
            "results": [
                *state.get("results", []),
                step_result,
            ],
            "used_tools": append_unique(
                state.get("used_tools", []),
                nested_tools,
            ),
            "status": (
                "running"
                if nested_status in {"completed", "needs_input"}
                else "error"
            ),
            "pending_action": None,
        }

    def approval_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Запитує HITL-рішення перед ризиковим tool."""

        pending_action = state.get("pending_action")

        if not pending_action:
            return {
                "status": "error",
                "completed": True,
                "final_answer": (
                    "Не знайдено параметри ризикової дії."
                ),
            }

        tool_name = str(pending_action["tool_name"])
        original_arguments = dict(
            pending_action.get("arguments", {})
        )

        resolution = request_tool_approval(
            tool_name,
            original_arguments,
        )

        current_step = state.get("current_step", 0)
        step = str(pending_action.get("step", tool_name))

        if not resolution.approved:
            rejection_result = (
                f"Крок {current_step + 1} ({step}): "
                f"ризикову дію відхилено людиною. "
                f"Причина: {resolution.reason or 'не вказана'}"
            )

            return {
                "current_step": current_step + 1,
                "results": [
                    *state.get("results", []),
                    rejection_result,
                ],
                "status": "rejected",
                "pending_action": None,
            }

        try:
            validated_arguments = (
                SubmitEstimationRequestInput.model_validate(
                    resolution.arguments
                ).model_dump()
            )

            observation = submit_estimation_request.invoke(
                validated_arguments
            )

            try:
                observation_data = json.loads(str(observation))
            except json.JSONDecodeError:
                observation_data = {
                    "status": "error",
                    "data": None,
                    "error": str(observation),
                }

            observation_status = str(
                observation_data.get("status", "error")
            )

            step_result = (
                f"Крок {current_step + 1} ({step}): "
                f"{RISKY_TOOL_NAME}: "
                f"{json.dumps(observation_data, ensure_ascii=False)}"
            )

            return {
                "current_step": current_step + 1,
                "results": [
                    *state.get("results", []),
                    step_result,
                ],
                "used_tools": append_unique(
                    state.get("used_tools", []),
                    [RISKY_TOOL_NAME],
                ),
                "status": (
                    "running"
                    if observation_status == "success"
                    else "error"
                ),
                "pending_action": None,
            }

        except Exception as error:
            error_result = (
                f"Крок {current_step + 1} ({step}): "
                f"ризикову дію не виконано. Помилка: {error}"
            )

            return {
                "current_step": current_step + 1,
                "results": [
                    *state.get("results", []),
                    error_result,
                ],
                "status": "error",
                "pending_action": None,
            }

    def replanner_node(
        state: PlanExecuteState,
    ) -> dict[str, Any]:
        """Продовжує, перебудовує або завершує план."""

        replanner_input = {
            "user_request": state.get("user_request"),
            "goal": state.get("goal"),
            "plan": state.get("plan", []),
            "current_step": state.get("current_step", 0),
            "results": state.get("results", []),
            "status": state.get("status"),
            "replan_count": state.get("replan_count", 0),
        }

        raw_decision = active_replanner.invoke(
            [
                SystemMessage(content=REPLANNER_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        replanner_input,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )

        decision = ReplanDecision.model_validate(
            model_to_dict(raw_decision)
        )

        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])

        if decision.action == "finish":
            return {
                "completed": True,
                "status": (
                    "rejected"
                    if state.get("status") == "rejected"
                    else "completed"
                ),
                "final_answer": (
                    decision.final_answer
                    or fallback_final_answer(state)
                ),
                "pending_action": None,
            }

        if decision.action == "replan":
            replan_count = state.get("replan_count", 0) + 1

            if replan_count > 3:
                return {
                    "completed": True,
                    "status": "error",
                    "final_answer": (
                        "Досягнуто ліміт у три перебудови плану."
                    ),
                    "pending_action": None,
                }

            if not decision.updated_steps:
                return {
                    "completed": True,
                    "status": "error",
                    "final_answer": (
                        "Replanner обрав replan, але не надав "
                        "оновлених кроків."
                    ),
                    "pending_action": None,
                }

            completed_steps = plan[:current_step]
            updated_plan = [
                *completed_steps,
                *decision.updated_steps,
            ]

            return {
                "plan": updated_plan,
                "replan_count": replan_count,
                "status": "running",
                "completed": False,
                "pending_action": None,
            }

        if current_step >= len(plan):
            return {
                "completed": True,
                "status": (
                    "rejected"
                    if state.get("status") == "rejected"
                    else "completed"
                ),
                "final_answer": fallback_final_answer(state),
                "pending_action": None,
            }

        return {
            "status": "running",
            "completed": False,
            "pending_action": None,
        }

    def route_after_executor(
        state: PlanExecuteState,
    ) -> Literal["approval", "replanner"]:
        """Маршрутизує ризиковий або звичайний результат."""

        if state.get("pending_action"):
            return "approval"

        return "replanner"

    def route_after_replanner(
        state: PlanExecuteState,
    ) -> Literal["executor", "end"]:
        """Визначає, чи завершено виконання."""

        if state.get("completed"):
            return "end"

        return "executor"

    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("approval", approval_node)
    builder.add_node("replanner", replanner_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")

    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "approval": "approval",
            "replanner": "replanner",
        },
    )

    builder.add_edge("approval", "replanner")

    builder.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor": "executor",
            "end": END,
        },
    )

    return builder.compile(
    checkpointer=checkpointer,
    interrupt_after=interrupt_after,
)

def serialize_interrupts(
    graph_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Перетворює LangGraph interrupts на JSON-сумісний список."""

    serialized: list[dict[str, Any]] = []

    for interrupt_value in graph_result.get("__interrupt__", []):
        value = getattr(interrupt_value, "value", interrupt_value)

        if isinstance(value, dict):
            serialized.append(value)
        else:
            serialized.append({"message": str(value)})

    return serialized


def public_result(
    graph_result: dict[str, Any],
) -> dict[str, Any]:
    """Готує результат графа для відображення користувачу."""

    result = {
        key: value
        for key, value in graph_result.items()
        if not key.startswith("__")
    }

    result["interrupts"] = serialize_interrupts(graph_result)
    return result
def log_plan_execute_trajectory(
    user_request: str,
    graph_result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any] | None:
    """Зберігає завершену Plan-and-Execute trajectory."""

    if not graph_result.get("completed"):
        return None

    plan = [
        str(step)
        for step in graph_result.get("plan", [])
    ]
    results = [
        str(result)
        for result in graph_result.get("results", [])
    ]
    final_answer = str(
        graph_result.get("final_answer")
        or "Plan-and-Execute завершив роботу."
    )

    messages = [
        HumanMessage(content=user_request),
        AIMessage(
            content=(
                "Сформований план:\n"
                + "\n".join(
                    f"{index}. {step}"
                    for index, step in enumerate(
                        plan,
                        start=1,
                    )
                )
            )
        ),
    ]

    messages.extend(
        AIMessage(content=result)
        for result in results
    )

    messages.append(
        AIMessage(content=final_answer)
    )

    return save_trajectory(
        agent_type="plan_execute",
        user_input=user_request,
        messages=messages,
        final_response={
            key: value
            for key, value in graph_result.items()
            if not key.startswith("__")
        },
        safety={
            "planned_steps": len(plan),
            "executed_steps": int(
                graph_result.get("current_step", 0)
            ),
            "replan_count": int(
                graph_result.get("replan_count", 0)
            ),
        },
        metadata={
            "thread_id": thread_id,
            "goal": graph_result.get("goal"),
            "plan": plan,
            "results": results,
            "architecture": "Plan-and-Execute",
        },
    )

def read_human_decision() -> dict[str, Any]:
    """Зчитує та перевіряє approve, reject або edit у CLI."""

    while True:
        decision = input(
            "Decision [approve/reject/edit]: "
        ).strip().lower()

        if decision not in {"approve", "reject", "edit"}:
            print(
                "Введіть лише approve, reject або edit."
            )
            continue

        payload: dict[str, Any] = {
            "decision": decision,
        }

        if decision == "reject":
            reason = input("Reason: ").strip()

            if reason:
                payload["reason"] = reason

        elif decision == "edit":
            while True:
                edited_json = input(
                    "Edited arguments as JSON: "
                ).strip()

                try:
                    edited_args = json.loads(edited_json)
                except json.JSONDecodeError:
                    print(
                        "Некоректний JSON. Спробуйте ще раз."
                    )
                    continue

                if not isinstance(edited_args, dict):
                    print(
                        "Edited arguments мають бути JSON object."
                    )
                    continue

                payload["edited_args"] = edited_args
                break

            reason = input("Reason: ").strip()

            if reason:
                payload["reason"] = reason

        return payload


def main() -> None:
    """Запускає інтерактивний Plan-and-Execute агент."""

    print("Requirements & Estimation Plan-and-Execute Agent")
    print(f"Model: {MODEL_NAME}")
    print("Для завершення введіть: exit\n")

    with SqliteSaver.from_conn_string(
        str(DEFAULT_DATABASE_PATH)
    ) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        while True:
            user_request = input("You: ").strip()

            if user_request.lower() in {
                "exit",
                "quit",
                "вихід",
                "выход",
            }:
                print("Agent stopped.")
                break

            if not user_request:
                continue

            default_thread_id = f"practice-{uuid.uuid4().hex[:8]}"
            thread_id = (
                input(
                    f"Thread ID [{default_thread_id}]: "
                ).strip()
                or default_thread_id
            )

            config = {
                "configurable": {
                    "thread_id": thread_id,
                },
                "recursion_limit": 50,
            }

            result = graph.invoke(
                create_initial_state(user_request),
                config=config,
            )

            print(
                json.dumps(
                    public_result(result),
                    ensure_ascii=False,
                    indent=2,
                )
            )

            while result.get("__interrupt__"):
                decision_payload = read_human_decision()

                result = graph.invoke(
                    Command(resume=decision_payload),
                    config=config,
                )

                print(
                    json.dumps(
                        public_result(result),
                        ensure_ascii=False,
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()