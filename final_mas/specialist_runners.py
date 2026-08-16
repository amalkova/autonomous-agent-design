"""Адаптери спеціалізованих агентів Final MAS."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from plan_execute_agent import (
    build_graph as build_plan_execute_graph,
)
from plan_execute_agent import (
    create_initial_state as create_plan_state,
)
from plan_execute_agent import (
    log_plan_execute_trajectory,
    public_result,
)
from react_agent import run_react_agent


def result_to_text(
    result: Any,
) -> str:
    """Перетворює результат specialist agent на текст."""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in (
            "final_answer",
            "answer",
            "response",
            "message",
        ):
            value = result.get(key)

            if value:
                return str(value)

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return str(result)


async def run_solution_security_react(
    user_request: str,
) -> str:
    """Запускає Solution & Security Agent через ReAct.

    ReAct самостійно обирає дозволені tools, а
    SafetyController обмежує кількість кроків,
    повторні виклики та загальний час виконання.
    """

    result = await asyncio.to_thread(
        run_react_agent,
        user_request,
    )

    return result_to_text(result)


async def run_estimation_plan_execute(
    user_request: str,
) -> str:
    """Запускає Estimation Agent через Plan-and-Execute.

    Planner формує план, executor виконує кожний крок
    через вкладений ReAct, а replanner вирішує, чи
    потрібні наступні кроки.
    """

    thread_id = (
        "estimation-"
        + uuid.uuid4().hex
    )

    graph = build_plan_execute_graph(
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 40,
    }

    initial_state = create_plan_state(
        user_request
    )

    def invoke_graph() -> dict[str, Any]:
        return graph.invoke(
            initial_state,
            config=config,
        )

    graph_result = await asyncio.to_thread(
        invoke_graph
    )

    visible_result = public_result(
        graph_result
    )

    log_plan_execute_trajectory(
        user_request=user_request,
        graph_result=graph_result,
        thread_id=thread_id,
    )

    final_answer = visible_result.get(
        "final_answer"
    )

    if final_answer:
        return str(final_answer)

    interrupts = visible_result.get(
        "interrupts",
        [],
    )

    if interrupts:
        return json.dumps(
            {
                "status": "waiting_human",
                "message": (
                    "Plan-and-Execute очікує рішення "
                    "людини для risky action."
                ),
                "interrupts": interrupts,
                "thread_id": thread_id,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return result_to_text(
        visible_result
    )
