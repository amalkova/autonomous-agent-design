"""Тести LangGraph ReAct-агента без зовнішніх LLM-викликів."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from react_agent import (
    REACT_TOOLS,
    build_react_graph,
    run_react_agent,
)
from safety import SafetyController
from trajectory_logger import save_trajectory


class FakeSequenceModel:
    """Повертає заздалегідь визначені AI messages."""

    def __init__(
        self,
        responses: list[AIMessage],
    ) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.received_messages: list[list[Any]] = []

    def invoke(
        self,
        messages: list[Any],
    ) -> AIMessage:
        """Повертає наступну відповідь."""

        self.calls += 1
        self.received_messages.append(messages)

        if not self.responses:
            raise RuntimeError(
                "Fake model не має наступної відповіді."
            )

        return self.responses.pop(0)


class FakeStructuredModel:
    """Повертає фіксовану structured response."""

    def __init__(
        self,
        response: dict[str, Any],
    ) -> None:
        self.response = response
        self.calls = 0
        self.received_messages: list[list[Any]] = []

    def invoke(
        self,
        messages: list[Any],
    ) -> dict[str, Any]:
        """Повертає structured response."""

        self.calls += 1
        self.received_messages.append(messages)

        return self.response


def make_tool_call(
    name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> AIMessage:
    """Створює AIMessage з одним tool call."""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_react_graph_contains_required_nodes() -> None:
    """ReAct graph містить agent, tools і finalize."""

    graph = build_react_graph(
        safety=SafetyController(),
    )

    business_nodes = [
        node
        for node in graph.get_graph().nodes
        if not node.startswith("__")
    ]

    assert business_nodes == [
        "agent",
        "tools",
        "finalize",
    ]
    assert len(REACT_TOOLS) == 4


def test_blank_user_input_is_rejected() -> None:
    """Порожній запит не запускає graph."""

    model = FakeSequenceModel([])
    structured_model = FakeStructuredModel(
        {
            "status": "completed",
            "answer": "Unused",
            "requires_human_confirmation": False,
        }
    )

    response = run_react_agent(
        user_input="   ",
        model_with_tools=model,
        structured_model=structured_model,
        log_trajectory=False,
    )

    assert response["status"] == "error"
    assert model.calls == 0
    assert structured_model.calls == 0


def test_react_agent_can_answer_without_tool() -> None:
    """Agent може одразу перейти до finalize."""

    model = FakeSequenceModel(
        [
            AIMessage(
                content=(
                    "Потрібно надати більше інформації."
                )
            ),
        ]
    )
    structured_model = FakeStructuredModel(
        {
            "status": "needs_input",
            "answer": (
                "Надайте initiative ID та requirements."
            ),
            "requires_human_confirmation": True,
        }
    )

    response = run_react_agent(
        user_input="Перевір готовність",
        model_with_tools=model,
        structured_model=structured_model,
        log_trajectory=False,
    )

    assert response["status"] == "needs_input"
    assert response["used_tools"] == []
    assert response["safety"]["step_count"] == 2
    assert model.calls == 1
    assert structured_model.calls == 1


def test_react_agent_executes_tool_loop() -> None:
    """Agent проходить цикл LLM → tool → LLM."""

    model = FakeSequenceModel(
        [
            make_tool_call(
                name="classify_estimation_complexity",
                arguments={
                    "initiative_id": "DEM-004",
                    "systems_count": 5,
                    "integration_count": 4,
                    "nfr_criticality": "high",
                    "data_migration_required": True,
                    "security_review_required": True,
                    "dependency_count": 4,
                    "requirements_stability": "low",
                },
                call_id="call-1",
            ),
            AIMessage(
                content=(
                    "Розрахунок complexity завершено."
                )
            ),
        ]
    )
    structured_model = FakeStructuredModel(
        {
            "status": "completed",
            "answer": (
                "DEM-004 має High complexity "
                "та 13 Estimation Points."
            ),
            "requires_human_confirmation": True,
        }
    )

    response = run_react_agent(
        user_input=(
            "Розрахуй estimation complexity для DEM-004"
        ),
        model_with_tools=model,
        structured_model=structured_model,
        log_trajectory=False,
    )

    assert response["status"] == "completed"
    assert response["used_tools"] == [
        "classify_estimation_complexity"
    ]
    assert response["safety"]["step_count"] == 4
    assert response["safety"]["registered_tool_calls"] == 1
    assert "13" in response["answer"]


def test_react_agent_reports_missing_requirements() -> None:
    """Tool observation використовується для needs_input."""

    model = FakeSequenceModel(
        [
            make_tool_call(
                name="check_requirements_readiness",
                arguments={
                    "initiative_id": "DEM-008",
                    "business_objective": (
                        "Reduce processing time"
                    ),
                },
                call_id="call-1",
            ),
            AIMessage(
                content=(
                    "Requirements package is incomplete."
                )
            ),
        ]
    )
    structured_model = FakeStructuredModel(
        {
            "status": "needs_input",
            "answer": (
                "Надайте functional_requirements, "
                "non_functional_requirements, "
                "acceptance_criteria, integration_scope "
                "та data_requirements."
            ),
            "requires_human_confirmation": True,
        }
    )

    response = run_react_agent(
        user_input=(
            "Перевір readiness DEM-008. "
            "business_objective=Reduce processing time"
        ),
        model_with_tools=model,
        structured_model=structured_model,
        log_trajectory=False,
    )

    assert response["status"] == "needs_input"
    assert response["used_tools"] == [
        "check_requirements_readiness"
    ]
    assert "functional_requirements" in response["answer"]

    structured_messages = (
        structured_model.received_messages[0]
    )
    serialized_messages = " ".join(
        str(message.content)
        for message in structured_messages
    )

    assert "readiness_percent" in serialized_messages
    assert "missing_fields" in serialized_messages


def test_repeated_tool_call_triggers_safety_stop() -> None:
    """Третій однаковий tool call зупиняє ReAct-loop."""

    repeated_arguments = {
        "initiative_id": "DEM-009",
        "business_objective": "Reduce errors",
    }

    model = FakeSequenceModel(
        [
            make_tool_call(
                "check_requirements_readiness",
                repeated_arguments,
                "call-1",
            ),
            make_tool_call(
                "check_requirements_readiness",
                repeated_arguments,
                "call-2",
            ),
            make_tool_call(
                "check_requirements_readiness",
                repeated_arguments,
                "call-3",
            ),
        ]
    )
    structured_model = FakeStructuredModel(
        {
            "status": "completed",
            "answer": "Should not be used",
            "requires_human_confirmation": False,
        }
    )

    response = run_react_agent(
        user_input="Repeat tool",
        model_with_tools=model,
        structured_model=structured_model,
        log_trajectory=False,
    )

    assert response["status"] == "safety_stop"
    assert "повторний tool call" in response["answer"]
    assert response["safety"]["registered_tool_calls"] == 3
    assert structured_model.calls == 0


def test_trajectory_logger_appends_runs(
    tmp_path,
) -> None:
    """Trajectory logger зберігає декілька запусків."""

    trajectory_path = tmp_path / "trajectory.json"
    messages = [
        HumanMessage(content="Test request"),
        AIMessage(content="Test answer"),
    ]

    first_run = save_trajectory(
        agent_type="react",
        user_input="First request",
        messages=messages,
        final_response={
            "status": "completed",
            "answer": "First answer",
        },
        safety={
            "step_count": 2,
            "max_steps": 10,
        },
        path=trajectory_path,
    )
    second_run = save_trajectory(
        agent_type="react",
        user_input="Second request",
        messages=messages,
        final_response={
            "status": "completed",
            "answer": "Second answer",
        },
        safety={
            "step_count": 2,
            "max_steps": 10,
        },
        path=trajectory_path,
    )

    payload = json.loads(
        trajectory_path.read_text(encoding="utf-8")
    )

    assert first_run["run_id"] == "RUN-001"
    assert second_run["run_id"] == "RUN-002"
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["agent_type"] == "react"
    assert payload["runs"][0]["messages"][0][
        "message_type"
    ] == "human"