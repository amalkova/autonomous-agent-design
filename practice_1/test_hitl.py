"""Тести HITL approve, reject та edit."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator
from typing_extensions import TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

import tools as tools_module
from hitl import (
    HumanDecision,
    request_tool_approval,
)
from tools import (
    SubmitEstimationRequestInput,
    submit_estimation_request,
)


class HITLTestState(TypedDict, total=False):
    """Стан мінімального HITL-графа."""

    arguments: dict
    result: dict


def risky_node(
    state: HITLTestState,
) -> dict:
    """Запитує approval і умовно виконує risky tool."""

    resolution = request_tool_approval(
        tool_name="submit_estimation_request",
        arguments=state["arguments"],
    )

    if not resolution.approved:
        return {
            "result": {
                "status": "rejected",
                "decision": resolution.decision,
                "reason": resolution.reason,
                "tool_executed": False,
            }
        }

    validated_arguments = (
        SubmitEstimationRequestInput.model_validate(
            resolution.arguments
        )
    )

    observation = json.loads(
        submit_estimation_request.invoke(
            validated_arguments.model_dump()
        )
    )

    return {
        "result": {
            "status": observation["status"],
            "decision": resolution.decision,
            "tool_executed": (
                observation["status"] == "success"
            ),
            "observation": observation,
        }
    }


@contextmanager
def persistent_hitl_app(
    database_path,
) -> Iterator:
    """Створює HITL-граф із файловим checkpointer."""

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    try:
        checkpointer = SqliteSaver(connection)
        builder = StateGraph(HITLTestState)

        builder.add_node(
            "risky",
            risky_node,
        )
        builder.add_edge(
            START,
            "risky",
        )
        builder.add_edge(
            "risky",
            END,
        )

        yield builder.compile(
            checkpointer=checkpointer
        )

    finally:
        connection.close()


def valid_arguments() -> dict:
    """Повертає валідні аргументи risky tool."""

    return {
        "initiative_id": "DEM-010",
        "estimation_complexity": "High",
        "estimation_points": 8,
        "target_team": "Core Banking Team",
        "requested_by": "Lead Business Analyst",
        "estimation_summary": (
            "Scope includes four systems, security review "
            "and multiple external dependencies."
        ),
    }


def test_interrupt_exposes_risky_tool_details(
    tmp_path,
    monkeypatch,
) -> None:
    """До рішення людини tool не виконується."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    with persistent_hitl_app(
        tmp_path / "interrupt.db"
    ) as app:
        config = {
            "configurable": {
                "thread_id": "interrupt-flow",
            }
        }

        result = app.invoke(
            {
                "arguments": valid_arguments(),
            },
            config=config,
        )

        interrupts = result.get(
            "__interrupt__",
            [],
        )

        assert len(interrupts) == 1

        interrupt_value = interrupts[0].value

        assert interrupt_value["type"] == "tool_approval"
        assert interrupt_value["risk_level"] == "high"
        assert (
            interrupt_value["action"]
            == "submit_estimation_request"
        )
        assert interrupt_value["arguments"][
            "initiative_id"
        ] == "DEM-010"
        assert interrupt_value["allowed_decisions"] == [
            "approve",
            "reject",
            "edit",
        ]
        assert not submissions_path.exists()


def test_approve_executes_risky_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """Approve продовжує graph і створює request."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    with persistent_hitl_app(
        tmp_path / "approve.db"
    ) as app:
        config = {
            "configurable": {
                "thread_id": "approve-flow",
            }
        }

        app.invoke(
            {
                "arguments": valid_arguments(),
            },
            config=config,
        )
        resumed_result = app.invoke(
            Command(
                resume={
                    "decision": "approve",
                }
            ),
            config=config,
        )

        result = resumed_result["result"]

        assert result["status"] == "success"
        assert result["decision"] == "approve"
        assert result["tool_executed"] is True
        assert result["observation"]["data"][
            "request_id"
        ] == "EST-001"
        assert submissions_path.exists()


def test_reject_does_not_execute_risky_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """Reject завершує graph без виконання tool."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    with persistent_hitl_app(
        tmp_path / "reject.db"
    ) as app:
        config = {
            "configurable": {
                "thread_id": "reject-flow",
            }
        }

        app.invoke(
            {
                "arguments": valid_arguments(),
            },
            config=config,
        )
        resumed_result = app.invoke(
            Command(
                resume={
                    "decision": "reject",
                    "reason": (
                        "Потрібне погодження Solution Architect."
                    ),
                }
            ),
            config=config,
        )

        result = resumed_result["result"]

        assert result["status"] == "rejected"
        assert result["decision"] == "reject"
        assert result["tool_executed"] is False
        assert "Solution Architect" in result["reason"]
        assert not submissions_path.exists()


def test_edit_changes_arguments_and_executes_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """Edit змінює points перед виконанням tool."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    with persistent_hitl_app(
        tmp_path / "edit.db"
    ) as app:
        config = {
            "configurable": {
                "thread_id": "edit-flow",
            }
        }

        app.invoke(
            {
                "arguments": valid_arguments(),
            },
            config=config,
        )
        resumed_result = app.invoke(
            Command(
                resume={
                    "decision": "edit",
                    "edited_args": {
                        "estimation_points": 13,
                    },
                    "reason": (
                        "Підвищено через migration risk."
                    ),
                }
            ),
            config=config,
        )

        result = resumed_result["result"]
        saved_data = result["observation"]["data"]

        assert result["status"] == "success"
        assert result["decision"] == "edit"
        assert result["tool_executed"] is True
        assert saved_data["estimation_points"] == 13


def test_edit_without_arguments_is_rejected() -> None:
    """Decision edit вимагає edited_args."""

    with pytest.raises(
        ValidationError,
        match="edited_args",
    ):
        HumanDecision(
            decision="edit",
        )


def test_edit_cannot_add_unknown_argument(
    tmp_path,
    monkeypatch,
) -> None:
    """HITL edit не дозволяє ін'єкцію нового аргументу."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    with persistent_hitl_app(
        tmp_path / "unknown-argument.db"
    ) as app:
        config = {
            "configurable": {
                "thread_id": "unknown-argument-flow",
            }
        }

        app.invoke(
            {
                "arguments": valid_arguments(),
            },
            config=config,
        )

        with pytest.raises(
            ValueError,
            match="невідомі аргументи",
        ):
            app.invoke(
                Command(
                    resume={
                        "decision": "edit",
                        "edited_args": {
                            "admin_override": True,
                        },
                    }
                ),
                config=config,
            )

        assert not submissions_path.exists()