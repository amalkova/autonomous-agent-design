"""Unit-тести Human-in-the-Loop interrupt та resume."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError
from typing_extensions import TypedDict

import tools as tools_module
from hitl import HumanDecision, request_tool_approval
from tools import submit_discovery_assessment


class HITLTestState(TypedDict, total=False):
    """Стан мінімального тестового HITL-графа."""

    arguments: dict[str, Any]
    status: str
    result: dict[str, Any]


def build_hitl_test_graph(
    database_path: Path,
):
    """Створює тестовий граф із SQLite checkpointer."""

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(connection)

    def approval_node(
        state: HITLTestState,
    ) -> dict[str, Any]:
        """Запитує approval і виконує tool лише після згоди."""

        resolution = request_tool_approval(
            tool_name="submit_discovery_assessment",
            arguments=state["arguments"],
        )

        if not resolution.approved:
            return {
                "status": "rejected",
                "result": {
                    "decision": resolution.decision,
                    "reason": resolution.reason,
                },
            }

        tool_response = json.loads(
            submit_discovery_assessment.invoke(
                resolution.arguments
            )
        )

        return {
            "status": "submitted",
            "result": tool_response,
        }

    builder = StateGraph(HITLTestState)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)

    return (
        builder.compile(checkpointer=checkpointer),
        connection,
    )


def assessment_arguments() -> dict[str, Any]:
    """Повертає валідні параметри ризикового tool."""

    return {
        "initiative_id": "DEM-900",
        "discovery_scope": "Deep",
        "discovery_points": 8,
        "decision_summary": (
            "Deep discovery потрібен через високу складність "
            "та значну кількість залежностей."
        ),
    }


def test_interrupt_contains_action_details(
    tmp_path,
    monkeypatch,
) -> None:
    """Граф зупиняється до створення submission."""

    submissions_file = tmp_path / "submissions.json"

    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_FILE",
        submissions_file,
    )

    app, connection = build_hitl_test_graph(
        tmp_path / "interrupt.db"
    )
    config = {
        "configurable": {
            "thread_id": "interrupt-details",
        }
    }

    try:
        result = app.invoke(
            {
                "arguments": assessment_arguments(),
            },
            config=config,
        )

        assert "__interrupt__" in result
        assert not submissions_file.exists()

        interrupt_value = result["__interrupt__"][0].value

        assert interrupt_value["type"] == "tool_approval"
        assert interrupt_value["risk_level"] == "high"
        assert (
            interrupt_value["action"]
            == "submit_discovery_assessment"
        )
        assert (
            interrupt_value["arguments"]["initiative_id"]
            == "DEM-900"
        )
    finally:
        connection.close()


def test_approve_executes_risky_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """Approve відновлює граф і виконує ризиковий tool."""

    submissions_file = tmp_path / "submissions.json"

    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_FILE",
        submissions_file,
    )

    app, connection = build_hitl_test_graph(
        tmp_path / "approve.db"
    )
    config = {
        "configurable": {
            "thread_id": "approve-flow",
        }
    }

    try:
        paused_result = app.invoke(
            {
                "arguments": assessment_arguments(),
            },
            config=config,
        )

        assert "__interrupt__" in paused_result
        assert not submissions_file.exists()

        resumed_result = app.invoke(
            Command(
                resume={
                    "decision": "approve",
                    "reason": "Assessment перевірено LBA.",
                }
            ),
            config=config,
        )

        assert resumed_result["status"] == "submitted"
        assert (
            resumed_result["result"]["status"]
            == "success"
        )
        assert submissions_file.exists()

        stored_records = json.loads(
            submissions_file.read_text(encoding="utf-8")
        )

        assert len(stored_records) == 1
        assert (
            stored_records[0]["initiative_id"]
            == "DEM-900"
        )
    finally:
        connection.close()


def test_reject_does_not_execute_risky_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """Reject завершує flow без виконання tool."""

    submissions_file = tmp_path / "submissions.json"

    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_FILE",
        submissions_file,
    )

    app, connection = build_hitl_test_graph(
        tmp_path / "reject.db"
    )
    config = {
        "configurable": {
            "thread_id": "reject-flow",
        }
    }

    try:
        app.invoke(
            {
                "arguments": assessment_arguments(),
            },
            config=config,
        )

        resumed_result = app.invoke(
            Command(
                resume={
                    "decision": "reject",
                    "reason": (
                        "Потрібне додаткове погодження."
                    ),
                }
            ),
            config=config,
        )

        assert resumed_result["status"] == "rejected"
        assert (
            resumed_result["result"]["decision"]
            == "reject"
        )
        assert not submissions_file.exists()
    finally:
        connection.close()


def test_edit_changes_arguments_and_executes_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """Edit змінює параметри перед виконанням tool."""

    submissions_file = tmp_path / "submissions.json"

    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_FILE",
        submissions_file,
    )

    app, connection = build_hitl_test_graph(
        tmp_path / "edit.db"
    )
    config = {
        "configurable": {
            "thread_id": "edit-flow",
        }
    }

    try:
        app.invoke(
            {
                "arguments": assessment_arguments(),
            },
            config=config,
        )

        resumed_result = app.invoke(
            Command(
                resume={
                    "decision": "edit",
                    "edited_args": {
                        "discovery_points": 13,
                    },
                    "reason": (
                        "Регуляторний вплив потребує "
                        "вищої оцінки."
                    ),
                }
            ),
            config=config,
        )

        stored_records = json.loads(
            submissions_file.read_text(encoding="utf-8")
        )

        assert resumed_result["status"] == "submitted"
        assert stored_records[0]["discovery_points"] == 13
        assert (
            stored_records[0]["discovery_scope"]
            == "Deep"
        )
    finally:
        connection.close()


def test_edit_without_arguments_is_rejected() -> None:
    """Decision edit без edited_args не проходить validation."""

    with pytest.raises(
        ValidationError,
        match="edited_args",
    ):
        HumanDecision(decision="edit")