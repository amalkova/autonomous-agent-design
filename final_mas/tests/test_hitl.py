"""Тести approve, reject та edit HITL scenarios."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.types import Command

from hitl import (
    build_hitl_graph,
    create_submission_state,
)


def valid_submission() -> dict[str, Any]:
    return {
        "initiative_id": "DEM-020",
        "estimation_complexity": "High",
        "estimation_points": 8,
        "target_team": "Demand Platform Team",
        "requested_by": "requester-001",
        "estimation_summary": (
            "Scope, integrations, NFR and dependencies "
            "validated for estimation."
        ),
    }


@pytest.mark.asyncio
async def test_approve_executes_risky_tool() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_executor(
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
        return {
            "status": "success",
            "initiative_id": arguments[
                "initiative_id"
            ],
        }

    graph = build_hitl_graph(
        tool_executor=fake_executor
    )
    config = {
        "configurable": {
            "thread_id": "hitl-approve",
        }
    }

    paused = await graph.ainvoke(
        create_submission_state(
            valid_submission()
        ),
        config=config,
    )

    assert "__interrupt__" in paused
    assert calls == []

    result = await graph.ainvoke(
        Command(
            resume={
                "action": "approve",
                "reason": "Scope confirmed.",
            }
        ),
        config=config,
    )

    assert result["status"] == "executed"
    assert len(calls) == 1
    assert (
        calls[0]["tool_name"]
        == "submit_estimation_request"
    )


@pytest.mark.asyncio
async def test_reject_blocks_risky_tool() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_executor(
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(arguments)
        return {"status": "success"}

    graph = build_hitl_graph(
        tool_executor=fake_executor
    )
    config = {
        "configurable": {
            "thread_id": "hitl-reject",
        }
    }

    paused = await graph.ainvoke(
        create_submission_state(
            valid_submission()
        ),
        config=config,
    )

    assert "__interrupt__" in paused

    result = await graph.ainvoke(
        Command(
            resume={
                "action": "reject",
                "reason": (
                    "Потрібно уточнити dependencies."
                ),
            }
        ),
        config=config,
    )

    assert result["status"] == "rejected"
    assert result["tool_result"]["status"] == (
        "not_executed"
    )
    assert calls == []


@pytest.mark.asyncio
async def test_edit_revalidates_and_executes() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_executor(
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(arguments)
        return {"status": "success"}

    graph = build_hitl_graph(
        tool_executor=fake_executor
    )
    config = {
        "configurable": {
            "thread_id": "hitl-edit",
        }
    }

    await graph.ainvoke(
        create_submission_state(
            valid_submission()
        ),
        config=config,
    )

    result = await graph.ainvoke(
        Command(
            resume={
                "action": "edit",
                "reason": (
                    "Передати іншій команді."
                ),
                "edited_arguments": {
                    "target_team": (
                        "Architecture Team"
                    ),
                    "estimation_summary": (
                        "Updated scope, integrations, "
                        "NFR and dependencies confirmed."
                    ),
                },
            }
        ),
        config=config,
    )

    assert result["status"] == "executed"
    assert len(calls) == 1
    assert (
        calls[0]["target_team"]
        == "Architecture Team"
    )