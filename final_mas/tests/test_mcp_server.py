"""Тести FastMCP-сервера Requirements & Estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

import tools_legacy
from mcp_server import mcp


EXPECTED_TOOLS = {
    "check_requirements_readiness",
    "classify_estimation_complexity",
    "identify_handover_gaps",
    "submit_estimation_request",
}


def parse_result(result: Any) -> dict[str, Any]:
    """Перетворити MCP result.data на словник."""

    data = result.data

    if isinstance(data, str):
        return json.loads(data)

    assert isinstance(data, dict)
    return data


@pytest.mark.asyncio
async def test_mcp_registers_four_domain_tools() -> None:
    """Сервер публікує рівно чотири domain tools."""

    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_requirements_readiness_returns_missing_fields() -> None:
    """Readiness tool визначає неповний набір requirements."""

    async with Client(mcp) as client:
        result = await client.call_tool(
            "check_requirements_readiness",
            {
                "initiative_id": "DEM-001",
                "business_objective": (
                    "Скоротити час обробки demand request."
                ),
            },
        )

    payload = parse_result(result)

    assert payload["status"] == "success"
    assert payload["data"]["initiative_id"] == "DEM-001"
    assert payload["data"]["is_ready"] is False
    assert payload["data"]["readiness_percent"] == 17
    assert payload["data"]["completed_fields"] == 1
    assert len(payload["data"]["missing_fields"]) == 5


@pytest.mark.asyncio
async def test_invalid_initiative_id_is_rejected() -> None:
    """Pydantic-валідація блокує неправильний ID."""

    async with Client(mcp) as client:
        result = await client.call_tool(
            "check_requirements_readiness",
            {
                "initiative_id": "INIT-001",
                "business_objective": "Test objective.",
            },
        )

    payload = parse_result(result)

    assert payload["status"] == "error"
    assert payload["error_type"] == "ValidationError"
    assert "DEM-001" in payload["message"]


@pytest.mark.asyncio
async def test_complexity_tool_returns_high_estimation() -> None:
    """Складна ініціатива класифікується як High."""

    async with Client(mcp) as client:
        result = await client.call_tool(
            "classify_estimation_complexity",
            {
                "initiative_id": "DEM-002",
                "systems_count": 3,
                "integration_count": 2,
                "nfr_criticality": "high",
                "data_migration_required": True,
                "security_review_required": True,
                "dependency_count": 2,
                "requirements_stability": "partial",
            },
        )

    payload = parse_result(result)
    data = payload["data"]

    assert payload["status"] == "success"
    assert data["raw_score"] == 8
    assert data["estimation_points"] == 8
    assert data["estimation_complexity"] == "High"
    assert data["requires_human_confirmation"] is True


@pytest.mark.asyncio
async def test_handover_tool_identifies_blocking_gaps() -> None:
    """Handover tool повертає controls і known blockers."""

    async with Client(mcp) as client:
        result = await client.call_tool(
            "identify_handover_gaps",
            {
                "initiative_id": "DEM-003",
                "solution_scope_defined": False,
                "dependencies_confirmed": False,
                "nfr_reviewed": False,
                "acceptance_criteria_testable": False,
                "data_owners_confirmed": False,
                "security_classification_completed": False,
                "known_blockers": [
                    "Не підтверджено data owner.",
                ],
            },
        )

    payload = parse_result(result)
    data = payload["data"]

    assert payload["status"] == "success"
    assert data["can_proceed_to_estimation"] is False
    assert len(data["missing_controls"]) == 6
    assert data["gaps_count"] == 7
    assert data["requires_human_input"] is True


@pytest.mark.asyncio
async def test_submission_tool_persists_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Risky MCP tool зберігає запит після валідного виклику."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_legacy,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "submit_estimation_request",
            {
                "initiative_id": "DEM-004",
                "estimation_complexity": "High",
                "estimation_points": 8,
                "target_team": "Demand Platform Team",
                "requested_by": "requester-001",
                "estimation_summary": (
                    "Scope, integrations, NFR and dependencies "
                    "validated for estimation."
                ),
            },
        )

    payload = parse_result(result)

    assert payload["status"] == "success"
    assert payload["data"]["status"] == "submitted"
    assert payload["data"]["initiative_id"] == "DEM-004"
    assert submissions_path.exists()

@pytest.mark.asyncio
async def test_mcp_registers_two_resources() -> None:
    """Сервер публікує два read-only standards resources."""

    async with Client(mcp) as client:
        resources = await client.list_resources()

    assert {
        str(resource.uri)
        for resource in resources
    } == {
        "demand://standards/readiness",
        "demand://standards/complexity",
    }


@pytest.mark.asyncio
async def test_mcp_resources_return_valid_standards() -> None:
    """Resources повертають валідні JSON standards."""

    async with Client(mcp) as client:
        readiness_result = await client.read_resource(
            "demand://standards/readiness"
        )
        complexity_result = await client.read_resource(
            "demand://standards/complexity"
        )

    readiness = json.loads(
        readiness_result[0].text
    )
    complexity = json.loads(
        complexity_result[0].text
    )

    assert len(readiness["required_fields"]) == 6
    assert readiness["side_effects"] is False

    assert (
        complexity["raw_score_to_points"]["8-9"]
        == 8
    )
    assert (
        complexity["points_to_complexity"]["8-13"]
        == "High"
    )
    assert (
        complexity["human_confirmation_required"]
        is True
    )


@pytest.mark.asyncio
async def test_mcp_prompt_renders_handover_workflow() -> None:
    """Prompt містить initiative context і HITL rule."""

    async with Client(mcp) as client:
        result = await client.get_prompt(
            "prepare_estimation_handover",
            {
                "initiative_id": "DEM-903",
                "business_objective": (
                    "Скоротити demand lead time."
                ),
                "known_gaps": (
                    "Не визначено acceptance criteria."
                ),
            },
        )

    assert result.messages

    content = result.messages[0].content
    prompt_text = getattr(
        content,
        "text",
        str(content),
    )

    assert "DEM-903" in prompt_text
    assert "check_requirements_readiness" in prompt_text
    assert "identify_handover_gaps" in prompt_text
    assert "classify_estimation_complexity" in prompt_text
    assert "Human-in-the-Loop approval" in prompt_text
