"""Тести інструментів ReAct-агента."""

import json

from tools import (
    calculate_priority_score,
    check_intake_completeness,
    classify_discovery_scope,
    get_initiative_status,
)


def parse_result(tool, arguments: dict) -> dict:
    """Викликає інструмент і перетворює JSON-відповідь на словник."""

    return json.loads(tool.invoke(arguments))


def test_get_existing_initiative_status() -> None:
    result = parse_result(
        get_initiative_status,
        {"initiative_id": "dem-001"},
    )

    assert result["status"] == "success"
    assert result["data"]["initiative_id"] == "DEM-001"
    assert result["data"]["status"] == "Discovery"


def test_get_unknown_initiative_status() -> None:
    result = parse_result(
        get_initiative_status,
        {"initiative_id": "DEM-999"},
    )

    assert result["status"] == "error"
    assert result["data"] is None
    assert "не знайдено" in result["error"]


def test_incomplete_intake() -> None:
    result = parse_result(
        check_intake_completeness,
        {
            "initiative_id": "DEM-004",
            "business_owner": "Retail Director",
            "business_driver": "Reduce onboarding time",
        },
    )

    assert result["status"] == "success"
    assert result["data"]["is_complete"] is False
    assert result["data"]["completeness_percent"] == 40
    assert len(result["data"]["missing_fields"]) == 3


def test_complete_intake() -> None:
    result = parse_result(
        check_intake_completeness,
        {
            "initiative_id": "DEM-004",
            "business_owner": "Retail Director",
            "business_driver": "Reduce onboarding time",
            "success_metrics": "Reduce processing time by 30%",
            "financial_effect": "Save 1 million UAH annually",
            "constraints": "No known constraints",
        },
    )

    assert result["data"]["is_complete"] is True
    assert result["data"]["completeness_percent"] == 100
    assert result["data"]["missing_fields"] == []


def test_light_discovery_scope() -> None:
    result = parse_result(
        classify_discovery_scope,
        {
            "initiative_id": "DEM-005",
            "systems_count": 1,
            "ownership_clarity": "clear",
            "technical_uncertainty": "low",
            "dependency_count": 0,
            "regulatory_impact": "none",
            "data_readiness": "ready",
        },
    )

    assert result["data"]["discovery_points"] == 1
    assert result["data"]["recommended_scope"] == "Light"


def test_deep_discovery_scope() -> None:
    result = parse_result(
        classify_discovery_scope,
        {
            "initiative_id": "DEM-004",
            "systems_count": 4,
            "ownership_clarity": "partial",
            "technical_uncertainty": "high",
            "dependency_count": 3,
            "regulatory_impact": "possible",
            "data_readiness": "partial",
        },
    )

    assert result["data"]["discovery_points"] in {1, 2, 3, 5, 8, 13}
    assert result["data"]["discovery_points"] == 8
    assert result["data"]["recommended_scope"] == "Deep"
    assert result["data"]["requires_lba_confirmation"] is True


def test_high_priority_score() -> None:
    result = parse_result(
        calculate_priority_score,
        {
            "initiative_id": "DEM-004",
            "strategic_alignment": 5,
            "customer_impact": 4,
            "financial_impact": 3,
            "regulatory_urgency": 5,
            "implementation_feasibility": 2,
        },
    )

    assert result["data"]["priority_score"] == 4.05
    assert result["data"]["priority_level"] == "High"
    assert result["data"]["requires_human_validation"] is True