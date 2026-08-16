"""Тести Pydantic-схем і доменних tools."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import tools as tools_module
from tools import (
    CheckRequirementsReadinessInput,
    ClassifyEstimationComplexityInput,
    IdentifyHandoverGapsInput,
    SubmitEstimationRequestInput,
    check_requirements_readiness,
    classify_estimation_complexity,
    identify_handover_gaps,
    submit_estimation_request,
)


def parse_tool_response(response: str) -> dict:
    """Перетворює JSON-відповідь tool на словник."""

    return json.loads(response)


def test_requirements_schema_normalizes_input() -> None:
    """Schema нормалізує ID, пробіли та порожні значення."""

    data = CheckRequirementsReadinessInput(
        initiative_id=" dem-001 ",
        business_objective=" Reduce delivery time ",
        functional_requirements="   ",
    )

    assert data.initiative_id == "DEM-001"
    assert data.business_objective == "Reduce delivery time"
    assert data.functional_requirements is None


def test_requirements_schema_rejects_invalid_id() -> None:
    """Schema відхиляє некоректний initiative ID."""

    with pytest.raises(
        ValidationError,
        match="DEM-001",
    ):
        CheckRequirementsReadinessInput(
            initiative_id="REQ-1",
        )


def test_requirements_schema_forbids_extra_fields() -> None:
    """Schema не дозволяє невідомі аргументи."""

    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        CheckRequirementsReadinessInput(
            initiative_id="DEM-001",
            unexpected_field="value",
        )


def test_incomplete_requirements_are_reported() -> None:
    """Tool повертає missing fields і readiness percent."""

    response = parse_tool_response(
        check_requirements_readiness.invoke(
            {
                "initiative_id": "DEM-004",
                "business_objective": (
                    "Reduce onboarding time"
                ),
            }
        )
    )

    assert response["status"] == "success"
    assert response["data"]["is_ready"] is False
    assert response["data"]["readiness_percent"] == 17
    assert response["data"]["completed_fields"] == 1
    assert "functional_requirements" in (
        response["data"]["missing_fields"]
    )
    assert response["data"]["requires_human_input"] is True


def test_complete_requirements_are_ready() -> None:
    """Повний набір requirements проходить readiness gate."""

    response = parse_tool_response(
        check_requirements_readiness.invoke(
            {
                "initiative_id": "DEM-004",
                "business_objective": "Reduce onboarding time",
                "functional_requirements": (
                    "Create and update applications"
                ),
                "non_functional_requirements": (
                    "Availability 99.9 percent"
                ),
                "acceptance_criteria": (
                    "Application is processed within two minutes"
                ),
                "integration_scope": "CRM and Core Banking",
                "data_requirements": (
                    "Customer profile owned by Retail Data"
                ),
            }
        )
    )

    assert response["data"]["is_ready"] is True
    assert response["data"]["readiness_percent"] == 100
    assert response["data"]["missing_fields"] == []


def test_complexity_schema_normalizes_categories() -> None:
    """Категоріальні значення нормалізуються до lower case."""

    data = ClassifyEstimationComplexityInput(
        initiative_id=" dem-004 ",
        systems_count=3,
        integration_count=2,
        nfr_criticality=" HIGH ",
        data_migration_required=True,
        security_review_required=True,
        dependency_count=2,
        requirements_stability=" PARTIAL ",
    )

    assert data.initiative_id == "DEM-004"
    assert data.nfr_criticality == "high"
    assert data.requirements_stability == "partial"


def test_complexity_schema_rejects_invalid_count() -> None:
    """Кількість систем не може бути меншою за один."""

    with pytest.raises(ValidationError):
        ClassifyEstimationComplexityInput(
            initiative_id="DEM-004",
            systems_count=0,
            integration_count=0,
            nfr_criticality="low",
            data_migration_required=False,
            security_review_required=False,
            dependency_count=0,
            requirements_stability="high",
        )


def test_low_estimation_complexity() -> None:
    """Простий scope отримує Low та один point."""

    response = parse_tool_response(
        classify_estimation_complexity.invoke(
            {
                "initiative_id": "DEM-005",
                "systems_count": 1,
                "integration_count": 0,
                "nfr_criticality": "low",
                "data_migration_required": False,
                "security_review_required": False,
                "dependency_count": 0,
                "requirements_stability": "high",
            }
        )
    )

    assert response["status"] == "success"
    assert response["data"]["raw_score"] == 0
    assert response["data"]["estimation_points"] == 1
    assert response["data"]["estimation_complexity"] == "Low"


def test_high_estimation_complexity() -> None:
    """Складний scope отримує High та 13 points."""

    response = parse_tool_response(
        classify_estimation_complexity.invoke(
            {
                "initiative_id": "DEM-006",
                "systems_count": 5,
                "integration_count": 4,
                "nfr_criticality": "high",
                "data_migration_required": True,
                "security_review_required": True,
                "dependency_count": 4,
                "requirements_stability": "low",
            }
        )
    )

    assert response["status"] == "success"
    assert response["data"]["raw_score"] == 12
    assert response["data"]["estimation_points"] == 13
    assert response["data"]["estimation_complexity"] == "High"
    assert (
        response["data"]["requires_human_confirmation"]
        is True
    )


def test_handover_gaps_are_identified() -> None:
    """Tool повертає controls і blockers, що не завершені."""

    response = parse_tool_response(
        identify_handover_gaps.invoke(
            {
                "initiative_id": "DEM-007",
                "solution_scope_defined": True,
                "dependencies_confirmed": False,
                "nfr_reviewed": False,
                "acceptance_criteria_testable": True,
                "data_owners_confirmed": True,
                "security_classification_completed": False,
                "known_blockers": [
                    "Waiting for vendor API specification",
                ],
            }
        )
    )

    data = response["data"]

    assert data["can_proceed_to_estimation"] is False
    assert data["gaps_count"] == 4
    assert "dependencies_confirmed" in data["missing_controls"]
    assert "nfr_reviewed" in data["missing_controls"]
    assert data["known_blockers"] == [
        "Waiting for vendor API specification"
    ]


def test_handover_schema_rejects_non_list_blockers() -> None:
    """known_blockers має бути списком."""

    with pytest.raises(
        ValidationError,
        match="known_blockers",
    ):
        IdentifyHandoverGapsInput(
            initiative_id="DEM-007",
            solution_scope_defined=True,
            dependencies_confirmed=True,
            nfr_reviewed=True,
            acceptance_criteria_testable=True,
            data_owners_confirmed=True,
            security_classification_completed=True,
            known_blockers="No blockers",
        )


def test_submission_schema_rejects_inconsistent_points() -> None:
    """Points мають відповідати рівню complexity."""

    with pytest.raises(
        ValidationError,
        match="не відповідають",
    ):
        SubmitEstimationRequestInput(
            initiative_id="DEM-010",
            estimation_complexity="Medium",
            estimation_points=8,
            target_team="Core Banking Team",
            requested_by="Lead Business Analyst",
            estimation_summary=(
                "Scope is ready for estimation and review."
            ),
        )


def test_submit_estimation_request_creates_record(
    tmp_path,
    monkeypatch,
) -> None:
    """Підтверджений request зберігається у JSON-файлі."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    response = parse_tool_response(
        submit_estimation_request.invoke(
            {
                "initiative_id": "DEM-010",
                "estimation_complexity": "High",
                "estimation_points": 8,
                "target_team": "Core Banking Team",
                "requested_by": "Lead Business Analyst",
                "estimation_summary": (
                    "Scope includes four systems, security review "
                    "and several external dependencies."
                ),
            }
        )
    )

    assert response["status"] == "success"
    assert response["data"]["request_id"] == "EST-001"
    assert response["data"]["status"] == "submitted"
    assert submissions_path.exists()

    saved_records = json.loads(
        submissions_path.read_text(encoding="utf-8")
    )

    assert len(saved_records) == 1
    assert saved_records[0]["initiative_id"] == "DEM-010"


def test_duplicate_submission_is_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    """Повторна відправка тієї самої ініціативи блокується."""

    submissions_path = (
        tmp_path / "submitted_estimation_requests.json"
    )
    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_PATH",
        submissions_path,
    )

    arguments = {
        "initiative_id": "DEM-010",
        "estimation_complexity": "High",
        "estimation_points": 8,
        "target_team": "Core Banking Team",
        "requested_by": "Lead Business Analyst",
        "estimation_summary": (
            "Scope includes four systems, security review "
            "and several external dependencies."
        ),
    }

    first_response = parse_tool_response(
        submit_estimation_request.invoke(arguments)
    )
    second_response = parse_tool_response(
        submit_estimation_request.invoke(arguments)
    )

    assert first_response["status"] == "success"
    assert second_response["status"] == "error"
    assert "вже відправлено" in second_response["error"]