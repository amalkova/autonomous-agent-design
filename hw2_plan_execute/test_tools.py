"""Unit-тести доменних tools HW2."""

import json

import pytest
from pydantic import ValidationError

import tools as tools_module
from tools import (
    CheckIntakeCompletenessInput,
    ClassifyDiscoveryScopeInput,
    SubmitDiscoveryAssessmentInput,
    check_intake_completeness,
    classify_discovery_scope,
    submit_discovery_assessment,
)


def test_intake_input_normalizes_values() -> None:
    """ID нормалізується, а порожній текст перетворюється на None."""

    data = CheckIntakeCompletenessInput(
        initiative_id=" dem-004 ",
        business_owner="   ",
        business_driver=" Reduce onboarding time ",
    )

    assert data.initiative_id == "DEM-004"
    assert data.business_owner is None
    assert data.business_driver == "Reduce onboarding time"


def test_incomplete_intake_returns_missing_fields() -> None:
    """Tool повертає відсоток повноти та відсутні поля."""

    response = check_intake_completeness.invoke(
        {
            "initiative_id": "DEM-004",
            "business_owner": "Retail Director",
            "business_driver": "Reduce onboarding time",
        }
    )

    result = json.loads(response)

    assert result["status"] == "success"
    assert result["data"]["is_complete"] is False
    assert result["data"]["completeness_percent"] == 40
    assert result["data"]["missing_fields"] == [
        "success_metrics",
        "financial_effect",
        "constraints",
    ]


def test_invalid_initiative_id_is_rejected() -> None:
    """Некоректний initiative_id відхиляється Pydantic."""

    with pytest.raises(
        ValidationError,
        match="DEM-001",
    ):
        CheckIntakeCompletenessInput(
            initiative_id="initiative-4"
        )


def test_non_text_intake_value_is_rejected() -> None:
    """Нетекстове поле intake не приймається."""

    with pytest.raises(
        ValidationError,
        match="текстовими",
    ):
        CheckIntakeCompletenessInput(
            initiative_id="DEM-004",
            business_owner=123,
        )


def test_deep_discovery_scope_is_calculated() -> None:
    """Складна ініціатива отримує 8 points та Deep scope."""

    response = classify_discovery_scope.invoke(
        {
            "initiative_id": "DEM-004",
            "systems_count": 4,
            "ownership_clarity": "partial",
            "technical_uncertainty": "high",
            "dependency_count": 3,
            "regulatory_impact": "possible",
            "data_readiness": "partial",
        }
    )

    result = json.loads(response)

    assert result["status"] == "success"
    assert result["data"]["raw_score"] == 9
    assert result["data"]["discovery_points"] == 8
    assert result["data"]["recommended_scope"] == "Deep"
    assert result["data"]["requires_lba_confirmation"] is True


def test_invalid_systems_count_is_rejected() -> None:
    """Кількість систем менше одиниці відхиляється."""

    with pytest.raises(ValidationError):
        ClassifyDiscoveryScopeInput(
            initiative_id="DEM-004",
            systems_count=0,
            ownership_clarity="clear",
            technical_uncertainty="low",
            dependency_count=0,
            regulatory_impact="none",
            data_readiness="ready",
        )


def test_inconsistent_scope_and_points_are_rejected() -> None:
    """Light scope не може містити 8 Discovery Points."""

    with pytest.raises(
        ValidationError,
        match="не відповідають",
    ):
        SubmitDiscoveryAssessmentInput(
            initiative_id="DEM-004",
            discovery_scope="Light",
            discovery_points=8,
            decision_summary=(
                "Assessment підготовлено для перевірки."
            ),
        )


def test_submit_assessment_creates_record(
    tmp_path,
    monkeypatch,
) -> None:
    """Після виконання ризикового tool створюється запис."""

    submissions_file = tmp_path / "submitted_assessments.json"

    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_FILE",
        submissions_file,
    )

    response = submit_discovery_assessment.invoke(
        {
            "initiative_id": "DEM-004",
            "discovery_scope": "Deep",
            "discovery_points": 8,
            "decision_summary": (
                "Deep discovery потрібен через чотири системи "
                "та високу технічну невизначеність."
            ),
        }
    )

    result = json.loads(response)
    stored_records = json.loads(
        submissions_file.read_text(encoding="utf-8")
    )

    assert result["status"] == "success"
    assert result["data"]["status"] == "submitted"
    assert result["data"]["submission_id"] == "SUB-001"
    assert len(stored_records) == 1
    assert stored_records[0]["initiative_id"] == "DEM-004"


def test_duplicate_submission_is_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    """Повторна відправка assessment не створює дубль."""

    submissions_file = tmp_path / "submitted_assessments.json"

    monkeypatch.setattr(
        tools_module,
        "SUBMISSIONS_FILE",
        submissions_file,
    )

    arguments = {
        "initiative_id": "DEM-004",
        "discovery_scope": "Deep",
        "discovery_points": 8,
        "decision_summary": (
            "Deep discovery підтверджено відповідальним LBA."
        ),
    }

    first_response = json.loads(
        submit_discovery_assessment.invoke(arguments)
    )
    second_response = json.loads(
        submit_discovery_assessment.invoke(arguments)
    )

    stored_records = json.loads(
        submissions_file.read_text(encoding="utf-8")
    )

    assert first_response["status"] == "success"
    assert second_response["status"] == "error"
    assert "вже було відправлено" in second_response["error"]
    assert len(stored_records) == 1