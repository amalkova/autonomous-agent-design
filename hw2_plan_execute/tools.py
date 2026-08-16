"""Доменні tools для Plan-and-Execute агента."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Self

from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SUBMISSIONS_FILE = Path("submitted_assessments.json")

DISCOVERY_POINTS = (1, 2, 3, 5, 8, 13)

SCOPE_POINTS = {
    "Light": {1, 2},
    "Standard": {3, 5},
    "Deep": {8, 13},
}


def normalize_initiative_id(value: object) -> str:
    """Нормалізує та перевіряє ідентифікатор ініціативи."""

    if not isinstance(value, str):
        raise ValueError("initiative_id має бути текстовим значенням.")

    normalized_id = value.strip().upper()

    if (
        len(normalized_id) != 7
        or not normalized_id.startswith("DEM-")
        or not normalized_id[4:].isdigit()
    ):
        raise ValueError(
            "initiative_id має відповідати формату DEM-001."
        )

    return normalized_id


def build_tool_response(
    status: Literal["success", "error"],
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    """Формує стандартну JSON-відповідь інструмента."""

    return json.dumps(
        {
            "status": status,
            "data": data,
            "error": error,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class CheckIntakeCompletenessInput(BaseModel):
    """Вхідні дані для перевірки повноти intake."""

    model_config = ConfigDict(str_strip_whitespace=True)

    initiative_id: str = Field(
        description="Ідентифікатор ініціативи у форматі DEM-001"
    )
    business_owner: str | None = Field(
        default=None,
        description="Власник бізнес-ініціативи",
    )
    business_driver: str | None = Field(
        default=None,
        description="Бізнес-причина або проблема, яку треба вирішити",
    )
    success_metrics: str | None = Field(
        default=None,
        description="Вимірювані критерії успіху",
    )
    financial_effect: str | None = Field(
        default=None,
        description="Очікуваний фінансовий ефект",
    )
    constraints: str | None = Field(
        default=None,
        description="Відомі обмеження та залежності",
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє формат initiative_id."""

        return normalize_initiative_id(value)

    @field_validator(
        "business_owner",
        "business_driver",
        "success_metrics",
        "financial_effect",
        "constraints",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: object,
    ) -> str | None:
        """Перетворює порожній текст на None."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                "Поля intake мають бути текстовими значеннями."
            )

        normalized_value = value.strip()
        return normalized_value or None


class ClassifyDiscoveryScopeInput(BaseModel):
    """Вхідні дані для визначення Discovery scope."""

    model_config = ConfigDict(str_strip_whitespace=True)

    initiative_id: str = Field(
        description="Ідентифікатор ініціативи у форматі DEM-001"
    )
    systems_count: int = Field(
        ge=1,
        le=50,
        strict=True,
        description="Кількість систем, яких стосується ініціатива",
    )
    ownership_clarity: Literal[
        "clear",
        "partial",
        "unclear",
    ] = Field(
        description="Рівень зрозумілості ownership"
    )
    technical_uncertainty: Literal[
        "low",
        "medium",
        "high",
    ] = Field(
        description="Рівень технічної невизначеності"
    )
    dependency_count: int = Field(
        ge=0,
        le=50,
        strict=True,
        description="Кількість зовнішніх залежностей",
    )
    regulatory_impact: Literal[
        "none",
        "possible",
        "confirmed",
    ] = Field(
        description="Рівень регуляторного впливу"
    )
    data_readiness: Literal[
        "ready",
        "partial",
        "not_ready",
    ] = Field(
        description="Рівень готовності даних"
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє формат initiative_id."""

        return normalize_initiative_id(value)


class SubmitDiscoveryAssessmentInput(BaseModel):
    """Вхідні дані для фінальної відправки assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    initiative_id: str = Field(
        description="Ідентифікатор ініціативи у форматі DEM-001"
    )
    discovery_scope: Literal[
        "Light",
        "Standard",
        "Deep",
    ] = Field(
        description="Підтверджений Discovery scope"
    )
    discovery_points: int = Field(
        description=(
            "Підтверджені Discovery Points за шкалою Fibonacci: "
            "1, 2, 3, 5, 8 або 13"
        ),
    )
    decision_summary: str = Field(
        min_length=10,
        max_length=1000,
        description="Підсумок і обґрунтування assessment",
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє формат initiative_id."""

        return normalize_initiative_id(value)

    @field_validator("decision_summary", mode="before")
    @classmethod
    def validate_decision_summary(cls, value: object) -> str:
        """Перевіряє, що підсумок є непорожнім текстом."""

        if not isinstance(value, str):
            raise ValueError(
                "decision_summary має бути текстовим значенням."
            )

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "decision_summary не може бути порожнім."
            )

        return normalized_value

    @model_validator(mode="after")
    def validate_scope_points_consistency(self) -> Self:
        """Перевіряє відповідність scope та Discovery Points."""

        allowed_points = SCOPE_POINTS[self.discovery_scope]

        if self.discovery_points not in allowed_points:
            raise ValueError(
                "Discovery Points не відповідають обраному scope."
            )

        return self


@tool(args_schema=CheckIntakeCompletenessInput)
def check_intake_completeness(
    initiative_id: str,
    business_owner: str | None = None,
    business_driver: str | None = None,
    success_metrics: str | None = None,
    financial_effect: str | None = None,
    constraints: str | None = None,
) -> str:
    """Перевірити повноту intake за обов'язковими полями Gate 0.

    Використовуйте перед плануванням discovery, щоб визначити,
    чи достатньо бізнес-даних для подальшої оцінки.
    """

    intake_fields = {
        "business_owner": business_owner,
        "business_driver": business_driver,
        "success_metrics": success_metrics,
        "financial_effect": financial_effect,
        "constraints": constraints,
    }

    missing_fields = [
        field_name
        for field_name, value in intake_fields.items()
        if value is None
    ]

    completed_fields = len(intake_fields) - len(missing_fields)
    completeness_percent = round(
        completed_fields / len(intake_fields) * 100
    )

    return build_tool_response(
        status="success",
        data={
            "initiative_id": initiative_id,
            "is_complete": not missing_fields,
            "completeness_percent": completeness_percent,
            "missing_fields": missing_fields,
            "requires_human_input": bool(missing_fields),
        },
    )


@tool(args_schema=ClassifyDiscoveryScopeInput)
def classify_discovery_scope(
    initiative_id: str,
    systems_count: int,
    ownership_clarity: str,
    technical_uncertainty: str,
    dependency_count: int,
    regulatory_impact: str,
    data_readiness: str,
) -> str:
    """Розрахувати Discovery Points та рекомендувати Discovery scope.

    Використовуйте лише тоді, коли всі параметри складності
    надані користувачем. Результат потребує підтвердження LBA.
    """

    systems_score = (
        0
        if systems_count == 1
        else 1
        if systems_count <= 3
        else 2
    )

    dependency_score = (
        0
        if dependency_count == 0
        else 1
        if dependency_count <= 2
        else 2
    )

    scoring_breakdown = {
        "systems": systems_score,
        "ownership": {
            "clear": 0,
            "partial": 1,
            "unclear": 2,
        }[ownership_clarity],
        "technical_uncertainty": {
            "low": 0,
            "medium": 1,
            "high": 2,
        }[technical_uncertainty],
        "dependencies": dependency_score,
        "regulatory_impact": {
            "none": 0,
            "possible": 1,
            "confirmed": 2,
        }[regulatory_impact],
        "data_readiness": {
            "ready": 0,
            "partial": 1,
            "not_ready": 2,
        }[data_readiness],
    }

    raw_score = sum(scoring_breakdown.values())

    if raw_score <= 1:
        discovery_points = 1
    elif raw_score <= 3:
        discovery_points = 2
    elif raw_score <= 5:
        discovery_points = 3
    elif raw_score <= 7:
        discovery_points = 5
    elif raw_score <= 10:
        discovery_points = 8
    else:
        discovery_points = 13

    if systems_count >= 3 and discovery_points < 3:
        discovery_points = 3

    if discovery_points <= 2:
        recommended_scope = "Light"
    elif discovery_points <= 5:
        recommended_scope = "Standard"
    else:
        recommended_scope = "Deep"

    return build_tool_response(
        status="success",
        data={
            "initiative_id": initiative_id,
            "raw_score": raw_score,
            "discovery_points": discovery_points,
            "recommended_scope": recommended_scope,
            "scoring_breakdown": scoring_breakdown,
            "requires_lba_confirmation": True,
        },
    )


@tool(args_schema=SubmitDiscoveryAssessmentInput)
def submit_discovery_assessment(
    initiative_id: str,
    discovery_scope: str,
    discovery_points: int,
    decision_summary: str,
) -> str:
    """Фінально відправити Discovery assessment.

    РИЗИКОВА ДІЯ: змінює стан ініціативи та створює запис
    про відправлений assessment. Інструмент дозволено виконувати
    лише після approve через Human-in-the-Loop.
    """

    existing_records: list[dict[str, Any]] = []

    if SUBMISSIONS_FILE.exists():
        try:
            stored_data = json.loads(
                SUBMISSIONS_FILE.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            return build_tool_response(
                status="error",
                error=(
                    "Файл submitted_assessments.json "
                    "містить некоректний JSON."
                ),
            )

        if not isinstance(stored_data, list):
            return build_tool_response(
                status="error",
                error=(
                    "Файл submitted_assessments.json "
                    "має містити список записів."
                ),
            )

        existing_records = stored_data

    already_submitted = any(
        record.get("initiative_id") == initiative_id
        and record.get("status") == "submitted"
        for record in existing_records
    )

    if already_submitted:
        return build_tool_response(
            status="error",
            error=(
                f"Assessment для {initiative_id} "
                "вже було відправлено."
            ),
        )

    submission_record = {
        "submission_id": f"SUB-{len(existing_records) + 1:03d}",
        "initiative_id": initiative_id,
        "discovery_scope": discovery_scope,
        "discovery_points": discovery_points,
        "decision_summary": decision_summary,
        "status": "submitted",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    existing_records.append(submission_record)

    SUBMISSIONS_FILE.write_text(
        json.dumps(
            existing_records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return build_tool_response(
        status="success",
        data=submission_record,
    )


DOMAIN_TOOLS = [
    check_intake_completeness,
    classify_discovery_scope,
    submit_discovery_assessment,
]