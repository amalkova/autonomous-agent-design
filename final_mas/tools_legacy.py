"""Доменні tools для Requirements & Estimation Readiness Agent."""

from __future__ import annotations

import json
import re
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


BASE_DIR = Path(__file__).resolve().parent
SUBMISSIONS_PATH = BASE_DIR / "submitted_estimation_requests.json"

INITIATIVE_ID_PATTERN = re.compile(r"^DEM-\d{3}$")

REQUIRED_REQUIREMENTS_FIELDS = (
    "business_objective",
    "functional_requirements",
    "non_functional_requirements",
    "acceptance_criteria",
    "integration_scope",
    "data_requirements",
)

COMPLEXITY_POINTS = {
    "Low": {1, 2},
    "Medium": {3, 5},
    "High": {8, 13},
}


def build_tool_response(
    status: Literal["success", "error"],
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    """Формує стандартну JSON-відповідь tool."""

    return json.dumps(
        {
            "status": status,
            "data": data,
            "error": error,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def normalize_initiative_id(value: object) -> str:
    """Нормалізує та перевіряє initiative ID."""

    if not isinstance(value, str):
        raise ValueError(
            "initiative_id має бути текстовим значенням."
        )

    normalized_value = value.strip().upper()

    if not INITIATIVE_ID_PATTERN.fullmatch(normalized_value):
        raise ValueError(
            "initiative_id має відповідати формату DEM-001."
        )

    return normalized_value


def normalize_optional_text(value: object) -> str | None:
    """Перетворює порожній текст на None."""

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            "Значення має бути текстом або None."
        )

    normalized_value = value.strip()

    return normalized_value or None


class CheckRequirementsReadinessInput(BaseModel):
    """Вхідні дані для перевірки готовності requirements."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    initiative_id: str = Field(
        description="ID ініціативи у форматі DEM-001",
    )
    business_objective: str | None = Field(
        default=None,
        description="Підтверджена бізнес-мета ініціативи",
    )
    functional_requirements: str | None = Field(
        default=None,
        description="Узгоджені функціональні вимоги",
    )
    non_functional_requirements: str | None = Field(
        default=None,
        description="Визначені нефункціональні вимоги",
    )
    acceptance_criteria: str | None = Field(
        default=None,
        description="Тестовані критерії приймання",
    )
    integration_scope: str | None = Field(
        default=None,
        description="Перелік інтеграцій і систем у scope",
    )
    data_requirements: str | None = Field(
        default=None,
        description="Вимоги до даних і data ownership",
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє initiative ID."""

        return normalize_initiative_id(value)

    @field_validator(
        "business_objective",
        "functional_requirements",
        "non_functional_requirements",
        "acceptance_criteria",
        "integration_scope",
        "data_requirements",
        mode="before",
    )
    @classmethod
    def validate_optional_text(
        cls,
        value: object,
    ) -> str | None:
        """Нормалізує необов'язкові текстові поля."""

        return normalize_optional_text(value)


class ClassifyEstimationComplexityInput(BaseModel):
    """Вхідні дані для оцінки складності estimation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    initiative_id: str = Field(
        description="ID ініціативи у форматі DEM-001",
    )
    systems_count: int = Field(
        ge=1,
        le=30,
        description="Кількість систем у solution scope",
    )
    integration_count: int = Field(
        ge=0,
        le=30,
        description="Кількість інтеграцій",
    )
    nfr_criticality: Literal[
        "low",
        "medium",
        "high",
    ] = Field(
        description="Критичність NFR: low, medium або high",
    )
    data_migration_required: bool = Field(
        description="Чи потрібна міграція даних",
    )
    security_review_required: bool = Field(
        description="Чи потрібен security review",
    )
    dependency_count: int = Field(
        ge=0,
        le=30,
        description="Кількість зовнішніх залежностей",
    )
    requirements_stability: Literal[
        "high",
        "partial",
        "low",
    ] = Field(
        description=(
            "Стабільність вимог: high, partial або low"
        ),
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє initiative ID."""

        return normalize_initiative_id(value)

    @field_validator(
        "nfr_criticality",
        "requirements_stability",
        mode="before",
    )
    @classmethod
    def normalize_category(cls, value: object) -> str:
        """Нормалізує категоріальні значення."""

        if not isinstance(value, str):
            raise ValueError(
                "Категоріальне значення має бути текстом."
            )

        return value.strip().lower()


class IdentifyHandoverGapsInput(BaseModel):
    """Вхідні дані для пошуку прогалин handover."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    initiative_id: str = Field(
        description="ID ініціативи у форматі DEM-001",
    )
    solution_scope_defined: bool = Field(
        description="Чи визначено solution scope",
    )
    dependencies_confirmed: bool = Field(
        description="Чи підтверджено зовнішні залежності",
    )
    nfr_reviewed: bool = Field(
        description="Чи виконано review нефункціональних вимог",
    )
    acceptance_criteria_testable: bool = Field(
        description="Чи є acceptance criteria тестованими",
    )
    data_owners_confirmed: bool = Field(
        description="Чи підтверджено data owners",
    )
    security_classification_completed: bool = Field(
        description="Чи завершено security classification",
    )
    known_blockers: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Відомі блокери для estimation",
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє initiative ID."""

        return normalize_initiative_id(value)

    @field_validator("known_blockers", mode="before")
    @classmethod
    def validate_known_blockers(
        cls,
        value: object,
    ) -> list[str]:
        """Нормалізує список відомих блокерів."""

        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                "known_blockers має бути списком."
            )

        normalized_blockers: list[str] = []

        for blocker in value:
            if not isinstance(blocker, str):
                raise ValueError(
                    "Кожен blocker має бути текстом."
                )

            normalized_blocker = blocker.strip()

            if normalized_blocker:
                normalized_blockers.append(
                    normalized_blocker
                )

        return normalized_blockers


class SubmitEstimationRequestInput(BaseModel):
    """Вхідні дані ризикової відправки на estimation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    initiative_id: str = Field(
        description="ID ініціативи у форматі DEM-001",
    )
    estimation_complexity: Literal[
        "Low",
        "Medium",
        "High",
    ] = Field(
        description="Підтверджена складність estimation",
    )
    estimation_points: int = Field(
        ge=1,
        le=13,
        description=(
            "Підтверджені Fibonacci points: "
            "1, 2, 3, 5, 8 або 13"
        ),
    )
    target_team: str = Field(
        min_length=2,
        max_length=100,
        description="Команда, яка має виконати estimation",
    )
    requested_by: str = Field(
        min_length=2,
        max_length=100,
        description="Особа або роль, яка підтвердила запит",
    )
    estimation_summary: str = Field(
        min_length=20,
        max_length=1000,
        description=(
            "Підсумок scope, залежностей і очікуваного estimation"
        ),
    )

    @field_validator("initiative_id", mode="before")
    @classmethod
    def validate_initiative_id(cls, value: object) -> str:
        """Перевіряє initiative ID."""

        return normalize_initiative_id(value)

    @field_validator(
        "target_team",
        "requested_by",
        "estimation_summary",
        mode="before",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: object,
    ) -> str:
        """Перевіряє обов'язкові текстові поля."""

        if not isinstance(value, str):
            raise ValueError(
                "Обов'язкове поле має бути текстом."
            )

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "Обов'язкове поле не може бути порожнім."
            )

        return normalized_value

    @field_validator(
        "estimation_complexity",
        mode="before",
    )
    @classmethod
    def normalize_complexity(
        cls,
        value: object,
    ) -> str:
        """Нормалізує назву рівня складності."""

        if not isinstance(value, str):
            raise ValueError(
                "estimation_complexity має бути текстом."
            )

        return value.strip().title()

    @model_validator(mode="after")
    def validate_complexity_points_consistency(
        self,
    ) -> Self:
        """Перевіряє відповідність complexity та points."""

        allowed_points = COMPLEXITY_POINTS[
            self.estimation_complexity
        ]

        if self.estimation_points not in allowed_points:
            raise ValueError(
                "Estimation Points не відповідають "
                "обраному рівню complexity."
            )

        return self


@tool(args_schema=CheckRequirementsReadinessInput)
def check_requirements_readiness(
    initiative_id: str,
    business_objective: str | None = None,
    functional_requirements: str | None = None,
    non_functional_requirements: str | None = None,
    acceptance_criteria: str | None = None,
    integration_scope: str | None = None,
    data_requirements: str | None = None,
) -> str:
    """Перевірити готовність requirements до estimation handover."""

    values = {
        "business_objective": business_objective,
        "functional_requirements": functional_requirements,
        "non_functional_requirements": (
            non_functional_requirements
        ),
        "acceptance_criteria": acceptance_criteria,
        "integration_scope": integration_scope,
        "data_requirements": data_requirements,
    }

    missing_fields = [
        field_name
        for field_name in REQUIRED_REQUIREMENTS_FIELDS
        if not values[field_name]
    ]
    completed_fields = (
        len(REQUIRED_REQUIREMENTS_FIELDS)
        - len(missing_fields)
    )
    readiness_percent = round(
        completed_fields
        / len(REQUIRED_REQUIREMENTS_FIELDS)
        * 100
    )

    return build_tool_response(
        status="success",
        data={
            "initiative_id": initiative_id,
            "is_ready": not missing_fields,
            "readiness_percent": readiness_percent,
            "completed_fields": completed_fields,
            "total_fields": len(
                REQUIRED_REQUIREMENTS_FIELDS
            ),
            "missing_fields": missing_fields,
            "requires_human_input": bool(missing_fields),
        },
    )


@tool(args_schema=ClassifyEstimationComplexityInput)
def classify_estimation_complexity(
    initiative_id: str,
    systems_count: int,
    integration_count: int,
    nfr_criticality: str,
    data_migration_required: bool,
    security_review_required: bool,
    dependency_count: int,
    requirements_stability: str,
) -> str:
    """Розрахувати complexity та Fibonacci points для estimation."""

    if systems_count == 1:
        systems_score = 0
    elif systems_count <= 3:
        systems_score = 1
    else:
        systems_score = 2

    if integration_count == 0:
        integrations_score = 0
    elif integration_count <= 2:
        integrations_score = 1
    else:
        integrations_score = 2

    nfr_score = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }[nfr_criticality]

    data_migration_score = int(data_migration_required)
    security_score = int(security_review_required)

    if dependency_count == 0:
        dependencies_score = 0
    elif dependency_count <= 2:
        dependencies_score = 1
    else:
        dependencies_score = 2

    stability_score = {
        "high": 0,
        "partial": 1,
        "low": 2,
    }[requirements_stability]

    scoring_breakdown = {
        "systems": systems_score,
        "integrations": integrations_score,
        "nfr_criticality": nfr_score,
        "data_migration": data_migration_score,
        "security_review": security_score,
        "dependencies": dependencies_score,
        "requirements_stability": stability_score,
    }

    raw_score = sum(scoring_breakdown.values())

    if raw_score <= 1:
        estimation_points = 1
    elif raw_score <= 3:
        estimation_points = 2
    elif raw_score <= 5:
        estimation_points = 3
    elif raw_score <= 7:
        estimation_points = 5
    elif raw_score <= 9:
        estimation_points = 8
    else:
        estimation_points = 13

    if estimation_points <= 2:
        estimation_complexity = "Low"
    elif estimation_points <= 5:
        estimation_complexity = "Medium"
    else:
        estimation_complexity = "High"

    return build_tool_response(
        status="success",
        data={
            "initiative_id": initiative_id,
            "raw_score": raw_score,
            "estimation_points": estimation_points,
            "estimation_complexity": estimation_complexity,
            "scoring_breakdown": scoring_breakdown,
            "requires_human_confirmation": True,
        },
    )


@tool(args_schema=IdentifyHandoverGapsInput)
def identify_handover_gaps(
    initiative_id: str,
    solution_scope_defined: bool,
    dependencies_confirmed: bool,
    nfr_reviewed: bool,
    acceptance_criteria_testable: bool,
    data_owners_confirmed: bool,
    security_classification_completed: bool,
    known_blockers: list[str] | None = None,
) -> str:
    """Знайти прогалини, що блокують estimation handover."""

    checks = {
        "solution_scope_defined": solution_scope_defined,
        "dependencies_confirmed": dependencies_confirmed,
        "nfr_reviewed": nfr_reviewed,
        "acceptance_criteria_testable": (
            acceptance_criteria_testable
        ),
        "data_owners_confirmed": data_owners_confirmed,
        "security_classification_completed": (
            security_classification_completed
        ),
    }

    missing_controls = [
        control_name
        for control_name, is_complete in checks.items()
        if not is_complete
    ]
    blockers = known_blockers or []

    return build_tool_response(
        status="success",
        data={
            "initiative_id": initiative_id,
            "can_proceed_to_estimation": (
                not missing_controls and not blockers
            ),
            "missing_controls": missing_controls,
            "known_blockers": blockers,
            "gaps_count": (
                len(missing_controls) + len(blockers)
            ),
            "requires_human_input": bool(
                missing_controls or blockers
            ),
        },
    )


@tool(args_schema=SubmitEstimationRequestInput)
def submit_estimation_request(
    initiative_id: str,
    estimation_complexity: str,
    estimation_points: int,
    target_team: str,
    requested_by: str,
    estimation_summary: str,
) -> str:
    """Відправити підтверджений estimation request.

    Це ризикова операція: вона змінює стан ініціативи й повинна
    виконуватися лише після явного HITL-підтвердження.
    """

    try:
        if SUBMISSIONS_PATH.exists():
            raw_content = SUBMISSIONS_PATH.read_text(
                encoding="utf-8"
            )
            submissions = json.loads(raw_content)

            if not isinstance(submissions, list):
                return build_tool_response(
                    status="error",
                    error=(
                        "Файл submissions має некоректний формат."
                    ),
                )
        else:
            submissions = []

        duplicate = next(
            (
                submission
                for submission in submissions
                if submission.get("initiative_id")
                == initiative_id
                and submission.get("status") == "submitted"
            ),
            None,
        )

        if duplicate is not None:
            return build_tool_response(
                status="error",
                error=(
                    f"Estimation request для {initiative_id} "
                    "вже відправлено."
                ),
            )

        submission = {
            "request_id": (
                f"EST-{len(submissions) + 1:03d}"
            ),
            "initiative_id": initiative_id,
            "estimation_complexity": estimation_complexity,
            "estimation_points": estimation_points,
            "target_team": target_team,
            "requested_by": requested_by,
            "estimation_summary": estimation_summary,
            "status": "submitted",
            "submitted_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        submissions.append(submission)

        SUBMISSIONS_PATH.write_text(
            json.dumps(
                submissions,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return build_tool_response(
            status="success",
            data=submission,
        )

    except (OSError, json.JSONDecodeError) as error:
        return build_tool_response(
            status="error",
            error=(
                "Не вдалося зберегти estimation request: "
                f"{error}"
            ),
        )


DOMAIN_TOOLS = [
    check_requirements_readiness,
    classify_estimation_complexity,
    identify_handover_gaps,
    submit_estimation_request,
]