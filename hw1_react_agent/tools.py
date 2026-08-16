"""Інструменти ReAct-агента для роботи з demand-ініціативами."""

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Стандартна відповідь кожного інструмента."""

    status: Literal["success", "error"]
    data: dict[str, Any] | None = None
    error: str | None = None


class GetInitiativeStatusInput(BaseModel):
    """Вхідні дані для пошуку ініціативи."""

    initiative_id: str = Field(
        description="Ідентифікатор ініціативи, наприклад DEM-001"
    )


# Синтетичні дані для навчального проєкту.
INITIATIVES = {
    "DEM-001": {
        "title": "AI assistant for customer support",
        "status": "Discovery",
        "assignee": "Lead Business Analyst",
        "last_update": "2026-08-14",
    },
    "DEM-002": {
        "title": "Automated regulatory reporting",
        "status": "On Hold",
        "assignee": "Data Analyst",
        "last_update": "2026-08-10",
    },
    "DEM-003": {
        "title": "Mobile onboarding redesign",
        "status": "QBR Ready",
        "assignee": "Product Owner",
        "last_update": "2026-08-15",
    },
}


@tool(args_schema=GetInitiativeStatusInput)
def get_initiative_status(initiative_id: str) -> str:
    """Повертає поточний статус demand-ініціативи за її ID."""

    normalized_id = initiative_id.strip().upper()
    initiative = INITIATIVES.get(normalized_id)

    if initiative is None:
        return ToolResult(
            status="error",
            error=f"Ініціативу {normalized_id} не знайдено.",
        ).model_dump_json()

    return ToolResult(
        status="success",
        data={
            "initiative_id": normalized_id,
            **initiative,
        },
    ).model_dump_json()

class CheckIntakeCompletenessInput(BaseModel):
    """Вхідні дані для перевірки повноти intake."""

    initiative_id: str = Field(description="Ідентифікатор ініціативи")
    business_owner: str | None = Field(
        default=None,
        description="Власник ініціативи",
    )
    business_driver: str | None = Field(
        default=None,
        description="Бізнес-проблема або причина створення ініціативи",
    )
    success_metrics: str | None = Field(
        default=None,
        description="Очікувані вимірювані результати",
    )
    financial_effect: str | None = Field(
        default=None,
        description="Очікуваний фінансовий ефект",
    )
    constraints: str | None = Field(
        default=None,
        description="Відомі обмеження або явна вказівка, що їх немає",
    )


@tool(args_schema=CheckIntakeCompletenessInput)
def check_intake_completeness(
    initiative_id: str,
    business_owner: str | None = None,
    business_driver: str | None = None,
    success_metrics: str | None = None,
    financial_effect: str | None = None,
    constraints: str | None = None,
) -> str:
    """Перевіряє, чи заповнені обов'язкові поля intake для Gate 0."""

    fields = {
        "business_owner": business_owner,
        "business_driver": business_driver,
        "success_metrics": success_metrics,
        "financial_effect": financial_effect,
        "constraints": constraints,
    }

    missing_fields = [
        field_name
        for field_name, value in fields.items()
        if value is None or not value.strip()
    ]

    completed_count = len(fields) - len(missing_fields)
    completeness_percent = round(completed_count / len(fields) * 100)

    return ToolResult(
        status="success",
        data={
            "initiative_id": initiative_id.strip().upper(),
            "is_complete": not missing_fields,
            "completeness_percent": completeness_percent,
            "missing_fields": missing_fields,
        },
    ).model_dump_json()

class ClassifyDiscoveryScopeInput(BaseModel):
    """Фактори для оцінювання складності discovery."""

    initiative_id: str = Field(description="Ідентифікатор ініціативи")
    systems_count: int = Field(
        ge=1,
        description="Кількість систем, яких стосується ініціатива",
    )
    ownership_clarity: Literal["clear", "partial", "unclear"]
    technical_uncertainty: Literal["low", "medium", "high"]
    dependency_count: int = Field(
        ge=0,
        description="Кількість зовнішніх командних залежностей",
    )
    regulatory_impact: Literal["none", "possible", "confirmed"]
    data_readiness: Literal["ready", "partial", "unknown", "unavailable"]


@tool(args_schema=ClassifyDiscoveryScopeInput)
def classify_discovery_scope(
    initiative_id: str,
    systems_count: int,
    ownership_clarity: Literal["clear", "partial", "unclear"],
    technical_uncertainty: Literal["low", "medium", "high"],
    dependency_count: int,
    regulatory_impact: Literal["none", "possible", "confirmed"],
    data_readiness: Literal["ready", "partial", "unknown", "unavailable"],
) -> str:
    """Розраховує Discovery Points і рекомендує Light, Standard або Deep."""

    systems_score = (
        0 if systems_count == 1
        else 1 if systems_count == 2
        else 2 if systems_count <= 4
        else 3
    )

    ownership_score = {
        "clear": 0,
        "partial": 1,
        "unclear": 2,
    }[ownership_clarity]

    uncertainty_score = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }[technical_uncertainty]

    dependency_score = (
        0 if dependency_count == 0
        else 1 if dependency_count <= 2
        else 2
    )

    regulatory_score = {
        "none": 0,
        "possible": 1,
        "confirmed": 2,
    }[regulatory_impact]

    data_score = {
        "ready": 0,
        "partial": 1,
        "unknown": 2,
        "unavailable": 3,
    }[data_readiness]

    scoring_breakdown = {
        "systems": systems_score,
        "ownership": ownership_score,
        "technical_uncertainty": uncertainty_score,
        "dependencies": dependency_score,
        "regulatory_impact": regulatory_score,
        "data_readiness": data_score,
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

    # Три та більше систем завжди потребують щонайменше Standard discovery.
    if systems_count >= 3 and discovery_points < 3:
        discovery_points = 3

    if discovery_points <= 2:
        recommended_scope = "Light"
    elif discovery_points <= 5:
        recommended_scope = "Standard"
    else:
        recommended_scope = "Deep"

    return ToolResult(
        status="success",
        data={
            "initiative_id": initiative_id.strip().upper(),
            "discovery_points": discovery_points,
            "recommended_scope": recommended_scope,
            "raw_score": raw_score,
            "scoring_breakdown": scoring_breakdown,
            "requires_lba_confirmation": True,
        },
    ).model_dump_json()

class CalculatePriorityScoreInput(BaseModel):
    """Оцінки критеріїв, підтверджені людиною."""

    initiative_id: str = Field(description="Ідентифікатор ініціативи")
    strategic_alignment: int = Field(ge=1, le=5)
    customer_impact: int = Field(ge=1, le=5)
    financial_impact: int = Field(ge=1, le=5)
    regulatory_urgency: int = Field(ge=1, le=5)
    implementation_feasibility: int = Field(ge=1, le=5)


@tool(args_schema=CalculatePriorityScoreInput)
def calculate_priority_score(
    initiative_id: str,
    strategic_alignment: int,
    customer_impact: int,
    financial_impact: int,
    regulatory_urgency: int,
    implementation_feasibility: int,
) -> str:
    """Розраховує пріоритет за оцінками критеріїв, заданими людиною."""

    weights = {
        "strategic_alignment": 0.30,
        "customer_impact": 0.25,
        "financial_impact": 0.20,
        "regulatory_urgency": 0.15,
        "implementation_feasibility": 0.10,
    }

    scores = {
        "strategic_alignment": strategic_alignment,
        "customer_impact": customer_impact,
        "financial_impact": financial_impact,
        "regulatory_urgency": regulatory_urgency,
        "implementation_feasibility": implementation_feasibility,
    }

    priority_score = round(
        sum(scores[criterion] * weight for criterion, weight in weights.items()),
        2,
    )

    if priority_score >= 4:
        priority_level = "High"
    elif priority_score >= 2.5:
        priority_level = "Medium"
    else:
        priority_level = "Low"

    return ToolResult(
        status="success",
        data={
            "initiative_id": initiative_id.strip().upper(),
            "priority_score": priority_score,
            "priority_level": priority_level,
            "scores": scores,
            "weights": weights,
            "requires_human_validation": True,
        },
    ).model_dump_json()