"""ChromaDB knowledge base для requirements та estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from tools_legacy import build_tool_response


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "requirements_estimation_knowledge"


KNOWLEDGE_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "kb-001",
        "title": "Definition of Ready for estimation",
        "source": "Estimation Readiness Standard",
        "content": (
            "Ініціатива готова до estimation, коли визначено "
            "business objective, functional requirements, NFR, "
            "acceptance criteria, integration scope та data "
            "requirements. Definition of Ready зменшує ризик "
            "оцінювання невизначеного scope. Неповні обов'язкові "
            "поля потрібно повернути на уточнення."
        ),
    },
    {
        "id": "kb-002",
        "title": "Business objective and solution scope",
        "source": "Requirements Quality Guide",
        "content": (
            "Business objective має описувати вимірювану зміну, "
            "а не конкретну технологію. Solution scope визначає, "
            "які процеси, канали та системи входять або не входять "
            "до реалізації. Нечіткий scope збільшує estimation "
            "uncertainty."
        ),
    },
    {
        "id": "kb-003",
        "title": "Functional requirements quality",
        "source": "Business Analysis Handbook",
        "content": (
            "Functional requirements описують поведінку системи "
            "та очікувані користувацькі сценарії. Кожна вимога має "
            "бути однозначною, перевірюваною та пов'язаною з "
            "business objective. Дублікати й суперечності потрібно "
            "усунути до handover."
        ),
    },
    {
        "id": "kb-004",
        "title": "Non-functional requirements",
        "source": "NFR Review Checklist",
        "content": (
            "NFR охоплюють performance, availability, security, "
            "scalability, auditability та supportability. Для "
            "критичних сервісів мають бути визначені числові "
            "пороги, наприклад latency або availability. "
            "Невизначені NFR можуть суттєво змінити оцінку."
        ),
    },
    {
        "id": "kb-005",
        "title": "Testable acceptance criteria",
        "source": "Delivery Handover Guide",
        "content": (
            "Acceptance criteria мають описувати спостережуваний "
            "результат і бути придатними для перевірки QA. "
            "Формулювання на кшталт «працює швидко» не є "
            "тестованими без конкретної метрики. Неоднозначні "
            "criteria потрібно уточнити до estimation."
        ),
    },
    {
        "id": "kb-006",
        "title": "Integration scope",
        "source": "Solution Design Standard",
        "content": (
            "Для кожної інтеграції потрібно визначити source, "
            "target, protocol, data contract та owner. Зовнішні "
            "API й vendor dependencies мають бути підтверджені "
            "до estimation. Невідомий integration contract є "
            "окремим ризиком."
        ),
    },
    {
        "id": "kb-007",
        "title": "Data readiness and migration",
        "source": "Data Readiness Policy",
        "content": (
            "Data requirements мають визначати джерела, quality "
            "expectations, retention та data owner. Якщо потрібна "
            "міграція, команда оцінює volume, mapping, cleansing "
            "і reconciliation. Відсутність підтвердженого data "
            "owner блокує фінальний handover."
        ),
    },
    {
        "id": "kb-008",
        "title": "Security classification",
        "source": "Security Review Policy",
        "content": (
            "Security classification визначає чутливість даних "
            "та необхідні controls. Ініціативи з PII, payment data "
            "або privileged access потребують security review. "
            "Невиконана classification має бути позначена як "
            "handover gap."
        ),
    },
    {
        "id": "kb-009",
        "title": "Dependency management",
        "source": "Delivery Dependency Guide",
        "content": (
            "Залежність вважається підтвердженою, коли відомі "
            "owner, deliverable та очікувана дата. Непідтверджені "
            "cross-team dependencies підвищують complexity і "
            "можуть блокувати estimation. Відомі blockers мають "
            "бути явно зафіксовані."
        ),
    },
    {
        "id": "kb-010",
        "title": "Estimation complexity and points",
        "source": "Estimation Sizing Rules",
        "content": (
            "Estimation Points використовують Fibonacci scale: "
            "1, 2, 3, 5, 8 або 13. Low complexity відповідає "
            "1–2 points, Medium — 3–5, High — 8–13. На оцінку "
            "впливають системи, інтеграції, NFR, migration, "
            "security, dependencies та stability requirements."
        ),
    },
    {
        "id": "kb-011",
        "title": "Estimation handover process",
        "source": "Portfolio Delivery Process",
        "content": (
            "Перед handover Lead Business Analyst перевіряє "
            "readiness, gaps і підтверджує estimation complexity. "
            "Пакет передається target delivery team лише після "
            "усунення критичних blockers. Команда estimation "
            "повертає оцінку або запит на clarification."
        ),
    },
    {
        "id": "kb-012",
        "title": "Human approval for submission",
        "source": "Estimation Governance Policy",
        "content": (
            "Submit estimation request є ризиковою операцією, "
            "оскільки змінює workflow state ініціативи. Перед "
            "відправкою система показує initiative ID, team, "
            "complexity, points і summary. Операція виконується "
            "лише після approve або підтвердженого edit."
        ),
    },
]


class SearchDeliveryKnowledgeInput(BaseModel):
    """Вхідні дані semantic search."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "Пошуковий запит про requirements, "
            "handover або estimation"
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> str:
        """Перевіряє та нормалізує пошуковий запит."""

        if not isinstance(value, str):
            raise ValueError(
                "query має бути текстовим значенням."
            )

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "query не може бути порожнім."
            )

        return normalized_value


def get_knowledge_collection(
    path: Path = CHROMA_PATH,
) -> Collection:
    """Повертає persistent ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=str(path)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Requirements and estimation readiness knowledge"
            ),
        },
    )


def initialize_knowledge_base(
    path: Path = CHROMA_PATH,
) -> dict[str, Any]:
    """Створює або оновлює доменну knowledge base."""

    collection = get_knowledge_collection(path)

    collection.upsert(
        ids=[
            document["id"]
            for document in KNOWLEDGE_DOCUMENTS
        ],
        documents=[
            document["content"]
            for document in KNOWLEDGE_DOCUMENTS
        ],
        metadatas=[
            {
                "title": document["title"],
                "source": document["source"],
            }
            for document in KNOWLEDGE_DOCUMENTS
        ],
    )

    return {
        "collection": COLLECTION_NAME,
        "documents_count": collection.count(),
        "path": str(path),
    }


@tool(args_schema=SearchDeliveryKnowledgeInput)
def search_delivery_knowledge(query: str) -> str:
    """Знайти top-3 правила про requirements та estimation."""

    try:
        collection = get_knowledge_collection()

        if collection.count() == 0:
            initialize_knowledge_base()
            collection = get_knowledge_collection()

        results = collection.query(
            query_texts=[query],
            n_results=min(3, collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get(
            "documents",
            [[]],
        )[0]
        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]
        distances = results.get(
            "distances",
            [[]],
        )[0]

        matches = []

        for (
            document_id,
            content,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            matches.append(
                {
                    "id": document_id,
                    "title": metadata.get(
                        "title",
                        "",
                    ),
                    "source": metadata.get(
                        "source",
                        "",
                    ),
                    "content": content,
                    "distance": round(
                        float(distance),
                        4,
                    ),
                }
            )

        return build_tool_response(
            status="success",
            data={
                "query": query,
                "results_count": len(matches),
                "documents": matches,
            },
        )

    except Exception as error:
        return build_tool_response(
            status="error",
            error=(
                "Не вдалося виконати knowledge search: "
                f"{error}"
            ),
        )


if __name__ == "__main__":
    print(
        json.dumps(
            initialize_knowledge_base(),
            ensure_ascii=False,
            indent=2,
        )
    )