"""ChromaDB knowledge base та Agentic RAG tool."""

import json
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from tools import build_tool_response


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "demand_discovery_knowledge"

KNOWLEDGE_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "kb-001",
        "title": "Gate 0: мінімальний intake",
        "source": "Demand and Discovery Policy",
        "content": (
            "Gate 0 визначає мінімальні дані, потрібні для прийняття "
            "ініціативи в роботу. Intake має містити business owner, "
            "business driver, success metrics, financial effect та "
            "відомі constraints. Якщо обов'язкові поля відсутні, "
            "аналітик повертає запит на уточнення."
        ),
    },
    {
        "id": "kb-002",
        "title": "Первинний screening",
        "source": "Demand Intake SLA",
        "content": (
            "Duty analyst виконує первинний screening нової "
            "ініціативи протягом двох робочих днів. Під час screening "
            "перевіряються повнота intake, дублікати, домен та "
            "необхідність залучення Lead Business Analyst."
        ),
    },
    {
        "id": "kb-003",
        "title": "Коли потрібен discovery",
        "source": "Discovery Entry Rules",
        "content": (
            "Повний discovery зазвичай потрібен для ініціатив із "
            "трьома або більше системами, нечітким ownership або "
            "заблокованою технічною оцінкою. Локальна зміна в одній "
            "системі або зміна у двох системах із чітким ownership "
            "може не потребувати повного discovery."
        ),
    },
    {
        "id": "kb-004",
        "title": "Light discovery",
        "source": "Discovery Scope Guide",
        "content": (
            "Light discovery використовується для локальних і добре "
            "зрозумілих змін із низькою технічною невизначеністю. "
            "Орієнтовний timebox становить від двох до п'яти робочих "
            "днів. Результатом є стислий scope та перелік ключових "
            "вимог і залежностей."
        ),
    },
    {
        "id": "kb-005",
        "title": "Standard discovery",
        "source": "Discovery Scope Guide",
        "content": (
            "Standard discovery підходить для міжсистемних ініціатив "
            "із помірною складністю або частково визначеним "
            "ownership. Орієнтовний timebox становить від трьох до "
            "шести тижнів. Команда деталізує вимоги, інтеграції, "
            "NFR та попередній solution design."
        ),
    },
    {
        "id": "kb-006",
        "title": "Deep discovery",
        "source": "Discovery Scope Guide",
        "content": (
            "Deep discovery потрібен для стратегічних, регуляторних "
            "або технічно невизначених ініціатив із багатьма "
            "системами та залежностями. Орієнтовний timebox становить "
            "від шести до десяти тижнів. Результат має включати "
            "детальний scope, solution options, ризики, NFR та "
            "план подальшої оцінки."
        ),
    },
    {
        "id": "kb-007",
        "title": "Discovery Points",
        "source": "Discovery Estimation Rules",
        "content": (
            "Discovery Points розраховуються за шкалою Fibonacci: "
            "1, 2, 3, 5, 8 або 13. На оцінку впливають кількість "
            "систем, ownership clarity, technical uncertainty, "
            "залежності, regulatory impact та data readiness. "
            "Значення 1-2 відповідають Light, 3-5 — Standard, "
            "а 8-13 — Deep."
        ),
    },
    {
        "id": "kb-008",
        "title": "Priority assessment",
        "source": "Priority Assessment Policy",
        "content": (
            "Priority score розраховується лише за оцінками, "
            "підтвердженими людиною. Агент не повинен самостійно "
            "вигадувати strategic alignment, customer impact, "
            "financial impact, regulatory urgency або implementation "
            "feasibility. Результат потребує human validation."
        ),
    },
    {
        "id": "kb-009",
        "title": "QBR readiness",
        "source": "QBR Preparation Guide",
        "content": (
            "Ініціатива може бути позначена QBR Ready після "
            "завершення необхідного discovery та підготовки даних "
            "для пріоритизації. Повинні бути зрозумілі scope, "
            "очікуваний ефект, ключові ризики, залежності та "
            "попередня оцінка реалізації."
        ),
    },
    {
        "id": "kb-010",
        "title": "Роль Lead Business Analyst",
        "source": "Discovery Operating Model",
        "content": (
            "Lead Business Analyst відповідає за організацію "
            "discovery та узгодження бізнесових і системних вимог. "
            "LBA координує взаємодію між бізнесом, архітекторами, "
            "розробкою, QA та іншими залежними командами. "
            "Рекомендації агента не замінюють рішення LBA."
        ),
    },
    {
        "id": "kb-011",
        "title": "Hold та очікування інформації",
        "source": "Demand Status Guide",
        "content": (
            "Ініціатива переводиться на Hold, якщо для продовження "
            "потрібна інформація від requester або результат іншої "
            "команди. Для Hold обов'язково зазначаються причина, "
            "відповідальна сторона, коментар та refinement date. "
            "Статус має однозначно показувати, хто повинен діяти."
        ),
    },
    {
        "id": "kb-012",
        "title": "Підтвердження assessment",
        "source": "Discovery Governance Policy",
        "content": (
            "Фінальна відправка Discovery assessment є ризиковою "
            "дією, оскільки змінює стан ініціативи та фіксує "
            "рекомендований scope. Перед відправкою система повинна "
            "показати initiative ID, Discovery Points, scope та "
            "decision summary. Дія виконується лише після явного "
            "підтвердження людиною."
        ),
    },
]


class SearchKnowledgeInput(BaseModel):
    """Вхідні дані для пошуку у knowledge base."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "Запит для пошуку правил, політик або рекомендацій "
            "у базі знань Demand and Discovery"
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> str:
        """Перевіряє, що пошуковий запит є непорожнім текстом."""

        if not isinstance(value, str):
            raise ValueError(
                "query має бути текстовим значенням."
            )

        normalized_query = value.strip()

        if not normalized_query:
            raise ValueError("query не може бути порожнім.")

        return normalized_query


def get_knowledge_collection() -> Any:
    """Створює persistent ChromaDB collection та завантажує документи."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Demand and Discovery policies and operating rules"
            )
        },
    )

    existing_ids = set(collection.get()["ids"])

    missing_documents = [
        document
        for document in KNOWLEDGE_DOCUMENTS
        if document["id"] not in existing_ids
    ]

    if missing_documents:
        collection.add(
            ids=[
                document["id"]
                for document in missing_documents
            ],
            documents=[
                document["content"]
                for document in missing_documents
            ],
            metadatas=[
                {
                    "title": document["title"],
                    "source": document["source"],
                }
                for document in missing_documents
            ],
        )

    return collection


def initialize_knowledge_base() -> dict[str, Any]:
    """Ініціалізує knowledge base та повертає її метадані."""

    collection = get_knowledge_collection()

    return {
        "collection": COLLECTION_NAME,
        "documents_count": collection.count(),
        "path": str(CHROMA_PATH),
    }


@tool(args_schema=SearchKnowledgeInput)
def search_knowledge(query: str) -> str:
    """Знайти правила та довідкову інформацію у knowledge base.

    Використовуйте цей tool, коли потрібно знайти політику,
    timebox, правило процесу, роль, критерій або рекомендацію
    з Demand and Discovery. Не використовуйте його для виконання
    дій, математичних розрахунків або фінальної відправки assessment.
    """

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

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    result_ids = results.get("ids", [[]])[0]

    relevant_documents = []

    for document_id, content, metadata, distance in zip(
        result_ids,
        documents,
        metadatas,
        distances,
        strict=True,
    ):
        relevant_documents.append(
            {
                "id": document_id,
                "title": metadata.get("title"),
                "source": metadata.get("source"),
                "content": content,
                "distance": round(float(distance), 4),
            }
        )

    return build_tool_response(
        status="success",
        data={
            "query": query,
            "results_count": len(relevant_documents),
            "documents": relevant_documents,
        },
    )


if __name__ == "__main__":
    print(
        json.dumps(
            initialize_knowledge_base(),
            ensure_ascii=False,
            indent=2,
        )
    )