"""Тести Agentic RAG та ChromaDB knowledge base."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import knowledge as knowledge_module
from knowledge import (
    KNOWLEDGE_DOCUMENTS,
    SearchDeliveryKnowledgeInput,
    get_knowledge_collection,
    initialize_knowledge_base,
    search_delivery_knowledge,
)


@pytest.fixture()
def isolated_collection(tmp_path):
    """Створює ізольовану ChromaDB для одного тесту."""

    database_path = tmp_path / "chroma_db"
    metadata = initialize_knowledge_base(
        database_path
    )
    collection = get_knowledge_collection(
        database_path
    )

    return collection, metadata


def test_knowledge_base_contains_required_documents(
    isolated_collection,
) -> None:
    """Knowledge base містить більше восьми документів."""

    collection, metadata = isolated_collection

    assert len(KNOWLEDGE_DOCUMENTS) == 12
    assert len(KNOWLEDGE_DOCUMENTS) >= 8
    assert metadata["documents_count"] == 12
    assert collection.count() == 12


def test_knowledge_documents_have_required_quality() -> None:
    """Кожен документ має ID, metadata та кілька речень."""

    document_ids = set()

    for document in KNOWLEDGE_DOCUMENTS:
        assert document["id"] not in document_ids
        document_ids.add(document["id"])

        assert document["title"].strip()
        assert document["source"].strip()
        assert len(document["content"]) >= 100
        assert document["content"].count(".") >= 2


def test_search_returns_top_three_documents(
    isolated_collection,
    monkeypatch,
) -> None:
    """RAG-tool повертає рівно top-3 результати."""

    collection, _ = isolated_collection

    monkeypatch.setattr(
        knowledge_module,
        "get_knowledge_collection",
        lambda *args, **kwargs: collection,
    )

    response = json.loads(
        search_delivery_knowledge.invoke(
            {
                "query": (
                    "security review and classification "
                    "before estimation"
                ),
            }
        )
    )

    assert response["status"] == "success"
    assert response["data"]["results_count"] == 3
    assert len(response["data"]["documents"]) == 3


def test_search_result_contains_metadata(
    isolated_collection,
    monkeypatch,
) -> None:
    """Кожен знайдений документ містить source та title."""

    collection, _ = isolated_collection

    monkeypatch.setattr(
        knowledge_module,
        "get_knowledge_collection",
        lambda *args, **kwargs: collection,
    )

    response = json.loads(
        search_delivery_knowledge.invoke(
            {
                "query": (
                    "Fibonacci estimation complexity points"
                ),
            }
        )
    )

    first_document = response["data"]["documents"][0]

    assert first_document["id"].startswith("kb-")
    assert first_document["title"]
    assert first_document["source"]
    assert first_document["content"]
    assert isinstance(
        first_document["distance"],
        float,
    )


def test_search_schema_normalizes_query() -> None:
    """Search schema прибирає зайві пробіли."""

    data = SearchDeliveryKnowledgeInput(
        query="  Definition of Ready  ",
    )

    assert data.query == "Definition of Ready"


def test_search_schema_rejects_blank_query() -> None:
    """Порожній search query не проходить validation."""

    with pytest.raises(ValidationError):
        SearchDeliveryKnowledgeInput(
            query="   ",
        )