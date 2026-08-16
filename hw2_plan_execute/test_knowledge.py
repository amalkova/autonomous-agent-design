"""Unit-тести ChromaDB knowledge base та RAG tool."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import knowledge
from knowledge import (
    KNOWLEDGE_DOCUMENTS,
    SearchKnowledgeInput,
    initialize_knowledge_base,
    search_knowledge,
)


@pytest.fixture(scope="module", autouse=True)
def isolated_knowledge_base(
    tmp_path_factory,
):
    """Створює окрему ChromaDB для тестового модуля."""

    original_path = knowledge.CHROMA_PATH
    test_path: Path = tmp_path_factory.mktemp("chroma_test")

    knowledge.CHROMA_PATH = test_path
    initialize_knowledge_base()

    yield

    knowledge.CHROMA_PATH = original_path


def test_knowledge_base_contains_at_least_eight_documents() -> None:
    """Knowledge base відповідає мінімальній вимозі ДЗ."""

    metadata = initialize_knowledge_base()

    assert len(KNOWLEDGE_DOCUMENTS) >= 8
    assert metadata["documents_count"] == 12


def test_search_knowledge_returns_top_three_documents() -> None:
    """RAG tool повертає три семантично релевантні документи."""

    response = search_knowledge.invoke(
        {
            "query": (
                "Який timebox та результати потрібні "
                "для Deep discovery?"
            )
        }
    )

    result = json.loads(response)
    documents = result["data"]["documents"]

    assert result["status"] == "success"
    assert result["data"]["results_count"] == 3
    assert len(documents) == 3
    assert any(
        document["title"] == "Deep discovery"
        for document in documents
    )


def test_search_results_have_required_metadata() -> None:
    """Кожен результат містить source, content та distance."""

    response = search_knowledge.invoke(
        {
            "query": (
                "Які обов'язкові поля потрібні для Gate 0?"
            )
        }
    )

    result = json.loads(response)

    for document in result["data"]["documents"]:
        assert document["id"]
        assert document["title"]
        assert document["source"]
        assert document["content"]
        assert isinstance(document["distance"], float)


def test_search_query_is_normalized() -> None:
    """Пошуковий запит очищується від зайвих пробілів."""

    data = SearchKnowledgeInput(
        query="  правила QBR readiness  "
    )

    assert data.query == "правила QBR readiness"


def test_empty_search_query_is_rejected() -> None:
    """Порожній пошуковий запит відхиляється Pydantic."""

    with pytest.raises(
        ValidationError,
        match="не може бути порожнім",
    ):
        SearchKnowledgeInput(query="   ")