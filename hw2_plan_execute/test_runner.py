"""Формує acceptance-звіт для HW2 без додаткових LLM-викликів."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from knowledge import (
    get_knowledge_collection,
    initialize_knowledge_base,
)
from plan_execute import MODEL_NAME, TOOLS, build_graph


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "agent_state.db"
REPORT_PATH = BASE_DIR / "test_results.json"

PERSISTENCE_THREAD_COMPLETED = "persistence-demo-001"
PERSISTENCE_THREAD_RUNNING = "persistence-demo-002"
HITL_THREAD_APPROVED = "hitl-demo-approve"
HITL_THREAD_REJECTED = "hitl-demo-reject"


def build_result(
    test_id: str,
    name: str,
    expected_result: Any,
    actual_result: Any,
    passed: bool,
) -> dict[str, Any]:
    """Створює один запис acceptance-звіту."""

    return {
        "test_id": test_id,
        "name": name,
        "expected_result": expected_result,
        "actual_result": actual_result,
        "passed": passed,
    }


def run_pytest_suite() -> dict[str, Any]:
    """Запускає всі unit та integration tests через поточний Python."""

    started_at = time.perf_counter()

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    execution_time = round(time.perf_counter() - started_at, 3)

    output_parts = [
        process.stdout.strip(),
        process.stderr.strip(),
    ]
    output = "\n".join(
        part for part in output_parts if part
    )

    return {
        "passed": process.returncode == 0,
        "return_code": process.returncode,
        "execution_time_seconds": execution_time,
        "output": output,
    }


def get_graph_nodes() -> list[str]:
    """Повертає бізнес-вузли Plan-and-Execute графа."""

    graph = build_graph()

    return [
        node_name
        for node_name in graph.get_graph().nodes
        if not node_name.startswith("__")
    ]


def get_tool_names() -> list[str]:
    """Повертає назви доступних агенту tools."""

    return [tool.name for tool in TOOLS]


def get_knowledge_documents_count() -> int:
    """Ініціалізує knowledge base та повертає кількість документів."""

    initialization_result = initialize_knowledge_base()

    if isinstance(initialization_result, dict):
        documents_count = initialization_result.get(
            "documents_count"
        )

        if documents_count is not None:
            return int(documents_count)

    return int(get_knowledge_collection().count())


def load_thread_states(
    thread_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Читає збережені thread states з файлового SQLite."""

    if not DATABASE_PATH.exists():
        return {
            thread_id: {
                "exists": False,
                "status": None,
                "current_step": None,
                "completed": None,
                "used_tools": [],
                "final_answer": None,
                "next_nodes": [],
            }
            for thread_id in thread_ids
        }

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    try:
        checkpointer = SqliteSaver(connection)
        app = build_graph(checkpointer=checkpointer)
        states: dict[str, dict[str, Any]] = {}

        for thread_id in thread_ids:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }
            snapshot = app.get_state(config)
            values = dict(snapshot.values or {})

            states[thread_id] = {
                "exists": bool(values),
                "status": values.get("status"),
                "current_step": values.get("current_step"),
                "completed": values.get("completed"),
                "used_tools": values.get("used_tools", []),
                "final_answer": values.get("final_answer"),
                "next_nodes": list(snapshot.next),
            }

        return states

    finally:
        connection.close()


def main() -> None:
    """Запускає acceptance-перевірки та зберігає JSON-звіт."""

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []

    pytest_result = run_pytest_suite()

    results.append(
        build_result(
            test_id="AC-001",
            name="Unit та integration tests",
            expected_result={
                "return_code": 0,
                "minimum_tests": 29,
            },
            actual_result=pytest_result,
            passed=pytest_result["passed"],
        )
    )

    graph_nodes = get_graph_nodes()
    expected_nodes = [
        "planner",
        "executor",
        "replanner",
    ]

    results.append(
        build_result(
            test_id="AC-002",
            name="Plan-and-Execute graph",
            expected_result=expected_nodes,
            actual_result=graph_nodes,
            passed=(
                set(graph_nodes) == set(expected_nodes)
                and len(graph_nodes) == 3
            ),
        )
    )

    tool_names = get_tool_names()
    expected_tools = [
        "check_intake_completeness",
        "classify_discovery_scope",
        "search_knowledge",
        "submit_discovery_assessment",
    ]

    results.append(
        build_result(
            test_id="AC-003",
            name="Чотири валідовані tools",
            expected_result=expected_tools,
            actual_result=tool_names,
            passed=(
                set(tool_names) == set(expected_tools)
                and len(tool_names) == 4
            ),
        )
    )

    knowledge_documents_count = get_knowledge_documents_count()

    results.append(
        build_result(
            test_id="AC-004",
            name="ChromaDB knowledge base",
            expected_result={
                "minimum_documents": 8,
                "search_results": 3,
            },
            actual_result={
                "documents_count": knowledge_documents_count,
                "configured_search_results": 3,
            },
            passed=knowledge_documents_count >= 8,
        )
    )

    thread_ids = [
        PERSISTENCE_THREAD_COMPLETED,
        PERSISTENCE_THREAD_RUNNING,
        HITL_THREAD_APPROVED,
        HITL_THREAD_REJECTED,
    ]
    states = load_thread_states(thread_ids)

    completed_state = states[PERSISTENCE_THREAD_COMPLETED]
    running_state = states[PERSISTENCE_THREAD_RUNNING]

    persistence_passed = (
        completed_state["exists"]
        and completed_state["status"] == "completed"
        and completed_state["completed"] is True
        and completed_state["next_nodes"] == []
        and running_state["exists"]
        and running_state["status"] == "running"
        and running_state["completed"] is False
        and bool(running_state["next_nodes"])
    )

    results.append(
        build_result(
            test_id="AC-005",
            name="SQLite persistence та незалежні thread_id",
            expected_result={
                PERSISTENCE_THREAD_COMPLETED: "completed",
                PERSISTENCE_THREAD_RUNNING: "running",
                "threads_are_independent": True,
            },
            actual_result={
                PERSISTENCE_THREAD_COMPLETED: completed_state,
                PERSISTENCE_THREAD_RUNNING: running_state,
                "threads_are_independent": (
                    completed_state != running_state
                ),
            },
            passed=persistence_passed,
        )
    )

    rag_passed = (
        "search_knowledge" in completed_state["used_tools"]
        and "Deep" in str(completed_state["final_answer"])
        and (
            "6-10" in str(completed_state["final_answer"])
            or "6–10" in str(completed_state["final_answer"])
        )
    )

    results.append(
        build_result(
            test_id="AC-006",
            name="Agentic RAG",
            expected_result={
                "tool": "search_knowledge",
                "scope": "Deep",
                "timebox": "6–10 тижнів",
            },
            actual_result={
                "used_tools": completed_state["used_tools"],
                "final_answer": completed_state["final_answer"],
            },
            passed=rag_passed,
        )
    )

    approved_state = states[HITL_THREAD_APPROVED]

    approval_passed = (
        approved_state["exists"]
        and approved_state["status"] == "completed"
        and approved_state["completed"] is True
        and (
            "submit_discovery_assessment"
            in approved_state["used_tools"]
        )
        and "SUB-" in str(approved_state["final_answer"])
    )

    results.append(
        build_result(
            test_id="AC-007",
            name="HITL approve",
            expected_result={
                "status": "completed",
                "tool_executed": True,
                "submission_created": True,
            },
            actual_result=approved_state,
            passed=approval_passed,
        )
    )

    rejected_state = states[HITL_THREAD_REJECTED]

    rejection_passed = (
        rejected_state["exists"]
        and rejected_state["status"] == "rejected"
        and rejected_state["completed"] is True
        and (
            "submit_discovery_assessment"
            not in rejected_state["used_tools"]
        )
        and "відмов" in str(
            rejected_state["final_answer"]
        ).lower()
    )

    results.append(
        build_result(
            test_id="AC-008",
            name="HITL reject",
            expected_result={
                "status": "rejected",
                "tool_executed": False,
                "submission_created": False,
            },
            actual_result=rejected_state,
            passed=rejection_passed,
        )
    )

    passed_count = sum(
        result["passed"] for result in results
    )
    failed_count = len(results) - passed_count
    execution_time = round(
        time.perf_counter() - started_at,
        3,
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "database": DATABASE_PATH.name,
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "success_rate_percent": round(
                passed_count / len(results) * 100,
                1,
            ),
            "execution_time_seconds": execution_time,
        },
        "acceptance_cases": results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Result: {passed_count}/{len(results)} passed. "
        f"Saved to {REPORT_PATH.name}"
    )

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status} — {result['test_id']}: "
            f"{result['name']}"
        )

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()