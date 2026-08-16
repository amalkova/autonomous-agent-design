"""Acceptance-перевірка Практичного завдання №1."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from knowledge import initialize_knowledge_base
from persistence_demo import (
    compare_threads,
    inspect_thread,
)
from plan_execute import build_graph
from react_agent import MODEL_NAME, build_react_graph
from safety import SafetyConfig, SafetyController
from tools import DOMAIN_TOOLS


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "test_results.json"


def load_json(filename: str) -> dict[str, Any]:
    """Читає JSON artifact."""

    return json.loads(
        (PROJECT_DIR / filename).read_text(
            encoding="utf-8"
        )
    )


def run_pytest() -> dict[str, Any]:
    """Запускає весь pytest suite у поточному environment."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    output = "\n".join(
        part.strip()
        for part in [
            result.stdout,
            result.stderr,
        ]
        if part.strip()
    )

    return {
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "output": output,
        "python": sys.executable,
    }


def check_domain_tools() -> dict[str, Any]:
    """Перевіряє кількість і schemas domain tools."""

    names = [tool.name for tool in DOMAIN_TOOLS]
    schemas = [
        tool.args_schema.model_json_schema()
        for tool in DOMAIN_TOOLS
    ]

    strict_schemas = [
        schema.get("additionalProperties") is False
        for schema in schemas
    ]

    return {
        "passed": (
            len(names) == 4
            and all(strict_schemas)
        ),
        "tool_names": names,
        "tools_count": len(names),
        "strict_schemas": strict_schemas,
    }


def check_react_architecture() -> dict[str, Any]:
    """Перевіряє ReAct graph та safety."""

    graph = build_react_graph(
        SafetyController()
    )
    nodes = list(graph.get_graph().nodes)

    safety = SafetyConfig()

    trajectory = load_json("trajectory.json")
    react_runs = [
        run
        for run in trajectory["runs"]
        if run["agent_type"] == "react"
    ]

    required_nodes = {
        "__start__",
        "agent",
        "tools",
        "finalize",
        "__end__",
    }

    return {
        "passed": (
            required_nodes.issubset(nodes)
            and safety.max_steps == 10
            and safety.timeout_seconds == 120
            and safety.max_identical_tool_calls == 2
            and bool(react_runs)
        ),
        "nodes": nodes,
        "max_steps": safety.max_steps,
        "timeout_seconds": safety.timeout_seconds,
        "max_identical_tool_calls": (
            safety.max_identical_tool_calls
        ),
        "trajectory_runs": len(react_runs),
    }


def check_plan_execute_architecture() -> dict[str, Any]:
    """Перевіряє Plan-and-Execute graph."""

    graph = build_graph()
    nodes = list(graph.get_graph().nodes)

    source = (
        PROJECT_DIR / "plan_execute.py"
    ).read_text(encoding="utf-8")

    required_nodes = {
        "__start__",
        "planner",
        "executor",
        "approval",
        "replanner",
        "__end__",
    }

    return {
        "passed": (
            required_nodes.issubset(nodes)
            and "run_react_agent(" in source
            and "ReplanDecision" in source
            and "interrupt_after" in source
        ),
        "nodes": nodes,
        "nested_react": "run_react_agent(" in source,
        "structured_replanner": (
            "ReplanDecision" in source
        ),
    }


def check_agentic_rag() -> dict[str, Any]:
    """Перевіряє ChromaDB та agent-selected search."""

    knowledge_info = initialize_knowledge_base()
    trajectory = load_json("trajectory.json")

    rag_runs = [
        run
        for run in trajectory["runs"]
        if "search_delivery_knowledge"
        in run["final_response"].get(
            "used_tools",
            [],
        )
    ]

    documents_count = int(
        knowledge_info["documents_count"]
    )

    return {
        "passed": (
            documents_count >= 8
            and bool(rag_runs)
        ),
        "collection": knowledge_info["collection"],
        "documents_count": documents_count,
        "rag_trajectory_runs": len(rag_runs),
    }


def check_persistence() -> dict[str, Any]:
    """Перевіряє SQLite та незалежні threads."""

    comparison = compare_threads(
        "practice-persistence-001",
        "practice-persistence-002",
    )

    database_path = PROJECT_DIR / "agent_state.db"

    return {
        "passed": (
            database_path.exists()
            and database_path.stat().st_size > 0
            and comparison["threads_are_independent"]
            and comparison["state_values_differ"]
            and comparison["progress_is_independent"]
        ),
        "database": str(database_path),
        "database_size_bytes": (
            database_path.stat().st_size
            if database_path.exists()
            else 0
        ),
        "threads_are_independent": (
            comparison["threads_are_independent"]
        ),
        "state_values_differ": (
            comparison["state_values_differ"]
        ),
        "progress_is_independent": (
            comparison["progress_is_independent"]
        ),
    }


def check_hitl() -> dict[str, Any]:
    """Перевіряє persisted approve та reject results."""

    approved = inspect_thread(
        "practice-hitl-approve-002"
    )["state"]

    rejected = inspect_thread(
        "practice-hitl-reject"
    )["state"]

    return {
        "passed": (
            approved.get("status") == "completed"
            and approved.get("completed") is True
            and "submit_estimation_request"
            in approved.get("used_tools", [])
            and rejected.get("status") == "rejected"
            and rejected.get("completed") is True
            and rejected.get("used_tools") == []
        ),
        "approved": {
            "status": approved.get("status"),
            "used_tools": approved.get(
                "used_tools",
                [],
            ),
            "final_answer": approved.get(
                "final_answer"
            ),
        },
        "rejected": {
            "status": rejected.get("status"),
            "used_tools": rejected.get(
                "used_tools",
                [],
            ),
            "final_answer": rejected.get(
                "final_answer"
            ),
        },
    }


def check_trajectories() -> dict[str, Any]:
    """Перевіряє runs обох архітектур."""

    trajectory = load_json("trajectory.json")
    runs = trajectory["runs"]

    agent_types = {
        run["agent_type"]
        for run in runs
    }

    nonempty_messages = all(
        bool(run.get("messages"))
        for run in runs
    )

    runs_with_tools = all(
        bool(
            run["final_response"].get(
                "used_tools",
                [],
            )
        )
        for run in runs
    )

    return {
        "passed": (
            {"react", "plan_execute"}
            .issubset(agent_types)
            and nonempty_messages
            and runs_with_tools
        ),
        "runs_count": len(runs),
        "agent_types": sorted(agent_types),
        "messages_per_run": [
            len(run["messages"])
            for run in runs
        ],
        "tools_per_run": [
            run["final_response"].get(
                "used_tools",
                [],
            )
            for run in runs
        ],
    }


def check_comparison() -> dict[str, Any]:
    """Перевіряє числове порівняння."""

    report = load_json(
        "comparison_results.json"
    )

    runs = report["runs"]
    architectures = {
        run["architecture"]
        for run in runs
    }

    quality_scores = {
        run["architecture"]: (
            run["quality"][
                "quality_score_percent"
            ]
        )
        for run in runs
    }

    execution_times = {
        run["architecture"]: (
            run["execution_time_seconds"]
        )
        for run in runs
    }

    return {
        "passed": (
            architectures
            == {"ReAct", "Plan-and-Execute"}
            and all(
                score == 100.0
                for score in quality_scores.values()
            )
            and all(
                value > 0
                for value in execution_times.values()
            )
        ),
        "architectures": sorted(architectures),
        "quality_scores": quality_scores,
        "execution_times": execution_times,
        "comparison": report["comparison"],
    }


def check_documentation() -> dict[str, Any]:
    """Перевіряє README, notebook та artifacts."""

    readme_path = PROJECT_DIR / "README.md"
    notebook_path = (
        PROJECT_DIR
        / "Task_001_Malkova_Requirements_Estimation.ipynb"
    )

    notebook = json.loads(
        notebook_path.read_text(
            encoding="utf-8"
        )
    )

    artifact_names = [
        "README.md",
        "trajectory.json",
        "agent_state.db",
        "comparison_results.json",
        "react_graph.mmd",
        "plan_execute_graph.mmd",
        "Task_001_Malkova_Requirements_Estimation.ipynb",
    ]

    artifacts = {
        name: (
            (PROJECT_DIR / name).exists()
            and (PROJECT_DIR / name).stat().st_size > 0
        )
        for name in artifact_names
    }

    return {
        "passed": (
            readme_path.stat().st_size > 5000
            and notebook["nbformat"] == 4
            and len(notebook["cells"]) >= 15
            and all(artifacts.values())
        ),
        "readme_size_bytes": readme_path.stat().st_size,
        "notebook_cells": len(notebook["cells"]),
        "notebook_format": notebook["nbformat"],
        "artifacts": artifacts,
    }


def check_bonus_features() -> dict[str, Any]:
    """Перевіряє comparison, visualization і fallback."""

    plan_source = inspect.getsource(build_graph)

    react_graph_path = PROJECT_DIR / "react_graph.mmd"
    plan_graph_path = (
        PROJECT_DIR / "plan_execute_graph.mmd"
    )
    comparison_path = (
        PROJECT_DIR / "comparison_results.json"
    )

    fallback_present = (
        "fallback_final_answer" in plan_source
        or "fallback_final_answer"
        in (
            PROJECT_DIR / "plan_execute.py"
        ).read_text(encoding="utf-8")
    )

    return {
        "passed": (
            react_graph_path.stat().st_size > 0
            and plan_graph_path.stat().st_size > 0
            and comparison_path.stat().st_size > 0
            and fallback_present
        ),
        "numeric_comparison": comparison_path.exists(),
        "react_visualization_bytes": (
            react_graph_path.stat().st_size
        ),
        "plan_visualization_bytes": (
            plan_graph_path.stat().st_size
        ),
        "fallback_strategy": fallback_present,
    }


def evaluate_case(
    test_id: str,
    name: str,
    expected_result: str,
    check: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Безпечно виконує acceptance check."""

    try:
        actual_result = check()
        passed = bool(
            actual_result.pop("passed")
        )
        error = None

    except Exception as exception:
        passed = False
        actual_result = {}
        error = (
            f"{type(exception).__name__}: "
            f"{exception}"
        )

    return {
        "test_id": test_id,
        "name": name,
        "expected_result": expected_result,
        "passed": passed,
        "error": error,
        "actual_result": actual_result,
    }


def main() -> None:
    """Формує test_results.json."""

    pytest_result = run_pytest()

    acceptance_cases = [
        {
            "test_id": "AC-001",
            "name": "Unit та integration tests",
            "expected_result": (
                "Увесь pytest suite завершується успішно."
            ),
            "passed": pytest_result["passed"],
            "error": (
                None
                if pytest_result["passed"]
                else pytest_result["output"]
            ),
            "actual_result": pytest_result,
        },
        evaluate_case(
            "AC-002",
            "Чотири валідовані domain tools",
            "Рівно 4 tools зі strict Pydantic schemas.",
            check_domain_tools,
        ),
        evaluate_case(
            "AC-003",
            "ReAct graph та safety",
            "LLM–tools–LLM graph із safety limits.",
            check_react_architecture,
        ),
        evaluate_case(
            "AC-004",
            "Plan-and-Execute graph",
            "Planner, nested ReAct executor та replanner.",
            check_plan_execute_architecture,
        ),
        evaluate_case(
            "AC-005",
            "Agentic RAG",
            "ChromaDB має ≥8 документів і agent-selected search.",
            check_agentic_rag,
        ),
        evaluate_case(
            "AC-006",
            "SQLite persistence",
            "State відновлюється, threads незалежні.",
            check_persistence,
        ),
        evaluate_case(
            "AC-007",
            "Human-in-the-Loop",
            "Approve виконує tool, reject блокує tool.",
            check_hitl,
        ),
        evaluate_case(
            "AC-008",
            "JSON trajectories",
            "Збережені ReAct та Plan-and-Execute runs.",
            check_trajectories,
        ),
        evaluate_case(
            "AC-009",
            "Числове порівняння",
            "Обидві архітектури мають числові метрики.",
            check_comparison,
        ),
        evaluate_case(
            "AC-010",
            "Документація та deliverables",
            "README, notebook та artifacts присутні.",
            check_documentation,
        ),
        evaluate_case(
            "AC-011",
            "Bonus features",
            "Comparison, visualization та fallback.",
            check_bonus_features,
        ),
    ]

    passed_count = sum(
        case["passed"]
        for case in acceptance_cases
    )
    failed_count = (
        len(acceptance_cases) - passed_count
    )

    report = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": MODEL_NAME,
        "summary": {
            "total": len(acceptance_cases),
            "passed": passed_count,
            "failed": failed_count,
            "success_rate_percent": round(
                passed_count
                / len(acceptance_cases)
                * 100,
                2,
            ),
        },
        "pytest": pytest_result,
        "acceptance_cases": acceptance_cases,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Result: {passed_count}/"
        f"{len(acceptance_cases)} passed. "
        "Saved to test_results.json"
    )

    for case in acceptance_cases:
        status = (
            "PASS"
            if case["passed"]
            else "FAIL"
        )

        print(
            f"{status} — "
            f"{case['test_id']}: "
            f"{case['name']}"
        )

        if case["error"]:
            print(f"  {case['error']}")

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()