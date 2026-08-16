"""Тести generated evaluation artifacts."""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def load_json(name: str):
    return json.loads(
        (BASE_DIR / name).read_text(
            encoding="utf-8"
        )
    )


def test_evaluation_artifact_has_five_scenarios() -> None:
    results = load_json(
        "eval_results.json"
    )

    assert len(results) >= 5
    assert all(
        result["pass"]
        for result in results
    )
    assert all(
        result["agents_used"]
        or result["actual"]["blocked"]
        for result in results
    )


def test_red_team_artifact_has_required_attacks() -> None:
    results = load_json(
        "red_team_results.json"
    )
    attack_types = {
        result["attack_type"]
        for result in results
    }

    assert {
        "prompt_injection",
        "pii_leakage",
        "scope_confusion",
        "tool_misuse",
        "jailbreak",
    } <= attack_types

    assert all(
        result["pass"]
        for result in results
    )


def test_framework_benchmark_has_same_queries() -> None:
    artifact = load_json(
        "artifacts/framework_benchmark.json"
    )

    assert len(artifact["scenarios"]) == 3
    assert all(
        scenario["pass"]
        for scenario in artifact["scenarios"]
    )

    for scenario in artifact["scenarios"]:
        assert (
            scenario["langgraph"][
                "selected_agent"
            ]
            == scenario["ag2"][
                "selected_agent"
            ]
        )


def test_langsmith_artifact_has_hierarchy() -> None:
    artifact = load_json(
        "artifacts/langsmith_trace.json"
    )
    summary = artifact["summary"]

    assert summary["runs_count"] >= 3
    assert summary["max_depth"] >= 1
    assert summary["trace_id"]
    assert summary["trace_url"]
