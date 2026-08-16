"""Тести числового порівняння агентних архітектур."""

from __future__ import annotations

from typing import Any

from compare_agents import (
    EXPECTED_TOOLS,
    build_comparison_summary,
    calculate_quality,
    run_plan_execute_comparison,
    run_react_comparison,
)


def test_quality_is_100_for_complete_result() -> None:
    """Повна tool та keyword coverage дає 100%."""

    quality = calculate_quality(
        status="completed",
        answer=(
            "Requirements готові на 100%. "
            "Estimation complexity — High."
        ),
        used_tools=EXPECTED_TOOLS,
    )

    assert quality["quality_score_percent"] == 100.0
    assert quality["tool_coverage_percent"] == 100.0
    assert quality["keyword_coverage_percent"] == 100.0
    assert quality["missing_tools"] == []


def test_quality_detects_missing_tools() -> None:
    """Метрика фіксує відсутній tool call."""

    quality = calculate_quality(
        status="completed",
        answer=(
            "Requirements готові на 100%. "
            "Estimation complexity — High."
        ),
        used_tools=[
            "check_requirements_readiness",
            "classify_estimation_complexity",
        ],
    )

    assert quality["quality_score_percent"] < 100.0
    assert quality["tool_coverage_percent"] == 66.67
    assert quality["missing_tools"] == [
        "search_delivery_knowledge",
    ]


def test_react_comparison_uses_response_metrics() -> None:
    """ReAct comparison читає status, safety та tools."""

    def fake_react_runner(
        request: str,
    ) -> dict[str, Any]:
        assert "DEM-050" in request

        return {
            "status": "completed",
            "answer": (
                "Requirements готові на 100%. "
                "Estimation complexity — High."
            ),
            "used_tools": EXPECTED_TOOLS,
            "safety": {
                "step_count": 7,
            },
        }

    result = run_react_comparison(
        "Перевір DEM-050.",
        runner=fake_react_runner,
    )

    assert result["architecture"] == "ReAct"
    assert result["completed"] is True
    assert result["reasoning_steps"] == 7
    assert result["tool_calls_count"] == 3
    assert result["quality"]["quality_score_percent"] == 100.0
    assert result["execution_time_seconds"] >= 0


def test_plan_execute_comparison_reads_graph_state() -> None:
    """Plan comparison читає plan, progress та replanning."""

    class FakeGraph:
        """Мінімальний fake compiled graph."""

        def invoke(
            self,
            state: dict[str, Any],
            config: dict[str, Any],
        ) -> dict[str, Any]:
            assert "DEM-050" in state["user_request"]
            assert config["recursion_limit"] == 50

            return {
                "status": "completed",
                "completed": True,
                "goal": "Перевірити readiness та estimation",
                "plan": [
                    "check_requirements_readiness: DEM-050",
                    "classify_estimation_complexity: DEM-050",
                    "search_delivery_knowledge: estimation",
                ],
                "current_step": 3,
                "results": [
                    "Readiness complete.",
                    "Complexity High.",
                    "Knowledge знайдено.",
                ],
                "final_answer": (
                    "Requirements готові на 100%. "
                    "Estimation complexity — High."
                ),
                "used_tools": EXPECTED_TOOLS,
                "replan_count": 1,
            }

    result = run_plan_execute_comparison(
        "Перевір DEM-050.",
        graph=FakeGraph(),
    )

    assert result["architecture"] == "Plan-and-Execute"
    assert result["completed"] is True
    assert result["planned_steps"] == 3
    assert result["executed_plan_steps"] == 3
    assert result["replan_count"] == 1
    assert result["tool_calls_count"] == 3
    assert result["quality"]["quality_score_percent"] == 100.0


def test_comparison_summary_selects_winners() -> None:
    """Summary визначає швидшу та якіснішу архітектуру."""

    react_result = {
        "execution_time_seconds": 2.0,
        "quality": {
            "quality_score_percent": 90.0,
            "tool_coverage_percent": 100.0,
        },
    }

    plan_result = {
        "execution_time_seconds": 5.5,
        "quality": {
            "quality_score_percent": 100.0,
            "tool_coverage_percent": 100.0,
        },
    }

    summary = build_comparison_summary(
        react_result,
        plan_result,
    )

    assert summary["faster_architecture"] == "ReAct"
    assert summary["higher_quality_architecture"] == (
        "Plan-and-Execute"
    )
    assert summary["latency_difference_seconds"] == 3.5
    assert len(summary["conclusions"]) == 4