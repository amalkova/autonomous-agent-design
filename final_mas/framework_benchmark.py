"""Three-query LangGraph vs AG2 benchmark."""

from __future__ import annotations

import asyncio
import ast
import json
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from pydantic import BaseModel

from mas_ag2 import (
    RouteDecision as AG2RouteDecision,
)
from mas_ag2 import (
    build_ag2_agents,
    extract_reply_text,
)
from mas_langgraph import (
    build_production_mas,
    run_mas_query,
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUT_PATH = (
    ARTIFACTS_DIR
    / "framework_benchmark.json"
)
REPORT_PATH = (
    ARTIFACTS_DIR
    / "framework_comparison.md"
)

load_dotenv(
    BASE_DIR / ".env",
    override=True,
)


SCENARIOS = [
    {
        "scenario_id": "BENCH-001",
        "expected_agent": "requirements_agent",
        "query": (
            "For DEM-981, check requirements readiness. "
            "Business objective is to reduce demand lead "
            "time. Functional requirements are drafted, "
            "but NFR, acceptance criteria, integration "
            "scope and data requirements are missing."
        ),
    },
    {
        "scenario_id": "BENCH-002",
        "expected_agent": (
            "solution_security_agent"
        ),
        "query": (
            "For DEM-982, analyse solution and security "
            "risks: three systems, two APIs, personal "
            "data, high availability, mandatory security "
            "review and an unconfirmed data owner."
        ),
    },
    {
        "scenario_id": "BENCH-003",
        "expected_agent": "estimation_agent",
        "query": (
            "For DEM-983, estimate complexity and "
            "Fibonacci points: three systems, two "
            "integrations, high NFR criticality, "
            "security review required, one dependency, "
            "no migration and partially stable "
            "requirements."
        ),
    },
]


def usage_to_dict(
    report: Any,
) -> dict[str, Any]:
    """Serialize AG2 UsageReport."""

    total = report.total

    return {
        "model_calls": len(
            report.records
        ),
        "prompt_tokens": int(
            total.prompt_tokens or 0
        ),
        "completion_tokens": int(
            total.completion_tokens or 0
        ),
        "total_tokens": int(
            total.total_tokens or 0
        ),
    }


def add_usage(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    """Sum two token usage dictionaries."""

    return {
        key: int(first.get(key, 0))
        + int(second.get(key, 0))
        for key in (
            "model_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }


async def run_ag2_case(
    query: str,
) -> dict[str, Any]:
    """Run one AG2 supervisor-specialist case."""

    agents = build_ag2_agents()
    supervisor = agents[
        "demand_supervisor"
    ]

    started = time.perf_counter()

    route_reply = await supervisor.ask(
        query
    )
    route_content = await route_reply.content()

    if isinstance(
        route_content,
        AG2RouteDecision,
    ):
        decision = route_content
    else:
        decision = (
            AG2RouteDecision.model_validate(
                route_content
            )
        )

    specialist = agents[decision.action]

    specialist_reply = await specialist.ask(
        "User request:\n"
        f"{query}\n\n"
        "Supervisor routing reason:\n"
        f"{decision.reasoning}"
    )

    answer = await extract_reply_text(
        specialist_reply
    )

    route_usage = usage_to_dict(
        await route_reply.usage()
    )
    specialist_usage = usage_to_dict(
        await specialist_reply.usage()
    )

    return {
        "selected_agent": decision.action,
        "route_reasoning": decision.reasoning,
        "final_answer": answer,
        "latency_ms": round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            3,
        ),
        "usage": add_usage(
            route_usage,
            specialist_usage,
        ),
    }


def query_langsmith_trace(
    *,
    client: Client,
    project: str,
    started_after: datetime,
) -> dict[str, Any] | None:
    """Find the newest LangGraph root after timestamp."""

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            DeprecationWarning,
        )

        roots = list(
            client.list_runs(
                project_name=project,
                is_root=True,
                start_time=(
                    started_after
                    - timedelta(seconds=2)
                ),
            )
        )

    roots = [
        run
        for run in roots
        if run.name == "LangGraph"
        and run.start_time >= (
            started_after
            - timedelta(seconds=2)
        )
    ]

    if not roots:
        return None

    root = max(
        roots,
        key=lambda run: run.start_time,
    )
    trace_id = str(
        getattr(root, "trace_id", None)
        or root.id
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            DeprecationWarning,
        )

        runs = list(
            client.list_runs(
                project_name=project,
                trace_id=trace_id,
            )
        )

    llm_runs = [
        run
        for run in runs
        if run.run_type == "llm"
    ]

    return {
        "trace_id": trace_id,
        "runs_count": len(runs),
        "model_calls": len(llm_runs),
        "prompt_tokens": sum(
            int(
                getattr(
                    run,
                    "prompt_tokens",
                    0,
                )
                or 0
            )
            for run in llm_runs
        ),
        "completion_tokens": sum(
            int(
                getattr(
                    run,
                    "completion_tokens",
                    0,
                )
                or 0
            )
            for run in llm_runs
        ),
        "total_tokens": sum(
            int(
                getattr(
                    run,
                    "total_tokens",
                    0,
                )
                or 0
            )
            for run in llm_runs
        ),
    }


async def wait_for_trace(
    *,
    client: Client,
    project: str,
    started_after: datetime,
) -> dict[str, Any]:
    """Wait briefly for asynchronous trace ingestion."""

    for _attempt in range(10):
        trace = await asyncio.to_thread(
            query_langsmith_trace,
            client=client,
            project=project,
            started_after=started_after,
        )

        if (
            trace is not None
            and trace["model_calls"] > 0
        ):
            return trace

        await asyncio.sleep(1)

    raise RuntimeError(
        "LangSmith trace was not available "
        "after 10 seconds."
    )


def source_metrics(
    paths: list[Path],
) -> dict[str, int]:
    """Calculate reproducible source-code metrics."""

    total_lines = 0
    code_lines = 0
    functions = 0
    classes = 0

    for path in paths:
        source = path.read_text(
            encoding="utf-8"
        )
        lines = source.splitlines()
        tree = ast.parse(source)

        total_lines += len(lines)
        code_lines += sum(
            1
            for line in lines
            if line.strip()
            and not line.lstrip().startswith(
                "#"
            )
        )
        functions += sum(
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            for node in ast.walk(tree)
        )
        classes += sum(
            isinstance(node, ast.ClassDef)
            for node in ast.walk(tree)
        )

    return {
        "total_lines": total_lines,
        "code_lines": code_lines,
        "functions": functions,
        "classes": classes,
    }


def aggregate_usage(
    cases: list[dict[str, Any]],
) -> dict[str, int]:
    """Aggregate framework token usage."""

    return {
        key: sum(
            int(case["usage"].get(key, 0))
            for case in cases
        )
        for key in (
            "model_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }


def build_report(
    artifact: dict[str, Any],
) -> str:
    """Build Markdown comparison report."""

    lg = artifact["summary"]["LangGraph"]
    ag2 = artifact["summary"]["AG2"]

    scenario_rows = "\n".join(
        (
            f"| {item['scenario_id']} | "
            f"{item['langgraph']['selected_agent']} | "
            f"{item['ag2']['selected_agent']} | "
            f"{item['langgraph']['latency_ms']:.1f} | "
            f"{item['ag2']['latency_ms']:.1f} | "
            f"{item['pass']} |"
        )
        for item in artifact["scenarios"]
    )

    return f"""# LangGraph vs AG2 — Final MAS

## Same-query benchmark

| Scenario | LangGraph agent | AG2 agent | LangGraph ms | AG2 ms | Pass |
|---|---|---|---:|---:|---|
{scenario_rows}

Both frameworks processed the same three queries using the same
Gemini model and the same Requirements & Estimation business domain.

## Aggregate metrics

| Metric | LangGraph | AG2 |
|---|---:|---:|
| Model calls | {lg['usage']['model_calls']} | {ag2['usage']['model_calls']} |
| Prompt tokens | {lg['usage']['prompt_tokens']} | {ag2['usage']['prompt_tokens']} |
| Completion tokens | {lg['usage']['completion_tokens']} | {ag2['usage']['completion_tokens']} |
| Total tokens | {lg['usage']['total_tokens']} | {ag2['usage']['total_tokens']} |
| Framework code lines | {lg['source']['code_lines']} | {ag2['source']['code_lines']} |
| Development time, minutes | {lg['development_time_minutes']} | {ag2['development_time_minutes']} |
| Control, 1–5 | {lg['control_rating']} | {ag2['control_rating']} |
| Debugging, 1–5 | {lg['debugging_rating']} | {ag2['debugging_rating']} |

Development time is the recorded coursework implementation estimate:
LangGraph includes explicit orchestration, persistence, nested patterns
and HITL; AG2 is an adaptation over the shared MCP/domain layer.

## Conclusion

LangGraph provides stronger workflow control, durable state,
breakpoints, replay and hierarchical tracing. It requires more code and
its nested Plan-and-Execute path can consume more tokens.

AG2 expresses supervisor-to-specialist delegation more compactly and is
easier to read as ordinary asynchronous Python. However, checkpoint
semantics, graph replay and node-level control are less explicit.

LangGraph remains the production choice for this case. AG2 is the
preferred compact alternative for simpler stateless collaboration.
"""


async def main() -> None:
    """Run the complete three-query benchmark."""

    project = os.getenv(
        "LANGSMITH_PROJECT",
        "hw3-malkova-demand-mas",
    )
    client = Client()
    langgraph = await build_production_mas()

    scenario_results = []

    for scenario in SCENARIOS:
        lg_started_at = datetime.now(
            timezone.utc
        )
        lg_clock = time.perf_counter()

        lg_result = await run_mas_query(
            langgraph,
            scenario["query"],
            (
                "benchmark-langgraph-"
                + scenario["scenario_id"]
            ),
            log_trajectory=False,
        )

        lg_latency = round(
            (
                time.perf_counter()
                - lg_clock
            )
            * 1000,
            3,
        )

        lg_usage = await wait_for_trace(
            client=client,
            project=project,
            started_after=lg_started_at,
        )

        ag2_result = await run_ag2_case(
            scenario["query"]
        )

        passed = (
            lg_result.get("current_agent")
            == scenario["expected_agent"]
            and ag2_result["selected_agent"]
            == scenario["expected_agent"]
        )

        scenario_results.append(
            {
                "scenario_id": (
                    scenario["scenario_id"]
                ),
                "query": scenario["query"],
                "expected_agent": scenario[
                    "expected_agent"
                ],
                "langgraph": {
                    "selected_agent": (
                        lg_result.get(
                            "current_agent"
                        )
                    ),
                    "final_answer": (
                        lg_result.get(
                            "final_answer"
                        )
                    ),
                    "latency_ms": lg_latency,
                    "usage": lg_usage,
                },
                "ag2": ag2_result,
                "pass": passed,
            }
        )

    langgraph_cases = [
        {
            "usage": item[
                "langgraph"
            ]["usage"]
        }
        for item in scenario_results
    ]
    ag2_cases = [
        {
            "usage": item["ag2"]["usage"]
        }
        for item in scenario_results
    ]

    artifact = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": os.getenv(
            "MODEL_NAME",
            "gemini-3.1-flash-lite",
        ),
        "scenarios": scenario_results,
        "summary": {
            "LangGraph": {
                "usage": aggregate_usage(
                    langgraph_cases
                ),
                "source": source_metrics(
                    [
                        BASE_DIR
                        / "mas_langgraph.py",
                        BASE_DIR
                        / "specialist_runners.py",
                    ]
                ),
                "development_time_minutes": 420,
                "control_rating": 5,
                "debugging_rating": 5,
            },
            "AG2": {
                "usage": aggregate_usage(
                    ag2_cases
                ),
                "source": source_metrics(
                    [
                        BASE_DIR / "mas_ag2.py",
                    ]
                ),
                "development_time_minutes": 90,
                "control_rating": 3,
                "debugging_rating": 3,
            },
        },
    }

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        build_report(artifact),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "scenarios": len(
                    scenario_results
                ),
                "passed": sum(
                    item["pass"]
                    for item
                    in scenario_results
                ),
                "langgraph_usage": (
                    artifact["summary"]
                    ["LangGraph"]["usage"]
                ),
                "ag2_usage": (
                    artifact["summary"]
                    ["AG2"]["usage"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
