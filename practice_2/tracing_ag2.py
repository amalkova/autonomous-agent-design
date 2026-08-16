"""Створення та експорт інструментованого AG2 trace у LangSmith."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

load_dotenv(BASE_DIR / ".env")


def usage_dict(usage: Any) -> dict[str, Any]:
    """Перетворити AG2 token usage на JSON-safe словник."""

    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "thinking_tokens",
    )

    return {
        field: getattr(usage, field, None)
        for field in fields
    }


async def flush_client(client: Any) -> None:
    """Надіслати відкладені LangSmith runs."""

    result = client.flush()

    if inspect.isawaitable(result):
        await result


async def main() -> None:
    """Запустити та експортувати AG2 trace."""

    required = (
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    )

    missing = [
        key
        for key in required
        if not os.getenv(key)
    ]

    if missing:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    os.environ["LANGSMITH_TRACING"] = "true"

    import langsmith as ls
    from langsmith import Client, traceable

    import mas_ag2

    project_name = os.environ["LANGSMITH_PROJECT"]

    client = Client(
        api_key=os.environ["LANGSMITH_API_KEY"],
        api_url=os.environ["LANGSMITH_ENDPOINT"],
    )

    # AG2 не має автоматичної інструментації LangSmith.
    # Обгортання MCP-функцій створює реальні tool spans.
    mas_ag2.check_requirements_readiness = traceable(
        name="check_requirements_readiness",
        run_type="tool",
        client=client,
    )(
        mas_ag2.check_requirements_readiness
    )

    mas_ag2.identify_handover_gaps = traceable(
        name="identify_handover_gaps",
        run_type="tool",
        client=client,
    )(
        mas_ag2.identify_handover_gaps
    )

    mas_ag2.classify_estimation_complexity = traceable(
        name="classify_estimation_complexity",
        run_type="tool",
        client=client,
    )(
        mas_ag2.classify_estimation_complexity
    )

    agents = mas_ag2.build_ag2_agents()

    request = (
        "Estimate demand DEM-025. "
        "systems_count=3; integration_count=2; "
        "nfr_criticality=high; "
        "data_migration_required=false; "
        "security_review_required=true; "
        "dependency_count=1; "
        "requirements_stability=partial. "
        "Return complexity, Fibonacci points and main drivers. "
        "Do not submit the estimation request."
    )

    @traceable(
        name="ag2_demand_supervisor",
        run_type="chain",
        client=client,
    )
    async def run_supervisor(
        user_request: str,
    ) -> dict[str, Any]:
        reply = await agents[
            "demand_supervisor"
        ].ask(user_request)

        decision = await reply.content()
        usage_report = await reply.usage()

        return {
            "decision": decision.model_dump(),
            "model_calls": len(
                usage_report.records
            ),
            "usage": usage_dict(
                usage_report.total
            ),
        }

    @traceable(
        name="ag2_specialist",
        run_type="chain",
        client=client,
    )
    async def run_specialist(
        agent_name: str,
        user_request: str,
        route_reasoning: str,
    ) -> dict[str, Any]:
        reply = await agents[
            agent_name
        ].ask(
            "User request:\n"
            f"{user_request}\n\n"
            "Supervisor routing reason:\n"
            f"{route_reasoning}"
        )

        answer = await mas_ag2.extract_reply_text(
            reply
        )
        usage_report = await reply.usage()

        return {
            "agent_name": agent_name,
            "answer": answer,
            "model_calls": len(
                usage_report.records
            ),
            "usage": usage_dict(
                usage_report.total
            ),
        }

    @traceable(
        name="ag2_requirements_estimation_mas",
        run_type="chain",
        client=client,
        tags=[
            "practice-2",
            "ag2",
            "requirements-estimation",
        ],
        metadata={
            "assignment": "Practical Assignment 2",
            "framework": "AG2 v1",
            "contains_real_pii": False,
        },
    )
    async def run_traced_ag2(
        user_request: str,
    ) -> dict[str, Any]:
        supervisor_result = (
            await run_supervisor(
                user_request
            )
        )

        decision = mas_ag2.RouteDecision.model_validate(
            supervisor_result["decision"]
        )

        specialist_result = (
            await run_specialist(
                decision.action,
                user_request,
                decision.reasoning,
            )
        )

        return {
            "framework": "AG2 v1",
            "selected_agent": decision.action,
            "route_reasoning": (
                decision.reasoning
            ),
            "final_answer": (
                specialist_result["answer"]
            ),
            "model_calls": (
                supervisor_result["model_calls"]
                + specialist_result[
                    "model_calls"
                ]
            ),
            "prompt_tokens": (
                supervisor_result[
                    "usage"
                ]["prompt_tokens"]
                + specialist_result[
                    "usage"
                ]["prompt_tokens"]
            ),
            "completion_tokens": (
                supervisor_result[
                    "usage"
                ]["completion_tokens"]
                + specialist_result[
                    "usage"
                ]["completion_tokens"]
            ),
            "total_tokens": (
                supervisor_result[
                    "usage"
                ]["total_tokens"]
                + specialist_result[
                    "usage"
                ]["total_tokens"]
            ),
        }

    with ls.tracing_context(
        enabled=True,
        project_name=project_name,
    ):
        result = await run_traced_ag2(
            request
        )

    await flush_client(client)

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
    )

    roots = list(
        client.list_runs(
            project_name=project_name,
            is_root=True,
            limit=10,
        )
    )

    root = next(
        (
            run
            for run in roots
            if run.name
            == "ag2_requirements_estimation_mas"
        ),
        None,
    )

    if root is None:
        raise RuntimeError(
            "AG2 root trace was not found."
        )

    trace_id = root.trace_id or root.id

    runs = list(
        client.list_runs(
            project_name=project_name,
            trace_id=trace_id,
        )
    )

    exported_runs = [
        run.model_dump(
            mode="json",
            exclude_none=False,
        )
        for run in runs
    ]

    exported_runs.sort(
        key=lambda run: str(
            run.get("start_time", "")
        )
    )

    artifact = {
        "summary": {
            "project": project_name,
            "trace_id": str(trace_id),
            "root_run_name": root.name,
            "runs_count": len(
                exported_runs
            ),
            "model_calls": result[
                "model_calls"
            ],
            "prompt_tokens": result[
                "prompt_tokens"
            ],
            "completion_tokens": result[
                "completion_tokens"
            ],
            "total_tokens": result[
                "total_tokens"
            ],
        },
        "runs": exported_runs,
    }

    ARTIFACTS_DIR.mkdir(
        exist_ok=True
    )

    output_path = (
        ARTIFACTS_DIR
        / "ag2_langsmith_trace.json"
    )

    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            artifact["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())