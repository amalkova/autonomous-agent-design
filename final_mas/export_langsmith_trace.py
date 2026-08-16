"""Експорт останнього LangSmith trace у JSON artifact."""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = (
    BASE_DIR
    / "artifacts"
    / "langsmith_trace.json"
)

load_dotenv(
    BASE_DIR / ".env",
    override=True,
)


def json_safe(
    value: Any,
) -> Any:
    """Перетворити LangSmith values на JSON-safe data."""

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):
        return json_safe(
            value.model_dump()
        )

    return str(value)


def calculate_depth(
    run: Any,
    by_id: dict[str, Any],
) -> int:
    """Розрахувати depth у trace hierarchy."""

    depth = 0
    parent_id = getattr(
        run,
        "parent_run_id",
        None,
    )
    visited: set[str] = set()

    while parent_id is not None:
        parent_key = str(parent_id)

        if parent_key in visited:
            break

        visited.add(parent_key)
        parent = by_id.get(parent_key)

        if parent is None:
            break

        depth += 1
        parent_id = getattr(
            parent,
            "parent_run_id",
            None,
        )

    return depth


def token_value(
    run: Any,
    name: str,
) -> int:
    """Безпечно прочитати token counter."""

    value = getattr(
        run,
        name,
        None,
    )

    if isinstance(value, (int, float)):
        return int(value)

    return 0


def export_latest_trace() -> dict[str, Any]:
    """Знайти та експортувати останній root trace."""

    project = os.getenv(
        "LANGSMITH_PROJECT",
        "hw3-malkova-demand-mas",
    )

    client = Client()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            DeprecationWarning,
        )

        roots = list(
            client.list_runs(
                project_name=project,
                is_root=True,
            )
        )

    if not roots:
        raise RuntimeError(
            f"No root runs found in project {project!r}."
        )

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

    runs.sort(
        key=lambda run: (
            run.start_time,
            str(run.id),
        )
    )

    by_id = {
        str(run.id): run
        for run in runs
    }

    exported_runs = []

    for run in runs:
        end_time = getattr(
            run,
            "end_time",
            None,
        )

        latency_ms = None

        if end_time is not None:
            latency_ms = round(
                (
                    end_time
                    - run.start_time
                ).total_seconds()
                * 1000,
                3,
            )

        exported_runs.append(
            {
                "id": str(run.id),
                "trace_id": str(
                    getattr(
                        run,
                        "trace_id",
                        trace_id,
                    )
                ),
                "parent_run_id": (
                    str(run.parent_run_id)
                    if run.parent_run_id
                    else None
                ),
                "depth": calculate_depth(
                    run,
                    by_id,
                ),
                "name": run.name,
                "run_type": run.run_type,
                "status": (
                    "error"
                    if run.error
                    else "success"
                ),
                "start_time": (
                    run.start_time.isoformat()
                ),
                "end_time": (
                    end_time.isoformat()
                    if end_time
                    else None
                ),
                "latency_ms": latency_ms,
                "prompt_tokens": token_value(
                    run,
                    "prompt_tokens",
                ),
                "completion_tokens": token_value(
                    run,
                    "completion_tokens",
                ),
                "total_tokens": token_value(
                    run,
                    "total_tokens",
                ),
                "inputs": json_safe(
                    run.inputs
                ),
                "outputs": json_safe(
                    run.outputs
                ),
                "error": run.error,
            }
        )

    run_types = Counter(
        run.run_type
        for run in runs
    )

    summary = {
        "project": project,
        "trace_id": trace_id,
        "root_run_id": str(root.id),
        "root_run_name": root.name,
        "runs_count": len(runs),
        "max_depth": max(
            (
                item["depth"]
                for item in exported_runs
            ),
            default=0,
        ),
        "run_types": dict(run_types),
        "llm_runs_count": run_types.get(
            "llm",
            0,
        ),
        "tool_runs_count": run_types.get(
            "tool",
            0,
        ),
        "prompt_tokens": sum(
            item["prompt_tokens"]
            for item in exported_runs
        ),
        "completion_tokens": sum(
            item["completion_tokens"]
            for item in exported_runs
        ),
        "total_tokens": sum(
            item["total_tokens"]
            for item in exported_runs
        ),
    }

    try:
        summary["trace_url"] = (
            client.get_run_url(run=root)
        )
    except Exception:
        summary["trace_url"] = None

    artifact = {
        "summary": summary,
        "runs": exported_runs,
    }

    OUTPUT_PATH.parent.mkdir(
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

    return artifact


def main() -> None:
    """Експортувати trace та вивести summary."""

    artifact = export_latest_trace()

    print(
        json.dumps(
            artifact["summary"],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
