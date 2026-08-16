"""JSON-логування траєкторій автономних агентів."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


BASE_DIR = Path(__file__).resolve().parent
TRAJECTORY_PATH = BASE_DIR / "trajectory.json"


def make_json_safe(value: Any) -> Any:
    """Перетворює значення на JSON-сумісний формат."""

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())

    return str(value)


def serialize_message(
    message: BaseMessage,
    step: int,
) -> dict[str, Any]:
    """Серіалізує LangChain message для trajectory log."""

    tool_calls = getattr(
        message,
        "tool_calls",
        [],
    )

    return {
        "step": step,
        "message_type": message.type,
        "name": getattr(message, "name", None),
        "content": make_json_safe(message.content),
        "tool_calls": make_json_safe(tool_calls),
        "tool_call_id": getattr(
            message,
            "tool_call_id",
            None,
        ),
    }


def load_trajectory(
    path: Path = TRAJECTORY_PATH,
) -> dict[str, Any]:
    """Завантажує існуючий trajectory log."""

    if not path.exists():
        return {
            "runs": [],
        }

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return {
            "runs": [],
        }

    if not isinstance(payload, dict):
        return {
            "runs": [],
        }

    runs = payload.get("runs")

    if not isinstance(runs, list):
        payload["runs"] = []

    return payload


def save_trajectory(
    agent_type: str,
    user_input: str,
    messages: list[BaseMessage],
    final_response: dict[str, Any],
    safety: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    path: Path = TRAJECTORY_PATH,
) -> dict[str, Any]:
    """Додає один повний запуск агента до JSON-логу."""

    trajectory = load_trajectory(path)

    run = {
        "run_id": (
            f"RUN-{len(trajectory['runs']) + 1:03d}"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "agent_type": agent_type,
        "user_input": user_input,
        "messages": [
            serialize_message(message, step)
            for step, message in enumerate(
                messages,
                start=1,
            )
        ],
        "final_response": make_json_safe(
            final_response
        ),
        "safety": make_json_safe(safety),
        "metadata": make_json_safe(metadata or {}),
    }

    trajectory["runs"].append(run)

    path.write_text(
        json.dumps(
            trajectory,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return run