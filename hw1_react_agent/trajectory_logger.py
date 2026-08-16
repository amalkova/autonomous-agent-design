"""Збереження траєкторії виконання ReAct-агента."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.messages import AnyMessage


def extract_text(message: AnyMessage) -> str:
    """Витягує текст без службових thought signatures Gemini."""

    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(str(block.get("text", "")))

        return "\n".join(text_blocks)

    return str(content)


def serialize_message(
    message: AnyMessage,
    index: int,
) -> dict[str, Any]:
    """Перетворює повідомлення LangChain на JSON-сумісний крок."""

    return {
        "step": index,
        "message_type": message.type,
        "name": getattr(message, "name", None),
        "content": extract_text(message),
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def save_trajectory(
    user_input: str,
    messages: list[AnyMessage],
    final_response: dict[str, Any],
    path: str | Path = "trajectory.json",
) -> Path:
    """Додає один запуск агента до trajectory.json."""

    trajectory_path = Path(path)

    history: dict[str, Any] = {"runs": []}

    if trajectory_path.exists() and trajectory_path.stat().st_size > 0:
        try:
            loaded_history = json.loads(
                trajectory_path.read_text(encoding="utf-8")
            )

            if (
                isinstance(loaded_history, dict)
                and isinstance(loaded_history.get("runs"), list)
            ):
                history = loaded_history
        except json.JSONDecodeError:
            history = {"runs": []}

    run = {
        "run_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "trajectory": [
            serialize_message(message, index)
            for index, message in enumerate(messages, start=1)
        ],
        "final_response": final_response,
    }

    history["runs"].append(run)

    trajectory_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return trajectory_path