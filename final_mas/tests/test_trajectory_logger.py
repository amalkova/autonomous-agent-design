"""Тести trajectory logger з agent_name."""

import json
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from trajectory_logger import save_trajectory


def test_trajectory_persists_agent_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory.json"

    save_trajectory(
        agent_type="multi_agent_system",
        agent_name="requirements_agent",
        user_input="Перевір DEM-990.",
        messages=[
            HumanMessage(
                content="Перевір DEM-990."
            ),
            AIMessage(
                content="Перевірено."
            ),
        ],
        final_response={
            "completed": True,
        },
        safety={
            "blocked": False,
        },
        metadata={
            "thread_id": "trajectory-test",
        },
        path=path,
    )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    run = payload["runs"][0]

    assert run["agent_name"] == (
        "requirements_agent"
    )
    assert all(
        message["agent_name"]
        == "requirements_agent"
        for message in run["messages"]
    )
