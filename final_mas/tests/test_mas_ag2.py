from typing import Any

import pytest
from pydantic import ValidationError

from mas_ag2 import RouteDecision, run_ag2_mas


class FakeReply:
    def __init__(
        self,
        *,
        body: str | None = None,
        content: Any = None,
    ) -> None:
        self.body = body
        self._content = content

    async def content(self) -> Any:
        return self._content


class FakeAgent:
    def __init__(self, reply: FakeReply) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def ask(self, message: str) -> FakeReply:
        self.calls.append(message)
        return self.reply


def build_fake_registry(
    selected_agent: str,
) -> dict[str, FakeAgent]:
    return {
        "demand_supervisor": FakeAgent(
            FakeReply(
                content=RouteDecision(
                    action=selected_agent,
                    reasoning="Offline routing decision.",
                )
            )
        ),
        "requirements_agent": FakeAgent(
            FakeReply(body="Requirements result.")
        ),
        "solution_security_agent": FakeAgent(
            FakeReply(body="Solution and security result.")
        ),
        "estimation_agent": FakeAgent(
            FakeReply(body="Estimation result.")
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected_agent, expected_answer",
    [
        (
            "requirements_agent",
            "Requirements result.",
        ),
        (
            "solution_security_agent",
            "Solution and security result.",
        ),
        (
            "estimation_agent",
            "Estimation result.",
        ),
    ],
)
async def test_ag2_routes_to_selected_specialist(
    selected_agent: str,
    expected_answer: str,
) -> None:
    agents = build_fake_registry(selected_agent)

    result = await run_ag2_mas(
        "Offline demand request.",
        agents=agents,
    )

    assert result.framework == "AG2"
    assert result.selected_agent == selected_agent
    assert result.final_answer == expected_answer
    assert result.handoff_count == 1

    assert len(agents["demand_supervisor"].calls) == 1
    assert len(agents[selected_agent].calls) == 1

    other_agents = {
        "requirements_agent",
        "solution_security_agent",
        "estimation_agent",
    } - {selected_agent}

    for agent_name in other_agents:
        assert agents[agent_name].calls == []


@pytest.mark.asyncio
async def test_ag2_serializes_structured_specialist_output() -> None:
    agents = build_fake_registry("requirements_agent")
    agents["requirements_agent"] = FakeAgent(
        FakeReply(
            content={
                "status": "success",
                "requires_human_input": True,
            }
        )
    )

    result = await run_ag2_mas(
        "Check requirements.",
        agents=agents,
    )

    assert '"status": "success"' in result.final_answer
    assert '"requires_human_input": true' in result.final_answer


@pytest.mark.asyncio
async def test_ag2_rejects_unknown_route() -> None:
    agents = build_fake_registry("requirements_agent")
    agents["demand_supervisor"] = FakeAgent(
        FakeReply(
            content={
                "action": "unknown_agent",
                "reasoning": "Invalid route.",
            }
        )
    )

    with pytest.raises(ValidationError):
        await run_ag2_mas(
            "Invalid routing test.",
            agents=agents,
        )