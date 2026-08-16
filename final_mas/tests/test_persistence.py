"""Тести SQLite persistence та static breakpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver,
)

import persistence_demo
from guardrails import RollingWindowRateLimiter
from mas_langgraph import (
    RouteDecision,
    build_mas_graph,
    create_initial_state,
)


def test_persistence_resumes_same_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Checkpoint переживає reopen та resume."""

    database = tmp_path / "agent_state.db"

    monkeypatch.setattr(
        persistence_demo,
        "DATABASE_PATH",
        database,
    )

    thread_id = "persistence-test"

    before = persistence_demo.start(
        thread_id
    )
    inspected = persistence_demo.inspect(
        thread_id
    )
    after = persistence_demo.resume(
        thread_id
    )

    comparison = persistence_demo.compare(
        before,
        inspected,
        after,
    )

    assert database.exists()
    assert comparison[
        "checkpoint_survived_restart"
    ] is True
    assert comparison[
        "paused_before_executor"
    ] is True
    assert comparison[
        "resumed_with_same_thread"
    ] is True


def test_static_interrupt_pauses_before_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """interrupt_before зупиняє graph до approval node."""

    database = tmp_path / "approval_state.db"

    monkeypatch.setattr(
        persistence_demo,
        "DATABASE_PATH",
        database,
    )

    result = (
        persistence_demo
        .demonstrate_approval_breakpoint()
    )

    assert result["paused_before"] == "approval"
    assert result[
        "dynamic_interrupt_entered"
    ] is False
    assert "approval" in result["state"]["next"]


@pytest.mark.asyncio
async def test_supervisor_mas_resumes_after_restart(
    tmp_path: Path,
) -> None:
    """Верхній MAS відновлюється з SQLite checkpoint."""

    database = tmp_path / "mas_state.db"
    thread_id = "mas-persistence-test"
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 20,
    }
    specialist_calls: list[str] = []

    async def route_selector(
        _request: str,
    ) -> RouteDecision:
        return RouteDecision(
            action="requirements_agent",
            reasoning="Requirements persistence test.",
        )

    async def requirements_runner(
        _request: str,
    ) -> str:
        specialist_calls.append(
            "requirements_agent"
        )
        return (
            "Requirements Agent resumed from SQLite."
        )

    async def unexpected_runner(
        _request: str,
    ) -> str:
        raise AssertionError(
            "Unexpected specialist route."
        )

    def build_graph(checkpointer):
        return build_mas_graph(
            route_selector=route_selector,
            agent_runners={
                "requirements_agent": (
                    requirements_runner
                ),
                "solution_security_agent": (
                    unexpected_runner
                ),
                "estimation_agent": (
                    unexpected_runner
                ),
            },
            checkpointer=checkpointer,
            rate_limiter=(
                RollingWindowRateLimiter(
                    max_requests=100,
                    window_seconds=60,
                )
            ),
            interrupt_before=[
                "requirements_agent",
            ],
        )

    async with AsyncSqliteSaver.from_conn_string(
        str(database)
    ) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer)

        await graph.ainvoke(
            create_initial_state(
                "Перевір requirements DEM-940.",
                session_id=thread_id,
            ),
            config=config,
        )
        before_crash = await graph.aget_state(
            config
        )

    assert "requirements_agent" in (
        before_crash.next
    )
    assert specialist_calls == []

    # Новий connection і graph імітують новий process.
    async with AsyncSqliteSaver.from_conn_string(
        str(database)
    ) as checkpointer:
        await checkpointer.setup()
        restored_graph = build_graph(
            checkpointer
        )
        after_restart = (
            await restored_graph.aget_state(
                config
            )
        )

        assert after_restart.next == (
            before_crash.next
        )
        assert after_restart.values[
            "current_agent"
        ] == "requirements_agent"

        result = await restored_graph.ainvoke(
            None,
            config=config,
        )

    assert result["completed"] is True
    assert result["current_agent"] == (
        "requirements_agent"
    )
    assert result["handoff_count"] == 1
    assert specialist_calls == [
        "requirements_agent",
    ]