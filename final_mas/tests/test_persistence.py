"""Тести SQLite persistence та static breakpoints."""

from __future__ import annotations

from pathlib import Path

import persistence_demo


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
