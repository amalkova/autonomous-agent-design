"""Відтворюване порівняння LangGraph та AG2."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@dataclass
class SourceMetrics:
    framework: str
    file: str
    total_lines: int
    code_lines: int
    functions: int
    classes: int


def collect_source_metrics(
    framework: str,
    path: Path,
) -> SourceMetrics:
    """Зібрати прості відтворювані метрики вихідного коду."""

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    code_lines = sum(
        1
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
    )

    functions = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )

    classes = sum(
        isinstance(node, ast.ClassDef)
        for node in ast.walk(tree)
    )

    return SourceMetrics(
        framework=framework,
        file=path.name,
        total_lines=len(lines),
        code_lines=code_lines,
        functions=functions,
        classes=classes,
    )


def build_markdown(
    metrics: list[SourceMetrics],
) -> str:
    """Сформувати звіт порівняння фреймворків."""

    rows = "\n".join(
        (
            f"| {item.framework} | `{item.file}` | "
            f"{item.total_lines} | {item.code_lines} | "
            f"{item.functions} | {item.classes} |"
        )
        for item in metrics
    )

    return f"""# LangGraph vs AG2 Comparison

## Quantitative source metrics

| Framework | Main file | Total lines | Code lines | Functions | Classes |
|---|---|---:|---:|---:|---:|
{rows}

The metrics are generated directly from the submitted source files and
can be reproduced with `python compare_frameworks.py`.

## Architectural comparison

| Criterion | LangGraph | AG2 v1 |
|---|---|---|
| Coordination | Explicit state graph with nodes and conditional edges | Programmatic supervisor followed by a selected specialist |
| Control | High: every transition and terminal state is declared | Medium-high: routing is explicit, but agent execution is encapsulated |
| State | Typed shared `MASState` | Lightweight `AgentReply` and `AG2MASResult` |
| Handoff | Conditional graph edge | Coordinator calls the selected `Agent` |
| Tool integration | LangChain tools loaded through the MCP adapter | AG2 callable tools delegate to the same MCP client |
| Tool scope | Explicit allowlist per agent | Tools are supplied only to the appropriate AG2 agent |
| Debugging | Graph topology, state and trajectory events | Direct Python call stack, structured route and agent replies |
| HITL | Native interrupt plus checkpointer | Supported by AG2 hooks/middleware; shared project HITL workflow is reused |
| Best fit | Deterministic, stateful and branching workflows | Compact agent-centric orchestration |
| Main trade-off | More orchestration boilerplate | Less visual workflow structure |

## Measured model calls and token usage

Both frameworks were measured using the same Gemini model and an
equivalent estimation request containing the same structured parameters.
Both requests required supervisor routing, an MCP tool call and a final
specialist response.

| Framework | LLM calls | Prompt tokens | Completion tokens | Total tokens |
|---|---:|---:|---:|---:|
| LangGraph | 3 | 1,214 | 464 | 1,678 |
| AG2 v1 | 3 | 1,274 | 413 | 1,687 |

The LangGraph values were exported from the LangSmith trace stored in
`artifacts/langsmith_trace.json`. The AG2 values were collected through
the native asynchronous `AgentReply.usage()` API and stored in
`artifacts/ag2_usage.json`.

AG2 consumed 60 more prompt tokens and 51 fewer completion tokens. Its
total was 9 tokens higher, a difference of approximately 0.54 percent.
For this single measured run, token efficiency is therefore practically
equivalent.

These measurements are a reproducible snapshot rather than a
statistically significant benchmark. Model output length is stochastic,
so repeated runs could change the small difference. The more meaningful
architectural distinction is that LangGraph exposes the complete
execution as a trace with graph nodes, model calls and tool calls,
whereas AG2 provides compact per-agent usage reports directly from each
`AgentReply`.

## Conclusion

LangGraph is the stronger option when the workflow requires explicit
branching, durable state, deterministic transitions and native
interrupt/resume behavior. AG2 v1 is more concise for a supervisor plus
specialists pattern and is easier to follow as ordinary asynchronous
Python.

For the Requirements & Estimation case, LangGraph is selected as the
primary implementation because guardrails, trajectory logging and HITL
are visible parts of the workflow. AG2 is retained as the alternative
implementation because it expresses the same multi-agent case with less
orchestration code.
"""


def main() -> None:
    """Зберегти порівняльні артефакти у Markdown та JSON."""

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    metrics = [
        collect_source_metrics(
            "LangGraph",
            BASE_DIR / "mas_langgraph.py",
        ),
        collect_source_metrics(
            "AG2 v1",
            BASE_DIR / "mas_ag2.py",
        ),
    ]

    markdown_path = (
        ARTIFACTS_DIR / "framework_comparison.md"
    )
    json_path = (
        ARTIFACTS_DIR / "framework_metrics.json"
    )

    markdown_path.write_text(
        build_markdown(metrics),
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(
            [asdict(item) for item in metrics],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(build_markdown(metrics))
    print(f"\nSaved: {markdown_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()