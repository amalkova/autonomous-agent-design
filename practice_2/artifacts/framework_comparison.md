# LangGraph vs AG2 Comparison

## Quantitative source metrics

| Framework | Main file | Total lines | Code lines | Functions | Classes |
|---|---|---:|---:|---:|---:|
| LangGraph | `mas_langgraph.py` | 666 | 570 | 20 | 2 |
| AG2 v1 | `mas_ag2.py` | 304 | 246 | 8 | 2 |

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
