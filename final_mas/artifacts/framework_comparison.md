# LangGraph vs AG2 — Final MAS

## Same-query benchmark

| Scenario | LangGraph agent | AG2 agent | LangGraph ms | AG2 ms | Pass |
|---|---|---|---:|---:|---|
| BENCH-001 | requirements_agent | requirements_agent | 7912.4 | 19820.4 | True |
| BENCH-002 | solution_security_agent | solution_security_agent | 26911.8 | 12948.1 | True |
| BENCH-003 | estimation_agent | estimation_agent | 35467.0 | 12676.1 | True |

Both frameworks processed the same three queries using the same
Gemini model and the same Requirements & Estimation business domain.

## Aggregate metrics

| Metric | LangGraph | AG2 |
|---|---:|---:|
| Model calls | 13 | 8 |
| Prompt tokens | 8608 | 3059 |
| Completion tokens | 1346 | 1996 |
| Total tokens | 9954 | 5055 |
| Framework code lines | 788 | 246 |
| Development time, minutes | 420 | 90 |
| Control, 1–5 | 5 | 3 |
| Debugging, 1–5 | 5 | 3 |

Development time is the recorded coursework implementation estimate:
LangGraph includes explicit orchestration, persistence, nested patterns
and HITL; AG2 is an adaptation over the shared MCP/domain layer.

## Conclusion

LangGraph provides stronger workflow control, durable state,
breakpoints, replay and hierarchical tracing. It requires more code and
its nested Plan-and-Execute path can consume more tokens.

AG2 expresses supervisor-to-specialist delegation more compactly and is
easier to read as ordinary asynchronous Python. However, checkpoint
semantics, graph replay and node-level control are less explicit.

LangGraph remains the production choice for this case. AG2 is the
preferred compact alternative for simpler stateless collaboration.
