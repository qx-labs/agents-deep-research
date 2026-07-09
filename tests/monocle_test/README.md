# Deep-research behavioural tests (Monocle Test Tools)

Trace-based tests that lock in this deep-research agent's behaviour. Monocle records
each run as a structured trace -- every agent invocation, LLM token usage, and timings
-- and each test asserts against that trace: which agents ran, what the run was asked,
what it produced, and its token/duration cost. A later prompt, model, or pipeline change
that regresses the behaviour fails here.

The app is built on the OpenAI Agents SDK. A single research request fans out across a
fixed five-agent pipeline rather than a tool-calling loop: `ThinkingAgent` plans the run,
`KnowledgeGapAgent` finds what is still missing, `ToolSelectorAgent` decides where to
look, `WebSearchAgent` gathers sources (invoked several times per run), and `WriterAgent`
composes the final report. Each shows up as its own agent invocation span, so behaviour
is asserted through the agents the run invokes. The pipeline emits no
`agentic.tool.invocation` spans, so `called_tool` is not used.

## Layout

- `test_openaiagents.py` — the suite: one offline test per research question + one live test
- `conftest.py` — Monocle setup, `.env` loading, and the `run_openaiagents()` runner the
  live test uses to drive the agent under instrumentation
- `traces/` — recorded good-trace fixtures, one per question, that the offline tests replay
- `requirements.txt` — dependencies

## Tests

The four offline tests each load their question's recorded trace by file
(`with_trace_source("file", id=..., trace_path=...)`) and assert structure + budget —
keyless and deterministic. The live test runs the agent end-to-end and asserts the same
shape on a freshly emitted trace.

| Test | Question | What it shows |
|---|---|---|
| `test_quantum_computing_survey` | current state of quantum computing | per-agent invocation, `WebSearchAgent` min count, a negative assertion, input/output, budget |
| `test_intermittent_fasting_health_review` | health benefits of intermittent fasting | aggregate agent count, input/output on a different topic, budget |
| `test_grid_scale_energy_storage` | latest advances in grid-scale energy storage | engineering topic, structure + budget |
| `test_mediterranean_diet_cardiovascular` | Mediterranean diet & cardiovascular disease risk | public-health topic, structure + budget |
| `test_carbon_capture_live` | carbon capture and storage (run live) | live end-to-end run, structure + budget only |

Offline budgets are the real numbers measured from each recorded run (~31–32k tokens,
~20–26s workflow), rounded up with headroom (50k tokens, 60s). The live test uses wider
headroom (120k tokens, 240s) since a fresh run varies.

## Run

```bash
pip install -r requirements.txt

pytest tests/monocle_test/ -k "not live"   # offline only — no keys, no network
pytest tests/monocle_test/                 # includes the live run
```

The offline tests replay committed traces (no keys needed). The live test
(`test_carbon_capture_live`) drives the agent for real, so run it in an environment where
this app is installed, with `OPENAI_API_KEY` in the repo `.env` and web-search access.

## Add your own test

1. Run the agent under Monocle and capture a trace of a run you're happy with
   (Monocle writes trace JSON to `.monocle/` by default).
2. Move it into `traces/` and load it with
   `monocle_trace_asserter.with_trace_source("file", id="<trace_id>", trace_path="<path>")`.
3. Assert with the fluent API — `called_agent(...)`, `contains_input/output(...)`,
   `contains_any_output(...)`, `under_token_limit(...)`,
   `under_duration(..., span_type="workflow")` — then add it alongside the others.

## Evaluations (optional)

Content is checked with the fluent `contains_output` / `contains_any_output` assertions
(keyless, deterministic). For LLM-judge evaluations, `test_quantum_computing_survey`
carries a commented `with_evaluation("okahu").check_eval("hallucination", ...)` chain —
set `OKAHU_API_KEY` (account at https://www.okahu.ai) and uncomment to enable.
