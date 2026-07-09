"""Trace-based behavioural tests for the deep-research agent, using Monocle Test Tools.

One offline test per research question. Each loads the Monocle trace that question
emitted (via ``with_trace_source("file", ...)``) and asserts against it -- which agents
ran, what the run was asked, what it produced, and its token/duration cost. A later
prompt, model, or pipeline change that regresses the behaviour fails here. The offline
tests are keyless: they replay recorded good traces, so they are deterministic and
CI-safe. One live test drives the agent end-to-end and asserts the same shape on a
freshly emitted trace (needs OPENAI_API_KEY + web search, so it is opt-in).

    pytest tests/monocle_test/ -k "not live"   # offline, no keys
    pytest tests/monocle_test/                 # includes the live run

The app is built on the OpenAI Agents SDK. A single research request fans out across a
fixed five-agent pipeline rather than a tool-calling loop: ThinkingAgent plans the run,
KnowledgeGapAgent finds what is still missing, ToolSelectorAgent decides where to look,
WebSearchAgent gathers sources (invoked several times per run), and WriterAgent composes
the final report. Each shows up as its own agentic.invocation span, so behaviour is
asserted through the agents the run invokes. The pipeline emits no agentic.tool.invocation
spans (the web reach-out is modelled as the WebSearchAgent invocation, not a separate tool
span), so called_tool is not used here.
"""
from monocle_test_tools import TraceAssertion

from conftest import TRACES, run_openaiagents

# Each trace was captured from a single real run of the curated question below,
# under monocle_apptrace instrumentation, and committed as a fixture.
TRACE_QUANTUM = {
    "id": "aeabde98c6845bd1c773e92a4d060896",
    "path": str(TRACES / "monocle_trace_openai-agents-deep-research_aeabde98c6845bd1c773e92a4d060896_2026-07-09_12.27.36.json"),
}
TRACE_FASTING = {
    "id": "9b32998e45513f846c8bc785c588864f",
    "path": str(TRACES / "monocle_trace_openai-agents-deep-research_9b32998e45513f846c8bc785c588864f_2026-07-09_12.28.03.json"),
}
TRACE_STORAGE = {
    "id": "1d870c9d5c73be705232773ad40be343",
    "path": str(TRACES / "monocle_trace_openai-agents-deep-research_1d870c9d5c73be705232773ad40be343_2026-07-09_12.28.31.json"),
}
TRACE_MEDITERRANEAN = {
    "id": "f8edfda0c437698cf256ae21cf2d1123",
    "path": str(TRACES / "monocle_trace_openai-agents-deep-research_f8edfda0c437698cf256ae21cf2d1123_2026-07-09_12.28.58.json"),
}


def test_quantum_computing_survey(monocle_trace_asserter: TraceAssertion):
    """Q: "What is the current state of quantum computing?"

    Real trace: ~31.8k tokens, ~20.7s workflow; the full five-agent pipeline with
    WebSearchAgent invoked 3 times.
    """
    monocle_trace_asserter.with_trace_source("file", id=TRACE_QUANTUM["id"], trace_path=TRACE_QUANTUM["path"])

    monocle_trace_asserter.called_agent("ThinkingAgent").contains_input("quantum computing")
    monocle_trace_asserter.called_agent("KnowledgeGapAgent")
    monocle_trace_asserter.called_agent("ToolSelectorAgent")
    monocle_trace_asserter.called_agent("WebSearchAgent", min_count=3)
    monocle_trace_asserter.called_agent("WriterAgent")
    monocle_trace_asserter.does_not_call_agent("PlannerAgent")
    monocle_trace_asserter.contains_output("quantum")
    monocle_trace_asserter.contains_any_output("quantum", "qubit", "computing")
    monocle_trace_asserter.under_token_limit(50_000)
    monocle_trace_asserter.under_duration(60, units="seconds", span_type="workflow")

    # Eval layer (deferred -- set OKAHU_API_KEY and uncomment to enable):
    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")


def test_intermittent_fasting_health_review(monocle_trace_asserter: TraceAssertion):
    """Q: "What are the health benefits of intermittent fasting?"

    A different topic than the quantum survey, exercising the same pipeline on
    lifestyle/health content. Real trace: ~32.2k tokens, ~26.3s workflow; seven agent
    invocations across the five agents.
    """
    monocle_trace_asserter.with_trace_source("file", id=TRACE_FASTING["id"], trace_path=TRACE_FASTING["path"])

    monocle_trace_asserter.called_agents(count=7)
    monocle_trace_asserter.called_agent("ThinkingAgent").contains_input("intermittent fasting")
    monocle_trace_asserter.called_agent("WebSearchAgent", min_count=3)
    monocle_trace_asserter.called_agent("WriterAgent")
    monocle_trace_asserter.contains_output("fasting")
    monocle_trace_asserter.contains_any_output("fasting", "health", "metabolic", "weight")
    monocle_trace_asserter.under_token_limit(50_000)
    monocle_trace_asserter.under_duration(60, units="seconds", span_type="workflow")

    # See test_quantum_computing_survey for the optional Okahu eval layer.


def test_grid_scale_energy_storage(monocle_trace_asserter: TraceAssertion):
    """Q: "What are the latest advances in grid-scale energy storage?"

    An engineering/technology topic. Real trace: ~31.8k tokens, ~25.0s workflow;
    WebSearchAgent invoked 3 times.
    """
    monocle_trace_asserter.with_trace_source("file", id=TRACE_STORAGE["id"], trace_path=TRACE_STORAGE["path"])

    monocle_trace_asserter.called_agent("ThinkingAgent").contains_input("energy storage")
    monocle_trace_asserter.called_agent("WebSearchAgent", min_count=3)
    monocle_trace_asserter.called_agent("WriterAgent")
    monocle_trace_asserter.contains_any_output("energy", "storage", "battery", "grid")
    monocle_trace_asserter.under_token_limit(50_000)
    monocle_trace_asserter.under_duration(60, units="seconds", span_type="workflow")


def test_mediterranean_diet_cardiovascular(monocle_trace_asserter: TraceAssertion):
    """Q: "What does the evidence say about the Mediterranean diet and cardiovascular
    disease risk?"

    A public-health topic. Real trace: ~30.7k tokens, ~20.7s workflow; WebSearchAgent
    invoked 3 times.
    """
    monocle_trace_asserter.with_trace_source("file", id=TRACE_MEDITERRANEAN["id"], trace_path=TRACE_MEDITERRANEAN["path"])

    monocle_trace_asserter.called_agent("ThinkingAgent").contains_input("Mediterranean diet")
    monocle_trace_asserter.called_agent("WebSearchAgent", min_count=3)
    monocle_trace_asserter.called_agent("WriterAgent")
    monocle_trace_asserter.contains_any_output("Mediterranean", "diet", "cardiovascular", "heart")
    monocle_trace_asserter.under_token_limit(50_000)
    monocle_trace_asserter.under_duration(60, units="seconds", span_type="workflow")


# --- Live: run the agent end-to-end on a fresh question -----------------------
# Drives the real pipeline under instrumentation and asserts the same structure +
# budget on the freshly emitted trace. Output text varies run to run, so this asserts
# structure and topic keywords (contains_any_output), not exact output. The runner is
# async, so it is driven via validator.test_workflow_async. Needs OPENAI_API_KEY and
# web search; opt-in via the "live" name (pytest -k "not live" skips it).

def test_carbon_capture_live(monocle_trace_asserter: TraceAssertion):
    """Live research path: methods of carbon capture and storage."""
    import asyncio

    asyncio.run(
        monocle_trace_asserter.validator.test_workflow_async(
            run_openaiagents,
            {"test_input": (
                "What are the main methods used for carbon capture and storage?",
            )},
        )
    )

    monocle_trace_asserter.called_agent("ThinkingAgent").contains_input("carbon capture")
    monocle_trace_asserter.called_agent("WebSearchAgent")
    monocle_trace_asserter.called_agent("WriterAgent")
    monocle_trace_asserter.contains_any_output("carbon", "capture", "storage", "CO2")
    monocle_trace_asserter.under_token_limit(120_000)
    monocle_trace_asserter.under_duration(240, units="seconds", span_type="workflow")

    # Okahu eval layer (deferred -- set OKAHU_API_KEY and uncomment to enable):
    # monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination") \
    #     .check_eval("contextual_precision", "high_precision") \
    #     .check_eval("sentiment", "positive") \
    #     .check_eval("bias", "unbiased")
