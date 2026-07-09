"""Pytest scaffold for the deep-research Monocle test suite.

Enables Monocle tracing, loads the repo `.env`, and exposes ``run_openaiagents``
-- the single entry the live tests use to drive the agent under instrumentation.
"""
from pathlib import Path

from dotenv import load_dotenv
from monocle_apptrace import setup_monocle_telemetry

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
REPO_ROOT = HERE.parent.parent

setup_monocle_telemetry(workflow_name="openai-agents-deep-research")
load_dotenv(REPO_ROOT / ".env")


async def run_openaiagents(message: str) -> str:
    """Run the deep-research agent once and return its final report text."""
    from deep_researcher.iterative_research import IterativeResearcher

    researcher = IterativeResearcher(
        max_iterations=1,
        max_time_minutes=3,
        verbose=False,
        tracing=False,
    )
    return await researcher.run(message)
