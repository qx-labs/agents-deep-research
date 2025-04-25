"""
Test suite for different model providers, as specified in tests/config.py.

These tests check that:
- The model providers correctly return a response when run by a simple agent
- The model returns the expected content in its output
"""
import pytest
from deep_researcher.agents.writer_agent import init_writer_agent
from deep_researcher import ResearchRunner
from deep_researcher import LLMConfig
from .config import PROVIDERS_TO_TEST, SEARCH_PROVIDER


@pytest.mark.parametrize("provider, model", list(PROVIDERS_TO_TEST.items()))
@pytest.mark.asyncio
async def test_model_provider(provider, model):
    config = LLMConfig(
        search_provider=SEARCH_PROVIDER,
        reasoning_model_provider=provider,
        reasoning_model=model,
        main_model_provider=provider,
        main_model=model,
        fast_model_provider=provider,
        fast_model=model,
    )
    agent = init_writer_agent(config)

    query = "What is the temperature in London today?"
    findings = "The temperature in London today is 36 degrees Celsius. Source: weather.com"

    input_str = f"""
    QUERY: {query}

    FINDINGS: {findings}
    """
    result = await ResearchRunner.run(agent, input_str)
    output_str = result.final_output

    print(f"Testing {provider}/{model}:\n{output_str}\n")

    assert isinstance(output_str, str), f"The model {provider}/{model} failed to return a string"
    assert len(output_str) > 0, f"The model {provider}/{model} failed to return any output"
    assert "36" in output_str, f"The model {provider}/{model} failed to return the correct output"
