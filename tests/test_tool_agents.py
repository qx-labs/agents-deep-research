"""
Test suite for the tool agents (search_agent and crawl_agent).

These tests check that:
- The tool agents successfully call the search/crawl tools
- The search and crawl tools correctly return results
- The tool agents correctly process the tool output and parse it into the correct format
"""
import pytest
from .config import \
    SEARCH_PROVIDER, \
    REASONING_MODEL_PROVIDER, \
    REASONING_MODEL, \
    MAIN_MODEL_PROVIDER, \
    MAIN_MODEL, \
    FAST_MODEL_PROVIDER, \
    FAST_MODEL
from deep_researcher import LLMConfig, ResearchRunner
from deep_researcher.agents.tool_agents import ToolAgentOutput
from deep_researcher.agents.tool_agents.search_agent import init_search_agent
from deep_researcher.agents.tool_agents.crawl_agent import init_crawl_agent
from deep_researcher.agents.tool_selector_agent import AgentTask

config = LLMConfig(
    search_provider=SEARCH_PROVIDER,
    reasoning_model_provider=REASONING_MODEL_PROVIDER,
    reasoning_model=REASONING_MODEL,
    main_model_provider=MAIN_MODEL_PROVIDER,
    main_model=MAIN_MODEL,
    fast_model_provider=FAST_MODEL_PROVIDER,
    fast_model=FAST_MODEL
)


@pytest.mark.asyncio
async def test_search_agent():
    """Test the WebSearchAgent."""
    search_agent = init_search_agent(config)
    agent_task = AgentTask(
        gap="Need to determine the capital of France",
        agent="WebSearchAgent",
        query="France capital city"
    )
    result = await ResearchRunner.run(search_agent, agent_task.model_dump_json())
    agent_output = result.final_output_as(ToolAgentOutput)

    assert isinstance(agent_output, ToolAgentOutput), "The WebSearchAgent is not correctly formatting its output as a ToolAgentOutput"
    assert len(agent_output.output) > 0, "The WebSearchAgent is not correctly retrieving and parsing data from the search tool"
    assert "paris" in agent_output.output.lower(), "The WebSearchAgent is not correctly retrieving and parsing data from the search tool"


@pytest.mark.asyncio
async def test_crawl_agent():
    """Test the SiteCrawlerAgent."""
    crawl_agent = init_crawl_agent(config)
    agent_task = AgentTask(
        gap="Need to determine what the website is about",
        agent="SiteCrawlerAgent",
        query="What is the purpose of this website?",
        entity_website="https://crawler-test.com/"
    )
    result = await ResearchRunner.run(crawl_agent, agent_task.model_dump_json())
    agent_output = result.final_output_as(ToolAgentOutput)

    assert isinstance(agent_output, ToolAgentOutput), "The SiteCrawlerAgent is not correctly formatting its output as a ToolAgentOutput"
    assert len(agent_output.output) > 0, "The SiteCrawlerAgent is not correctly retrieving and parsing data from the crawl tool"
    assert "test" in agent_output.output.lower(), "The SiteCrawlerAgent is not correctly retrieving and parsing data from the crawl tool"
