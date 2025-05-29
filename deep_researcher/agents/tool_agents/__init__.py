from typing import List
from pydantic import BaseModel, Field

class ToolAgentOutput(BaseModel):
    """Output from a tool agent"""
    output: str = Field(description="The output from the tool agent")
    sources: List[str] = Field(description="List of source URLs used by the tool agent", default_factory=list)
    metadata: dict = Field(description="Additional metadata about the tool execution", default_factory=dict)

from .search_agent import init_search_agent
from .crawl_agent import init_crawl_agent
from ...llm_config import LLMConfig
from ..baseclass import ResearchAgent

def init_tool_agents(config: LLMConfig) -> dict[str, ResearchAgent]:
    search_agent = init_search_agent(config)
    crawl_agent = init_crawl_agent(config)

    return {
        "WebSearchAgent": search_agent,
        "SiteCrawlerAgent": crawl_agent,
    }
