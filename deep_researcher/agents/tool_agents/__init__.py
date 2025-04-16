from pydantic import BaseModel, Field

class ToolAgentOutput(BaseModel):
    """Standard output for all tool agents"""
    output: str
    sources: list[str] = Field(default_factory=list)

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
