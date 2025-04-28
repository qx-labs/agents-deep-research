import json
import os
import ssl
import aiohttp
import asyncio
from agents import function_tool
from ..agents.baseclass import ResearchAgent, ResearchRunner
from ..agents.utils.parse_output import create_type_parser
from typing import List, Union, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ..llm_config import LLMConfig, model_supports_structured_output

load_dotenv()
CONTENT_LENGTH_LIMIT = 10000  # Trim scraped content to this length to avoid large context / token limit issues

# ------- DEFINE TYPES -------

class ScrapeResult(BaseModel):
    url: str = Field(description="The URL of the webpage")
    text: str = Field(description="The full text content of the webpage")
    title: str = Field(description="The title of the webpage")
    description: str = Field(description="A short description of the webpage")


class WebpageSnippet(BaseModel):
    url: str = Field(description="The URL of the webpage")
    title: str = Field(description="The title of the webpage")
    description: Optional[str] = Field(description="A short description of the webpage")

class SearchResults(BaseModel):
    results_list: List[WebpageSnippet]

# ------- DEFINE TOOL -------

def create_web_search_tool(config: LLMConfig) -> function_tool:
    filter_agent = init_filter_agent(config)
    serper_client = SerperClient(filter_agent)

    @function_tool
    async def web_search(query: str) -> Union[List[ScrapeResult], str]:
        """Perform a web search for a given query and get back the URLs along with their titles, descriptions and text contents.
        
        Args:
            query: The search query
            
        Returns:
            List of ScrapeResult objects which have the following fields:
                - url: The URL of the search result
                - title: The title of the search result
                - description: The description of the search result
                - text: The full text content of the search result
        """
        try:
            search_results = await serper_client.search(query, filter_for_relevance=True, max_results=5)
            results = await scrape_urls(search_results)
            return results
        except Exception as e:
            # Return a user-friendly error message
            return f"Sorry, I encountered an error while searching: {str(e)}"

    return web_search

# ------- DEFINE AGENT FOR FILTERING SEARCH RESULTS BY RELEVANCE -------

FILTER_AGENT_INSTRUCTIONS = f"""
You are a search result filter. Your task is to analyze a list of SERP search results and determine which ones are relevant
to the original query based on the link, title and snippet. Return only the relevant results in the specified format. 

- Remove any results that refer to entities that have similar names to the queried entity, but are not the same.
- E.g. if the query asks about a company "Amce Inc, acme.com", remove results with "acmesolutions.com" or "acme.net" in the link.

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{SearchResults.model_json_schema()}
"""

def init_filter_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.reasoning_model

    return ResearchAgent(
        name="SearchFilterAgent",
        instructions=FILTER_AGENT_INSTRUCTIONS,
        model=selected_model,
        output_type=SearchResults if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(SearchResults) if not model_supports_structured_output(selected_model) else None
    )

# ------- DEFINE UNDERLYING TOOL LOGIC -------

# Create a shared connector
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')  # Add this line to allow older cipher suites


class SerperClient:
    """A client for the Serper API to perform Google searches."""

    def __init__(self, filter_agent: ResearchAgent, api_key: str = None):
        self.filter_agent = filter_agent
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set SERPER_API_KEY environment variable.")
        
        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

    async def search(self, query: str, filter_for_relevance: bool = True, max_results: int = 5) -> List[WebpageSnippet]:
        """Perform a Google search using Serper API and fetch basic details for top results.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return (max 10)
            
        Returns:
            Dictionary with search results
        """
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self.url,
                headers=self.headers,
                json={"q": query, "autocorrect": False}
            ) as response:
                response.raise_for_status()
                results = await response.json()
                results_list = [
                    WebpageSnippet(
                        url=result.get('link', ''),
                        title=result.get('title', ''),
                        description=result.get('snippet', '')
                    )
                    for result in results.get('organic', [])
                ]
                
        if not results_list:
            return []
            
        if not filter_for_relevance:
            return results_list[:max_results]
            
        return await self._filter_results(results_list, query, max_results=max_results)

    async def _filter_results(self, results: List[WebpageSnippet], query: str, max_results: int = 5) -> List[WebpageSnippet]:
        serialized_results = [result.model_dump() if isinstance(result, WebpageSnippet) else result for result in results]
        
        user_prompt = f"""
        Original search query: {query}
        
        Search results to analyze:
        {json.dumps(serialized_results, indent=2)}
        
        Return {max_results} search results or less.
        """
        
        try:
            result = await ResearchRunner.run(self.filter_agent, user_prompt)
            output = result.final_output_as(SearchResults)
            return output.results_list
        except Exception as e:
            print("Error filtering results:", str(e))
            return results[:max_results]


async def scrape_urls(items: List[WebpageSnippet]) -> List[ScrapeResult]:
    """Fetch text content from provided URLs.
    
    Args:
        items: List of SearchEngineResult items to extract content from
        
    Returns:
        List of ScrapeResult objects which have the following fields:
            - url: The URL of the search result
            - title: The title of the search result
            - description: The description of the search result
            - text: The full text content of the search result
    """
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create list of tasks for concurrent execution
        tasks = []
        for item in items:
            if item.url:  # Skip empty URLs
                tasks.append(fetch_and_process_url(session, item))
                
        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and return successful results
        return [r for r in results if isinstance(r, ScrapeResult)]


async def fetch_and_process_url(session: aiohttp.ClientSession, item: WebpageSnippet) -> ScrapeResult:
    """Helper function to fetch and process a single URL."""

    if not is_valid_url(item.url):
        return ScrapeResult(
            url=item.url,
            title=item.title,
            description=item.description,
            text=f"Error fetching content: URL contains restricted file extension"
        )

    try:
        async with session.get(item.url, timeout=8) as response:
            if response.status == 200:
                content = await response.text()
                # Run html_to_text in a thread pool to avoid blocking
                text_content = await asyncio.get_event_loop().run_in_executor(
                    None, html_to_text, content
                )
                text_content = text_content[:CONTENT_LENGTH_LIMIT]  # Trim content to avoid exceeding token limit
                return ScrapeResult(
                    url=item.url,
                    title=item.title,
                    description=item.description,
                    text=text_content
                )
            else:
                # Instead of raising, return a WebSearchResult with an error message
                return ScrapeResult(
                    url=item.url,
                    title=item.title,
                    description=item.description,
                    text=f"Error fetching content: HTTP {response.status}"
                )
    except Exception as e:
        # Instead of raising, return a WebSearchResult with an error message
        return ScrapeResult(
            url=item.url,
            title=item.title,
            description=item.description,
            text=f"Error fetching content: {str(e)}"
        )


def html_to_text(html_content: str) -> str:
    """
    Strips out all of the unnecessary elements from the HTML context to prepare it for text extraction / LLM processing.
    """
    # Parse the HTML using lxml for speed
    soup = BeautifulSoup(html_content, 'lxml')

    # Extract text from relevant tags
    tags_to_extract = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'blockquote')

    # Use a generator expression for efficiency
    extracted_text = "\n".join(element.get_text(strip=True) for element in soup.find_all(tags_to_extract) if element.get_text(strip=True))

    return extracted_text


def is_valid_url(url: str) -> bool:
    """Check that a URL does not contain restricted file extensions."""
    if any(ext in url for ext in [
        ".pdf", 
        ".doc", 
        ".xls",
        ".ppt",
        ".zip",
        ".rar",
        ".7z",
        ".txt", 
        ".js", 
        ".xml", 
        ".css", 
        ".png", 
        ".jpg", 
        ".jpeg", 
        ".gif", 
        ".ico", 
        ".svg", 
        ".webp", 
        ".mp3", 
        ".mp4", 
        ".avi", 
        ".mov", 
        ".wmv", 
        ".flv", 
        ".wma", 
        ".wav", 
        ".m4a", 
        ".m4v", 
        ".m4b", 
        ".m4p", 
        ".m4u"
    ]):
        return False
    return True