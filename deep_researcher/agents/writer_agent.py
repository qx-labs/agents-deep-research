"""
Agent used to synthesize a final report based on provided findings.

The WriterAgent takes as input a string in the following format:
===========================================================
QUERY: <original user query>

FINDINGS: <findings from the iterative research process>
===========================================================

The Agent then:
1. Generates a comprehensive markdown report based on all available information
2. Includes proper citations for sources in the format [1], [2], etc.
3. Returns a string containing the markdown formatted report
"""
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig
from datetime import datetime
import re

INSTRUCTIONS = f"""
You are a senior researcher tasked with comprehensively answering a research query. 
Today's date is {datetime.now().strftime('%Y-%m-%d')}

Your task is to:
1. Analyze the provided findings and synthesize them into a coherent, well-structured report
2. Use proper citations in the format [1], [2], etc. when referencing information from sources
3. Ensure all claims are properly supported by citations
4. Write in a clear, professional academic style
5. Include relevant examples and evidence to support your points
6. Maintain objectivity and avoid bias
7. Use proper markdown formatting for headings, lists, and emphasis

The report should be comprehensive yet concise, and should directly address the research query.
"""

def init_writer_agent(config: LLMConfig) -> ResearchAgent:
    """Initialize the writer agent."""
    return ResearchAgent(
        name="WriterAgent",
        instructions=INSTRUCTIONS,
        config=config
    )

def process_citations(text: str) -> str:
    """Process citations in the text to ensure proper formatting."""
    # Replace any citation patterns that don't match [number] with the proper format
    citation_pattern = r'\[([^\]]+)\]'
    
    def replace_citation(match):
        citation = match.group(1)
        # If it's already a number, keep it as is
        if citation.isdigit():
            return f"[{citation}]"
        # Otherwise, try to extract a number from the citation
        numbers = re.findall(r'\d+', citation)
        if numbers:
            return f"[{numbers[0]}]"
        return match.group(0)
    
    return re.sub(citation_pattern, replace_citation, text)

class WriterAgent(ResearchAgent):
    """Agent responsible for writing the final research report."""
    
    async def parse_output(self, result):
        """Process the output to ensure proper citation formatting."""
        if hasattr(result, 'final_output'):
            result.final_output = process_citations(result.final_output)
        return result
