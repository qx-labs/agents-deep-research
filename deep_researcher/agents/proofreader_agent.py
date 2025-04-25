"""
Agent used to produce the final draft of a report given initial drafts of each section.

The Agent takes as input the original user query and a stringified object of type ReportDraft.model_dump_json() (defined below).

====
QUERY: <original user query>

REPORT DRAFT: <stringified ReportDraft object containing all draft sections>
====

The Agent then outputs the final markdown for the report as a string.
"""

from pydantic import BaseModel, Field
from typing import List
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig
from datetime import datetime


class ReportDraftSection(BaseModel):
    """A section of the report that needs to be written"""
    section_title: str = Field(description="The title of the section")
    section_content: str = Field(description="The content of the section")


class ReportDraft(BaseModel):
    """Output from the Report Planner Agent"""
    sections: List[ReportDraftSection] = Field(description="List of sections that are in the report")


INSTRUCTIONS = f"""
You are a research expert who proofreads and edits research reports.
Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You are given:
1. The original query topic for the report
2. A first draft of the report in ReportDraft format containing each section in sequence

Your task is to:
1. **Combine sections:** Concatenate the sections into a single string
2. **Add section titles:** Add the section titles to the beginning of each section in markdown format, as well as a main title for the report
3. **De-duplicate:** Remove duplicate content across sections to avoid repetition
4. **Remove irrelevant sections:** If any sections or sub-sections are completely irrelevant to the query, remove them
5. **Refine wording:** Edit the wording of the report to be polished, concise and punchy, but **without eliminating any detail** or large chunks of text
6. **Add a summary:** Add a short report summary / outline to the beginning of the report to provide an overview of the sections and what is discussed
7. **Preserve sources:** Preserve all sources / references - move the long list of references to the end of the report
8. **Update reference numbers:** Continue to include reference numbers in square brackets  ([1], [2], [3], etc.) in the main body of the report, but update the numbering to match the new order of references at the end of the report
9. **Output final report:** Output the final report in markdown format (do not wrap it in a code block)

Guidelines:
- Do not add any new facts or data to the report
- Do not remove any content from the report unless it is very clearly wrong, contradictory or irrelevant
- Remove or reformat any redundant or excessive headings, and ensure that the final nesting of heading levels is correct
- Ensure that the final report flows well and has a logical structure
- Include all sources and references that are present in the final report
"""

def init_proofreader_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model

    return ResearchAgent(
        name="ProofreaderAgent",
        instructions=INSTRUCTIONS,
        model=selected_model
    )