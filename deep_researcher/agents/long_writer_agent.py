"""
Agent used to synthesize a final report by iteratively writing each section of the report.
Used to produce long reports given drafts of each section. Broadly aligned with the methodology described here:


The LongWriterAgent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

CURRENT REPORT DRAFT: <current working draft of the report, all sections up to the current one being written>

TITLE OF NEXT SECTION TO WRITE: <title of the next section of the report to be written>

DRAFT OF NEXT SECTION: <draft of the next section of the report>
===========================================================

The Agent then:
1. Reads the current draft and the draft of the next section
2. Writes the next section of the report
3. Produces an updated draft of the new section to fit the flow of the report
4. Returns the updated draft of the new section along with references/citations
"""
from .baseclass import ResearchAgent, ResearchRunner
from ..llm_config import LLMConfig, model_supports_structured_output
from .utils.parse_output import create_type_parser
from datetime import datetime
from pydantic import BaseModel, Field
from .proofreader_agent import ReportDraft
from typing import List, Tuple, Dict
import re


class LongWriterOutput(BaseModel):
    next_section_markdown: str = Field(description="The final draft of the next section in markdown format")
    references: List[str] = Field(description="A list of URLs and their corresponding reference numbers for the section")


INSTRUCTIONS = f"""
You are an expert report writer tasked with iteratively writing each section of a report. 
Today's date is {datetime.now().strftime('%Y-%m-%d')}.
You will be provided with:
1. The original research query
3. A final draft of the report containing the table of contents and all sections written up until this point (in the first iteration there will be no sections written yet)
3. A first draft of the next section of the report to be written

OBJECTIVE:
1. Write a final draft of the next section of the report with numbered citations in square brackets in the body of the report
2. Produce a list of references to be appended to the end of the report

CITATIONS/REFERENCES:
The citations should be in numerical order, written in numbered square brackets in the body of the report.
Separately, a list of all URLs and their corresponding reference numbers will be included at the end of the report.
Follow the example below for formatting.

LongWriterOutput(
    next_section_markdown="The company specializes in IT consulting [1](https://example.com/first-source-url). It operates in the software services market which is expected to grow at 10% per year [2](https://example.com/second-source-url).",
    references=["[1] https://example.com/first-source-url", "[2] https://example.com/second-source-url"]
)

GUIDELINES:
- You can reformat and reorganize the flow of the content and headings within a section to flow logically, but DO NOT remove details that were included in the first draft
- Only remove text from the first draft if it is already mentioned earlier in the report, or if it should be covered in a later section per the table of contents
- Ensure the heading for the section matches the table of contents
- Format the final output and references section as markdown
- Do not include a title for the reference section, just a list of numbered references

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{LongWriterOutput.model_json_schema()}
"""

def init_long_writer_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model

    return ResearchAgent(
        name="LongWriterAgent",
        instructions=INSTRUCTIONS,
        model=selected_model,
        output_type=LongWriterOutput if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(LongWriterOutput) if not model_supports_structured_output(selected_model) else None
    )


async def write_next_section(
    long_writer_agent: ResearchAgent,
    original_query: str,
    report_draft: str,
    next_section_title: str,
    next_section_draft: str,
) -> LongWriterOutput:
    """Write the next section of the report"""

    user_message = f"""
    <ORIGINAL QUERY>
    {original_query}
    </ORIGINAL QUERY>

    <CURRENT REPORT DRAFT>
    {report_draft or "No draft yet"}
    </CURRENT REPORT DRAFT>

    <TITLE OF NEXT SECTION TO WRITE>
    {next_section_title}
    </TITLE OF NEXT SECTION TO WRITE>

    <DRAFT OF NEXT SECTION>
    {next_section_draft}
    </DRAFT OF NEXT SECTION>
    """

    result = await ResearchRunner.run(
        long_writer_agent,
        user_message,
    )

    return result.final_output_as(LongWriterOutput)


async def write_report(
    long_writer_agent: ResearchAgent,
    original_query: str,
    report_title: str,
    report_draft: ReportDraft,
) -> str:
    """Write the final report by iteratively writing each section"""

    # Initialize the final draft of the report with the title and table of contents
    final_draft = f"# {report_title}\n\n" + "## Table of Contents\n\n" + "\n".join([f"{i+1}. {section.section_title}" for i, section in enumerate(report_draft.sections)]) + "\n\n"
    all_references = []

    for section in report_draft.sections:
        # Produce the final draft of each section and add it to the report with corresponding references
        next_section_draft = await write_next_section(long_writer_agent, original_query, final_draft, section.section_title, section.section_content)
        section_markdown, all_references = reformat_references(
            next_section_draft.next_section_markdown, 
            next_section_draft.references,
            all_references
        )
        section_markdown = reformat_section_headings(section_markdown)
        final_draft += section_markdown + '\n\n'

    # Add the final references to the end of the report
    final_draft += '## References:\n\n' + '  \n'.join(all_references)
    return final_draft


def reformat_references(
        section_markdown: str, 
        section_references: List[str], 
        all_references: List[str] 
    ) -> Tuple[str, List[str]]:
    """
    This method gracefully handles the re-numbering, de-duplication and re-formatting of references as new sections are added to the report draft.
    It takes as input:
    1. The markdown content of the new section containing inline references in square brackets, e.g. [1], [2]
    2. The list of references for the new section, e.g. ["[1] https://example1.com", "[2] https://example2.com"]
    3. The list of references covering all prior sections of the report

    It returns:
    1. The updated markdown content of the new section with the references re-numbered and de-duplicated, such that they increment from the previous references
    2. The updated list of references for the full report, to include the new section's references
    """
    def convert_ref_list_to_map(ref_list: List[str]) -> Dict[str, str]:
        ref_map = {}
        for ref in ref_list:
            try:
                ref_num = int(ref.split(']')[0].strip('['))
                url = ref.split(']', 1)[1].strip()
                ref_map[url] = ref_num
            except ValueError:
                print(f"Invalid reference format: {ref}")
                continue
        return ref_map

    section_ref_map = convert_ref_list_to_map(section_references)
    report_ref_map = convert_ref_list_to_map(all_references)
    section_to_report_ref_map = {}

    report_urls = set(report_ref_map.keys())
    ref_count = max(report_ref_map.values() or [0])
    for url, section_ref_num in section_ref_map.items():
        if url in report_urls:
            section_to_report_ref_map[section_ref_num] = report_ref_map[url]
        else:
            # If the reference is not in the report, add it to the report
            ref_count += 1
            section_to_report_ref_map[section_ref_num] = ref_count
            all_references.append(f"[{ref_count}] {url}")

    def replace_reference(match):
        # Extract the reference number from the match
        ref_num = int(match.group(1))
        # Look up the new reference number
        mapped_ref_num = section_to_report_ref_map.get(ref_num)
        if mapped_ref_num:
            return f'[{mapped_ref_num}]'
        return ''
    
    # Replace all references in a single pass using a replacement function
    section_markdown = re.sub(r'\[(\d+)\]', replace_reference, section_markdown)

    return section_markdown, all_references


def reformat_section_headings(section_markdown: str) -> str:
    """
    Reformat the headings of a section to be consistent with the report, by rebasing the section's heading to be a level-2 heading

    E.g. this:
    # Big Title
    Some content
    ## Subsection

    Becomes this:
    ## Big Title
    Some content
    ### Subsection
    """
    # If the section is empty, return as-is
    if not section_markdown.strip():
        return section_markdown

    # Find the first heading level
    first_heading_match = re.search(r'^(#+)\s', section_markdown, re.MULTILINE)
    if not first_heading_match:
        return section_markdown

    # Calculate the level adjustment needed
    first_heading_level = len(first_heading_match.group(1))
    level_adjustment = 2 - first_heading_level

    def adjust_heading_level(match):
        hashes = match.group(1)
        content = match.group(2)
        new_level = max(2, len(hashes) + level_adjustment)
        return '#' * new_level + ' ' + content

    # Apply the heading adjustment to all headings in one pass
    return re.sub(r'^(#+)\s(.+)$', adjust_heading_level, section_markdown, flags=re.MULTILINE)
