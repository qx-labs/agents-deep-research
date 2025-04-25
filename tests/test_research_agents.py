"""
Test suite for the main research agents.

These tests check that:
- The agents correctly format their output as the correct type (e.g. structured outputs where applicable)
- The agents produce the expected output content
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
from deep_researcher.agents.knowledge_gap_agent import init_knowledge_gap_agent, KnowledgeGapOutput
from deep_researcher.agents.long_writer_agent import init_long_writer_agent, LongWriterOutput, write_report, ReportDraft
from deep_researcher.agents.planner_agent import init_planner_agent, ReportPlan
from deep_researcher.agents.proofreader_agent import init_proofreader_agent, ReportDraft, ReportDraftSection
from deep_researcher.agents.writer_agent import init_writer_agent
from deep_researcher.agents.thinking_agent import init_thinking_agent
from deep_researcher.agents.tool_selector_agent import init_tool_selector_agent, AgentSelectionPlan, AgentTask

# Configuration for LLMs
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
async def test_knowledge_gap_agent():
    """Test the KnowledgeGapAgent."""
    agent = init_knowledge_gap_agent(config)

    # Example of a user input that requires research to complete (i.e. knowledge gaps exist)
    initial_user_input = """
    ORIGINAL QUERY: What is the founding history of the company QX Labs (qxlabs.com)?

    HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
    - No research has been carried out yet.
    """
    result = await ResearchRunner.run(agent, initial_user_input)
    agent_output = result.final_output_as(KnowledgeGapOutput)

    assert isinstance(agent_output, KnowledgeGapOutput), "The KnowledgeGapAgent is not correctly formatting its structured output"
    assert agent_output.research_complete is False, "The KnowledgeGapAgent is not correctly evaluating research as incomplete"
    assert len(agent_output.outstanding_gaps) > 0, "The KnowledgeGapAgent is not correctly identifying knowledge gaps"
    
    # Example of a user input that doesn't require further research to complete (i.e. no knowledge gaps exist)
    final_user_input = """
    ORIGINAL QUERY: What is the capital of France?

    HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
    - Thinking: I need to run a web search to find the capital of France
    - Action: Running WebSearchAgent with query 'France capital city'
    - Findings: The capital of France is Paris
    """
    result = await ResearchRunner.run(agent, final_user_input)
    agent_output = result.final_output_as(KnowledgeGapOutput)

    assert isinstance(agent_output, KnowledgeGapOutput), "The KnowledgeGapAgent is not correctly formatting its structured output"
    assert agent_output.research_complete is True, "The KnowledgeGapAgent is not correctly evaluating research as complete"
    assert len(agent_output.outstanding_gaps) == 0, "The KnowledgeGapAgent is not correctly identifying knowledge gaps"


@pytest.mark.asyncio
async def test_long_writer_agent():
    """Test the LongWriterAgent."""
    agent = init_long_writer_agent(config)
    user_input = """
    ===========================================================
    ORIGINAL QUERY: Tell me about common dog breeds.

    CURRENT REPORT DRAFT: # Dog Breeds Report\n\n## Table of Contents\n1. Introduction\n2. Golden Retrievers\n\n## Introduction\n\nThis is a report about dog breeds.

    TITLE OF NEXT SECTION TO WRITE: Golden Retrievers

    DRAFT OF NEXT SECTION: Golden Retrievers are friendly dogs [1](https://goldies.com/dogs). They are good family pets.
    ===========================================================
    """
    result = await ResearchRunner.run(agent, user_input)
    agent_output = result.final_output_as(LongWriterOutput)

    assert isinstance(agent_output, LongWriterOutput), "The LongWriterAgent is not correctly formatting its structured output"
    assert "golden retrievers" in agent_output.next_section_markdown.lower(), "The LongWriterAgent is not correctly writing markdown content"
    assert len(agent_output.references) >= 1, "No references are present in the output of the LongWriterAgent"
    assert "goldies.com" in agent_output.references[0], "The correct reference is not present in the output of the LongWriterAgent"


@pytest.mark.asyncio
async def test_planner_agent():
    """Test the PlannerAgent."""
    agent = init_planner_agent(config)
    # NOTE: This agent uses tools (search/crawl) which might make the test slow/flaky
    # For a robust unit test, we might mock these tools, but for now, we'll run it.
    user_input = """
    QUERY: What are the main features of the Python programming language?
    """
    result = await ResearchRunner.run(agent, user_input)
    agent_output = result.final_output_as(ReportPlan)

    assert isinstance(agent_output, ReportPlan), "The PlannerAgent is not correctly formatting its structured output"
    assert len(agent_output.report_outline) > 0, "The PlannerAgent is not producing a report outline"
    assert "python" in agent_output.report_title.lower(), "The PlannerAgent is not correctly naming the report"


@pytest.mark.asyncio
async def test_proofreader_agent():
    """Test the ProofreaderAgent."""
    agent = init_proofreader_agent(config)
    query = "Write a short summary of the sun."
    draft = ReportDraft(
        sections=[
            ReportDraftSection(section_title="Introduction", section_content="The sun is a star. It is big [1]."),
            ReportDraftSection(section_title="Details", section_content="The sun provides light [2]. It is very hot."),
            ReportDraftSection(section_title="References", section_content="[1] https://example.com/sun [2] https://example.com/light")
        ]
    )
    user_input = f"""
    ====
    QUERY:
    {query}

    REPORT DRAFT:
    {draft.model_dump_json()}
    ====
    """
    result = await ResearchRunner.run(agent, user_input)
    final_report = result.final_output

    assert isinstance(final_report, str), "The ProofreaderAgent is not correctly formatting its output as a string"
    assert "## Introduction" in final_report, "The ProofreaderAgent is not correctly adding section headings"


@pytest.mark.asyncio
async def test_writer_agent():
    """Test the WriterAgent."""
    agent = init_writer_agent(config)
    user_input = """
    ===========================================================
    QUERY: Who is the founder of QX Labs?

    FINDINGS:
    - QX Labs was founded by Jai Juneja [1].
    - It is based in London[2].
    - Sources:
      [1] https://qxlabs.com
      [2] https://qxlabs.com/about
    ===========================================================
    """
    result = await ResearchRunner.run(agent, user_input)
    final_report = result.final_output # Writer outputs raw markdown

    assert isinstance(final_report, str), "The WriterAgent is not correctly formatting its output as a string"
    assert "jai" in final_report.lower(), "The WriterAgent is not producing the expected markdown content"
    assert "qxlabs.com" in final_report, "The WriterAgent is not correctly citing references"


@pytest.mark.asyncio
async def test_thinking_agent():
    """Test the ThinkingAgent."""
    agent = init_thinking_agent(config)
    user_input = """
    ORIGINAL QUERY: What is the story of QX Labs (qxlabs.com)?

    HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
    - Iteration 1:
        - Thought: I need learn more about what QX Labs is and what they do
        - Action: Run SiteCrawlerAgent for qxlabs.com
        - Findings: QX Labs is an AI research company that develops workflow automation tools.
    - Iteration 2:
        - Thought: I need to find out more about the founding history of QX Labs.
        - Action: Run WebSearchAgent for 'QX Labs founding history'
        - Findings: QX Labs was founded in 2023 by Jai Juneja and is based in London.
    """
    result = await ResearchRunner.run(agent, user_input)
    thoughts = result.final_output # Thinking agent outputs raw text

    assert isinstance(thoughts, str), "The ThinkingAgent is not correctly formatting its output as a string"
    assert len(thoughts) > 0, "The ThinkingAgent is not producing any output"
    assert any([keyword in thoughts.lower() for keyword in ['jai', 'iteration', 'qx', 'london', '2023', 'found', 'automation']]), "The ThinkingAgent is not producing any relevant thoughts"


@pytest.mark.asyncio
async def test_tool_selector_agent():
    """Test the ToolSelectorAgent."""
    agent = init_tool_selector_agent(config)
    user_input = """
    ORIGINAL QUERY: Provide a comprehensive overview of the company QX Labs (qxlabs.com)

    KNOWLEDGE GAP TO ADDRESS: Need to find out what products or services QX Labs (qxlabs.com) offers.

    BACKGROUND CONTEXT: QX Labs is an AI research company that develops workflow automation tools.

    HISTORY OF ACTIONS, FINDINGS AND THOUGHTS: No research has been carried out yet.
    """
    result = await ResearchRunner.run(agent, user_input)
    agent_output = result.final_output_as(AgentSelectionPlan)

    assert isinstance(agent_output, AgentSelectionPlan), "The ToolSelectorAgent is not correctly formatting its output as an AgentSelectionPlan"
    assert len(agent_output.tasks) > 0, "The ToolSelectorAgent is not producing any tasks"
    assert agent_output.tasks[0].agent in ["WebSearchAgent", "SiteCrawlerAgent"], "The ToolSelectorAgent is not correctly specifying an agent to hand off to"
