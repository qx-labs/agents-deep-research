import asyncio
import time
from .iterative_research import IterativeResearcher
from .agents.planner_agent import init_planner_agent, ReportPlan, ReportPlanSection
from .agents.proofreader_agent import ReportDraftSection, ReportDraft, init_proofreader_agent
from .agents.long_writer_agent import init_long_writer_agent, write_report
from .agents.baseclass import ResearchRunner
from typing import List, Optional
from agents.tracing import trace, gen_trace_id, custom_span
from .llm_config import LLMConfig, create_default_config


class DeepResearcher:
    """
    Manager for the deep research workflow that breaks down a query into a report plan with sections and then runs an iterative research loop for each section.
    """
    def __init__(
            self, 
            max_iterations: int = 5,
            max_time_minutes: int = 10,
            verbose: bool = True,
            tracing: bool = False,
            config: Optional[LLMConfig] = None
        ):
        self.max_iterations = max_iterations
        self.max_time_minutes = max_time_minutes
        self.verbose = verbose
        self.tracing = tracing
        self.config = create_default_config() if not config else config
        self.planner_agent = init_planner_agent(self.config)
        self.proofreader_agent = init_proofreader_agent(self.config)
        self.long_writer_agent = init_long_writer_agent(self.config)

    async def run(self, query: str) -> str:
        """Run the deep research workflow"""
        start_time = time.time()

        if self.tracing:
            trace_id = gen_trace_id()
            workflow_trace = trace("deep_researcher", trace_id=trace_id)
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            workflow_trace.start(mark_as_current=True)

        # First build the report plan which outlines the sections and compiles any relevant background context on the query
        report_plan: ReportPlan = await self._build_report_plan(query)

        # Run the independent research loops concurrently for each section and gather the results
        research_results: List[str] = await self._run_research_loops(report_plan)

        # Create the final report from the original report plan and the drafts of each section
        final_report: str = await self._create_final_report(query, report_plan, research_results)

        elapsed_time = time.time() - start_time
        self._log_message(f"DeepResearcher completed in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds")

        if self.tracing:
            workflow_trace.finish(reset_current=True)

        return final_report

    async def _build_report_plan(self, query: str) -> ReportPlan:
        """Build the initial report plan including the report outline (sections and key questions) and background context"""
        if self.tracing:
            span = custom_span(name="build_report_plan")
            span.start(mark_as_current=True)

        self._log_message("=== Building Report Plan ===")
        user_message = f"QUERY: {query}"
        result = await ResearchRunner.run(
            self.planner_agent,
            user_message
        )
        report_plan = result.final_output_as(ReportPlan)

        if self.verbose:
            num_sections = len(report_plan.report_outline)
            message_log = '\n\n'.join(f"Section: {section.title}\nKey question: {section.key_question}" for section in report_plan.report_outline)
            if report_plan.background_context:
                message_log += f"\n\nThe following background context has been included for the report build:\n{report_plan.background_context}"
            else:
                message_log += "\n\nNo background context was provided for the report build.\n"
            self._log_message(f"Report plan created with {num_sections} sections:\n{message_log}")

        if self.tracing:
            span.finish(reset_current=True)

        return report_plan

    async def _run_research_loops(
        self, 
        report_plan: ReportPlan
    ) -> List[str]:
        """For a given ReportPlan, run a research loop concurrently for each section and gather the results"""
        async def run_research_for_section(section: ReportPlanSection):
            iterative_researcher = IterativeResearcher(
                max_iterations=self.max_iterations,
                max_time_minutes=self.max_time_minutes,
                verbose=self.verbose,
                tracing=False,  # Do not trace as this will conflict with the tracing we already have set up for the deep researcher
                config=self.config
            )
            args = {
                "query": section.key_question,
                "output_length": "",
                "output_instructions": "",
                "background_context": report_plan.background_context,
            }
            
            # Only use custom span if tracing is enabled
            if self.tracing:
                with custom_span(
                    name=f"iterative_researcher:{section.title}", 
                    data={"key_question": section.key_question}
                ):
                    return await iterative_researcher.run(**args)
            else:
                return await iterative_researcher.run(**args)
        
        self._log_message("=== Initializing Research Loops ===")
        # Run all research loops concurrently in a single gather call
        research_results = await asyncio.gather(
            *(run_research_for_section(section) for section in report_plan.report_outline)
        )
        return research_results

    async def _create_final_report(
        self, 
        query: str, 
        report_plan: ReportPlan, 
        section_drafts: List[str],
        use_long_writer: bool = True
    ) -> str:
        """Create the final report from the original report plan and the drafts of each section"""
        if self.tracing:
            span = custom_span(name="create_final_report")
            span.start(mark_as_current=True)

        # Each section is a string containing the markdown for the section
        # From this we need to build a ReportDraft object to feed to the final proofreader agent
        report_draft = ReportDraft(
            sections=[]
        )
        for i, section_draft in enumerate(section_drafts):
            report_draft.sections.append(
                ReportDraftSection(
                    section_title=report_plan.report_outline[i].title,
                    section_content=section_draft
                )
            )

        self._log_message("\n=== Building Final Report ===")

        if use_long_writer:
            final_output = await write_report(self.long_writer_agent, query, report_plan.report_title, report_draft)
        else:
            user_prompt = f"QUERY:\n{query}\n\nREPORT DRAFT:\n{report_draft.model_dump_json()}"
            # Run the proofreader agent to produce the final report
            final_report = await ResearchRunner.run(
                self.proofreader_agent,
                user_prompt
            )
            final_output = final_report.final_output

        self._log_message(f"Final report completed")

        if self.tracing:
            span.finish(reset_current=True)

        return final_output

    def _log_message(self, message: str) -> None:
        """Log a message if verbose is True"""
        if self.verbose:
            print(message)