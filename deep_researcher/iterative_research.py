from __future__ import annotations
import asyncio
import time
from typing import Dict, List, Optional
from agents import custom_span, gen_trace_id, trace
from .agents.baseclass import ResearchRunner
from .agents.writer_agent import init_writer_agent
from .agents.knowledge_gap_agent import KnowledgeGapOutput, init_knowledge_gap_agent
from .agents.tool_selector_agent import AgentTask, AgentSelectionPlan, init_tool_selector_agent
from .agents.thinking_agent import init_thinking_agent
from .agents.tool_agents import init_tool_agents, ToolAgentOutput
from pydantic import BaseModel, Field
from .llm_config import LLMConfig, create_default_config


class IterationData(BaseModel):
    """Data for a single iteration of the research loop."""
    gap: str = Field(description="The gap addressed in the iteration", default_factory=list)
    tool_calls: List[str] = Field(description="The tool calls made", default_factory=list)
    findings: List[str] = Field(description="The findings collected from tool calls", default_factory=list)
    thought: List[str] = Field(description="The thinking done to reflect on the success of the iteration and next steps", default_factory=list)


class Conversation(BaseModel):
    """A conversation between the user and the iterative researcher."""
    history: List[IterationData] = Field(description="The data for each iteration of the research loop", default_factory=list)

    def add_iteration(self, iteration_data: Optional[IterationData] = None):
        if iteration_data is None:
            iteration_data = IterationData()
        self.history.append(iteration_data)
    
    def set_latest_gap(self, gap: str):
        self.history[-1].gap = gap

    def set_latest_tool_calls(self, tool_calls: List[str]):
        self.history[-1].tool_calls = tool_calls

    def set_latest_findings(self, findings: List[str]):
        self.history[-1].findings = findings

    def set_latest_thought(self, thought: str):
        self.history[-1].thought = thought

    def get_latest_gap(self) -> str:
        return self.history[-1].gap
    
    def get_latest_tool_calls(self) -> List[str]:
        return self.history[-1].tool_calls
    
    def get_latest_findings(self) -> List[str]:
        return self.history[-1].findings
    
    def get_latest_thought(self) -> str:
        return self.history[-1].thought
    
    def get_all_findings(self) -> List[str]:
        return [finding for iteration_data in self.history for finding in iteration_data.findings]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""
        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"{self.get_thought_string(iteration_num)}\n\n"
            if iteration_data.gap:
                conversation += f"{self.get_task_string(iteration_num)}\n\n"
            if iteration_data.tool_calls:
                conversation += f"{self.get_action_string(iteration_num)}\n\n"
            if iteration_data.findings:
                conversation += f"{self.get_findings_string(iteration_num)}\n\n"

        return conversation
    
    def get_task_string(self, iteration_num: int) -> str:
        """Get the task for the current iteration."""
        if self.history[iteration_num].gap:
            return f"<task>\nAddress this knowledge gap: {self.history[iteration_num].gap}\n</task>"
        return ""
    
    def get_action_string(self, iteration_num: int) -> str:
        """Get the action for the current iteration."""
        if self.history[iteration_num].tool_calls:
            joined_calls = '\n'.join(self.history[iteration_num].tool_calls)
            return (
                "<action>\nCalling the following tools to address the knowledge gap:\n"
                f"{joined_calls}\n</action>"
            )
        return ""
        
    def get_findings_string(self, iteration_num: int) -> str:
        """Get the findings for the current iteration."""
        if self.history[iteration_num].findings:
            joined_findings = '\n\n'.join(self.history[iteration_num].findings)
            return f"<findings>\n{joined_findings}\n</findings>"
        return ""
    
    def get_thought_string(self, iteration_num: int) -> str:
        """Get the thought for the current iteration."""
        if self.history[iteration_num].thought:
            return f"<thought>\n{self.history[iteration_num].thought}\n</thought>"
        return ""
    
    def latest_task_string(self) -> str:
        """Get the latest task."""
        return self.get_task_string(len(self.history) - 1)
    
    def latest_action_string(self) -> str:
        """Get the latest action."""
        return self.get_action_string(len(self.history) - 1)
    
    def latest_findings_string(self) -> str:
        """Get the latest findings."""
        return self.get_findings_string(len(self.history) - 1)
    
    def latest_thought_string(self) -> str:
        """Get the latest thought."""
        return self.get_thought_string(len(self.history) - 1)
    

class IterativeResearcher:
    """Manager for the iterative research workflow that conducts research on a topic or subtopic by running a continuous research loop."""

    def __init__(
        self, 
        max_iterations: int = 5,
        max_time_minutes: int = 10,
        verbose: bool = True,
        tracing: bool = False,
        config: Optional[LLMConfig] = None
    ):
        self.max_iterations: int = max_iterations
        self.max_time_minutes: int = max_time_minutes
        self.start_time: float = None
        self.iteration: int = 0
        self.conversation: Conversation = Conversation()
        self.should_continue: bool = True
        self.verbose: bool = verbose
        self.tracing: bool = tracing
        self.config: LLMConfig = create_default_config() if not config else config
        self.knowledge_gap_agent = init_knowledge_gap_agent(self.config)
        self.tool_selector_agent = init_tool_selector_agent(self.config)
        self.thinking_agent = init_thinking_agent(self.config)
        self.writer_agent = init_writer_agent(self.config)
        self.tool_agents = init_tool_agents(self.config)
        
    async def run(
            self, 
            query: str,
            output_length: str = "",  # A text description of the desired output length, can be left blank
            output_instructions: str = "",  # Instructions for the final report (e.g. don't include any headings, just a couple of paragraphs of text)
            background_context: str = "",
        ) -> str:
        """Run the deep research workflow for a given query."""
        self.start_time = time.time()

        if self.tracing:
            trace_id = gen_trace_id()
            workflow_trace = trace("iterative_researcher", trace_id=trace_id)
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            workflow_trace.start(mark_as_current=True)

        self._log_message("=== Starting Iterative Research Workflow ===")
        
        # Iterative research loop
        while self.should_continue and self._check_constraints():
            self.iteration += 1
            self._log_message(f"\n=== Starting Iteration {self.iteration} ===")

            # Set up blank IterationData for this iteration
            self.conversation.add_iteration()

            # 1. Generate observations
            observations: str = await self._generate_observations(query, background_context=background_context)

            # 2. Evaluate current gaps in the research
            evaluation: KnowledgeGapOutput = await self._evaluate_gaps(query, background_context=background_context)
            
            # Check if we should continue or break the loop
            if not evaluation.research_complete:
                next_gap = evaluation.outstanding_gaps[0]

                # 3. Select agents to address knowledge gap
                selection_plan: AgentSelectionPlan = await self._select_agents(next_gap, query, background_context=background_context)

                # 4. Run the selected agents to gather information
                results: Dict[str, ToolAgentOutput] = await self._execute_tools(selection_plan.tasks)
            else:
                self.should_continue = False
                self._log_message("=== IterativeResearcher Marked As Complete - Finalizing Output ===")
        
        # Create final report
        report = await self._create_final_report(query, length=output_length, instructions=output_instructions)
        
        elapsed_time = time.time() - self.start_time
        self._log_message(f"IterativeResearcher completed in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds after {self.iteration} iterations.")
        
        if self.tracing:
            workflow_trace.finish(reset_current=True)

        return report
    
    def _check_constraints(self) -> bool:
        """Check if we've exceeded our constraints (max iterations or time)."""
        if self.iteration >= self.max_iterations:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum iterations ({self.max_iterations})")
            return False
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes >= self.max_time_minutes:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum time ({self.max_time_minutes} minutes)")
            return False
        
        return True
    
    async def _evaluate_gaps(
        self, 
        query: str,
        background_context: str = ""
    ) -> KnowledgeGapOutput:
        """Evaluate the current state of research and identify knowledge gaps."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        input_str = f"""
        Current Iteration Number: {self.iteration}
        Time Elapsed: {(time.time() - self.start_time) / 60:.2f} minutes of maximum {self.max_time_minutes} minutes

        ORIGINAL QUERY:
        {query}

        {background}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}        
        """

        result = await ResearchRunner.run(
            self.knowledge_gap_agent,
            input_str,
        )
        
        evaluation = result.final_output_as(KnowledgeGapOutput)

        if not evaluation.research_complete:
            next_gap = evaluation.outstanding_gaps[0]
            self.conversation.set_latest_gap(next_gap)
            self._log_message(self.conversation.latest_task_string())
        
        return evaluation
    
    async def _select_agents(
        self, 
        gap: str, 
        query: str,
        background_context: str = ""
    ) -> AgentSelectionPlan:
        """Select agents to address the identified knowledge gap."""
        
        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        input_str = f"""
        ORIGINAL QUERY:
        {query}

        KNOWLEDGE GAP TO ADDRESS:
        {gap}

        {background}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """
        
        result = await ResearchRunner.run(
            self.tool_selector_agent,
            input_str,
        )
        
        selection_plan = result.final_output_as(AgentSelectionPlan)

        # Add the tool calls to the conversation
        self.conversation.set_latest_tool_calls([
            f"[Agent] {task.agent} [Query] {task.query} [Entity] {task.entity_website if task.entity_website else 'null'}" for task in selection_plan.tasks
        ])
        self._log_message(self.conversation.latest_action_string())
        
        return selection_plan
    
    async def _execute_tools(self, tasks: List[AgentTask]) -> Dict[str, ToolAgentOutput]:
        """Execute the selected tools concurrently to gather information."""
        with custom_span("Execute Tool Agents"):
            # Create a task for each agent
            async_tasks = []
            for task in tasks:
                async_tasks.append(self._run_agent_task(task))
            
            # Run all tasks concurrently
            num_completed = 0
            results = {}
            for future in asyncio.as_completed(async_tasks):
                gap, agent_name, result = await future
                results[f"{agent_name}_{gap}"] = result
                num_completed += 1
                self._log_message(f"<processing>\nTool execution progress: {num_completed}/{len(async_tasks)}\n</processing>")

            # Add findings from the tool outputs to the conversation
            findings = []
            for tool_output in results.values():
                findings.append(tool_output.output)
            self.conversation.set_latest_findings(findings)

            return results
    
    async def _run_agent_task(self, task: AgentTask) -> tuple[str, str, ToolAgentOutput]:
        """Run a single agent task and return the result."""
        try:
            agent_name = task.agent
            agent = self.tool_agents.get(agent_name)
            if agent:
                result = await ResearchRunner.run(
                    agent,
                    task.model_dump_json(),
                )
                # Extract ToolAgentOutput from RunResult
                output = result.final_output_as(ToolAgentOutput)
            else:
                output = ToolAgentOutput(
                    output=f"No implementation found for agent {agent_name}",
                    sources=[]
                )
            
            return task.gap, agent_name, output
        except Exception as e:
            error_output = ToolAgentOutput(
                output=f"Error executing {task.agent} for gap '{task.gap}': {str(e)}",
                sources=[]
            )
            return task.gap, task.agent, error_output
        
    async def _generate_observations(self, query: str, background_context: str = "") -> str:
        """Generate observations from the current state of the research."""
        
        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        input_str = f"""
        You are starting iteration {self.iteration} of your research process.

        ORIGINAL QUERY:
        {query}

        {background}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """
        result = await ResearchRunner.run(
            self.thinking_agent,
            input_str,
        )

        # Add the observations to the conversation
        observations = result.final_output
        self.conversation.set_latest_thought(observations)
        self._log_message(self.conversation.latest_thought_string())
        return observations

    async def _create_final_report(
        self, 
        query: str,
        length: str = "",
        instructions: str = ""
        ) -> str:
        """Create the final response from the completed draft."""
        self._log_message("=== Drafting Final Response ===")

        length_str = f"* The full response should be approximately {length}.\n" if length else ""
        instructions_str = f"* {instructions}" if instructions else ""
        guidelines_str = ("\n\nGUIDELINES:\n" + length_str + instructions_str).strip('\n') if length or instructions else ""

        all_findings = '\n\n'.join(self.conversation.get_all_findings()) or "No findings available yet."

        input_str = f"""
        Provide a response based on the query and findings below with as much detail as possible. {guidelines_str}

        QUERY: {query}

        FINDINGS:
        {all_findings}
        """

        result = await ResearchRunner.run(
            self.writer_agent,
            input_str,
        )
        
        self._log_message("Final response from IterativeResearcher created successfully")
        
        return result.final_output
    
    def _log_message(self, message: str) -> None:
        """Log a message if verbose is True"""
        if self.verbose:
            print(message)