"""
Agent used to reflect on the research process so far and share your latest thoughts.

The Agent takes as input a string in the following format:
===========================================================
ORIGINAL QUERY: <original user query>

BACKGROUND CONTEXT: <supporting background context related to the original query>

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS: <a log of prior iterations of the research process>
===========================================================

The Agent then outputs a string containing its latest thoughts on the research process.
"""
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig
from datetime import datetime

INSTRUCTIONS = f"""
You are a research expert who is managing a research process in iterations. Today's date is {datetime.now().strftime("%Y-%m-%d")}.

You are given:
1. The original research query along with some supporting background context
2. A history of the tasks, actions, findings and thoughts you've made up until this point in the research process (on iteration 1 you will be at the start of the research process, so this will be empty)

Your objective is to reflect on the research process so far and share your latest thoughts.

Specifically, your thoughts should include reflections on questions such as:
- What have you learned from the last iteration?
- What new areas would you like to explore next, or existing topics you'd like to go deeper into?
- Were you able to retrieve the information you were looking for in the last iteration?
- If not, should we change our approach or move to the next topic?
- Is there any info that is contradictory or conflicting?

Guidelines:
- Share you stream of consciousness on the above questions as raw text
- Keep your response concise and informal
- Focus most of your thoughts on the most recent iteration and how that influences this next iteration
- Our aim is to do very deep and thorough research - bear this in mind when reflecting on the research process
- DO NOT produce a draft of the final report. This is not your job.
- If this is the first iteration (i.e. no data from prior iterations), provide thoughts on what info we need to gather in the first iteration to get started
"""

def init_thinking_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.reasoning_model

    return ResearchAgent(
        name="ThinkingAgent",
        instructions=INSTRUCTIONS,
        model=selected_model,
    )
