from typing import Sequence
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

from vaspgo.planning_agent import create_planning_agent
from vaspgo.review_agent import create_review_agent
from vaspgo.vaspflow.flow import create_flow
from vaspgo.model_client import create_model_client
from vaspgo.submit_agent import create_submit_agent

selector_prompt = """
Select an agent to perform task.
{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

def create_selector():
    model_client = create_model_client(configs={"temperature": 1.0})   
    flow = create_flow()
    planning_agent = create_planning_agent()
    review_agent = create_review_agent()
    submit_agent = create_submit_agent()

    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if messages[-1].source != planning_agent.name:
            return planning_agent.name
        return None

    return SelectorGroupChat(
        name="SELECTOR",
        participants=[flow, planning_agent, review_agent, submit_agent],
        selector_prompt=selector_prompt,
        model_client=model_client,
        description="A selector for selecting the next agent to perform the task.",
        termination_condition=TextMessageTermination(termination_keywords=["TERMINATE"]),
        selector_func=selector_func
    )

__all__ = ["create_selector"]
