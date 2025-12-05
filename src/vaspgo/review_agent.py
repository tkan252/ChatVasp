from autogen_agentchat.agents import AssistantAgent

from vaspgo.model_client import create_model_client

model_client = create_model_client(configs={"temperature": 1.0})

SYSTEM_PROMPT = """

"""

def create_review_agent():
    review_agent = AssistantAgent(
        name="REVIEW_AGENT",
        description="An agent for reviewing the calculation results, this agent should be the last to engage when given a new task.",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
    )
    return review_agent

__all__ = ["create_review_agent"]
