from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import Workbench
from autogen_ext.tools.mcp import McpWorkbench

from vaspgo.model_client import create_model_client
from tools.structure_tools import create_structure_workbench

load_dotenv()
model_client = create_model_client(configs={"temperature": 0.0})

SYSTEM_PROMPT = """
You are an expert in crystal structures, proficient in crystallography, geometry, and group theory. Your task is to query, generate, or edit crystal structure files.

# You must follow the procedure below to complete the task:
1. Analyze the user's intent to determine whether a crystal structure needs to be queried or generated. If so, proceed as follows:
- Query: Retrieve all structures meeting the specified criteria from the appropriate database and collect them into a subdirectory.
- Generate: Use the designated tool to generate structures with the desired properties and collect them into a subdirectory.
2. Identify the source of the crystal structure files and determine whether prototype structures need editing.
- Edit: Use the appropriate tool to modify the crystal structures.
3. Output the locations of all resulting crystal structure files.

# Rules you must adhere to:
1. You must always output the file paths of the crystal structures.
- If no editing was performed, output the paths of the original (prototype) structures.
- If editing was performed, output only the paths of the edited crystal structure files.
"""


def create_poscar_agent(workbench: Workbench) -> AssistantAgent:
    """
    """
    tools_workbench = create_structure_workbench()

    return AssistantAgent(
        name="POSCAR_AGENT",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
        workbench=[tools_workbench, workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=20,
        description="A VASP POTCAR agent that generates POTCAR files based on structure files."
    )


__all__ = ["create_poscar_agent"]

if __name__ == '__main__':
    import asyncio
    from autogen_agentchat.ui import Console
    from tools.mcp import files_mcp

    async def main():
        task = r""""""
        async with McpWorkbench(files_mcp) as wb:
            agent = create_poscar_agent(wb)
            await Console(agent.run_stream(task=task))

    asyncio.run(main())
