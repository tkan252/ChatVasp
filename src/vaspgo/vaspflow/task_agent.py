import json
from typing import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool, Workbench, StaticWorkbench

from vaspgo.model_client import create_model_client

model_client = create_model_client(configs={"temperature": 0.0})

# Path to task examples JSON file
TASK_EXAMPLES_PATH = "src/vaspgo/data/task_examples.json"

# Load task examples at module level for reuse
_EXAMPLES_CACHE: dict | None = None


def _load_task_examples() -> dict:
    """
    Load task examples from the JSON file with caching.
    
    Returns:
        dict: Dictionary containing all task examples organized by category.
    """
    global _EXAMPLES_CACHE
    if _EXAMPLES_CACHE is None:
        with open(TASK_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            _EXAMPLES_CACHE = json.load(f)
    return _EXAMPLES_CACHE


def _get_available_keys() -> list[str]:
    """
    Get all available category keys from the examples.
    
    Returns:
        list[str]: List of all available category keys.
    """
    return list(_load_task_examples().keys())


def get_task_examples(
    category_key: Annotated[
        str,
        "The category key to retrieve examples for. Must be one of the available keys."
    ]
) -> str:
    """
    Retrieve task examples for a specific VASP calculation category.
    
    Use this tool to get detailed workflow examples for a given calculation type.
    You can call this tool multiple times with different keys to combine examples
    for novel or complex tasks.
    
    Args:
        category_key: The exact category key to retrieve examples for.
    
    Returns:
        str: Formatted string containing the task workflow details.
    """
    examples_data = _load_task_examples()
    
    if category_key not in examples_data:
        available = _get_available_keys()
        return f"Error: Key '{category_key}' not found.\n\nAvailable keys:\n" + \
               "\n".join(f"  - {key}" for key in available)
    
    tasks = examples_data[category_key]
    return _format_category_details(category_key, tasks)


def _format_category_details(category_name: str, tasks: list) -> str:
    """
    Format detailed task information for a category.
    
    Args:
        category_name: Name of the category.
        tasks: List of tasks in the category.
    
    Returns:
        str: Formatted string with detailed task information.
    """
    result = f"Category: {category_name}\n"
    result += "=" * 50 + "\n\n"
    
    for idx, task in enumerate(tasks, 1):
        result += f"Step {idx}: {task['Task Name']}\n"
        result += f"  Precision Level: {task['Precision Level']}\n"
        
        if task['Dependencies']:
            result += "  Dependencies:\n"
            for dep in task['Dependencies']:
                result += f"    - {dep}\n"
        else:
            result += "  Dependencies: None (starting step)\n"
        
        if task['Special Requirements']:
            result += f"  Special Requirements: {task['Special Requirements']}\n"
        else:
            result += "  Special Requirements: Default settings\n"
        
        result += "\n"
    
    return result


SYSTEM_PROMPT = """
You are a world-class VASP expert and computational materials science workflow engineer. Your job is to take any user request involving VASP calculations (structure relaxation, DOS, band structure, HSE06, phonon, optical properties, surface energy, defect formation energy, etc.) and automatically break it down into a complete, logically ordered task list that can be directly executed in sequence.

# Available Example Categories
Use `get_task_examples` tool with these keys to retrieve workflow examples:
{available_keys}

# Strictly follow the procedure below to execute the task:
1. Check whether task_note.md exists in the target directory.
- If it exists: Read task_note.md and mark the tasks declared by the user as completed.
- If it does not exist: Analyze the user's intent, generate a checklist, and record it in the notes.
2. Communicate the pending (incomplete) tasks according to the order specified in task_note.md.
3. Create a subdirectory for the next computational task and output the path to this directory.

# Task Analysis Guidelines
1. Analyze the user's intention and use `get_task_examples` tool to retrieve the most relevant examples based on the task. If no single example matches, query multiple examples and combine them.
2. Simple task splitting
3. Consider the rationality of each subtask and make modifications accordingly
   - What is the minimal complete set of calculations needed?
   - Which steps can reuse charge density/wavefunctions?
   - Where must we switch functional (e.g., PBE → R2SCAN or R2SCAN → HSE06)?
   - What precision is reasonable by default if user didn't specify? (high is safe default for publication)
   - Which parameters must depend on the previous output
4. Write the task list to task_note.md for progress tracking using this format:
```
# VASP Task Checklist

1.  Task Name [precision_level] - [❌] Incomplete
    deps: [Task X: FILE → FILE, type], [Task Y: FILE, type], ...
    reqs: requirement description
    dir: path to the subdirectory for the computational task (Leave it blank if not created yet)
```
*When a task is executed by other agents, update the status to [✅]*
*The checklist should only contain English*

Always return the result in the following concise format. Never add extra explanations outside the list unless the user explicitly asks.

# Fields:
1. **Task Name** (e.g., Structure Relaxation, SCF Calculation, NSCF for DOS, HSE06 Band Structure)
2. **Precision Level** (in brackets after task name):
   - ultrahigh / high / medium / low / only-gamma / null
   - null: for non-computational tasks (pre/post-processing)
3. **deps** (dependencies):
   - Format: [Task N: SOURCE → DEST, type] or [Task N: FILE, type]
   - type: copy / softlink / hardlink
   - Use [] for no dependencies
4. **reqs** (requirements):
   - User-specified requirements or "Default recommended settings"
   - Include skip instructions if applicable

*Within reqs, NO parameters other than those provided in the examples or explicitly requested by the user should be included.*
You MUST ONLY return the complete information of the next task to be executed and the path of the subdirectory for the computational task.
"""


def create_task_agent(workbench: Workbench) -> AssistantAgent:
    """
    Create task agent instance with file-based note-taking capability.
    
    The agent is equipped with:
    - `get_task_examples` tool to retrieve workflow examples by category key
    - MCP workbench for file operations (writing task_note.md)
    
    This is useful for:
    - Handling novel tasks that don't match existing examples exactly
    - Combining multiple examples to construct new calculation workflows
    - Tracking the decomposition process step by step
    
    Args:
        workbench: MCP workbench for file system operations.
    
    Returns:
        AssistantAgent: Configured task agent for VASP workflow decomposition.
    """
    # Get available keys and format them for the prompt
    available_keys = _get_available_keys()
    keys_formatted = "\n".join(f"  - {key}" for key in available_keys)
    
    # Create tool workbench
    task_examples_tool = FunctionTool(get_task_examples, description=get_task_examples.__doc__)
    tools_workbench = StaticWorkbench([task_examples_tool])
    
    return AssistantAgent(
        name="TASK_AGENT",
        model_client=model_client,
        system_message=SYSTEM_PROMPT.format(available_keys=keys_formatted),
        workbench=[tools_workbench, workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=20,
        description="A task agent that decomposes user requests into a list of VASP tasks."
    )


__all__ = ["create_task_agent", "get_task_examples"]

if __name__ == '__main__':
    import asyncio
    from autogen_agentchat.ui import Console
    from autogen_ext.tools.mcp import McpWorkbench
    from tools.mcp import files_mcp

    async def main():
        task = "我想计算HSE06级别的能带，并且结构优化需要分为两步，由低精度到高精度逐步进行，ENCUT全程使用500eV，除了第一步粗优化"
        async with McpWorkbench(files_mcp) as wb:
            agent = create_task_agent(wb)
            await Console(agent.run_stream(task=task))

    asyncio.run(main())

