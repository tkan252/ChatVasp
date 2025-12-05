import json
from typing import Annotated, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool, Workbench, StaticWorkbench
from autogen_ext.tools.mcp import McpWorkbench

from vaspgo.model_client import create_model_client

model_client = create_model_client(configs={"temperature": 0.0})

INCAR_EXAMPLES_PATH = "src/vaspgo/data/incar_examples.json"

_INCAR_CACHE: dict | None = None


def _load_incar_examples() -> dict:
    """
    Load INCAR examples from the JSON file with caching.
    
    Returns:
        dict: Dictionary containing all INCAR examples organized by category.
    """
    global _INCAR_CACHE
    if _INCAR_CACHE is None:
        with open(INCAR_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            _INCAR_CACHE = json.load(f)
    return _INCAR_CACHE


def _get_available_keys() -> list[str]:
    """
    Get all available category keys from the INCAR examples.
    
    Returns:
        list[str]: List of all available category keys.
    """
    return list(_load_incar_examples().keys())


def get_incar_examples(
    category_key: Annotated[
        str,
        "The category key to retrieve INCAR examples for. Must be one of the available keys."
    ]
) -> str:
    """
    Retrieve INCAR parameter examples for a specific VASP calculation type.
    
    Use this tool to get reference INCAR parameters for a given calculation type.
    You can call this tool multiple times with different keys to understand
    parameter settings for different calculation scenarios.
    
    Args:
        category_key: The exact category key to retrieve INCAR examples for.
    
    Returns:
        str: Formatted string containing the INCAR parameters.
    """
    examples_data = _load_incar_examples()
    
    if category_key not in examples_data:
        available = _get_available_keys()
        return f"Error: Key '{category_key}' not found.\n\nAvailable keys:\n" + \
               "\n".join(f"  - {key}" for key in available)
    
    params = examples_data[category_key]
    return _format_incar_details(category_key, params)


def _format_incar_details(category_name: str, params: dict) -> str:
    """
    Format INCAR parameters for a category.
    
    Args:
        category_name: Name of the calculation category.
        params: Dictionary of INCAR parameters.
    
    Returns:
        str: Formatted string with INCAR parameters.
    """
    result = f"INCAR Example: {category_name}\n"
    result += "=" * 50 + "\n\n"
    
    for key, value in params.items():
        result += f"  {key} = {value}\n"
    
    result += "\n"
    return result


SYSTEM_PROMPT = """
You are an expert VASP workflow automation assistant. Your job is to generate a correct INCAR file for **every task** in the provided "VASP Task Checklist" table based on its Task Name, Precision Level, and Special Requirements.

### Available INCAR Example Categories
Use `get_incar_examples` tool with these keys to retrieve parameter references:
{available_keys}

# Input format (example)
```
1.  Coarse Structure Relaxation [medium] - [❌] Incomplete
    deps: []
    reqs: ENCUT=500eV, Default recommended settings for coarse relaxation
```

# You must proceed as follows:
1. List all INCAR file names in the target directory, while avoiding duplicate names, record the list of INCAR file names that need to be generated and their function descriptions and precision, take notes, mark as incomplete, and leave note empty.
2. Select an uncompleted INCAR in order, and call the tool to query the corresponding example before generating. If there is no completely corresponding example, query similar examples or combine multiple examples.
### Reference values for parameters influenced by accuracy
- **only-gamma**:     equal to low
- **low**:     PREC = Med;  ENCUT = 300;  SIGMA = 0.1;  EDIFF = 1E-3; EDIFFG = -0.2   
- **medium**:  PREC = Normal; ENCUT = 400;  SIGMA = 0.05; EDIFF = 1E-4; EDIFFG = -0.05
- **high**:    PREC = Accurate; ENCUT = 500;  SIGMA = 0.05; EDIFF = 1E-5; EDIFFG = -0.02
- **ultrahigh**:    PREC = Accurate; ENCUT = 600;  SIGMA = 0.02; EDIFF = 1E-6; EDIFFG = -0.01
3. Generate INCAR and write it to the target directory, naming it exactly as in the notes.
4. Write a good note, focusing only on parameters that depend on other files, such as NBANDS in optical computing and OUTCAR in SCF calculation. After writing the note, mark the corresponding task as completed.
5. If there are still tasks to generate INCAR that have not been completed, return to step 2; otherwise, end the task.

# You must record tasks in the following way:
- The format is [...]: filename, desc: {{brief description of the file's purpose}}, note: {{additional parameters to be noted in the file}}. [✅]  Indicates completion[❌] Indicates incomplete
- The note content must be updated immediately
- The note file name must be incar_note.md

# Universal rules
- Always include: SYSTEM = <Task Name>

# Output format
Only inform the location of the output note file incar_note.md
"""

def create_incar_agent(workbench: Workbench) -> AssistantAgent:
    """
    Create INCAR agent instance.
    
    The agent is equipped with the `get_incar_examples` tool to dynamically
    retrieve INCAR parameter examples by category key. This is useful for:
    - Getting reference parameters for different calculation types
    - Combining parameters from multiple examples for complex tasks
    - Understanding the typical INCAR settings before generation
    
    Returns:
        AssistantAgent: Configured INCAR agent for VASP INCAR generation.
    """
    available_keys = _get_available_keys()
    keys_formatted = "\n".join(f"  - {key}" for key in available_keys)

    incar_tool = FunctionTool(get_incar_examples, description=get_incar_examples.__doc__)
    tools_workbench = StaticWorkbench([incar_tool])

    return AssistantAgent(
        name="INCAR_AGENT",
        model_client=model_client,
        system_message=SYSTEM_PROMPT.format(available_keys=keys_formatted),
        workbench=[tools_workbench, workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=50,
        description="A VASP INCAR agent that generates INCAR files for each task in the VASP Task Checklist."
    )


__all__ = ["create_incar_agent", "get_incar_examples"]

if __name__ == '__main__':
    import asyncio
    from autogen_agentchat.ui import Console
    from tools.mcp import files_mcp

    async def main():
        task = """
# VASP Task Checklist

1. Coarse Structure Relaxation [low]
   deps: []
   reqs: ENCUT=500eV, Default recommended settings for coarse relaxation

2. Fine Structure Relaxation [high]
   deps: [Task 1: CONTCAR to POSCAR, copy], [Task 1: WAVECAR, copy]
   reqs: ENCUT=500eV, Default recommended settings for fine relaxation

3. SCF Calculation [high]
   deps: [Task 2: CONTCAR to POSCAR, copy], [Task 2: WAVECAR, copy]
   reqs: ENCUT=500eV, Default recommended settings

4. HSE06 for Band Structure [high]
   deps: [Task 2: CONTCAR to POSCAR, copy], [Task 2: WAVECAR, softlink]
   reqs: ENCUT=500eV, ICHARG!=11 is very IMPORTANT        
"""
        async with McpWorkbench(files_mcp) as wb:
            agent = create_incar_agent(wb)
        await Console(agent.run_stream(task=task))

    asyncio.run(main())
