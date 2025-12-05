from typing import Annotated
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool, Workbench, StaticWorkbench
from autogen_ext.tools.mcp import McpWorkbench

from vaspgo.model_client import create_model_client

load_dotenv()
model_client = create_model_client(configs={"temperature": 0.0})


def gen_potcar(
    poscar_path: Annotated[
        str,
        "Path to the input POSCAR file. The file must exist and contain valid VASP structure data."
    ],
    potcar_path: Annotated[
        str,
        "Path where the generated POTCAR file will be saved. Parent directory must exist."
    ]
) -> str:
    """
    This tool reads the atomic species from a POSCAR file and automatically generates
    the corresponding POTCAR file using pymatgen's default pseudopotential settings.

    Note: Requires properly configured PMG_VASP_PSP_DIR environment variable pointing
    to the VASP pseudopotential directory.
    
    Args:
        poscar_path: Path to the input POSCAR file containing the crystal structure.
        potcar_path: Path where the output POTCAR file will be written.
    
    Returns:
        str: Success message with the output file path.
    
    Example:
        gen_potcar("./POSCAR", "./POTCAR")
        # Generates POTCAR based on elements found in POSCAR
    """
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Potcar

    structure = Structure.from_file(poscar_path)
    symbols = [s.specie.symbol for s in structure]
    potcar = Potcar(symbols=symbols)
    potcar.write_file(potcar_path)
    
    return f"POTCAR generated successfully at {potcar_path} for elements: {', '.join(dict.fromkeys(symbols))}"


def gen_potcar_batch(
    directory: Annotated[
        str,
        "Path to the directory containing structure files (POSCAR, CIF, VASP, etc.)."
    ],
    recursive: Annotated[
        bool,
        "Whether to search subdirectories recursively."
    ] = True
) -> str:
    """
    Generate POTCAR files for all crystal structure files in a directory.
    
    This tool scans a directory for structure files (POSCAR, *.cif, *.vasp, CONTCAR)
    and generates a corresponding POTCAR file in the same directory as each structure file.
    
    Note: Requires properly configured PMG_VASP_PSP_DIR environment variable pointing
    to the VASP pseudopotential directory.
    
    Args:
        directory: Path to the directory to scan for structure files.
        recursive: If True, search subdirectories recursively. Default is True.
    
    Returns:
        str: Summary of generated POTCAR files.
    
    Example:
        gen_potcar_batch("./calculations", recursive=True)
        # Generates POTCAR for all POSCAR/CIF files in ./calculations and subdirectories
    """
    import os
    import glob
    from pathlib import Path
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Potcar
    
    # Patterns to match structure files
    patterns = ["POSCAR", "CONTCAR", "*.cif", "*.vasp"]
    
    found_files = []
    for pattern in patterns:
        if recursive:
            search_pattern = os.path.join(directory, "**", pattern)
            found_files.extend(glob.glob(search_pattern, recursive=True))
        else:
            search_pattern = os.path.join(directory, pattern)
            found_files.extend(glob.glob(search_pattern))
    
    # Remove duplicates and sort
    found_files = sorted(set(found_files))
    
    if not found_files:
        return f"No structure files found in {directory}"
    
    results = []
    success_count = 0
    error_count = 0
    
    for structure_file in found_files:
        structure_path = Path(structure_file)
        potcar_path = structure_path.parent / "POTCAR"
        
        try:
            structure = Structure.from_file(str(structure_path))
            symbols = [s.specie.symbol for s in structure]
            unique_symbols = list(dict.fromkeys(symbols))
            potcar = Potcar(symbols=symbols)
            potcar.write_file(str(potcar_path))
            results.append(f"  [✅] {structure_file} → {potcar_path} (elements: {', '.join(unique_symbols)})")
            success_count += 1
        except Exception as e:
            results.append(f"  [❌] {structure_file} → Error: {str(e)}")
            error_count += 1
    
    summary = f"POTCAR batch generation completed:\n"
    summary += f"  Total: {len(found_files)}, Success: {success_count}, Failed: {error_count}\n\n"
    summary += "Details:\n" + "\n".join(results)
    
    return summary


SYSTEM_PROMPT = """
You are a professional expert in POTCAR generation. You generate POTCAR files based on structure files (POSCAR, CIF, etc.).

# You must proceed as follows:
1. Find all structure files that need POTCAR generation in the target directory.
2. Record the task list in potcar_note.md with format: [❌] structure_file → potcar_file (elements: ...)
3. For each uncompleted task, call `gen_potcar` tool to generate the POTCAR file.
4. After successful generation, update the note: [✅] structure_file → potcar_file (elements: ...)
5. Repeat until all tasks are completed.

# Note:
- You MUST NOT read the specific content of POSCAR
- You ONLY need to generate a POTCAR of the structure mentioned by the user

# Note format (potcar_note.md)
- [✅] POSCAR → POTCAR (elements: Si, O) 
- [❌] struct2.cif → POTCAR_2 (elements: pending)

# Output format
Only inform the location of the output note file potcar_note.md
"""


def create_potcar_agent(workbench: Workbench) -> AssistantAgent:
    """
    Create POTCAR agent instance.
    
    The agent is equipped with the `gen_potcar` tool to generate POTCAR files
    and uses MCP workbench for file operations (reading structure files, writing notes).
    
    Args:
        workbench: MCP workbench for file system operations.
    
    Returns:
        AssistantAgent: Configured POTCAR agent for VASP POTCAR generation.
    """
    potcar_tool = FunctionTool(gen_potcar, description=gen_potcar.__doc__)
    potcar_batch_tool = FunctionTool(gen_potcar_batch, description=gen_potcar_batch.__doc__)
    tools_workbench = StaticWorkbench([potcar_tool, potcar_batch_tool])

    return AssistantAgent(
        name="POTCAR_AGENT",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
        workbench=[tools_workbench, workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=20,
        description="A VASP POTCAR agent that generates POTCAR files based on structure files."
    )


__all__ = ["create_potcar_agent", "gen_potcar", "gen_potcar_batch"]

if __name__ == '__main__':
    import asyncio
    from autogen_agentchat.ui import Console
    from tools.mcp import files_mcp

    async def main():
        task = r"""仅有一个POSCAR需要计算，位于E:\PyCharmProject\ChatVasp\test"""
        async with McpWorkbench(files_mcp) as wb:
            agent = create_potcar_agent(wb)
            await Console(agent.run_stream(task=task))

    asyncio.run(main())
