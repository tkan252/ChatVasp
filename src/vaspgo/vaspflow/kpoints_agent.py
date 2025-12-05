from typing import Annotated, Literal
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool, Workbench, StaticWorkbench
from autogen_ext.tools.mcp import McpWorkbench

from vaspgo.model_client import create_model_client

model_client = create_model_client(configs={"temperature": 0.0})

# K-point density settings: [a, b, c] lengths for each precision level
KPOINT_DENSITY = {
    'only-gamma': 1,
    'low': 15,
    'medium': 30,
    'high': 45,
    'ultrahigh': 60
}


def gen_kpoints(
    poscar_path: Annotated[
        str,
        "Path to the input POSCAR file containing the crystal structure."
    ],
    kpoints_path: Annotated[
        str,
        "Path where the generated KPOINTS file will be saved."
    ],
    precision: Annotated[
        Literal['only-gamma', 'low', 'medium', 'high', 'ultrahigh'],
        "Precision level for k-point density. Higher precision means denser k-mesh."
    ],
    force_gamma: Annotated[
        bool,
        "If True, force Gamma-centered k-mesh. Recommended for hexagonal systems and HSE06 calculations."
    ] = True
) -> str:
    """
    Generate a KPOINTS file with automatic dimensionality detection.
    
    This tool analyzes the crystal structure to determine its dimensionality (0D/1D/2D/3D)
    and generates an appropriate k-mesh. For low-dimensional materials, k-point density
    is reduced along non-periodic directions (e.g., only 1 k-point along vacuum direction
    for 2D materials).
    
    Args:
        poscar_path: Path to the input POSCAR file.
        kpoints_path: Path where the output KPOINTS file will be written.
        precision: K-point density level (only-gamma/low/medium/high/ultrahigh).
        force_gamma: Whether to use Gamma-centered k-mesh.
    
    Returns:
        str: Success message with dimensionality info and k-mesh details.
    """
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Kpoints
    from pymatgen.analysis.dimensionality import get_dimensionality_larsen
    import numpy as np

    structure = Structure.from_file(poscar_path)
    base_density = KPOINT_DENSITY[precision]
    
    # Detect dimensionality using Larsen algorithm
    try:
        dim_info = get_dimensionality_larsen(structure)
        # dim_info returns a list of (dim, components) tuples
        # Take the highest dimensionality found
        dimensionality = max(d[0] for d in dim_info) if dim_info else 3
    except Exception:
        # Fallback to 3D if dimensionality detection fails
        dimensionality = 3
    
    # Get lattice vectors to identify non-periodic directions
    lattice = structure.lattice
    lengths = [lattice.a, lattice.b, lattice.c]
    
    # Determine k-point density for each direction based on dimensionality
    if precision == 'only-gamma':
        k_lengths = [1, 1, 1]
        dim_note = "Gamma-only calculation"
    elif dimensionality == 3:
        # 3D: uniform k-mesh
        k_lengths = [base_density, base_density, base_density]
        dim_note = "3D bulk material"
    elif dimensionality == 2:
        # 2D: find the vacuum direction (longest lattice vector or c-axis typically)
        # Use 1 k-point along vacuum direction
        vacuum_idx = np.argmax(lengths)
        k_lengths = [base_density, base_density, base_density]
        k_lengths[vacuum_idx] = 1
        dim_note = f"2D material (vacuum along {'abc'[vacuum_idx]}-axis)"
    elif dimensionality == 1:
        # 1D: find the periodic direction (shortest), use 1 k-point for other two
        periodic_idx = np.argmin(lengths)
        k_lengths = [1, 1, 1]
        k_lengths[periodic_idx] = base_density
        dim_note = f"1D material (periodic along {'abc'[periodic_idx]}-axis)"
    else:
        # 0D: molecule or cluster
        k_lengths = [1, 1, 1]
        dim_note = "0D molecule/cluster"
    
    kpoints = Kpoints.automatic_density_by_lengths(structure, k_lengths, force_gamma)
    kpoints.write_file(kpoints_path)
    
    # Get actual k-mesh from generated KPOINTS
    kpts_str = f"{kpoints.kpts[0][0]}x{kpoints.kpts[0][1]}x{kpoints.kpts[0][2]}"
    
    return (f"KPOINTS generated at {kpoints_path}\n"
            f"  Dimensionality: {dimensionality}D ({dim_note})\n"
            f"  Precision: {precision}\n"
            f"  K-mesh: {kpts_str}\n"
            f"  Gamma-centered: {force_gamma}")


SYSTEM_PROMPT = """
You are an expert VASP workflow automation assistant. Your job is to generate the correct KPOINTS file for **every task** in the provided "VASP Task Checklist" table.

# Input format (example)
```# VASP Task Checklist

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
```

# You must proceed as follows:
1. List all KPOINTS file names needed in the target directory, record them in kpoints_note.md with precision level and dimensionality info. Mark as incomplete.
2. Select an uncompleted KPOINTS in order, call `gen_kpoints` tool to generate it. The tool will auto-detect dimensionality (0D/1D/2D/3D) and adjust k-mesh accordingly.
3. After successful generation, update the note with k-mesh info and mark as completed.
4. If there are still tasks incomplete, return to step 2; otherwise, end the task.

# Rules
- precision level: only-gamma / low / medium / high / ultrahigh (from task's [precision])
- force_gamma=True recommended for hexagonal systems and HSE06 calculations
- Any task with Precision Level = null → Skip

# Note format (kpoints_note.md)
- [✅] KPOINTS_relax, precision: low, k-mesh: 4x4x4, dim: 3D
- [❌] KPOINTS_scf, precision: high, k-mesh: pending, dim: pending

# Output format
Only inform the location of the output note file kpoints_note.md
"""


def create_kpoints_agent(workbench: Workbench) -> AssistantAgent:
    """
    Create KPOINTS agent instance.
    
    The agent is equipped with the `gen_kpoints` tool to generate KPOINTS files
    with automatic dimensionality detection and uses MCP workbench for file operations.
    
    Args:
        workbench: MCP workbench for file system operations.
    
    Returns:
        AssistantAgent: Configured KPOINTS agent for VASP KPOINTS generation.
    """
    kpoints_tool = FunctionTool(gen_kpoints, description=gen_kpoints.__doc__)
    tools_workbench = StaticWorkbench([kpoints_tool])

    return AssistantAgent(
        name="KPOINTS_AGENT",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
        workbench=[tools_workbench, workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=30,
        description="A VASP KPOINTS agent that generates KPOINTS files for each task in the VASP Task Checklist."
    )


__all__ = ["create_kpoints_agent", "gen_kpoints"]

if __name__ == '__main__':
    import asyncio
    from autogen_agentchat.ui import Console
    from tools.mcp import files_mcp

    async def main():
        task = r"""
# VASP Task Checklist

1. Coarse Structure Relaxation [low]
   deps: []
   reqs: Default recommended settings for coarse relaxation

2. Fine Structure Relaxation [high]
   deps: [Task 1: CONTCAR to POSCAR, copy], [Task 1: WAVECAR, copy]
   reqs: Default recommended settings for fine relaxation

3. SCF Calculation [high]
   deps: [Task 2: CONTCAR to POSCAR, copy], [Task 2: WAVECAR, copy]
   reqs: Default recommended settings

POSCAR位于E:\PyCharmProject\ChatVasp\test，KPOINTS也输出到该目录
"""
        async with McpWorkbench(files_mcp) as wb:
            agent = create_kpoints_agent(wb)
            await Console(agent.run_stream(task=task))

    asyncio.run(main())
