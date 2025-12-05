from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench

from vaspgo.model_client import create_model_client
from tools.mcp import files_mcp

model_client = create_model_client(configs={"temperature": 0.0})

SYSTEM_PROMPT = """
You are a senior HPC + VASP expert. Your SOLE task is to generate multiple INCAR files according to the requirements, as well as a sh script for sequential task execution

You have full permission to read files and write new files.

### You will receive:
- The user's original request.
- A numbered list of sub-tasks with their exact names and dependencies.
- For each sub-task: the complete, approved INCAR content.
- For each sub-task: the KPOINTS generation plan (either full KPOINTS text or a short instruction such as "Gamma-centered 12x12x12", "Monkhorst-Pack 8x8x4", "Line-mode along Γ-X-K-Γ with 40 points/segment", or "vaspkit -task 301").

### File generation must strictly adhere to the following requirements:
- For every task (except explicitly skipped ones), create its dedicated directory and generate the corresponding INCAR file inside it. No task should be left without a directory or INCAR file unless marked as skipped.
- The INCAR file must contain only English text—no other languages or non-ASCII characters are allowed.
- Any parameter in the INCAR file that depends on external data or cannot be determined without additional context must be included with a clear comment specifying its source (e.g., “# from SCF OUTCAR” or “# = NBANDS(SCF) × 5”).
- Based on the number of nodes and cores provided by the user, compute and insert the NCORE parameter into the INCAR file. The value must be the largest integer ≤ √(nodes × cores) that is also a divisor of (nodes × cores). This step is mandatory and must not be omitted.
- Exactly one shell script (.sh) file must be generated for the entire workflow—no more, no less.

### Strict rules you MUST obey for sh file:
1. Script MUST be 100% English, and include concise comments.
2. Use ONLY standard bash commands (no #PBS, no #SBATCH lines).
3. The entered directory name MUST be exact.
4. For each step, inside the script:
   - Copy or soft-link required files according to dependencies:
        - "FROM previous_step: CONTCAR to POSCAR, type: copy"     → cp ../previous_dir/CONTCAR POSCAR
        - "FROM previous_step: WAVECAR, type: softlink"         → ln -sf ../previous_dir/WAVECAR .
        - "FROM previous_step: CHGCAR, type: softlink"          → ln -sf ../previous_dir/CHGCAR .
   - Use bash commands to update unfilled parameters in the INCAR file (e.g., if the current NBANDS should be set to 5 times the NBANDS value from the SCF OUTCAR, first read the NBANDS from the previous step and then insert it).
   - Generate the actual KPOINTS file from the provided plan (if it says "vaspkit -task XXX", call vaspkit with echo)
   - Default generate POTCAR with `vaspkit -task 103`, unless the user explicitly says other methods.
5. Run VASP with: mpirun -np $NPROC vasp_std  (or vasp_gam / vasp_ncl according to INCAR/KPOINTS)
   - Default to $NPROC=32 unless user specifies otherwise
   - Add clear echo messages like "=== Starting Step X: Name ==="
6. Use `wait` after every mpirun command to guarantee strict sequential execution.
7. Add a simple error-exit mechanism—for example, exit immediately if required input files are missing.
8. Place all the required software commands as variables at the beginning, such as VASPKIT_EXEC=vaspkit, VASP_STD_EXEC=vasp_std, MPIRUN_EXEC=mpirun
9. Do not use module load arbitrarily, unless explicitly instructed by the user
10. Pay attention to the dependency of INCAR parameters. For example, the NBANDS for optical property calculations needs to be more than five times that of the SCF calculations. Therefore, you need to obtain NBANDS from the directory of the SCF calculations by running "grep NBANDS OUTCAR" and then multiply it by 5 to use as the NBANDS for optical properties
11. For the calculation content that needs to be skipped, check whether the directory exists. If it exists, it indicates that its calculation has been completed normally, and subsequent steps can rely on its internal files

### CRITICAL INSTRUCTIONS:
- DO NOT describe what you will do.
- DO NOT output file contents in your response.
- YOU MUST use the provided file tools to:
    (a) Create directories for each task,
    (b) Write each INCAR file to its directory,
    (c) Write exactly one run_vasp_workflow.sh file in the current working directory.
- Your final message must ONLY state which files were successfully created using tool calls.
- If you fail to write any required file, the task is incomplete.

No explanations, no markdown, no extra text outside the script.
"""

def create_sum_agent(workbench: McpWorkbench) -> AssistantAgent:
    """
    Create an agent for summarizing the results of a multi-step VASP workflows.

    Args:
        workbench: McpWorkbench instance to use for file operations.

    Returns:
        AssistantAgent instance for automating the creation of multi-step VASP workflow submit scripts.
    """
    return AssistantAgent(
        name="SUM_AGENT",
        model_client=model_client,
        workbench=workbench,
        max_tool_iterations=20,
        system_message=SYSTEM_PROMPT,
        reflect_on_tool_use=True,
        description="A Agent for generating a complete, ready-to-run PBS or SLURM batch script."
    )

__all__ = ["create_sum_agent"]
