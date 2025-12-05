import asyncio
from autogen_agentchat.ui import Console

from vaspgo.submit_agent import create_submit_agent

MESSAGE = """
Created directories and INCAR files for all 4 tasks:
- Coarse_Structure_Relaxation/INCAR
- Fine_Structure_Relaxation/INCAR
- SCF_Calculation/INCAR
- HSE06_Band_Structure/INCAR

Created run_vasp_workflow.sh script for sequential execution.
"""

async def test():
    agent = await create_submit_agent()

    prompt = "当前计算环境使用PBS任务调度系统，我使用的是manycores节点，32核，用1个节点计算即可\n"
    task = prompt + "我想计算HSE06级别的能带，并且结构优化需要分为两步，由低精度到高精度逐步进行，ENCUT全程使用500eV，除了第一步粗优化"
    task += """
1. VASP_FLOW : 根据用户需求，设计一个两步结构优化流程（从低精度到高精度），然后进行HSE06能带计算。具体要求：ENCUT全程使用500eV（第一步粗优化除外），使用PBS任务调度系统，manycores节点，32核，1个节点。

2. SUBMIT_AGENT : 分析当前PBS环境和用户需求（manycores节点，32核，1个节点），为VASP_FLOW生成的脚本提供合适的提交方法。

3. REVIEW_AGENT : 在计算完成后，分析结构优化和HSE06能带计算的结果，检查是否有错误，并提供总结与建议。

请按顺序执行上述任务，完成后我将汇总结果。
"""
    task += "\n" + MESSAGE

    result = await Console(agent.run_stream(task=task))
    return result

if __name__ == "__main__":
    asyncio.run(test())
