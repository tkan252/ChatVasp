import asyncio
from autogen_agentchat.ui import Console

from vaspgo.planning_agent import create_planning_agent

async def test():
    agent = create_planning_agent()
    
    # 测试任务：可以根据需要修改
    prompt = "当前计算环境使用PBS任务调度系统，我使用的是manycores节点，32核，用4个节点计算\n"
    task = prompt + "我想计算HSE06级别的能带，并且结构优化需要分为两步，由低精度到高精度逐步进行，ENCUT全程使用500eV，除了第一步粗优化"

    result = await Console(agent.run_stream(task=task))
    return result

if __name__ == "__main__":
    asyncio.run(test())
