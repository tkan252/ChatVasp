from contextlib import asynccontextmanager
from autogen_agentchat.agents import MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.tools.mcp import McpWorkbench

from vaspgo.vaspflow.task_agent import create_task_agent
from vaspgo.vaspflow.incar_agent import create_incar_agent
from vaspgo.vaspflow.kpoints_agent import create_kpoints_agent
from vaspgo.vaspflow.submit_agent import create_sum_agent
from tools.mcp import files_mcp

@asynccontextmanager
async def create_flow():
    builder = DiGraphBuilder()
    task_agent = create_task_agent()
    incar_agent = create_incar_agent()
    kpoints_agent = create_kpoints_agent()

    async with McpWorkbench(files_mcp) as workbench:
        sum_agent = create_sum_agent(workbench=workbench)

        filtered_sum_agent = MessageFilterAgent(
            name="FILTERED_SUM_AGENT",
            wrapped_agent=sum_agent,
            filter=MessageFilterConfig(
                per_source=[
                    PerSourceFilter(source="user", position="last", count=1),
                    PerSourceFilter(source="TASK_AGENT", position="last", count=1),
                    PerSourceFilter(source="INCAR_AGENT", position="last", count=1),
                    PerSourceFilter(source="KPOINTS_AGENT", position="last", count=1),
                ],
            ),
        )

        builder.add_node(task_agent).add_node(incar_agent).add_node(incar_score_agent).add_node(kpoints_agent).add_node(filtered_sum_agent)
        builder.set_entry_point(task_agent) 
        builder.add_edge(task_agent, incar_agent)
        builder.add_edge(task_agent, kpoints_agent)
        builder.add_edge(incar_agent, incar_score_agent, activation_group="initial")
        builder.add_edge(incar_score_agent, incar_agent, condition="REJECT", activation_group="feedback")
        builder.add_edge(incar_score_agent, filtered_sum_agent, condition="APPROVE")
        builder.add_edge(kpoints_agent, filtered_sum_agent)
        graph = builder.build()

        flow = GraphFlow(
            name="VASP_FLOW",
            participants=builder.get_participants(),
            graph=graph,
            description="A flow for generating a complete, ready-to-run bash script for a multi-step VASP workflow.",
            termination_condition=MaxMessageTermination(max_messages=200),
        )
        yield flow

__all__ = ["create_flow"]
