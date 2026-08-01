from langgraph.graph import StateGraph, END
from src.graph.state import PlannerState
from src.agents.planner import Planner_agent, router
from src.agents.researcher import Researcher_Agent
from src.agents.coder import Coder_Agent
from src.agents.reviewer import Reviewer_Agent
from src.agents.writer import Writer_Agent

builder = StateGraph(PlannerState)

# Add Nodes
builder.add_node("planner", Planner_agent)
builder.add_node("researcher", Researcher_Agent)
builder.add_node("coder", Coder_Agent)
builder.add_node("reviewer", Reviewer_Agent)
builder.add_node("writer", Writer_Agent)

builder.set_entry_point("planner")

# Conditional edges using notebook's router function
routing_map = {
    "planner": "planner",
    "researcher": "researcher",
    "coder": "coder",
    "reviewer": "reviewer",
    "writer": "writer",
    "END": END,
    END: END
}

builder.add_conditional_edges("planner", router, routing_map)
builder.add_conditional_edges("researcher", router, routing_map)
builder.add_conditional_edges("coder", router, routing_map)
builder.add_conditional_edges("reviewer", router, routing_map)
builder.add_conditional_edges("writer", router, routing_map)

# Compiled graph export
app_graph = builder.compile()