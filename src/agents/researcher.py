from langchain_core.messages import  HumanMessage, AIMessage
from src.config import llm
from src.graph.state import PlannerState
#B. RESEACHER AGENT
def Researcher_Agent(state:PlannerState):
    task = state.get("curr_task", "research topic")
    prompt = f"""You are the Researcher Agent.

Goal:
Gather the most useful facts for the next agent.

Task:
{task}


Do:
- Identify the real task
- Search docs, APIs, libraries, best practices, datasets, or troubleshooting info
- For building tasks: technologies, approach, constraints, edge cases
- For debugging: likely causes, docs, fixes, expected behavior
- For analysis/design: facts, definitions, comparisons, recommendation
- Keep only relevant findings
- Be factual, concise, and actionable
- Do not write final code unless a tiny example is needed

Output:
- Brief task understanding
- Key findings
- Important constraints or risks
- Suggested direction for the next agent

If there is no useful research, say the task is ready for the next agent."""
    
    res= llm.invoke([HumanMessage(content = prompt)])
    res_data = res.content

    agent_mess = f"Research done, key findings on {task} is : {res_data[:500]} "

    return{
        "messages" : [AIMessage(content = agent_mess)],
        "next_agent" : "planner",
        "res_data" : res_data,
        "coder_retry_count": state.get("coder_retry_count", 0)
    }