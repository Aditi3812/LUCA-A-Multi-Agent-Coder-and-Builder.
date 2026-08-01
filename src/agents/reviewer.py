from langchain_core.messages import AIMessage,HumanMessage
from src.config import llm
from src.graph.state import PlannerState

#4. REVIEWER AGENT
def Reviewer_Agent(state:PlannerState):
    task = state.get("curr_task", "review")
    res_data = state.get("res_data", "")
    code_data = state.get("code_data", "")
    revi_data = state.get("revi_data", "")
    # REVIEWER AGENT PROMPT
    prompt = f"""You are the Reviewer Agent.

Task: {task}
Research: {res_data}
Code: {code_data}
Review: {revi_data}

Review the code against the task and research.
- If the code is correct and efficient, output only OK.DO NOT OUTPUT ANYTHING ELSE
- If it needs changes, give short, practical fix instructions only.
- Focus on the smallest optimized correction.
- Do not rewrite the whole code or add extra commentary.

Output:
- OK, or concise review instructions only."""

    revi = llm.invoke([HumanMessage(content = prompt)])
    revi_data = revi.content

    agent_mess = f"Reviewing done, onto the planner to see wether to route to coder or writer."

    return{
        "messages" : [AIMessage(content = agent_mess)],
        "next_agent" : "planner",
        "revi_data" : revi_data,
        "coder_retry_count": state.get("coder_retry_count", 0)
    }