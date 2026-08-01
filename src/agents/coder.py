from langchain_core.messages import AIMessage,HumanMessage
from src.config import llm
from src.graph.state import PlannerState

#3. CODER AGENT
def Coder_Agent(state:PlannerState):
    task = state.get("curr_task", "coding")
    res_data = state.get("res_data", "")
    revi_data = state.get("revi_data", "")
    # CODER AGENT PROMPT
    prompt = f"""You are the Coder Agent.

Task: {task}
Research: {res_data}
Review: {revi_data}

If review is empty, code from research and task only.
If review exists, use it to refine the code.

Rules:
- Write only the minimal correct runnable code.
- Follow the task and existing code style.
- Handle edge cases if needed.
- Do not explain, review, or add JSON.

Output:
- Return only code or the required code-ready result."""

    cod = llm.invoke([HumanMessage(content = prompt)])
    cod_data = cod.content

    agent_mess = f"Coding done, onto the reviewer. "

    return{
        "messages" : [AIMessage(content = agent_mess)],
        "next_agent" : "planner",
        "code_data" : cod_data,
        "coder_retry_count": state.get("coder_retry_count", 0)
    }