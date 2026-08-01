from datetime import datetime
from langchain_core.messages import AIMessage,HumanMessage
from src.config import llm
from src.graph.state import PlannerState
#4. Writer AGENT
def Writer_Agent(state:PlannerState):
    task = state.get("curr_task", "report")
    res_data = state.get("res_data", "")
    code_data = state.get("code_data", "")
    revi_data = state.get("revi_data", "")
    prompt = f"""You are the Writer Agent.

Task:
{task}

Research:
{res_data}

Code:
{code_data}

Review:
{revi_data}

Instructions:
- Write the final answer in a clean, readable format.
- If the task is coding, provide the final code and a short explanation.
- If the task is explanation-only, provide only the explanation.
- If the task is structured, return only the required structure.
- Keep the answer concise and ready to use.
- Do not mention internal reasoning or routing steps.

Formatting rules:
- Use clear section headings.
- Use fenced code blocks for code.
- Do not output JSON."""

    rep = llm.invoke([HumanMessage(content = prompt)])
    rep_data = rep.content
    final_report = f"""
    FINAL REPORT:
    {'='*50}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    Topic: {task}
    {'='*50}

    {rep_data}

    {'='*50}"""
    agent_mess = f"Report done, onto the planner. The final report is: {final_report} "

    return{
        "messages" : [AIMessage(content = agent_mess)],
        "next_agent" : "planner",
        "fin_rep_data" : final_report,
        "task_comp" : True,
        "coder_retry_count": state.get("coder_retry_count", 0)
    }