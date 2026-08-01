from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.graph import END
from src.config import llm
from src.graph.state import PlannerState


def superchain():
    sys_prompt = ChatPromptTemplate([
        ("system", """You are a routing agent for a chain with exactly 4 agents: research, coder, reviewer, writer.

Inputs:
- has_res_data: {has_res_data}
- has_code_data: {has_code_data}
- has_revi_data: {has_revi_data}
- has_rep: {has_rep}
- task: {task}
- task_comp: {task_comp}

Rules:
- If {task_comp} is true, output only:
done
- Otherwise, choose exactly one next agent name based on the current task and the available data:
research
coder
reviewer
writer

Output rules:
- Output only one word.
- The output must be exactly one of:
done
research
code
review
write
- Do not output any extra text, punctuation, explanation, quotes, or formatting.
- Do not output multiple lines.
- Do not output JSON.

Decision logic:
- If the task is complete, respond with 'done'.
- Based on the current state and conversation, decide which agent should work next.
- Use has_res_data, has_code_data, and has_revi_data to determine whether the task has enough information to move forward.
- Select the most appropriate next agent for the current task state."""),
        ("human", "{task}")
    ])

    return sys_prompt | llm


def Planner_agent(state: PlannerState):
    has_res_data = bool(state.get("res_data", ""))
    has_code_data = bool(state.get("code_data", ""))
    revi_text = state.get("revi_data", "").strip().upper()
    has_revi_data = bool(revi_text)
    has_rep = bool(state.get("fin_rep_data", ""))
    coder_retry_count = state.get("coder_retry_count", 0)

    # Get user task safely
    messages = state.get("messages", [])
    task_text = state.get("curr_task") or (messages[-1].content if messages else "")

    call = superchain()
    chainmess = call.invoke({
        "task": task_text,
        "has_res_data": has_res_data,
        "has_code_data": has_code_data,
        "has_revi_data": has_revi_data,
        "has_rep": has_rep,
        "task_comp": state.get("task_comp", False)
    })
    
    # Standardize string comparison
    chainmess = chainmess.content.strip().lower()
# 1. Termination condition
    if state.get("task_comp", False):
        mess = "TASK COMPLETED NOW END!!"
        next_agent = "END"

    # 2. Approved review OR manual write command -> Writer
    elif (revi_text == "OK" and not has_rep) or chainmess == "write":
        mess = "Passing to the writer agent"
        next_agent = "writer"

    # 3. Missing research
    elif chainmess == "research" or not has_res_data:
        mess = "Passing to the researcher agent"
        next_agent = "researcher"

    # 4. Coding phase (no code exists yet)
    elif chainmess == "code" or (has_res_data and not has_code_data):
        mess = "Passing to the Coding agent"
        next_agent = "coder"

    # 5. Review phase (code exists, but no review done yet)
    elif chainmess == "review" or (has_code_data and not has_revi_data):
        mess = "Passing to the reviewer agent"
        next_agent = "reviewer"

    # 6. Code fixes needed (retry count < 1) -> Send back to Coder & increment retry!
    elif has_code_data and has_revi_data and revi_text != "OK" and coder_retry_count < 1:
        mess = "Reviewer asked for fixes, sending coder one retry"
        next_agent = "coder"
        coder_retry_count += 1  # Increment local variable

    # 7. Max retries reached (coder_retry_count >= 1) -> Force route to Writer to prevent infinite loop!
    elif has_code_data and has_revi_data and revi_text != "OK" and coder_retry_count >= 1:
        mess = "Max retries reached. Forcing route to writer."
        next_agent = "writer"
    # Fallback
    else:
        mess = "Defaulting to planner agent"
        next_agent = "planner"

    return {
        "messages": [AIMessage(content=mess)],
        "next_agent": next_agent,
        "curr_task": task_text,
        "coder_retry_count": coder_retry_count,
    }


def router(state: PlannerState) -> Literal["planner", "researcher", "coder", "reviewer", "writer", "__end__"]:
    """Routes to next agent based on state"""
    next_agent = state.get("next_agent", "planner")
    
    if next_agent == "END" or state.get("task_comp", False):
        return END
        
    if next_agent in ["planner", "researcher", "coder", "reviewer", "writer"]:
        return next_agent
        
    return "planner"