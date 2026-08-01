from langgraph.graph import MessagesState

class PlannerState(MessagesState):
    next_agent: str = ""
    res_data: str = ""
    code_data: str = ""
    revi_data: str = ""
    fin_rep_data: str = ""
    curr_task: str = ""
    task_comp: bool = False
    coder_retry_count: int = 0