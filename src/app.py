import sys
from pathlib import Path

# Add the project root directory (LUCA) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now your existing imports will work fine!
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from src.graph.workflow import app_graph


st.set_page_config(page_title="LUCA", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #ffffff;
            --panel: #ffffff;
            --text: #050505;
            --muted: #666666;
            --gold: #b08d57;
            --gold-soft: rgba(176, 141, 87, 0.22);
            --border: #050505;
            --border-soft: rgba(5, 5, 5, 0.14);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #000000;
                --panel: #090909;
                --text: #ffffff;
                --muted: #a7a7a7;
                --gold: #d1b06c;
                --gold-soft: rgba(209, 176, 108, 0.22);
                --border: #ffffff;
                --border-soft: rgba(255, 255, 255, 0.18);
            }
        }

        html, body {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        .stApp {
            background: var(--bg) !important;
            color: var(--text);
        }

        .stApp [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
        }

        .stApp [data-testid="stHeader"] {
            background: transparent !important;
        }

        .block-container {
            max-width: 920px;
            padding-top: 2.1rem;
            padding-bottom: 3.2rem;
        }

        .luca-hero {
            text-align: center;
            margin: 0 0 2.1rem 0;
        }

        .luca-wordmark {
            margin: 0;
            color: var(--text);
            font-family: "Baskerville", "Bodoni MT", "Didot", "Times New Roman", serif;
            font-size: clamp(5.2rem, 12vw, 9.25rem);
            font-style: italic;
            font-weight: 400;
            letter-spacing: -0.1em;
            line-height: 0.88;
            text-rendering: geometricPrecision;
        }

        .luca-subtitle {
            margin-top: 0.65rem;
            font-size: 0.78rem;
            letter-spacing: 0.56em;
            text-transform: uppercase;
            color: var(--gold);
            font-weight: 500;
        }

        .luca-divider {
            width: 112px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--gold), transparent);
            margin: 1.1rem auto 0;
        }

        .stTextArea label {
            color: var(--text) !important;
            font-weight: 600 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.68rem !important;
            display: block;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .stTextArea textarea {
            background: var(--panel) !important;
            color: var(--text) !important;
            border: 2px solid var(--border) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 1.15rem 1.15rem !important;
            font-size: 1.02rem !important;
            line-height: 1.65 !important;
            caret-color: var(--gold) !important;
            transition: border-color 160ms ease, box-shadow 160ms ease, background-color 160ms ease;
        }

        .stTextArea textarea:focus {
            border-color: var(--gold) !important;
            box-shadow: 0 0 0 2px var(--gold-soft) !important;
        }

        .stTextArea textarea::placeholder {
            color: var(--muted) !important;
            font-size: 0.98rem !important;
            opacity: 1 !important;
        }

        .stTextArea textarea::-webkit-input-placeholder {
            color: var(--muted) !important;
            font-size: 0.98rem !important;
        }

        .stButton > button {
            background: var(--text) !important;
            color: var(--bg) !important;
            border: 1px solid var(--text) !important;
            border-radius: 0 !important;
            padding: 0.7rem 1.35rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            transition: transform 160ms ease, border-color 160ms ease, color 160ms ease, background 160ms ease;
        }

        .stButton > button:hover {
            background: var(--bg) !important;
            color: var(--text) !important;
            border-color: var(--gold) !important;
            transform: translateY(-1px);
        }

        .stAlert {
            border-radius: 0 !important;
            border: 1px solid var(--border) !important;
        }

        .stExpander {
            border: 1px solid var(--border-soft) !important;
            border-radius: 0 !important;
        }

        .stMarkdown, .stSubheader, .stJson {
            color: var(--text);
        }

        .stTextArea {
            margin-top: 0.25rem;
        }

        .stTextArea > div {
            border-radius: 0 !important;
        }

        .block-container hr {
            border-color: var(--gold);
            opacity: 0.55;
        }
    </style>
    <div class="luca-hero">
        <h1 class="luca-wordmark">LUCA</h1>
        <div class="luca-subtitle">Build and Code</div>
        <div class="luca-divider"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
# User Input
user_input = st.text_area(
    label="Prompt",
    placeholder="e.g., Build a spam email detector in Python, or write a web scraper...",
    height=140,
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 6])
with col1:
    submit_btn = st.button("Run", type="primary")

if submit_btn:
    if not user_input.strip():
        st.warning("Please enter a valid task.")
    else:
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "curr_task": user_input,
            "next_agent": "",
            "res_data": "",
            "code_data": "",
            "revi_data": "",
            "fin_rep_data": "",
            "task_comp": False,
            "coder_retry_count": 0
        }
        
        config = {"recursion_limit": 15}
        final_state = initial_state.copy()
        
        # Step counter variable
        total_steps_executed = 0

        with st.status("⚡ Agents collaborating...", expanded=True) as status:
            try:
                for step in app_graph.stream(initial_state, config=config):
                    total_steps_executed += 1
                    
                    for node_name, state_update in step.items():
                        st.write(f"✦ Active Agent: `{node_name.capitalize()}` (Step {total_steps_executed})")
                        final_state.update(state_update)
                
                status.update(label="✔ Execution Complete", state="complete")

            except GraphRecursionError:
                status.update(label="✖ Recursion Limit Reached", state="error")
                st.error("Stopped: Reached maximum allowed agent routing loops.")
            except Exception as e:
                status.update(label="✖ Error Occurred", state="error")
                st.error(f"An unexpected error occurred: {e}")

        # Display Final Output inside a Card
        st.markdown("### 📋 Final Output")
        
        output_content = ""
        if final_state.get("fin_rep_data"):
            output_content = final_state["fin_rep_data"]
        elif final_state.get("messages"):
            output_content = final_state["messages"][-1].content
        else:
            output_content = "No output generated."

        st.markdown(f'<div class="gemini-card">{output_content}</div>', unsafe_allow_html=True)

        with st.expander("🔍 System Execution Trace"):
            st.json({
                "next_agent": final_state.get("next_agent"),
                "has_res_data": bool(final_state.get("res_data")),
                "has_code_data": bool(final_state.get("code_data")),
                "revi_data": final_state.get("revi_data"),
                "has_fin_rep": bool(final_state.get("fin_rep_data")),
                "coder_retry_count": final_state.get("coder_retry_count", 0),
                "total_steps": total_steps_executed,
                "task_comp": final_state.get("task_comp"),
                "messages_count": len(final_state.get("messages", []))
            })