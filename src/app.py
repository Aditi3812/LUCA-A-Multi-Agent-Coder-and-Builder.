import sys
from pathlib import Path

# Add the project root directory (LUCA) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now your existing imports will work fine!
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from src.config import get_luca_user_id
from src.db import (
    archive_conversation,
    create_conversation,
    delete_conversation,
    get_messages,
    list_conversations,
    summarize_conversation,
    update_conversation_title,
    upsert_conversation_state,
    insert_message,
)
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


def _derive_title(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "New Chat"
    return cleaned[:48].rstrip()


def _conversation_label(conversation: dict) -> str:
    title = conversation.get("title") or "New Chat"
    preview = conversation.get("last_output_preview") or conversation.get("summary") or ""
    preview = " ".join(str(preview).split())
    if preview:
        return f"{title} — {preview[:42]}"
    return title


def _render_message_history(conversation_id: str) -> None:
    messages = get_messages(conversation_id, limit=30)
    if not messages:
        st.info("This chat is empty. Send the first message to begin.")
        return

    for message in messages:
        role = message.get("role", "assistant")
        chat_role = "user" if role == "user" else "assistant"
        with st.chat_message(chat_role):
            st.markdown(message.get("content", ""))


st.session_state.setdefault("user_id", get_luca_user_id())
st.session_state.setdefault("active_conversation_id", None)

user_id = st.session_state["user_id"]
conversation_rows = list_conversations(user_id)

if not conversation_rows:
    created_conversation = create_conversation(user_id=user_id, title="New Chat")
    conversation_rows = [created_conversation]
    st.session_state.active_conversation_id = str(created_conversation["id"])

active_conversation_id = st.session_state.get("active_conversation_id")
conversation_ids = [str(row["id"]) for row in conversation_rows]
if active_conversation_id not in conversation_ids:
    st.session_state.active_conversation_id = conversation_ids[0]
    active_conversation_id = conversation_ids[0]

conversation_map = {str(row["id"]): row for row in conversation_rows}

with st.sidebar:
    st.subheader("Chats")
    if st.button("+ New Chat", use_container_width=True):
        created_conversation = create_conversation(user_id=user_id, title="New Chat")
        st.session_state.active_conversation_id = str(created_conversation["id"])
        st.rerun()

    delete_confirmation = st.checkbox("Confirm delete active chat", value=False)
    delete_pressed = st.button("Delete active chat", use_container_width=True, type="secondary")
    if delete_pressed:
        if not delete_confirmation:
            st.warning("Check the confirmation box before deleting the active chat.")
        else:
            delete_conversation(st.session_state.active_conversation_id)
            remaining_conversations = list_conversations(user_id)
            if remaining_conversations:
                st.session_state.active_conversation_id = str(remaining_conversations[0]["id"])
            else:
                created_conversation = create_conversation(user_id=user_id, title="New Chat")
                st.session_state.active_conversation_id = str(created_conversation["id"])
            st.rerun()

    selected_conversation_id = st.selectbox(
        "Active conversation",
        options=conversation_ids,
        index=conversation_ids.index(st.session_state.active_conversation_id),
        format_func=lambda conversation_id: _conversation_label(conversation_map[conversation_id]),
        label_visibility="collapsed",
    )

    if selected_conversation_id != st.session_state.active_conversation_id:
        st.session_state.active_conversation_id = selected_conversation_id
        st.rerun()

active_conversation = conversation_map[st.session_state.active_conversation_id]

st.subheader(active_conversation.get("title") or "New Chat")
if active_conversation.get("summary"):
    st.caption(f"Memory summary: {active_conversation['summary']}")

with st.sidebar:
    if st.button("Archive active chat", use_container_width=True):
        archived_conversation = archive_conversation(st.session_state.active_conversation_id)
        remaining_conversations = [row for row in list_conversations(user_id) if str(row["id"]) != st.session_state.active_conversation_id]
        if remaining_conversations:
            st.session_state.active_conversation_id = str(remaining_conversations[0]["id"])
        else:
            created_conversation = create_conversation(user_id=user_id, title="New Chat")
            st.session_state.active_conversation_id = str(created_conversation["id"])
        st.rerun()

_render_message_history(st.session_state.active_conversation_id)

# User Input
user_input = st.text_area(
    label="Prompt",
    placeholder="Ask a follow-up, or start a new task...",
    height=140,
    label_visibility="collapsed",
    key="prompt_input",
)

col1, col2 = st.columns([1, 6])
with col1:
    submit_btn = st.button("Run", type="primary")

if submit_btn:
    if not user_input.strip():
        st.warning("Please enter a valid task.")
    else:
        conversation_id = str(active_conversation["id"])
        thread_id = str(active_conversation["thread_id"])

        insert_message(conversation_id, "user", user_input)

        if (active_conversation.get("title") or "New Chat") == "New Chat":
            update_conversation_title(conversation_id, _derive_title(user_input))
            active_conversation["title"] = _derive_title(user_input)

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "curr_task": user_input,
            "next_agent": "",
            "res_data": "",
            "code_data": "",
            "revi_data": "",
            "fin_rep_data": "",
            "task_comp": False,
            "coder_retry_count": 0,
        }

        config = {
            "recursion_limit": 15,
            "configurable": {
                "thread_id": thread_id,
            },
        }
        final_state = initial_state.copy()

        # Step counter variable
        total_steps_executed = 0
        run_succeeded = False

        with st.status("⚡ Agents collaborating...", expanded=True) as status:
            try:
                for step in app_graph.stream(initial_state, config=config):
                    total_steps_executed += 1

                    for node_name, state_update in step.items():
                        st.write(f"✦ Active Agent: `{node_name.capitalize()}` (Step {total_steps_executed})")
                        final_state.update(state_update)

                status.update(label="✔ Execution Complete", state="complete")
                run_succeeded = True

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

        if run_succeeded:
            insert_message(
                conversation_id,
                "assistant",
                output_content,
                agent_trace={
                    "next_agent": final_state.get("next_agent"),
                    "coder_retry_count": final_state.get("coder_retry_count", 0),
                    "task_comp": final_state.get("task_comp"),
                },
            )

            upsert_conversation_state(
                conversation_id,
                last_snapshot={
                    "next_agent": final_state.get("next_agent"),
                    "res_data": final_state.get("res_data"),
                    "code_data": final_state.get("code_data"),
                    "revi_data": final_state.get("revi_data"),
                    "fin_rep_data": final_state.get("fin_rep_data"),
                    "curr_task": final_state.get("curr_task"),
                    "task_comp": final_state.get("task_comp"),
                    "coder_retry_count": final_state.get("coder_retry_count", 0),
                    "messages_count": len(final_state.get("messages", [])),
                },
                last_next_agent=final_state.get("next_agent"),
                last_task_comp=bool(final_state.get("task_comp")),
            )

            refreshed_summary = summarize_conversation(conversation_id)
            if refreshed_summary:
                active_conversation["summary"] = refreshed_summary
        else:
            output_content = "No final assistant response was saved because the graph run did not complete."

        st.markdown(f'<div class="gemini-card">{output_content}</div>', unsafe_allow_html=True)

        with st.expander("🔍 System Execution Trace"):
            st.json({
                "conversation_id": conversation_id,
                "thread_id": thread_id,
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