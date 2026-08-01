from __future__ import annotations

import json
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from langchain_core.messages import HumanMessage

from src.config import get_database_url, llm

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT,
    last_output_preview TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
    ON conversations (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    agent_trace JSONB,
    token_usage JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON messages (conversation_id, created_at ASC);

CREATE TABLE IF NOT EXISTS conversation_state (
    conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    last_snapshot JSONB NOT NULL,
    last_next_agent TEXT,
    last_task_comp BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _connect() -> psycopg.Connection:
    return psycopg.connect(get_database_url(), row_factory=dict_row)


def init_db() -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


def create_conversation(
    user_id: str,
    title: str,
    thread_id: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    thread_id = thread_id or str(uuid.uuid4())

    query = """
    INSERT INTO conversations (user_id, thread_id, title, summary)
    VALUES (%s, %s, %s, %s)
    RETURNING id, user_id, thread_id, title, summary, last_output_preview,
              created_at, updated_at, archived;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, thread_id, title.strip() or "New Chat", summary))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError("Failed to create conversation.")

    return row


def list_conversations(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    query = """
    SELECT id, user_id, thread_id, title, summary, last_output_preview,
           created_at, updated_at, archived
    FROM conversations
    WHERE user_id = %s AND archived = FALSE
    ORDER BY updated_at DESC
    LIMIT %s;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, max(1, limit)))
            return list(cur.fetchall())


def archive_conversation(conversation_id: str) -> dict[str, Any]:
    query = """
    UPDATE conversations
    SET archived = TRUE,
        updated_at = now()
    WHERE id = %s
    RETURNING id, user_id, thread_id, title, summary, last_output_preview,
              created_at, updated_at, archived;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (conversation_id,))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError("Failed to archive conversation.")

    return row


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    query = """
    SELECT id, user_id, thread_id, title, summary, last_output_preview,
           created_at, updated_at, archived
    FROM conversations
    WHERE id = %s;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (conversation_id,))
            return cur.fetchone()


def update_conversation_title(conversation_id: str, title: str) -> dict[str, Any]:
    query = """
    UPDATE conversations
    SET title = %s,
        updated_at = now()
    WHERE id = %s
    RETURNING id, user_id, thread_id, title, summary, last_output_preview,
              created_at, updated_at, archived;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (title.strip() or "New Chat", conversation_id))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError("Failed to update conversation title.")

    return row


def insert_message(
    conversation_id: str,
    role: str,
    content: str,
    agent_trace: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query = """
    INSERT INTO messages (conversation_id, role, content, agent_trace, token_usage)
    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
    RETURNING id, conversation_id, role, content, agent_trace, token_usage, created_at;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(agent_trace) if agent_trace is not None else None,
                    json.dumps(token_usage) if token_usage is not None else None,
                ),
            )
            row = cur.fetchone()

            cur.execute(
                """
                UPDATE conversations
                SET updated_at = now(),
                    last_output_preview = CASE WHEN %s = 'assistant' THEN left(%s, 300) ELSE last_output_preview END
                WHERE id = %s;
                """,
                (role, content, conversation_id),
            )
        conn.commit()

    if not row:
        raise RuntimeError("Failed to insert message.")

    return row


def get_messages(conversation_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    query = """
    SELECT id, conversation_id, role, content, agent_trace, token_usage, created_at
    FROM messages
    WHERE conversation_id = %s
    ORDER BY created_at ASC;
    """

    if limit is not None and limit > 0:
        query = """
        SELECT id, conversation_id, role, content, agent_trace, token_usage, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """

    with _connect() as conn:
        with conn.cursor() as cur:
            if limit is not None and limit > 0:
                cur.execute(query, (conversation_id, limit))
                rows = list(cur.fetchall())[::-1]
            else:
                cur.execute(query, (conversation_id,))
                rows = list(cur.fetchall())

            return rows


def upsert_conversation_state(
    conversation_id: str,
    last_snapshot: dict[str, Any],
    last_next_agent: str | None,
    last_task_comp: bool,
) -> dict[str, Any]:
    query = """
    INSERT INTO conversation_state (conversation_id, last_snapshot, last_next_agent, last_task_comp, updated_at)
    VALUES (%s, %s::jsonb, %s, %s, now())
    ON CONFLICT (conversation_id)
    DO UPDATE SET
        last_snapshot = EXCLUDED.last_snapshot,
        last_next_agent = EXCLUDED.last_next_agent,
        last_task_comp = EXCLUDED.last_task_comp,
        updated_at = now()
    RETURNING conversation_id, last_snapshot, last_next_agent, last_task_comp, updated_at;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    conversation_id,
                    json.dumps(last_snapshot),
                    last_next_agent,
                    last_task_comp,
                ),
            )
            row = cur.fetchone()

            cur.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s;",
                (conversation_id,),
            )
        conn.commit()

    if not row:
        raise RuntimeError("Failed to upsert conversation_state.")

    return row


def get_conversation_state(conversation_id: str) -> dict[str, Any] | None:
    query = """
    SELECT conversation_id, last_snapshot, last_next_agent, last_task_comp, updated_at
    FROM conversation_state
    WHERE conversation_id = %s;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (conversation_id,))
            return cur.fetchone()


def update_conversation_summary(conversation_id: str, summary: str) -> dict[str, Any]:
    query = """
    UPDATE conversations
    SET summary = %s,
        updated_at = now()
    WHERE id = %s
    RETURNING id, user_id, thread_id, title, summary, last_output_preview,
              created_at, updated_at, archived;
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (summary.strip(), conversation_id))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError("Failed to update conversation summary.")

    return row


def summarize_conversation(conversation_id: str, recent_window: int = 8) -> str | None:
    conversation = get_conversation(conversation_id)
    if not conversation:
        return None

    messages = get_messages(conversation_id)
    if len(messages) <= recent_window:
        return conversation.get("summary")

    older_messages = messages[:-recent_window]
    transcript_lines = []
    for message in older_messages[-20:]:
        transcript_lines.append(f"{message['role']}: {message['content']}")

    existing_summary = conversation.get("summary") or ""
    prompt = f"""
You are compressing a chat history for long-term memory.

Existing summary:
{existing_summary or 'None'}

Chat excerpt to compress:
{chr(10).join(transcript_lines)}

Write a compact summary that preserves:
- the user goal
- important constraints
- decisions made
- code or implementation details that matter later

Keep it under 120 words. Do not use bullet points unless necessary.
"""

    summary = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if summary:
        update_conversation_summary(conversation_id, summary)
        return summary

    return conversation.get("summary")
