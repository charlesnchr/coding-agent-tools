"""Extract conversation-turn chunks from coding agent sessions for embedding."""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MAX_CHUNK_CHARS = 2000


@dataclass
class Chunk:
    """A conversation-turn chunk ready for embedding."""

    id: str  # {agent}:{session_id}:{turn_index}
    text: str  # the conversation turn text
    metadata: dict = field(default_factory=dict)


def _truncate(text: str, max_len: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return " ".join(text.split())


def _extract_text_from_content(content) -> str:
    """Extract plain text from a message content field (string or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text", "output_text"):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()
    return ""


def _is_system_message(text: str) -> bool:
    if not text or len(text.strip()) < 5:
        return True
    text = text.strip()
    if text.startswith("<") and ">" in text[:100]:
        return True
    return False


# -- Claude session chunking --


def chunk_claude_session(
    filepath: Path,
    session_id: str,
    project: str,
    cwd: str,
    branch: str,
) -> list[Chunk]:
    """Extract conversation-turn chunks from a Claude Code JSONL session file."""
    chunks: list[Chunk] = []
    stat = filepath.stat()
    base_metadata = {
        "session_id": session_id,
        "agent": "claude",
        "project": project,
        "cwd": cwd,
        "branch": branch,
        "file_path": str(filepath),
        "file_mtime": stat.st_mtime,
        "file_size": stat.st_size,
    }

    # Collect messages as (role, text) pairs
    messages: list[tuple[str, str, Optional[float]]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                message = data.get("message", {})
                content = message.get("content", "")
                text = _extract_text_from_content(content)

                if not text or _is_system_message(text):
                    continue

                timestamp = None
                if "timestamp" in data:
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")).timestamp()
                    except (ValueError, AttributeError):
                        pass

                messages.append((msg_type, text, timestamp))
    except (OSError, IOError):
        return []

    # Group into conversation turns (user + subsequent assistant messages)
    turn_index = 0
    i = 0
    while i < len(messages):
        role, text, ts = messages[i]

        if role == "user":
            user_text = _clean_text(text)
            assistant_texts = []
            i += 1

            # Collect following assistant messages
            while i < len(messages) and messages[i][0] == "assistant":
                assistant_texts.append(_clean_text(messages[i][1]))
                i += 1

            # Build turn text: prioritize user message
            if assistant_texts:
                assistant_combined = " ".join(assistant_texts)
                # Budget: user gets up to 1200 chars, assistant gets the rest
                user_part = _truncate(user_text, 1200)
                remaining = MAX_CHUNK_CHARS - len(user_part) - 20
                assistant_part = _truncate(assistant_combined, max(remaining, 200))
                turn_text = f"User: {user_part}\nAssistant: {assistant_part}"
            else:
                turn_text = f"User: {_truncate(user_text)}"

            meta = {**base_metadata, "turn_index": turn_index}
            if ts:
                meta["timestamp"] = ts

            chunks.append(Chunk(
                id=f"claude:{session_id}:{turn_index}",
                text=turn_text,
                metadata=meta,
            ))
            turn_index += 1
        else:
            # Orphan assistant message (no preceding user)
            i += 1

    return chunks


# -- Codex session chunking --


def chunk_codex_session(
    filepath: Path,
    session_id: str,
    project: str,
    cwd: str,
    branch: str,
) -> list[Chunk]:
    """Extract conversation-turn chunks from a Codex JSONL session file."""
    chunks: list[Chunk] = []
    stat = filepath.stat()
    base_metadata = {
        "session_id": session_id,
        "agent": "codex",
        "project": project,
        "cwd": cwd,
        "branch": branch,
        "file_path": str(filepath),
        "file_mtime": stat.st_mtime,
        "file_size": stat.st_size,
    }

    messages: list[tuple[str, str]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") != "response_item":
                    continue

                payload = entry.get("payload", {})
                role = payload.get("role")
                if role not in ("user", "assistant"):
                    continue

                content = payload.get("content", [])
                text = _extract_text_from_content(content)
                if not text or _is_system_message(text):
                    continue

                messages.append((role, text))
    except (OSError, IOError):
        return []

    # Group into turns
    turn_index = 0
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            user_text = _clean_text(text)
            assistant_texts = []
            i += 1
            while i < len(messages) and messages[i][0] == "assistant":
                assistant_texts.append(_clean_text(messages[i][1]))
                i += 1

            if assistant_texts:
                assistant_combined = " ".join(assistant_texts)
                user_part = _truncate(user_text, 1200)
                remaining = MAX_CHUNK_CHARS - len(user_part) - 20
                assistant_part = _truncate(assistant_combined, max(remaining, 200))
                turn_text = f"User: {user_part}\nAssistant: {assistant_part}"
            else:
                turn_text = f"User: {_truncate(user_text)}"

            chunks.append(Chunk(
                id=f"codex:{session_id}:{turn_index}",
                text=turn_text,
                metadata={**base_metadata, "turn_index": turn_index},
            ))
            turn_index += 1
        else:
            i += 1

    return chunks


# -- OpenCode session chunking --


def chunk_opencode_session(
    conn: sqlite3.Connection,
    session_id: str,
    project: str,
    cwd: str,
    time_updated: float,
) -> list[Chunk]:
    """Extract conversation-turn chunks from an OpenCode SQLite session."""
    chunks: list[Chunk] = []
    base_metadata = {
        "session_id": session_id,
        "agent": "opencode",
        "project": project,
        "cwd": cwd,
        "file_mtime": time_updated,
        "file_size": 0,
    }

    # Get message parts ordered by creation time
    rows = conn.execute(
        """
        SELECT p.data as part_data, m.data as msg_data, p.time_created
        FROM part p
        JOIN message m ON p.message_id = m.id
        WHERE p.session_id = ?
        ORDER BY p.time_created ASC
        """,
        (session_id,),
    ).fetchall()

    messages: list[tuple[str, str]] = []
    for row in rows:
        try:
            part_data = json.loads(row["part_data"]) if row["part_data"] else {}
            msg_data = json.loads(row["msg_data"]) if row["msg_data"] else {}
        except json.JSONDecodeError:
            continue

        role = msg_data.get("role")
        if role not in ("user", "assistant"):
            continue

        if part_data.get("type") == "text":
            text = part_data.get("text", "").strip()
            if text and not _is_system_message(text):
                messages.append((role, text))

    # Group into turns
    turn_index = 0
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            user_text = _clean_text(text)
            assistant_texts = []
            i += 1
            while i < len(messages) and messages[i][0] == "assistant":
                assistant_texts.append(_clean_text(messages[i][1]))
                i += 1

            if assistant_texts:
                assistant_combined = " ".join(assistant_texts)
                user_part = _truncate(user_text, 1200)
                remaining = MAX_CHUNK_CHARS - len(user_part) - 20
                assistant_part = _truncate(assistant_combined, max(remaining, 200))
                turn_text = f"User: {user_part}\nAssistant: {assistant_part}"
            else:
                turn_text = f"User: {_truncate(user_text)}"

            chunks.append(Chunk(
                id=f"opencode:{session_id}:{turn_index}",
                text=turn_text,
                metadata={**base_metadata, "turn_index": turn_index},
            ))
            turn_index += 1
        else:
            i += 1

    return chunks
