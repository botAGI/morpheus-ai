"""
Tests for OpenClaw session consolidation.
"""
import json

import click
import pytest

from morpheus.training.consolidate import (
    consolidate_sessions,
    extract_text_from_content,
    is_useful_message,
    messages_to_qa_pairs,
    parse_session_file,
)


def write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(entry) for entry in entries))


def message(role, content):
    return {
        "type": "message",
        "message": {
            "role": role,
            "content": content,
        },
    }


def test_extract_text_from_content_ignores_tool_blocks():
    content = [
        {"type": "text", "text": "Summarize the result."},
        {"type": "tool_use", "name": "exec_command", "input": {"cmd": "pytest"}},
        {"type": "tool_result", "content": "Chunk ID: abc"},
        {"type": "image"},
    ]

    assert extract_text_from_content(content) == "Summarize the result. [image]"


def test_is_useful_message_filters_openclaw_noise():
    assert not is_useful_message("HEARTBEAT_OK", "assistant")
    assert not is_useful_message("<environment_context><cwd>/tmp</cwd></environment_context>", "user")
    assert not is_useful_message("Chunk ID: abc Wall time: 0.1 Process exited with code 0", "assistant")
    assert not is_useful_message('{"tool_uses": [{"recipient_name": "functions.exec_command"}]}', "user")
    assert not is_useful_message("Thanks", "user")
    assert is_useful_message("Fix pytest failures", "user")


def test_parse_session_file_filters_noise_and_tool_calls(tmp_path):
    session_path = tmp_path / "session.jsonl"
    write_jsonl(
        session_path,
        [
            message("system", "You are an AI assistant"),
            message("assistant", [{"type": "text", "text": "HEARTBEAT_OK"}]),
            message("user", [{"type": "text", "text": "<environment_context><cwd>/tmp</cwd></environment_context>"}]),
            message(
                "user",
                [
                    {"type": "text", "text": "Fix pytest failures"},
                    {"type": "tool_use", "name": "exec_command", "input": {"cmd": "pytest"}},
                ],
            ),
            message(
                "assistant",
                [
                    {
                        "type": "text",
                        "text": "Implemented the fix and updated tests so the failing suite now passes.",
                    },
                    {"type": "tool_result", "content": "Chunk ID: abc"},
                ],
            ),
        ],
    )

    assert parse_session_file(session_path) == [
        {"role": "user", "content": "Fix pytest failures"},
        {
            "role": "assistant",
            "content": "Implemented the fix and updated tests so the failing suite now passes.",
        },
    ]


def test_messages_to_qa_pairs_only_uses_adjacent_assistant_turns():
    messages = [
        {"role": "user", "content": "First request needs no answer"},
        {"role": "user", "content": "Explain the parser fix"},
        {
            "role": "assistant",
            "content": "The parser now ignores tool calls, filters HEARTBEAT_OK noise, and keeps useful text turns for consolidation.",
        },
    ]

    pairs = messages_to_qa_pairs(messages)

    assert len(pairs) == 1
    assert pairs[0]["instruction"] == "Explain the parser fix"
    assert "ignores tool calls" in pairs[0]["output"]


def test_consolidate_sessions_writes_unique_pairs(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    session_path = sessions_dir / "session.jsonl"
    write_jsonl(
        session_path,
        [
            message("user", [{"type": "text", "text": "How should we improve consolidation?"}]),
            message(
                "assistant",
                [{"type": "text", "text": "Implemented stronger filtering and created tests for OpenClaw JSONL sessions."}],
            ),
            message("user", [{"type": "text", "text": "How should we improve consolidation?"}]),
            message(
                "assistant",
                [{"type": "text", "text": "Implemented stronger filtering and created tests for OpenClaw JSONL sessions."}],
            ),
        ],
    )
    session_path.touch()

    output_path = tmp_path / "dataset.jsonl"
    consolidate_sessions(
        sessions_dir=sessions_dir,
        output_path=output_path,
        days=1,
        min_pairs=1,
        verbose=True,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["instruction"] == "How should we improve consolidation?"
    assert "stronger filtering" in rows[0]["output"]


def test_consolidate_sessions_errors_when_no_pairs(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(sessions_dir / "session.jsonl", [message("assistant", "HEARTBEAT_OK")])

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=tmp_path / "dataset.jsonl",
            days=1,
            min_pairs=1,
        )
