"""
Tests for OpenClaw session consolidation.
"""
import json

import click
import pytest

from morpheus.training.consolidate import (
    ConsolidationStats,
    consolidate_sessions,
    deduplicate_pairs,
    extract_text_from_content,
    is_high_quality_pair,
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


def test_extract_text_from_content_accepts_single_content_block():
    content = {"type": "text", "text": "Summarize the result."}

    assert extract_text_from_content(content) == "Summarize the result."


def test_extract_text_from_content_skips_non_string_text_values():
    content = [
        {"type": "text", "text": None},
        {"type": "text", "text": ["not", "text"]},
        {"type": "text", "text": "Keep this text."},
    ]

    assert extract_text_from_content(content) == "Keep this text."


def test_is_useful_message_filters_openclaw_noise():
    assert not is_useful_message("HEARTBEAT_OK", "assistant")
    assert not is_useful_message("<environment_context><cwd>/tmp</cwd></environment_context>", "user")
    assert not is_useful_message("Chunk ID: abc Wall time: 0.1 Process exited with code 0", "assistant")
    assert not is_useful_message("Chunk ID: abc Wall time: 0.1 Process exited with code 0", "user")
    assert not is_useful_message('[{"tool_calls": [{"function": {"name": "exec_command"}}]}]', "assistant")
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


def test_parse_session_file_counts_malformed_lines_and_filtered_messages(tmp_path):
    session_path = tmp_path / "session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                "{bad json",
                json.dumps(message("user", "How do we handle malformed OpenClaw session lines?")),
                json.dumps(message("assistant", '[{"tool_calls": [{"name": "exec_command"}]}]')),
                json.dumps(
                    message(
                        "assistant",
                        "Implemented resilient JSONL parsing that skips malformed rows and still keeps valid adjacent user and assistant turns.",
                    )
                ),
            ]
        )
    )
    stats = ConsolidationStats()

    messages = parse_session_file(session_path, stats)

    assert stats.malformed_lines == 1
    assert stats.messages_seen == 3
    assert stats.messages_filtered == 1
    assert messages == [
        {"role": "user", "content": "How do we handle malformed OpenClaw session lines?"},
        {
            "role": "assistant",
            "content": "Implemented resilient JSONL parsing that skips malformed rows and still keeps valid adjacent user and assistant turns.",
        },
    ]


def test_parse_session_file_skips_non_object_json_entries(tmp_path):
    session_path = tmp_path / "session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(["not", "a", "message"]),
                json.dumps("not a message"),
                json.dumps(message("user", "How should parser handle odd JSONL rows?")),
                json.dumps(
                    message(
                        "assistant",
                        "Implemented validation that skips non-object JSONL rows without aborting session parsing.",
                    )
                ),
            ]
        )
    )

    messages = parse_session_file(session_path)

    assert messages == [
        {"role": "user", "content": "How should parser handle odd JSONL rows?"},
        {
            "role": "assistant",
            "content": "Implemented validation that skips non-object JSONL rows without aborting session parsing.",
        },
    ]


def test_is_high_quality_pair_rejects_low_signal_assistant_ack():
    assert not is_high_quality_pair("Fix the parser", "Working on it")
    assert is_high_quality_pair(
        "Fix the parser",
        "Implemented a parser fix that filters tool output, skips malformed session rows, and records diagnostics for verbose consolidation output.",
    )


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
    stats = consolidate_sessions(
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
    assert stats.files_found == 1
    assert stats.pairs_extracted == 2
    assert stats.pairs_unique == 1
    assert stats.pairs_duplicate == 1


def test_deduplicate_pairs_keeps_same_prompt_with_different_answers():
    stats = ConsolidationStats()
    pairs = [
        {
            "instruction": "How should we improve consolidation?",
            "input": "",
            "output": "Implemented stronger filtering and created tests for JSONL sessions.",
        },
        {
            "instruction": "How should we improve consolidation?",
            "input": "",
            "output": "Added a machine-readable stats report for automated training jobs.",
        },
        {
            "instruction": "How should we improve consolidation?",
            "input": "",
            "output": "Implemented stronger filtering and created tests for JSONL sessions.",
        },
    ]

    unique_pairs = deduplicate_pairs(pairs, stats)

    assert len(unique_pairs) == 2
    assert stats.pairs_unique == 2
    assert stats.pairs_duplicate == 1


def test_consolidate_sessions_writes_stats_report(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(
        sessions_dir / "session.jsonl",
        [
            message(
                "user",
                [{"type": "text", "text": "How should automation inspect consolidation?"}],
            ),
            message(
                "assistant",
                [
                    {
                        "type": "text",
                        "text": (
                            "Added a JSON stats report with file counters, message counters, "
                            "and pair counts for training automation."
                        ),
                    }
                ],
            ),
        ],
    )

    output_path = tmp_path / "dataset.jsonl"
    stats_path = tmp_path / "reports" / "stats.json"
    stats = consolidate_sessions(
        sessions_dir=sessions_dir,
        output_path=output_path,
        days=1,
        min_pairs=1,
        stats_output_path=stats_path,
    )

    report = json.loads(stats_path.read_text())
    assert report["sessions_dir"] == str(sessions_dir)
    assert report["output_path"] == str(output_path)
    assert report["days"] == 1
    assert report["stats"]["pairs_unique"] == 1
    assert report["stats"] == stats.to_dict()


def test_consolidate_sessions_reports_unwritable_dataset_output(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(
        sessions_dir / "session.jsonl",
        [
            message("user", [{"type": "text", "text": "How should we improve consolidation?"}]),
            message(
                "assistant",
                [{"type": "text", "text": "Implemented controlled output error handling for training datasets."}],
            ),
        ],
    )
    output_path = tmp_path / "dataset.jsonl"
    output_path.mkdir()

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=output_path,
            days=1,
            min_pairs=1,
        )


def test_consolidate_sessions_reports_unwritable_stats_output(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(
        sessions_dir / "session.jsonl",
        [
            message("user", [{"type": "text", "text": "How should automation inspect consolidation?"}]),
            message(
                "assistant",
                [
                    {
                        "type": "text",
                        "text": (
                            "Implemented controlled stats report write errors for automation "
                            "and recorded the failure as a user-facing training pipeline diagnostic."
                        ),
                    }
                ],
            ),
        ],
    )
    output_path = tmp_path / "dataset.jsonl"
    stats_path = tmp_path / "stats.json"
    stats_path.mkdir()

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=output_path,
            days=1,
            min_pairs=1,
            stats_output_path=stats_path,
        )


def test_consolidate_sessions_rejects_sessions_path_file(tmp_path, capsys):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.write_text("not a directory")
    output_path = tmp_path / "dataset.jsonl"

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=output_path,
            days=1,
            min_pairs=0,
        )

    captured = capsys.readouterr().out
    assert "Sessions path is not a directory" in captured
    assert "No session files found" not in captured
    assert not output_path.exists()


def test_consolidate_sessions_skips_symlinked_session_files(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    outside_session = tmp_path / "outside.jsonl"
    write_jsonl(
        outside_session,
        [
            message("user", [{"type": "text", "text": "How should symlinked sessions be handled?"}]),
            message(
                "assistant",
                [
                    {
                        "type": "text",
                        "text": (
                            "Implemented consolidation hardening that skips symlinked session "
                            "files instead of reading data outside the configured sessions directory."
                        ),
                    }
                ],
            ),
        ],
    )
    (sessions_dir / "session.jsonl").symlink_to(outside_session)
    output_path = tmp_path / "dataset.jsonl"

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=output_path,
            days=1,
            min_pairs=1,
        )

    assert not output_path.exists()


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


def test_consolidate_sessions_rejects_negative_min_pairs(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(sessions_dir / "session.jsonl", [message("assistant", "HEARTBEAT_OK")])
    output_path = tmp_path / "dataset.jsonl"

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=output_path,
            days=1,
            min_pairs=-1,
        )

    assert not output_path.exists()


def test_consolidate_sessions_rejects_negative_days(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(sessions_dir / "session.jsonl", [message("assistant", "HEARTBEAT_OK")])
    output_path = tmp_path / "dataset.jsonl"

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=output_path,
            days=-1,
            min_pairs=0,
        )

    assert not output_path.exists()


def test_consolidate_sessions_enforces_min_pairs_after_deduplication(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_jsonl(
        sessions_dir / "session.jsonl",
        [
            message("user", [{"type": "text", "text": "How should we improve consolidation?"}]),
            message(
                "assistant",
                [
                    {
                        "type": "text",
                        "text": "Implemented stronger filtering and created tests for OpenClaw JSONL sessions.",
                    }
                ],
            ),
            message("user", [{"type": "text", "text": "How should we improve consolidation?"}]),
            message(
                "assistant",
                [
                    {
                        "type": "text",
                        "text": "Implemented stronger filtering and created tests for OpenClaw JSONL sessions.",
                    }
                ],
            ),
        ],
    )

    with pytest.raises(click.exceptions.Exit):
        consolidate_sessions(
            sessions_dir=sessions_dir,
            output_path=tmp_path / "dataset.jsonl",
            days=1,
            min_pairs=2,
        )
