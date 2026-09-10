from __future__ import annotations

import json

import pytest

from web_agent.runtime.event_log import EpisodeEventLogs, HashChainedEventLog, verify_event_log


def test_event_log_is_append_only_hash_chained_redacted_and_episode_keyed(tmp_path):
    path = tmp_path / "actions.jsonl"
    log = HashChainedEventLog(
        path,
        stream="actions",
        episode_id="episode-1",
        context={"system_id": "E2"},
        timestamp_factory=lambda: "2025-01-01T00:00:00+00:00",
    )
    first = log.append("ACTION_REQUEST", {"action_id": "a1", "password": "do-not-store"})
    second = log.append("ACTION_RESULT", {"action_id": "a1", "status": "executed"})
    assert first["previous_record_hash"] == "0" * 64
    assert second["previous_record_hash"] == first["record_hash"]
    assert first["episode_id"] == "episode-1"
    assert "do-not-store" not in path.read_text(encoding="utf-8")

    verified = verify_event_log(
        path, expected_stream="actions", expected_episode_id="episode-1"
    )
    assert verified.records == 2
    assert verified.tail_sha256 == second["record_hash"]
    with pytest.raises(FileExistsError):
        HashChainedEventLog(path, stream="actions", episode_id="episode-1")


def test_type_action_values_and_private_form_contents_are_redacted_by_value(tmp_path):
    path = tmp_path / "actions.jsonl"
    log = HashChainedEventLog(path, stream="actions", episode_id="episode-type")
    record = log.append(
        "normal_action",
        {
            "parameters": {
                "action_type": "TYPE",
                "values": {"text": "private@example.test", "target": "email-field"},
            },
            "action": {
                "action_type": "TYPE",
                "parameters": {"text": "private@example.test"},
            },
            "observation": {
                "form_value": "private@example.test",
                "page_state": {
                    "aria_tree": [
                        {
                            "role": "textbox",
                            "name": "Email",
                            "value": "private-page-state@example.test",
                        }
                    ]
                },
            },
        },
    )
    serialized = path.read_text(encoding="utf-8")
    assert "private@example.test" not in serialized
    assert "private-page-state@example.test" not in serialized
    assert "email-field" not in serialized
    assert record["payload"]["parameters"]["values"]["text"]["redacted"] is True
    assert record["payload"]["observation"]["form_value"]["redacted"] is True


def test_url_query_credentials_and_userinfo_are_redacted_without_losing_location(tmp_path):
    path = tmp_path / "environment_events.jsonl"
    log = HashChainedEventLog(
        path,
        stream="environment_events",
        episode_id="episode-url-redaction",
        timestamp_factory=lambda: "2025-01-01T00:00:00+00:00",
    )
    record = log.append(
        "observation",
        {
            "url": (
                "https://alice:plain-password@example.test/path?view=compact&"
                "session_token=SHOULD_NOT_PERSIST#access_token=ALSO_SECRET"
            ),
            "links": [
                {
                    "href": (
                        "https://example.test/next?page=2&api_key=NESTED_SECRET"
                    )
                }
            ],
            "current_url": (
                "https://example.test/session_token/PATH_SECRET/dashboard"
            ),
        },
    )
    serialized = path.read_text(encoding="utf-8")
    for secret in (
        "alice",
        "plain-password",
        "SHOULD_NOT_PERSIST",
        "ALSO_SECRET",
        "NESTED_SECRET",
        "PATH_SECRET",
    ):
        assert secret not in serialized
    assert "example.test/path" in serialized
    assert "view=compact" in serialized
    assert "page=2" in serialized
    assert "sha256" in record["payload"]["url"]
    verify_event_log(
        path,
        expected_stream="environment_events",
        expected_episode_id="episode-url-redaction",
    )


def test_url_session_aliases_semicolon_pairs_and_nested_redirects_are_redacted(
    tmp_path,
):
    path = tmp_path / "nested_urls.jsonl"
    log = HashChainedEventLog(
        path,
        stream="environment_events",
        episode_id="episode-nested-url-redaction",
        timestamp_factory=lambda: "2025-01-01T00:00:00+00:00",
    )
    record = log.append(
        "observation",
        {
            "url": (
                "https://example.test/login?view=compact;"
                "session_token=SEMICOLON_SECRET;session=SESSION_SECRET&"
                "redirect=https%3A%2F%2Fnested.example.test%2Fcontinue%3Fpage%3D2%"
                "26session_token%3DNESTED_SECRET#panel=security;sid=SID_SECRET"
            ),
            "target_url": (
                "https://example.test/callback#redirect=https%3A%2F%2F"
                "nested.example.test%2Fdone%3Fx%3D1%3Baccess_token%3DFRAGMENT_SECRET"
            ),
        },
    )

    serialized = path.read_text(encoding="utf-8")
    for secret in (
        "SESSION_SECRET",
        "SEMICOLON_SECRET",
        "NESTED_SECRET",
        "SID_SECRET",
        "FRAGMENT_SECRET",
    ):
        assert secret not in serialized
    sanitized = record["payload"]["url"]
    assert "example.test/login" in sanitized
    assert "view=compact" in sanitized
    assert "nested.example.test" in sanitized
    assert "page%3D2" in sanitized
    assert "panel=security" in sanitized
    assert "sha256" in sanitized
    verify_event_log(
        path,
        expected_stream="environment_events",
        expected_episode_id="episode-nested-url-redaction",
    )


def test_benign_urls_are_preserved_exactly(tmp_path):
    path = tmp_path / "benign_urls.jsonl"
    benign = [
        "https://example.test/catalog?view=compact;sort=recent#panel=details",
        "https://example.test/outside/resident?session_type=guest&sidereal=true",
        "fixture://recovery/current?page=2",
    ]
    record = HashChainedEventLog(
        path,
        stream="environment_events",
        episode_id="episode-benign-urls",
        timestamp_factory=lambda: "2025-01-01T00:00:00+00:00",
    ).append("observation", {"urls": benign})

    assert record["payload"]["urls"] == benign
    serialized = path.read_text(encoding="utf-8")
    assert all(url in serialized for url in benign)


def test_event_log_tampering_is_detected(tmp_path):
    path = tmp_path / "transitions.jsonl"
    log = HashChainedEventLog(path, stream="transitions", episode_id="episode-1")
    log.append("TRANSITION", {"action_id": "a1"})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["action_id"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_event_log(path)


def test_e2_event_package_has_zero_memory_stream_and_e3_has_one(tmp_path):
    e2 = EpisodeEventLogs(tmp_path / "e2", episode_id="e2", include_memory=False)
    with pytest.raises(ValueError):
        e2.append("memory_queries", "QUERY", {"query_id": "forbidden"})

    e3 = EpisodeEventLogs(tmp_path / "e3", episode_id="e3", include_memory=True)
    record = e3.append("memory_queries", "QUERY", {"query_id": "q1"})
    assert record["episode_id"] == "e3"
