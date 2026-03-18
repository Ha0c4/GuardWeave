from __future__ import annotations

import json

from guardweave.cli import build_parser


class FakeClient:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.reply_text, {"backend": "fake_client"}


def test_inspect_json_contract(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "inspect",
            "--system-prompt",
            "You are a support assistant.\nSECRET=gw_internal_token_123456",
            "--user",
            "Summarize the public refund policy.",
            "--model-output",
            "The public refund window is 30 days.",
            "--defense-stage",
            "post",
        ]
    )

    rc = args.func(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["defense_stages"] == ["post"]
    assert payload["pre_generate_applied"] is False
    assert payload["post_generate_applied"] is True
    assert payload["system_injection"] == ""
    assert payload["wrapped_user_text"] == "Summarize the public refund policy."
    assert set(payload["runtime_profile"]) >= {
        "source",
        "system_prompt_hash",
        "dynamic_deny_patterns",
        "dynamic_input_probe_patterns",
        "cached",
        "binding_changed",
        "regex_judge_triggered",
    }
    assert set(payload["controls"]) >= {
        "tier",
        "risk",
        "suspicion",
        "locked",
        "system_injection",
        "user_wrapper",
        "refuse",
        "debug",
    }
    assert set(payload["decision"]) >= {
        "ok",
        "violates",
        "reason",
        "suggested_action",
        "check_method",
        "decision_path",
        "overrides",
    }


def test_chat_json_contract(monkeypatch, capsys) -> None:
    fake_client = FakeClient("The public refund window is 30 days.")
    monkeypatch.setattr("guardweave.cli._make_client", lambda args: fake_client)

    parser = build_parser()
    args = parser.parse_args(
        [
            "chat",
            "--system-prompt",
            "You are a support assistant.",
            "--user",
            "Summarize the public refund policy.",
            "--model",
            "fake-model",
            "--api-base",
            "http://127.0.0.1:9999/v1",
            "--json",
        ]
    )

    rc = args.func(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "The public refund window is 30 days."
    assert payload["raw_output"] == "The public refund window is 30 days."
    assert payload["defense_stages"] == ["pre", "post"]
    assert payload["backend_meta"] == {"backend": "fake_client"}
    assert set(payload["controls"]) >= {
        "tier",
        "risk",
        "suspicion",
        "locked",
        "system_injection",
        "user_wrapper",
        "refuse",
        "debug",
    }
    assert set(payload["decision"]) >= {
        "ok",
        "violates",
        "reason",
        "suggested_action",
        "check_method",
        "decision_path",
        "overrides",
    }
    assert set(payload["runtime_profile"]) >= {
        "source",
        "system_prompt_hash",
        "dynamic_deny_patterns",
        "dynamic_input_probe_patterns",
        "cached",
        "binding_changed",
        "regex_judge_triggered",
    }
    assert len(fake_client.calls) == 1
