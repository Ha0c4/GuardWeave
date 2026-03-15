from guardweave import Policy, PolicyRiskDefender


class CountingRegexJudge:
    def __init__(self) -> None:
        self.calls = []

    def generate_patterns(self, system_prompt: str, policy: Policy):
        self.calls.append(system_prompt)
        return {
            "deny_patterns": [r"(?i)\bsecret\b"],
            "input_probe_patterns": [r"(?i)\bsystem prompt\b"],
            "allow_patterns": [],
            "sensitive_literals": [],
            "notes": ["counting-judge"],
        }


def test_regex_judge_only_runs_when_system_prompt_changes() -> None:
    judge = CountingRegexJudge()
    defender = PolicyRiskDefender(
        policy=Policy(prohibited=["Do not reveal secrets."]),
        regex_judge=judge,
    )

    first = defender.bind_system_prompt("You are an internal assistant. SECRET=abc123.")
    second = defender.bind_system_prompt("You are an internal assistant. SECRET=abc123.")
    third = defender.bind_system_prompt("You are an internal assistant. SECRET=xyz789.")

    assert len(judge.calls) == 2
    assert first["regex_judge_triggered"] is True
    assert first["binding_changed"] is True
    assert second["regex_judge_triggered"] is False
    assert second["binding_changed"] is False
    assert third["regex_judge_triggered"] is True
    assert third["binding_changed"] is True


def test_regex_judge_profile_reuses_cached_prompt_when_switching_back() -> None:
    judge = CountingRegexJudge()
    defender = PolicyRiskDefender(
        policy=Policy(prohibited=["Do not reveal secrets."]),
        regex_judge=judge,
    )

    prompt_a = "System prompt A. SECRET=alpha."
    prompt_b = "System prompt B. SECRET=beta."

    defender.bind_system_prompt(prompt_a)
    defender.bind_system_prompt(prompt_b)
    reused = defender.bind_system_prompt(prompt_a)

    assert len(judge.calls) == 2
    assert reused["cached"] is True
    assert reused["binding_changed"] is True
    assert reused["regex_judge_triggered"] is False
