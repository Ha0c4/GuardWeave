from __future__ import annotations

import guardweave


def test_top_level_public_api_is_runtime_focused() -> None:
    assert set(guardweave.__all__) == {
        "ChatBackendJSONAdapter",
        "CallableChatBackend",
        "Controls",
        "DefenderConfig",
        "DefendedChatPipeline",
        "DefendedGenerationResult",
        "LLMJudgePrompts",
        "LLMOutputJudge",
        "LLMRegexJudge",
        "LLMRiskJudge",
        "OpenAICompatibleRESTClient",
        "OpenAICompatibleRESTConfig",
        "Policy",
        "PolicyRiskDefender",
        "RegexJudgePrompts",
        "TransformersChatBackend",
    }


def test_top_level_public_api_excludes_optional_training_and_local_judges() -> None:
    assert not hasattr(guardweave, "TrainJudgeConfig")
    assert not hasattr(guardweave, "LocalSequenceRiskJudge")
    assert not hasattr(guardweave, "LocalSequenceOutputJudge")
    assert not hasattr(guardweave, "JudgeArtifactConfig")
