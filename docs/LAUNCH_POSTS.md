# Launch Posts

Use these as starting points for GitHub, X, Reddit, Hacker News, or internal sharing.

## Short Post

GuardWeave is now public: a lightweight prompt-injection defense layer for hosted APIs and local LLMs.

- Heuristic-only mode uses the Python standard library only
- A local `Qwen2.5-3B` judge reduced malicious violations on a `Qwen2.5-7B-Instruct` base model by `63.96%`
- A `gemini-2.5-flash` judge reduced them by `92.33%`

Repo: https://github.com/Ha0c4/GuardWeave

## Technical Post

I just open-sourced GuardWeave, a risk-adaptive defense layer that can sit in front of commercial APIs, OpenAI-compatible local wrappers, and local Hugging Face models.

What I wanted was a reusable gating layer that is easy to drop into real deployments instead of another benchmark-only prototype.

Current repo includes:

- pre-generation risk scoring
- post-generation output checks
- judge-derived regex refresh only when the system prompt changes
- local classifier judge training hooks
- benchmark reports comparing a local `3B` judge and a remote `gemini-2.5-flash` judge on the same `7B` base model

The most useful result so far:

- local `Qwen2.5-3B` judge protecting `Qwen2.5-7B-Instruct`: malicious violation rate down by `63.96%`
- `gemini-2.5-flash` judge: malicious violation rate down by `92.33%`

Repo: https://github.com/Ha0c4/GuardWeave

## Chinese Post

GuardWeave 开源了，一个可直接套在商用 API 和本地模型前后的 prompt injection 防御层。

目前比较有说服力的一组结果：

- 本地 `Qwen2.5-3B` judge 保护 `Qwen2.5-7B-Instruct`，恶意样本违规率平均下降 `63.96%`
- `gemini-2.5-flash` judge 版本平均下降 `92.33%`

仓库地址：

https://github.com/Ha0c4/GuardWeave
