# Project Snapshot

## Identity

- Project name: `GuardWeave`
- Package name: `guardweave`
- Positioning: a lightweight, risk-adaptive prompt-injection defense layer for hosted APIs and local LLMs
- Public channels:
  - GitHub repo: `https://github.com/Ha0c4/GuardWeave`
  - PyPI package: `https://pypi.org/project/guardweave/`

## Current Product Shape

GuardWeave is no longer just an experiment script collection. It is currently structured as:

- an installable Python package
- a CLI with beginner-friendly entrypoints
- a reusable pre/post defense pipeline
- optional judge-assisted defense modes
- starter local-judge training helpers
- public docs, tests, release metadata, and community files

## Core Capabilities

- pre-generation risk scoring and refusal
- post-generation output verification
- stage selection: `pre`, `post`, or `pre + post`
- heuristic-only mode with no required runtime dependencies
- local judge support
- remote judge support
- regex generation tied to system-prompt changes
- system-prompt binding plus runtime regex profile refresh only when the system prompt changes
- reusable backends for hosted APIs, OpenAI-compatible services, custom callables, and local HF models

## Main Code Areas

- `guardweave/core.py`
  - core policy and defense logic
  - system-prompt binding
  - judge-assisted gating
  - defended pipeline
- `guardweave/cli.py`
  - CLI entrypoint
  - `init`, `inspect`, `chat`, `train-judge`, `eval-judge`
- `guardweave/local_judges.py`
  - trained local classifier judge loading
- `guardweave/training.py`
  - transformers/PEFT-based local judge training helpers

## New-User Onboarding State

Beginner onboarding was improved in this thread history:

- `guardweave init` creates a starter system-prompt file
- `PolicyRiskDefender.inspect_input()` provides a simpler first Python API
- README quick start was rewritten around PyPI-first usage

This means a first-use path now exists without depending on repo-only example files.

## Public Docs State

The public docs currently emphasize deployment and evidence, not just features:

- `README.md`
- `README.zh-CN.md`
- `docs/PERFORMANCE.md`
- `docs/PERFORMANCE.zh-CN.md`
- `docs/THREAT_MODEL.md`
- `docs/THREAT_MODEL.zh-CN.md`
- `docs/JUDGE_BENCHMARK_COMPARISON.md`
- `docs/JUDGE_BENCHMARK_COMPARISON.zh-CN.md`

## Public Evidence State

README now leads with a deployment trade-off table:

- `No defense`
- `Heuristic-only`
- `Local judge`
- `Remote judge`

Headline public numbers:

- heuristic-only malicious violation rate: `54.44%`
- local `Qwen/Qwen2.5-3B` judge malicious violation rate: `37.78%`
- remote `gemini-2.5-flash` judge malicious violation rate: `7.67%`

Lightweight claim is currently backed by:

- `0` required runtime dependencies for `pip install guardweave`
- `+148` median extra base prompt tokens in the published benchmark write-up
- local `3B` judge protecting a `7B` base model, about `42.9%` by parameter count

## Tests State

Current visible test coverage includes:

- regex-judge binding behavior
- benchmark helper behavior
- strategy-driven experiment helper behavior
- training helper behavior
- onboarding helper behavior
- public API surface behavior
- core decision contracts
- CLI JSON contracts

Recent improvement:

- library-level core decision tests and CLI JSON contract tests were added

## Architecture State

The README now includes a maintained SVG architecture diagram:

- `docs/assets/architecture-diagram.svg`

It communicates:

- `User Input -> Pre Gate -> Wrapped Prompt / Tiered Instruction -> Base Model -> Post Verifier -> Final Output`
- optional local judge
- optional remote judge
- system-prompt binder
- regex generation cache behavior

## Packaging and Release State

- MIT licensed
- GitHub Actions CI present
- GitHub release flow present
- PyPI publishing already set up and used
- community files already present:
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`

## Current Biggest Remaining Gaps

- full benchmark reproduction still depends on optional heavyweight dependencies and external setup
- first-token latency instrumentation is still not part of the public benchmark write-up
- the new FastAPI and RAG examples are illustrative, not production-hardened reference apps
