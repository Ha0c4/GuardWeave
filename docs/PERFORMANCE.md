# GuardWeave Performance Profile

[English](PERFORMANCE.md) | [简体中文](PERFORMANCE.zh-CN.md)

This document turns the repo's `lightweight` claim into a deployment-oriented cost profile.

## Measurement Basis

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Generation cap: `max_new_tokens = 128`
- Public attack protocol: `wambosec/prompt-injections` + local `attack_strategies.json`
- `No defense`, `Heuristic-only`, and `Local judge` rows below use the representative full-scan run with `90` effective attacks and `50` benign prompts.
- `Remote judge` uses the published 10-run `gemini-2.5-flash` suite summary under the same base-model and attack-selection protocol.
- The public benchmark currently records end-to-end turn latency, not first-token latency. For that reason, the public runtime table uses median end-to-end latency overhead on benign prompts.

## Deployment Overview

| Setup | Malicious violation rate | Benign false-refusal rate | Median latency overhead | Median extra base prompt tokens | Median extra model/API calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| No defense | `100.00%` | `0.00%` | `0.00s` | `0` | `0` |
| Heuristic-only | `54.44%` | `2.00%` | `3.58s` | `+148` | `0` |
| Local judge (`Qwen/Qwen2.5-3B`) | `37.78%` | `10.00%` | `1.45s` | `+148` | `+1` |
| Remote judge (`gemini-2.5-flash`) | `7.67%` | `9.00%` | `3.26s` | `+148` | `+1` |

## 1. Compute Cost

- `No defense`: one `7B` base-model generation only.
- `Heuristic-only`: still one `7B` base-model generation, plus local rule-based pre/post checks.
- `Local judge`: one `7B` base-model generation plus a `3B` classifier layer for risk/output judging.
- `Remote judge`: one `7B` base-model generation plus network-backed risk/output judging.

The local judge path is still materially smaller than the protected model:

- Judge model size: `3B`
- Base model size: `7B`
- Ratio: about `42.9%`

## 2. Latency

The runtime metric exposed in the public benchmark artifacts is median end-to-end overhead on benign prompts relative to the undefended base-model call.

| Setup | Median latency overhead |
| --- | ---: |
| No defense | `0.00s` |
| Heuristic-only | `3.58s` |
| Local judge | `1.45s` |
| Remote judge | `3.26s` |

Notes:

- The heuristic-only path is not automatically the fastest defended path. In the representative run, it lets more requests flow through the full defended prompt path instead of pre-refusing early.
- The local judge path adds extra classification work, but still landed below the heuristic-only median overhead in the representative run.
- The remote judge path adds network/API latency on top of the same defended base-model generation path.

## 3. Token / API Overhead

The median extra base-model prompt cost on benign prompts is `+148` tokens across the defended setups reported here. That extra prompt budget comes from:

- the injected defense policy block
- tiered threat instructions
- wrapped user-input separation when the tier escalates

Judge-assisted modes also add a median of `+1` extra judge call on benign traffic in the current benchmark.

Interpretation:

- `Heuristic-only` changes prompt structure but does not add any extra model/API calls.
- `Local judge` adds local classifier calls without introducing network/API billing.
- `Remote judge` converts that extra call budget into external API traffic and credentials/cost requirements.

## 4. Dependency Footprint

The package is intentionally split so the default install stays minimal.

| Install path | Declared required packages | What it enables |
| --- | ---: | --- |
| `pip install guardweave` | `0` | Core library, CLI, heuristic-only path, remote REST-judge integration |
| `pip install guardweave[train]` | `7` optional ML packages | Local judge training and local classifier runtime |

The optional training stack currently adds:

- `accelerate`
- `datasets`
- `numpy`
- `peft`
- `pyyaml`
- `torch`
- `transformers`

## Takeaways

- The lightest meaningful deployment is `heuristic-only`: no extra model/API calls, no extra runtime dependencies, and a real drop in malicious violation rate.
- The best local trade-off in the current public benchmark is the `3B` local judge path: stronger blocking than heuristic-only, with a median `+1` extra local judge call on benign traffic.
- The strongest published blocking result is still the remote Gemini judge, but it buys that improvement with external API latency, credentials, and provider cost.
