# GuardWeave Judge Benchmark Comparison

[English](JUDGE_BENCHMARK_COMPARISON.md) | [简体中文](JUDGE_BENCHMARK_COMPARISON.zh-CN.md)

This report compares two 10-run benchmark suites under the same base-model and attack-selection protocol:

- `Local judge`: `Qwen/Qwen2.5-3B`
- `Remote judge`: `gemini-2.5-flash`

The goal is to show how the judge choice changes GuardWeave's prompt-injection defense performance while keeping the rest of the evaluation setup fixed.

## Setup

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Runs: `10`
- Seeds: `42-51`
- Candidate attacks per run: `200`
- Minimum effective attacks per run: `80`
- Effective attacks actually selected: `90-93`
- Benign prompts per run: `50`
- Generation cap: `max_new_tokens = 128`
- Attack source: `wambosec/prompt-injections`
- Violation scoring: local `attack_strategies.json` plus GuardWeave's system-prompt violation checks

## Aggregate Results

| Metric | Local Judge (`Qwen2.5-3B`) | Gemini Judge (`gemini-2.5-flash`) |
| --- | ---: | ---: |
| Effective attacks mean | `91.3` | `91.3` |
| Effective attack rate mean | `45.65%` | `45.65%` |
| Malicious defended violation rate mean | `36.04%` | `7.67%` |
| Malicious violation-rate reduction mean | `63.96%` | `92.33%` |
| Malicious post-block rate mean | `11.93%` | `35.38%` |
| Benign false-refusal rate mean | `11.60%` | `9.00%` |

## Stability Across 10 Runs

| Metric | Local Judge Range | Gemini Judge Range |
| --- | ---: | ---: |
| Effective attacks | `90-93` | `90-93` |
| Malicious defended violation rate | `35.16%-38.46%` | `7.53%-7.78%` |
| Malicious post-block rate | `10.99%-13.19%` | `34.07%-37.36%` |
| Benign false-refusal rate | `8%-18%` | `6%-12%` |

## Interpretation

- The attack pool is matched. Both suites operate on the same base model and the same per-run attack-selection budget, so the effective-attack counts stay aligned.
- The main difference comes from the judge layer. `gemini-2.5-flash` reduces the final malicious violation rate much more aggressively than the local `Qwen2.5-3B` classifier judges.
- The post-generation verifier becomes substantially stronger with Gemini. The `post-block` rate rises from about `11.9%` to about `35.4%`.
- False refusals also improve slightly. The Gemini suite lowers the benign false-refusal mean from `11.6%` to `9.0%`.

## Public Repo Policy

- The public repo keeps the benchmark scripts and this high-level summary.
- A tiny checked-in smoke pack is available in [`docs/BENCHMARK_REPRO.md`](BENCHMARK_REPRO.md).
- Generated artifacts under `benchmarks/results/` are intentionally excluded from GitHub.
- Generated benchmark datasets under `benchmarks/data/qwen_attack_guardweave/` and `benchmarks/data/strategy_guardweave*/` are also excluded.

## Notes

- This comparison is about judge choice, not about changing the protected base model.
- The Gemini suite depends on an external API and therefore introduces network, cost, and credential requirements that the local suite does not have.
- The local suite remains useful when offline deployment or fixed-cost inference matters more than maximum blocking strength.
