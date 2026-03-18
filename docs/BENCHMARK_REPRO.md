# Benchmark Reproduction Pack

This repository does not keep the full generated benchmark outputs on GitHub.

What it does keep now is a tiny checked-in smoke pack so a new reader can understand the flow in a few minutes:

- sample dataset: `benchmarks/data/sample_prompt_benchmark.jsonl`
- sample generated records: `benchmarks/results/sample_prompt_benchmark/sample_records.jsonl`
- sample summary artifact: `benchmarks/results/sample_prompt_benchmark/sample_summary.json`
- sample summary CSV: `benchmarks/results/sample_prompt_benchmark/sample_summary.csv`

## What This Pack Is For

This pack is intentionally small. It is not meant to reproduce the headline published numbers in `README.md`.

It is meant to answer these faster questions:

1. What does a benchmark input row look like?
2. What does a GuardWeave evaluation record look like?
3. What kind of summary metrics come out of a smoke run?

## Smoke Command

From the repo root:

```bash
python benchmarks/run_sample_benchmark.py \
  --dataset benchmarks/data/sample_prompt_benchmark.jsonl \
  --output-dir benchmarks/results/sample_prompt_benchmark
```

This command:

- runs a tiny no-network smoke benchmark
- uses only GuardWeave's built-in heuristic path
- writes deterministic JSONL/JSON/CSV artifacts

## Expected Output Files

After the smoke command finishes, inspect:

- `benchmarks/results/sample_prompt_benchmark/sample_records.jsonl`
- `benchmarks/results/sample_prompt_benchmark/sample_summary.json`
- `benchmarks/results/sample_prompt_benchmark/sample_summary.csv`

## Sample Dataset Format

Each JSONL record includes:

- `id`
- `label`
- `category`
- `user_text`
- `model_output`

The checked-in sample mixes:

- explicit prompt injection
- secret exfiltration
- chunked extraction
- benign support requests

## Sample Summary Metrics

The smoke summary currently reports:

- `malicious_pre_refusal_rate`
- `malicious_block_or_refusal_rate`
- `malicious_post_block_rate`
- `benign_false_refusal_rate`
- `malicious_mean_risk`
- `benign_mean_risk`

These are deliberately simple metrics so the artifact is easy to read without the full benchmark stack.

## Relation To Full Benchmarks

The smoke pack is separate from the larger benchmark scripts under `benchmarks/`.

Use the smoke pack when you want:

- a fast local sanity check
- a concrete example of benchmark I/O
- a docs/demo-friendly artifact

Use the larger scripts when you want:

- full attack-pool scanning
- repeated-run evaluation
- local or remote judge comparisons

## Important Limitations

- The smoke pack does not use the full public benchmark dataset.
- The smoke pack does not reproduce the headline numbers from the README.
- The smoke pack is intentionally heuristic-only and no-network.
- The full benchmark scripts may still require optional dependencies such as `datasets`, `transformers`, and `torch`.
