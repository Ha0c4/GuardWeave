# Next Steps

This file is the practical continuation guide for a new thread.

## Recently Completed

The previous P1/P2 checklist was completed in the follow-up thread:

- top-level API was narrowed in `guardweave/__init__.py`
- library-level core decision tests were added
- CLI JSON contract tests were added
- a tiny public benchmark reproduction pack was added
- `examples/fastapi_middleware.py` was added
- `examples/rag_wrapper.py` was added

## Recommended Immediate Starting Point

If a new thread needs to continue from here, start with narrower polish instead of reopening the old checklist:

1. improve public benchmark instrumentation where evidence is still thin
2. harden the new integration examples if a concrete deployment target is requested
3. tighten package/documentation consistency around the new stable runtime API

Reason:

- the review-driven maturity checklist is no longer the main blocker
- the remaining work is now more about depth and production polish than missing basics

## Suggested Next Improvements

### A. Benchmark Instrumentation

Current issue:

- public docs still do not include first-token latency instrumentation
- the tiny smoke pack is useful for understanding flow, but it is not a substitute for the full benchmark protocol

### B. Production-Hardened Integrations

Current issue:

- `examples/fastapi_middleware.py` and `examples/rag_wrapper.py` are intentionally minimal
- they show placement and control flow, but not auth, streaming, persistence, tracing, or deployment concerns

### C. Public API And Docs Consistency

Current issue:

- the runtime API is now narrower, but future work should keep top-level imports stable
- optional training and local-classifier paths should continue to use explicit submodule imports in docs and examples

## Things A New Thread Should Avoid Repeating

- do not redo the repo naming/package/release setup work
- do not reopen the P0 homepage/performance/threat-model work unless the user asks
- do not assume missing benchmark result artifacts are accidental; many are intentionally excluded from GitHub
- do not expand the architecture diagram unless there is a clear visual issue or new architecture change

## Useful Files To Read Before Editing

- `README.md`
- `README.zh-CN.md`
- `docs/PERFORMANCE.md`
- `docs/THREAT_MODEL.md`
- `docs/JUDGE_BENCHMARK_COMPARISON.md`
- `guardweave/core.py`
- `guardweave/cli.py`
- `tests/test_onboarding_helpers.py`
- `.tmp/EXECUTION_CHECKLIST_FROM_REVIEW.md` if available locally

## Success Condition For The Next Thread

If the next thread only completes one of these, it will still move the project forward:

- add stronger public benchmark instrumentation or reporting depth
- harden one real integration example against a concrete deployment scenario
