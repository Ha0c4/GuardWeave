# Decisions And History

This file records the decisions that matter for future work, not every single edit.

## 1. Project Was Repositioned For Public Use

The repo moved from an internal/experimental baseline shape to a public open-source package shape.

That included:

- renaming the project to `GuardWeave`
- packaging it as `guardweave`
- adding PyPI support
- adding release/community files
- cleaning out obviously obsolete or throwaway files

Why it matters:

- future work should preserve the public package identity
- compatibility and documentation quality now matter more than one-off experiment convenience

## 2. Core Defense Direction Was Centered On `policy_risk_defense`

The project evolved around that defense logic, then the core was renamed and reorganized under `guardweave/core.py`.

Key design direction:

- defend before generation and after generation
- support both hosted and local model deployments
- support both heuristic-only and judge-assisted paths
- generate extra regex patterns from the bound system prompt via a judge when enabled
- only refresh judge-derived regex data when the bound system prompt changes

Why it matters:

- future work should treat `core.py` as the source of truth
- new logic should preserve the system-prompt-bound regex lifecycle

## 3. Beginner Experience Was Intentionally Simplified

A novice onboarding simulation was run during this thread history. It exposed two blockers:

- PyPI users were implicitly pointed at repo-local example files
- first Python API usage was too low-level

The fixes were:

- add `guardweave init`
- add `PolicyRiskDefender.inspect_input()`
- rewrite README quick start around those two entrypoints

Why it matters:

- future onboarding work should keep the first experience repo-independent
- documentation should prefer the simple path first, then advanced paths later

## 4. README Was Reorganized Around Evidence, Not Just Features

An external-style review concluded that the project had enough content, but its proof structure was weak.

P0 work that was already completed:

- homepage trade-off table with `No defense / Heuristic-only / Local judge / Remote judge`
- separate performance doc for the `lightweight` claim
- separate threat-model doc for scope and non-goals

Why it matters:

- new work should continue improving proof clarity
- avoid adding feature text that weakens the landing-page evidence path

## 5. Public Benchmark Policy Is Deliberately Split

The repo keeps:

- benchmark scripts
- public high-level benchmark summaries
- docs-level comparison tables

The repo intentionally does not keep:

- most generated benchmark artifacts under `benchmarks/results/`
- generated benchmark datasets under `benchmarks/data/strategy_guardweave*`

Why it matters:

- new threads should not assume raw benchmark outputs are checked into GitHub
- if a new public reproducibility effort is started, it should be done as a minimal sample pack, not by dumping full artifacts

## 6. Remote Judge And Local Judge Were Both Benchmarked

The public benchmark narrative currently compares:

- local judge: `Qwen/Qwen2.5-3B`
- remote judge: `gemini-2.5-flash`

Same protected base model:

- `Qwen/Qwen2.5-7B-Instruct`

Why it matters:

- future benchmark changes should preserve apples-to-apples comparisons
- the public positioning already uses these two judge modes as the main contrast

## 7. Current Review-Derived Priorities

An execution checklist was created from a review.

Current status:

- `P0`: done
- `P1`: done
  - architecture diagram: done
  - public API narrowing: done
  - core/CLI contract tests: done
- `P2`: done
  - sample benchmark reproduction pack
  - FastAPI integration example
  - RAG/agent wrapper example

Why it matters:

- the next thread can move on to narrower polish work instead of reopening the old P1/P2 checklist

## 8. Architecture Diagram Exists And Was Manually Refined

The architecture SVG in `docs/assets/architecture-diagram.svg` was iterated several times to fix:

- alignment
- spacing
- overflowing labels
- visual balance

Why it matters:

- if the diagram is edited again, treat it like a maintained asset, not a placeholder
- README already depends on it

## 9. Public API Was Narrowed To Stable Runtime Imports

`guardweave/__init__.py` now keeps a stable runtime-focused surface and no longer re-exports training/local-judge helpers from the top level.

Explicit submodule imports are now the intended path for:

- `guardweave.training`
- `guardweave.local_judges`

Why it matters:

- future runtime examples can depend on the top-level package with lower compatibility risk
- optional training and classifier APIs now have room to evolve independently
