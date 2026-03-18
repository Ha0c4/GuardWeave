# GuardWeave Migration Pack

This folder is a handoff bundle for a new Codex thread.

Purpose:

- sync the current project state without relying on prior chat history
- keep only repo-relevant context
- avoid machine-specific environment details, cached artifacts, or secrets

Read order:

1. `PROJECT_SNAPSHOT.md`
2. `DECISIONS_AND_HISTORY.md`
3. `NEXT_STEPS.md`
4. `context.json`

Rules for the next thread:

- Prefer repo source files over this folder if there is any conflict.
- Treat all paths in this folder as repo-relative.
- Do not assume benchmark artifacts under `benchmarks/results/` are available on GitHub.
- Do not assume prior runtime environment, local model cache, API keys, or OS-specific setup.

What this pack is for:

- quickly understanding what GuardWeave currently is
- knowing what was recently changed and why
- resuming work after the review-driven P1/P2 checklist without rereading the full conversation

What this pack intentionally excludes:

- local credentials or API keys
- absolute filesystem paths
- machine-specific shell state
- unpublished temporary experiment outputs
