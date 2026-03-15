# Contributing to GuardWeave

Thanks for considering a contribution.

## What Is Most Helpful

- Bug reports with a minimal reproduction
- Integration examples for hosted APIs or local model stacks
- New benchmark strategies that reflect real prompt-injection behavior
- Documentation fixes that reduce setup friction

## Before You Open a Pull Request

1. Open an issue first for large API or behavior changes.
2. Keep one change set per pull request.
3. Update docs or examples when the user-facing behavior changes.
4. Add or update tests when possible.

## Local Setup

```bash
git clone https://github.com/Ha0c4/GuardWeave.git
cd GuardWeave
pip install -e .[dev,train]
pytest
```

## Pull Request Checklist

- The change is scoped and explained
- New behavior is covered by tests, or the gap is called out
- README or docs were updated if the public interface changed
- Sensitive data, local paths, and generated benchmark artifacts are not included

## Benchmark and Model Contributions

- Keep benchmark inputs small enough to review in git
- Do not commit model weights, local caches, or generated result bundles
- If a benchmark depends on an external dataset, document the source and license

## Security Issues

Please do not open public issues for suspected vulnerabilities or credential exposure.
Use the process in [SECURITY.md](SECURITY.md) instead.
