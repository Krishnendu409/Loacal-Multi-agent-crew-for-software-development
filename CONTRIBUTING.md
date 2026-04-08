# Contributing

Thanks for contributing to Local Multi-Agent Software Development Crew.

## Development setup

1. Install dependencies:

```bash
pip install -r requirements.txt
pip install pytest
```

2. Run tests:

```bash
python -m pytest tests/ -v
```

## Coding standards

- Use Python 3.12+ compatible syntax.
- Keep changes focused and minimal.
- Prefer clear, typed, and testable code.
- Preserve the existing architecture and role naming conventions.

## Branching and release policy

- Branch from `main` using descriptive names.
- Open PRs early as draft for larger work.
- Keep PRs small and scoped to one objective.
- Require green CI checks before merge.
- Use squash merge with a clear commit title.

## Pull request checklist

- [ ] Added/updated tests for behavior changes
- [ ] Updated docs/config examples when needed
- [ ] Verified `python -m pytest tests/ -v`
- [ ] Confirmed no secrets or sensitive data added
