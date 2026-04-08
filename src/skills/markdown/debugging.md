# Debugging

## When to activate
- Build/test execution fails or critic reports blocking defects.

## Workflow
1. Reproduce failure from logs.
2. Isolate root cause.
3. Patch minimal surface.
4. Re-run verification.

## Rules
- Fix root cause, not symptom.
- Keep changes scoped.

## Output format
- Return changed files, steps taken, and residual risks.

## Failure conditions
- Patch without verification.
- Incomplete root-cause analysis.
