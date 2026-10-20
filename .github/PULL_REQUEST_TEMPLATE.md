## Summary

What does this PR change, and why?

## Impact

One of: No public API change / New feature / Bug fix /
**Breaking:** what changed — how to migrate (see "Versioning & Backward
Compatibility" in `CLAUDE.md`). User-visible changes also add a line to
`CHANGELOG.md` under `[Unreleased]`.

## Related issues

Closes #... (if applicable)

## Checklist

- [ ] Builds on the backends I touched (CPU Rust / GPU CUDA)
- [ ] Existing tests pass; new behavior has tests
- [ ] CPU and GPU results agree to tolerance (if both touched)
- [ ] No test was weakened to make CI pass
- [ ] New public APIs have docstrings; `docs/` and `STATE.md` updated when
      features changed
- [ ] No leftover debug output, commented-out code, or unresolved TODOs

## Notes for reviewers

Anything reviewers should focus on, trade-offs made, or follow-ups deferred.
