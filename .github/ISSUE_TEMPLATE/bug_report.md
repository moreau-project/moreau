---
name: Bug report
about: Report incorrect results, a crash, or other unexpected behavior
title: ""
labels: bug
assignees: ""
---

## Description

A clear and concise description of the bug.

## Reproduction

A minimal, self-contained example that reproduces the problem (problem data,
cones, settings). Smaller is better.

```python
import moreau
# ...
```

## Expected vs. actual

- **Expected:** what you expected to happen.
- **Actual:** what actually happened (include the full error message / traceback
  or the wrong output, and the `SolverStatus` if relevant).

## Environment

- Moreau version: (`python -c "import moreau; print(moreau.__version__)"`)
- Backend: CPU / CUDA (and GPU model + CUDA version if CUDA)
- OS and Python version:
- Installed via: pip wheel / source build

## Additional context

Anything else that might help — does it reproduce on CPU and CUDA, does a
reference solver (Clarabel/Mosek) solve the same problem, etc.
