"""Shared utilities for moreau example scripts."""

import moreau


def check_status(info) -> bool:
    """Return True if all problems in info solved successfully.

    Accepts a batched info object (info.status is an array) or a single
    info object (info.status is a SolverStatus).  Prints a warning for
    any problem that did not reach Solved / AlmostSolved.
    """
    ok_statuses = {moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved}

    statuses = info.status
    # Scalar (single-problem) case
    if not hasattr(statuses, "__len__"):
        statuses = [statuses]

    all_ok = True
    for i, s in enumerate(statuses):
        if s not in ok_statuses:
            print(f"  WARNING: problem {i} status = {s.name}")
            all_ok = False
    return all_ok


def get_torch_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
