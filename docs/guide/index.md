# User Guide

This guide covers the core concepts and features of Moreau.

## Contents

```{toctree}
:maxdepth: 2

basic-usage
psd-cones
batching
direct-cones
warm-starting
device-selection
solver-settings
algorithms
smoothed-differentiation
pytorch-integration
jax-integration
cvxpy-integration
cvxpylayers-integration
jump-integration
julia-integration
testing-diagnostics
```

## Overview

Moreau solves convex conic optimization problems:

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{subject to} \quad & Ax + s = b \\
& x \in \mathcal{K}_1, \; s \in \mathcal{K}_2
\end{aligned}
$$

Where $\mathcal{K}_2$ constrains the slack $s$ and $\mathcal{K}_1$ constrains $x$
directly (direct conic constraints); each is a Cartesian product of convex cones.

## Key Concepts

1. **Single vs Batched**: Use `Solver` for one problem, `CompiledSolver` for many with shared structure
2. **Multiple APIs**: Python (NumPy, PyTorch, JAX) and Julia (JuMP/MOI, ChainRules) with autograd support
3. **Device Selection**: Automatic CPU/GPU selection, or manual override
4. **Cone Specification**: Define constraint types via the `Cones` class (Python) or MOI cone sets (Julia)

## Data Flow

```
Problem Data (P, q, A, b, cones)
    |
    v
+-------------------+
| Solver/           |  Validates dimensions, converts to CSR
| CompiledSolver    |
+-------------------+
    |
    v
+-------------------+
| Backend           |  CPU (Rust) or CUDA (C++)
+-------------------+
    |
    v
+-------------------+
| Solution          |  x, z, s arrays + metadata
+-------------------+
```

## Solver vs CompiledSolver

| Feature | Solver | CompiledSolver |
|---------|--------|----------------|
| Use case | Single problem | Batched problems |
| Input format | Dense or sparse matrices (NumPy/SciPy) | CSR structure |
| Batching | No | Yes |
| Setup | Implicit | Explicit `setup()` call |
| Best for | Prototyping | Production |
