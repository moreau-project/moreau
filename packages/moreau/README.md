# moreau

Unified Python interface for **Moreau**, a batched differentiable conic
optimization solver with CPU and optional GPU (CUDA) backends.

Solves quadratic conic programs of the form:

```
minimize    (1/2) xᵀPx + qᵀx
subject to  Ax + s = b
            x ∈ K₁,  s ∈ K₂
```

where `K₂` constrains the slack `s` and `K₁` constrains `x` directly (direct-x
cones); each is a product of convex cones (zero, nonnegative, second-order,
exponential, power, generalized power, PSD).

```bash
pip install moreau            # CPU only
pip install moreau[cuda]      # CUDA 12 GPU backend (requires Python 3.12+)
```

This package depends on `moreau-cpu` and optionally `moreau-cuda`. See the
[main repository](https://github.com/moreau-project/moreau) for documentation,
examples, and source.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms and
upstream attribution (Clarabel.rs, diffqcp, DAQP).
