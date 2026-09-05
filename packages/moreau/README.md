# moreau

Unified Python interface for **Moreau**, a batched differentiable conic
optimization solver with CPU and optional GPU (CUDA) backends.

Solves quadratic conic programs of the form:

```
minimize    (1/2) xᵀPx + qᵀx
subject to  Ax + s = b
            x ∈ K₁,  s ∈ K₂
```

where `s ∈ K₂` constrains the slack and `x ∈ K₁` is a **Direct Conic
Constraint**. Both cones are products of nonnegative, second-order, exponential,
power, generalized power, and PSD cones. `K₂` also admits the zero cone;
`K₁` admits the free cone instead.

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
