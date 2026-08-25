# moreau-cpu

CPU backend for **Moreau**, a batched differentiable conic optimization solver.
Implemented in Rust (built with maturin) and based on Clarabel.rs, with a
DAQP-derived active-set solver for small QPs.

Most users should install the unified [`moreau`](https://pypi.org/project/moreau/)
package, which depends on this backend:

```bash
pip install moreau
```

See the [main repository](https://github.com/moreau-project/moreau) for
documentation, examples, and source.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms and
upstream attribution (Clarabel.rs, diffqcp, DAQP).
