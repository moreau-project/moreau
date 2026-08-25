# moreau-cuda

GPU (CUDA) backend for **Moreau**, a batched differentiable conic optimization
solver. Implemented in C++/CUDA with nanobind bindings. Requires Python 3.12+.

Most users should install the unified
[`moreau`](https://pypi.org/project/moreau/) package with the CUDA extra, which
pulls in this backend:

```bash
pip install moreau[cuda]      # CUDA 12 (requires Python 3.12+)
pip install moreau[cuda13]    # CUDA 13 (requires Python 3.12+)
```

See the [main repository](https://github.com/moreau-project/moreau) for
documentation, examples, and source.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms and
upstream attribution (Clarabel.rs, diffqcp).
