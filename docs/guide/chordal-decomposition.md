# Chordal Decomposition

Chordal decomposition replaces one positive semidefinite (PSD) matrix constraint
with several smaller PSD constraints and variables that connect them. The
reformulation is exact: it changes the work the solver does, while preserving
the feasible values of the original variables and their objective.

The opportunity is **small cliques with small overlaps**, rather than sparsity
by itself. A banded matrix can be an excellent candidate; a matrix with many
zeros can still give an expensive decomposition, or none at all.

This page explains the mathematics, works through a complete example, and
describes the CPU and CUDA paths. See [PSD cones](psd-cones.md) for the `svec`
format and basic API.

Both backends use matching approximate minimum degree (AMD) ordering and
support the same three clique merge strategies. Given the same aggregate
sparsity pattern and merge setting, they produce the same cliques. To use
that structure, supply the positions where `b` may be nonzero when compiling
a fixed-structure problem. The sections below explain how this pattern,
ordering, and merging determine the reformulation and its cost.

## From an affine matrix to a graph

Consider a PSD slack block of matrix order $d$:

$$
S(x) = B - \sum_{j=1}^{n} x_j A_j \succeq 0.
$$

Here $B$ is the matrix represented by that block of `b`, and $A_j$ is the
matrix represented by the corresponding rows of column $j$ of `A`. They are
obtained by undoing the off-diagonal $\sqrt{2}$ scaling in `svec`.
The quadratic objective matrix `P` is separate from this PSD sparsity graph.

Make one graph vertex for each matrix index, and an edge $(i,j)$ whenever
**any** $A_j$ or $B$ can have a nonzero in that off-diagonal position. Include
the diagonal positions. This is the **aggregate sparsity pattern**: the union
over all decision variables, not the nonzeros of one evaluated $S(x)$.

An entry is a *structural zero* only if it stays zero throughout the family of
problems being compiled. A coefficient that happens to be zero in one solve
does not establish structural sparsity for later solves.

The graph determines the PSD structure that chordal analysis can exploit:

- A **clique** is a set of mutually adjacent vertices. A maximal clique is
  one that cannot be enlarged by adding another vertex.
- A graph is **chordal** if every cycle of length at least four has an edge
  joining two nonconsecutive vertices of the cycle.
- A **chordal extension** adds edges until the graph is chordal. These added
  edges are called *fill*. An elimination ordering determines which fill is
  introduced; different orderings can give different clique sizes.
- A **separator** is an overlap between neighboring cliques in a clique tree.
  The tree can be chosen so that the cliques containing any one vertex form
  a connected subtree, called the running-intersection property.

```{figure} ../_static/figures/chordal-cycle.svg
:alt: A four-cycle, the same graph with a dashed diagonal added, and its two triangular cliques sharing vertices 1 and 3.
:width: 100%

A four-cycle is sparse but not chordal. Adding edge (1,3) gives two cliques,
{1,2,3} and {1,3,4}, with separator {1,3}. This is one possible extension.
```

Fill belongs to the reformulation. It does **not** make an originally fixed
zero into an unconstrained entry of the original matrix: the additional
variables and equations still enforce the original matrix equality.

## What the smaller PSD constraints actually mean

Two related statements must be kept distinct.

For a chordal graph with maximal cliques $C_1,\ldots,C_p$, let $E_C$ select the
coordinates indexed by $C$.

**A sparse matrix with fixed zeros.** A symmetric matrix $S$ that is zero
outside the graph is PSD if and only if there exist PSD matrices $W_k$ such
that

$$
S = \sum_{k=1}^{p} E_{C_k}^{\mathsf T} W_k E_{C_k},
\qquad W_k \succeq 0.
$$

The $W_k$ are contributions that add up to $S$. They are generally **not**
the principal submatrices $S[C_k,C_k]$. This sparse-PSD decomposition is the
relevant statement for Moreau's slack block, whose structural zeros are fixed.

**A partial matrix with unspecified entries.** If entries outside the graph
are free to be chosen, a PSD completion exists if and only if each specified
clique principal submatrix is PSD. This completion statement explains the
dual side of the decomposition. These two cones are dual to each other; see
[Vandenberghe and Andersen, Sections 9.2 and 10.1](https://www.seas.ucla.edu/~vandenbe/publications/chordalsdp.pdf).

### A three-by-three example: the overlap matters

Take the tridiagonal matrix

$$
S(a,b) =
\begin{bmatrix}
1 & a & 0 \\
a & 1 & b \\
0 & b & 1
\end{bmatrix}.
$$

Its graph is the path $1$–$2$–$3$, so the cliques are {1,2} and {2,3}.
An exact decomposition introduces one scalar $u$:

$$
\begin{bmatrix}1 & a\\a & u\end{bmatrix} \succeq 0,
\qquad
\begin{bmatrix}1-u & b\\b & 1\end{bmatrix} \succeq 0.
$$

Embedding and adding these two matrices gives $S(a,b)$. Their PSD conditions
are $u\ge a^2$ and $1-u\ge b^2$. Such a $u$ exists exactly when
$a^2+b^2\le 1$, which is also the PSD condition for $S$.

Checking the two original principal submatrices instead would require only
$|a|\le1$ and $|b|\le1$. For example, $a=b=0.9$ passes those checks but gives
$\det S=-0.62$, so the original matrix is indefinite. Allowing a completion
in the (1,3) entry is a different problem from fixing that entry to zero.

```{figure} ../_static/figures/chordal-exactness.svg
:alt: A blue unit disk of feasible a and b lies inside an orange square from checking only principal submatrices. The red point (0.9, 0.9) lies outside the disk. The shared middle vertex carries the allocations u and 1 minus u.
:width: 100%

Overlap variables preserve the original constraint. The disk is the true
feasible set; the larger square corresponds to a PSD-completion condition.
```

Moreau's default compact conversion implements this kind of allocation with
additional decision variables and pairs of $+1$/$-1$ coefficients. It reorders
the original data into clique cones and maps the solution back afterwards.
The original optimization variables are retained; callers do not manage $u$.

## Where the savings come from

A PSD cone of matrix order $d$ has $d(d+1)/2$ scalar slack entries. Dense
spectral operations have a leading cubic dependence on $d$. A useful first
estimate of the change in **cone work** is therefore

$$
d^3 \quad\longrightarrow\quad \sum_{k=1}^{p}|C_k|^3.
$$

This is a cost proxy, not a runtime prediction. Moreau must also assemble and
factor a KKT system. The separators introduce variables and coupling; a
separator of order $r$ can carry $r(r+1)/2$ scalar overlap coordinates. More
small cones can mean a larger linear system and more bookkeeping.

For example, an order-24 path with no additional clique merging has 23 cliques
of order 2. Its cone-work proxy falls from $24^3=13{,}824$ to
$23\cdot2^3=184$. Its compact PSD rows fall from 300 to 69, while 22 overlap
variables are added. Those counts explain an opportunity, not a measured
75-fold speedup of the whole solver.

Block diagonal structure gives disjoint cliques and no separators. A
half-bandwidth-$w$ pattern gives cliques of order at most $w+1$ in the natural
ordering. A narrow arrow pattern—a small dense hub connected to otherwise
separate leaves—also has small cliques when the hub is small. The original
optimization variables or objective can still couple these cones, so smaller
PSD blocks do not imply independent optimization problems.

```{figure} ../_static/figures/chordal-patterns.svg
:alt: Six aggregate matrix patterns: block diagonal, a narrow band, a cycle, a complete bipartite pattern, a dense aggregate pattern, and a two-dimensional grid.
:width: 100%

The graph of the aggregate matrix matters. Block and narrow-band patterns
have small cliques; a cycle needs fill; bipartite and grid patterns need a
closer look. A dense aggregate cannot split even when its data has a concise
description. Blue squares mark structurally permitted entries, not values.
```

### Why merge cliques again?

Keeping every maximal clique is not always cheapest. If two cliques overlap
heavily, merging them removes separator variables while increasing the size
of one cone. Moreau exposes three choices through
`IPMSettings.chordal_decomposition_merge_method`:

| Method | What it does | What to look for |
|---|---|---|
| `'clique_graph'` | The default. Searches for permissible merges that reduce a cubic cone-cost estimate. | A compromise between small cones and heavy overlap. |
| `'parent_child'` | Traverses the clique tree and uses local fill and supernode-size thresholds of 8. A supernode is the part of a clique outside its parent separator. | Can merge aggressively, including back into the original cone. |
| `'none'` | Keeps the cliques after symbolic factorization, without the additional merge stage. | Useful for understanding the initial decomposition; it still includes fill and supernode construction. |

For clique orders $a,b$ and union order $c$, the cubic score is
$a^3+b^3-c^3$; a positive value favors a merge. It does not model the complete
KKT factorization or GPU launch costs. The
[clique-graph merging paper](https://arxiv.org/abs/1911.05615) explains this
trade-off and the role of overlap.

## Non-examples: structured sparsity that does not pay

There are several different ways for decomposition to buy nothing. These
should be distinguished from a successful reduction that merely runs slower
on a particular machine.

### A bipartite pattern with expensive fill

Consider

$$
S = \begin{bmatrix}D_1 & B\\B^{\mathsf T} & D_2\end{bmatrix},
$$

where $D_1,D_2$ are diagonal of order $k$ and $B$ has a general $k\times k$
pattern. All within-group off-diagonal entries are zero. Nevertheless, the
graph is the complete bipartite graph $K_{k,k}$ and has many chordless
four-cycles.

One valid chordal extension makes the first group a clique. It has $k$ maximal
cliques of order $k+1$, with a common separator of order $k$. Keeping these
cliques separately gives

$$
\text{cone-work proxy}=k(k+1)^3,
\qquad
\text{PSD scalar rows}=k\frac{(k+1)(k+2)}2.
$$

For $k=8$, the original PSD(16) has 136 scalar rows and proxy 4,096.
This extension has eight PSD(9) cones, 360 rows, 252 overlap variables, and
proxy 5,832. The decomposition has made both of these cost measures worse.
Merging the cliques back can recover the original cone and remove the
opportunity entirely.

```{figure} ../_static/figures/chordal-bipartite.svg
:alt: The aggregate K8,8 matrix pattern, an extension with one diagonal block filled, and a comparison showing 136 versus 360 scalar rows and cone-work proxies 4096 versus 5832.
:width: 100%

A visible pattern with many zeros is not enough. This particular valid
extension increases both PSD scalar rows and the cubic cost proxy. It is an
illustration, not a prediction of the ordering chosen by either backend.
```

The backend ordering makes this example more instructive. With
$D_1=D_2=I_8$, one variable for each entry of $B$ in row-major order, and an
explicit diagonal `b` pattern, both backends produce the following structures
using AMD ordering:

| Backend and merge choice | PSD matrix orders | Scalar PSD rows | Added overlap variables | Cubic cone-work proxy |
|---|---|---:|---:|---:|
| Original, decomposition off | 16 | 136 | 0 | 4,096 |
| Both backends, `'none'` | 15 and 9 | 165 | 36 | 4,104 |
| Both backends, `'clique_graph'` | 16 | 136 | 0 | 4,096 |
| Both backends, `'parent_child'` | 16 | 136 | 0 | 4,096 |

These are structural counts, not timing measurements. Both backends' default
strategy recombines to the original cone, buying no decomposition. With no
merging, the smaller blocks add rows and variables without reducing the cubic
cone-work proxy. The ordering and merge sequence determine this trade-off.

This is a finite, structured counterexample to treating “smaller blocks” as
an automatic improvement. It has roughly half its matrix entries zero; it is
not an asymptotically vanishing-density family. For a stronger sparse-family
counterexample, see the treewidth discussion below.

### A small band that the merge heuristic recombines

A tridiagonal PSD(8) has seven cliques of order 2 before additional merging.
On both backends, `'none'` and `'clique_graph'` keep those cliques, while
`'parent_child'` recombines them into a single PSD(8). With that choice there
is no smaller cone left to exploit.

Even keeping the seven small cones does not guarantee a speedup for such a
small problem: analysis, mappings, and the enlarged variable set can cost
more than the spectral work saved.

### A sparse family with no small separators

The minimum possible largest clique order among all chordal extensions is
the graph's **treewidth plus one**. A low nonzero count does not bound it.
Bounded-degree expander families have only $O(d)$ edges but treewidth
$\Omega(d)$: every chordal extension still contains a clique whose order
grows linearly with the original PSD order. There is no asymptotic reduction
of the largest dense PSD block, even though the pattern becomes arbitrarily
sparse as $d$ grows. Constant-factor or instance-specific gains remain
possible. See the primary results on
[expansion, treewidth, and separators](https://arxiv.org/abs/0910.3014).

A two-dimensional nearest-neighbor grid is an intermediate case: it is sparse,
but cycles create fill and separators grow with the grid size. It can still
benefit from decomposition; it should not be mistaken for a path with
bounded-size cliques.

### Sparse storage, but a dense aggregate

An `A` matrix can have very few nonzeros while `b` represents a dense PSD
matrix. For instance, $S(x)=vv^{\mathsf T}+\operatorname{diag}(x)$ with all
entries of $v$ nonzero has a dense aggregate graph. Its rank-one-plus-diagonal
description is useful structure, but not structure exploited by this chordal
conversion. The graph has one clique of order $d$.

Likewise, a union of individually sparse coefficient matrices can cover every
matrix position. Moreau uses the union, rather than decomposing each
coefficient matrix separately. Explicit zero entries stored in `A` also count
as structural entries; remove them before fixing a sparse pattern if they
are guaranteed to stay zero.

### Sparse values, but no declared `b` pattern

At compilation time the solver must know which positions of `b` may be
nonzero. The native compiled paths conservatively assume that every row not
occupied by `A` might have a nonzero `b` when no pattern is supplied. Together
with `A`, that assumption fills the aggregate graph. A numerically banded
matrix can therefore produce no decomposition at all.

Supply `b_sparsity_pattern` at entry points that accept it when the zeros are
structural. Use the union of allowed positions across the batch and future
solves, rather than the nonzeros of one convenient example. Single-problem
paths can infer a pattern from the first `b`, but that is not a promise that
every batched or framework path does the same.

## A runnable Moreau example

This solves the three-by-three example above with objective
$\max(a+b)$. The analytic optimum is $a=b=1/\sqrt{2}$ and $a+b=\sqrt{2}$.

```python
import moreau
import numpy as np
from scipy import sparse

# svec order: (0,0), (0,1), (1,1), (0,2), (1,2), (2,2).
# Ax + s = b, so negative A entries put +a and +b in the slack.
P = sparse.csr_matrix((2, 2))
q = -np.ones(2)
A = sparse.csr_matrix(
    (-np.sqrt(2) * np.ones(2), ([1, 4], [0, 1])), shape=(6, 2)
)
b = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
cones = moreau.Cones(psd_dims=[3])

for chordal in (False, True):
    solver = moreau.Solver(
        P, q, A, b, cones,
        settings=moreau.Settings(
            device="cpu", solver="ipm", verbose=True,
            ipm_settings=moreau.IPMSettings(
                chordal_decomposition_enable=chordal,
                chordal_decomposition_merge_method="none",
            ),
        ),
    )
    solution = solver.solve()
    a, beta = np.asarray(solution.x)
    S = np.array([[1.0, a, 0.0], [a, 1.0, beta], [0.0, beta, 1.0]])
    print(solver.info.status, a + beta, np.linalg.eigvalsh(S).min())
    np.testing.assert_allclose([a, beta], np.full(2, 1 / np.sqrt(2)), atol=1e-5)
    assert np.linalg.eigvalsh(S).min() >= -1e-6
```

Both solves should reach the same optimum to numerical tolerance. With
`'none'`, the decomposed solver has two PSD(2) cones and one extra variable.
Both versions have six scalar PSD rows: a reduction in matrix order does not
always reduce the number of rows. Change `device` to `'cuda'` to use the GPU
binding path when a compatible CUDA backend is installed.

For fixed-structure batches, the same data can be compiled with an explicit
`b` mask:

```python
compiled = moreau.CompiledSolver(
    n=2, m=6,
    P_row_offsets=P.indptr, P_col_indices=P.indices,
    A_row_offsets=A.indptr, A_col_indices=A.indices,
    cones=cones,
    b_sparsity_pattern=(b != 0).tolist(),  # Valid for this fixed-diagonal family.
    settings=moreau.Settings(
        device="cpu", solver="ipm", batch_size=2,
        ipm_settings=moreau.IPMSettings(
            chordal_decomposition_merge_method="none",
        ),
    ),
)
compiled.setup(P_values=P.data, A_values=A.data)
solutions = compiled.solve(qs=np.stack([q, 2 * q]), bs=np.stack([b, b]))
np.testing.assert_allclose(
    solutions.x, np.full((2, 2), 1 / np.sqrt(2)), atol=1e-5
)
```

The extra dimensions are internal. Inputs and returned `x`, `s`, and `z` use
the original problem's dimensions. Recovering a full PSD dual `z` can involve
PSD completion; its entries outside the aggregate pattern need not be zero.

## Backend behavior

Both backends use a compact clique reformulation on their chordal-enabled
paths. CPU graph analysis uses the Rust `amd` crate, a port of SuiteSparse
AMD; CUDA graph analysis uses vendored SuiteSparse AMD. Both use default
controls and matching merge semantics, so the same aggregate pattern and
merge setting give the same clique partition. Graph analysis runs on the
host for both backends; CUDA accelerates the numerical cone operations and
KKT solves.

| Stage | CPU | CUDA |
|---|---|---|
| Graph analysis | Rust host code; AMD ordering through the Rust `amd` crate. | C++ host code; vendored SuiteSparse AMD with default controls, matching the Rust `amd` ordering. |
| Clique merging | Honors `'clique_graph'`, `'parent_child'`, and `'none'`. | Honors the same strategies through C++ implementations with matching semantics. |
| Conversion | Rust builds the compact augmented problem and its value maps. | The C++ Python binding analyzes the structure and constructs augmented CSR data before creating the GPU solver. |
| Numerical iterations | CPU cone operations and KKT solves. | CUDA cone operations and GPU KKT solves; splitting cones does not move the graph analysis to the GPU. |
| Recovery | Maps the primal/slack results back and can complete the dual. | Binding-side host code maps results back and can complete the dual; this adds work around the GPU solve. |

Different entry points can supply different structural information, especially
for `b`, and therefore build different aggregate graphs. Match that information
before comparing clique partitions. Even with matching cliques, compare final
objective values and feasibility in the original problem: floating-point
arithmetic, KKT solvers, and recovery costs can produce different iteration
counts, timings, and numerical solutions.

### Framework and differentiation details

The entry point matters as well as the selected device:

- The CPU Python single-problem adapter can infer the `b` pattern from the
  supplied values. Fixed-structure compiled batches and framework solvers
  should receive an explicit valid pattern when relying on chordal savings.
- CUDA's Python binding performs the host-side chordal conversion. The
  JAX CUDA **XLA FFI** path constructs the core C++ solver directly and bypasses
  that conversion. Selecting CUDA under JAX therefore does not by itself
  establish that chordal decomposition ran. The callback path and FFI path
  should not be assumed to have the same decomposition.
- Differentiation maps through the conversion and recovery. CPU stateless
  backward paths can differentiate the original system rather than reuse the
  decomposed one. Gradients outside a declared `b` pattern are currently
  inconsistent across paths; [issue #78](https://github.com/moreau-project/moreau/issues/78)
  tracks the convention. Restrict comparisons to the parameters that are
  actually allowed to vary.
- Dual completion is a separate numerical step after the solve. The CUDA
  recovery path accounts for solver scaling and handles singular separator
  blocks. A completion failure is different from the interior-point method
  failing to converge.

Small cones can be particularly cheap on the CPU, while GPU efficiency also
depends on having enough work and batching to amortize overhead. Neither
backend is guaranteed to win because the input matrix is sparse.

## How to evaluate a decomposition on your problem

1. **Check the pattern.** Inspect the aggregate support of the PSD rows of
   `[A b]`, including constants and all allowed batch values. Verify the `svec`
   ordering and distinguish structural zeros from zeros in one solve.
2. **Check that it ran.** Enable `verbose=True` and inspect the solver header's
   cone counts and dimensions. A setting alone does not prove a split
   happened. Count both matrix orders and total scalar rows/extra variables.
3. **Compare all four choices.** Try decomposition off, then on with
   `'none'`, `'clique_graph'`, and `'parent_child'`. Keep the remaining solver
   settings and data fixed.
4. **Validate the original answer.** Check status, objective, `s = b - A @ x`,
   and the eigenvalues of the reconstructed original PSD matrix. For gradient
   applications, compare derivatives on allowed parameters with appropriately
   converged finite differences.
5. **Measure the workload you will use.** Separate construction, setup, solve,
   recovery, and backward time. Warm up first; synchronize when timing CUDA.
   Include your intended batch size and number of repeated solves so that
   one-time analysis is amortized honestly.

Prefer the choice that gives reliable original-problem answers and lower
end-to-end cost. Cone sizes and cubic scores explain the trade-offs; timing
and residuals decide the practical result.

## Further reading and implementation

- Vandenberghe and Andersen, [*Chordal Graphs and Semidefinite Optimization*](https://www.seas.ucla.edu/~vandenbe/publications/chordalsdp.pdf): Sections 9–10 distinguish sparse PSD matrices from PSD completion.
- Sun, Andersen, and Vandenberghe, [*Decomposition in Conic Optimization with Partially Separable Structure*](https://arxiv.org/abs/1306.0057): clique conversion and the coupling it introduces.
- Garstka, Cannon, and Goulart, [*A Clique Graph Based Merging Strategy for Decomposable SDPs*](https://arxiv.org/abs/1911.05615): choosing which overlapping cliques to recombine.
- Böttcher, Pruessmann, Taraz, and Würfl, [*Bandwidth, Expansion, Treewidth, Separators, and Universality for Bounded Degree Graphs*](https://arxiv.org/abs/0910.3014): why low degree alone does not give small separators.
- CPU implementation: [`solver/chordal/`](https://github.com/moreau-project/moreau/tree/main/packages/moreau-cpu/src/solver/chordal).
- CUDA implementation: [`chordal_info.cpp`](https://github.com/moreau-project/moreau/blob/main/packages/moreau-cuda/src/chordal/chordal_info.cpp), [`merge_strategy.hpp`](https://github.com/moreau-project/moreau/blob/main/packages/moreau-cuda/include/moreau/chordal/merge_strategy.hpp), and the [`Python bindings`](https://github.com/moreau-project/moreau/blob/main/packages/moreau-cuda/bindings/moreau_bindings.cpp).

The figures show graph structure and arithmetic cost estimates, not measured
benchmarks. Their {download}`generation script <../_scripts/chordal_figures.py>` is included
with the documentation.
