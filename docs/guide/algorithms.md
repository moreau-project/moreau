# Algorithms in Moreau

Moreau solves the conic QP

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{1}{2}\, x^\top P x + q^\top x \\
\text{subject to} \quad & A x + s = b \\
& x \in \mathcal{K}_1, \quad s \in \mathcal{K}_2,
\end{aligned}
$$

with $\mathcal{K}_1, \mathcal{K}_2$ products of convex cones: $\mathcal{K}_2$
constrains the slack $s = b - Ax$ (the usual conic form), while $\mathcal{K}_1$
constrains $x$ through direct conic constraints. It then differentiates through the
solution map. Most pieces below come from published papers (cited); the parts
original to Moreau — the **direct extension**, a primal–dual scaling and
higher-order corrector for the **generalized power cone**, and the
structure-exploiting GPU KKT backends — each get their own section.

## Forward solve

- **Interior-point method (IPM).** Default solver: a primal-dual IPM on the
  homogeneous embedding, based on **Clarabel** [1] (GPU port **CuClarabel** [2])
  and extended for direct conic constraints. The embedding and the Newton iteration are
  spelled out in the next two sections.
- **Active-set.** Handles only the zero+nonneg special case (CPU) — i.e. a
  dense, two-sided-inequality QP, no general cones. The vendored solver is
  **DAQP** [5] (dual active-set, recursive $LDL^\top$ updates), stripped to the
  general-constraint core. Note DAQP does *not* solve the conic form above:
  its problem is a nonnegative least-distance problem (LDP), so Moreau densifies
  the zero+nonneg QP and reformulates it to LDP form (`transform.cpp`) before
  the active-set loop runs.

## The homogeneous embedding

The IPM does not solve the QP's KKT system directly. It adjoins two scalars
$\tau \ge 0$ (homogenizer) and $\kappa \ge 0$ and finds a **root of the
homogeneous embedding** — the residual map $F(x, s, z, \tau, \kappa)$ whose
zeros are

$$
\begin{aligned}
r_x &= Px + q\tau + A^\top z - \sum_J E_J^\top z_x &&= 0 && \text{(stationarity)} \\
r_z &= Ax + s - b\tau &&= 0 && \text{(primal feasibility)} \\
r_\tau &= q^\top x + b^\top z + \kappa + \tfrac{x^\top P x}{\tau} &&= 0 && \text{(gap row)}
\end{aligned}
$$

together with the conic complementarity on **both** cone families,

$$
\begin{aligned}
& s \in \mathcal{K}_2, && z \in \mathcal{K}_2^*, && s \circ z = 0, \\
& x_J \in \mathcal{K}_1, && z_x \in \mathcal{K}_1^*, && x_J \circ z_x = 0,
\end{aligned}
\qquad \tau, \kappa \ge 0, \;\; \tau\kappa = 0,
$$

where $x_J = E_J x$ is the sub-vector of $x$ that direct conic constraint $J$ constrains,
$z_x$ its dual, and $E_J$ the selection of that cone's coordinates.

The two cone families enter the embedding differently, and that difference is
the subject of this note. A **slack** cone constrains $s = b - Ax$, so its dual
$z$ reaches $x$ only through the constraint matrix — the $A^\top z$ term in
$r_x$. A **direct** cone constrains a slice of $x$ itself, so its dual $z_x$
enters $r_x$ directly, through $-\sum_J E_J^\top z_x$. Everything in the
direct extension section below is a consequence of that one structural
difference. Two further properties make this an *embedding* rather than the bare
KKT system:

- **One map covers every outcome.** At a root with $\tau > 0$ the QP solution is
  recovered by dividing through, $(x, s, z)/\tau$. A root with
  $\tau = 0,\ \kappa > 0$ is an infeasibility/unboundedness certificate
  (recovered via $1/\kappa$). Feasible, infeasible, and unbounded problems are
  all just roots of the *same* $F$ — no separate phase-I, no special-casing.
- **Homogeneous but *not* self-dual — $x^\top P x/\tau$ is why.** With $P = 0$
  (an LP or linear-objective conic program) this is the classic *self-dual*
  embedding: $r_\tau$ is the gap row $q^\top x + b^\top z + \kappa = 0$ and the
  embedding's system matrix is skew-symmetric. A genuine quadratic objective
  puts the symmetric block $P$ in the $(1,1)$ position and adds the
  $x^\top P x/\tau$ term, breaking that skew-symmetry — the embedding stays
  homogeneous but is no longer self-dual. Clarabel [1] handles exactly this
  non-self-dual quadratic case.

## The IPM is Newton's method on the embedding

The only nonsmooth piece of $F$ is the complementarity $s \circ z = 0$. The IPM
relaxes it to the smooth **central path** $s \circ z = \mu e$ and drives
$\mu \to 0$, where the duality measure is

$$
\mu = \frac{s^\top z + \sum_J x_J^\top z_x + \tau\kappa}{\nu + 1},
\qquad \nu = \text{total cone degree}.
$$

Each iteration is a **damped Newton step** on this relaxed system $F_\mu$:

1. **Linearize.** The only nonlinear rows are the cone-complementarity rows.
   Symmetric cones are linearized in **Nesterov–Todd-scaled** coordinates: the
   scaling $W$ satisfies $Ws = W^{-\top}z = \lambda$, and the row becomes a
   linear relation between the scaled steps $W\Delta z$, $W^{-1}\Delta s$ and the
   scaled iterate $\lambda$. After eliminating $\Delta s$, the step reduces to
   one **symmetric quasidefinite KKT system** in $(\Delta x, \Delta z)$ (the
   $\Delta\tau, \Delta\kappa$ rows are resolved by a scalar back-substitution
   from two solves sharing the same factorization). This KKT solve is the
   per-iteration cost — see the next section. Asymmetric cones
   (exp/power/gen-power) carry no $W$; they linearize with the barrier Hessian
   $\nabla^2 F$ directly.
2. **Predictor–corrector (Mehrotra).** Solve once with the affine target
   $s \circ z = 0$ (predictor); from how much that step would shrink $\mu$, pick
   a centering parameter $\sigma \in [0, 1]$; solve again with target
   $s \circ z = \sigma\mu e$ plus a second-order correction (corrector). Both
   solves reuse the one factorization.
3. **Stay interior.** A fraction-to-boundary rule caps the step length so
   $s, z, x_J, z_x, \tau, \kappa$ stay strictly inside their cones.

Iterate until the relative residuals $r_x, r_z, r_\tau$ and the gap fall below
tolerance, then recover $(x, s, z)/\tau$.

## Direct conic constraints (new in Moreau)

Direct conic constraints also change the linearized Newton system. A slack
cone's scaling
reaches the $x$-block of the KKT system only through $A$ (as $A^\top H_s^{-1} A$);
the scaling for a direct conic constraint enters that block directly:

- **Linearizing its complementarity.** Each direct conic constraint carries the (relaxed)
  complementarity $x_J \circ z_x = \mu e$, where $x_J = E_J x$ is the constrained
  slice and $z_x$ its dual. This is linearized *exactly* like a slack cone's
  $s \circ z$: NT scaling for symmetric cones, the barrier Hessian for
  asymmetric ones. The linearized complementarity is
  $H_{s,x}\,\Delta x_J + \Delta z_x = -c$, i.e.
  $\Delta z_x = -H_{s,x}\,\Delta x_J - c$, with $H_{s,x}$ the cone's scaling
  Hessian and $c$ its affine/combined-step shift — the same shape as the slack
  cone's recovery $\Delta s = -H_s\,\Delta z - c$, with $x_J$ in the role of $s$
  and $z_x$ in the role of $z$.
- **Why it lands in the $(1,1)$ block.** Unlike a slack dual, $z_x$ appears in
  the embedding *only* in the stationarity row, as the $-E_J^\top z_x$ term, and
  $x_J$ is a slice of $x$ — there is no slack and no $A$-row carrying it. So
  eliminating $\Delta z_x$ with the relation above substitutes straight back into
  the $x$-row, adding $E_J^\top H_{s,x} E_J$ to the $(1,1)$ block (and
  $-E_J^\top c$ to its RHS) — directly, on the selected coordinates, no routing
  through $A$. (A slack dual $z$ instead enters via the constraint row
  $A\Delta x - H_s\,\Delta z = r$; eliminating it gives the Schur term
  $A^\top H_s^{-1} A$.) Dodging that $A$-detour — and the extra row/slack/cone-
  block a slack reformulation would need — is the payoff.
- **Which Hessian, and the free lunch.** The $(1,1)$ block needs the *dual-side*
  Hessian, $H_{s,x} = H_s^{-1}$. For self-dual cones (nonneg, SOC, PSD) the
  log-barrier is self-conjugate, so calling Clarabel's slack NT-scaling code with
  the primal/dual arguments swapped, $(s, z) \to (z, x)$, returns exactly that —
  $H_s \leftrightarrow H_s^{-1}$ — with no new scaling math. (Scalar check:
  nonneg slack stores $s/z$; fed $(z, x)$ it returns $z/x = H_s^{-1}$.) For
  exp/power/gen-power cones $F \ne F^*$, the swap is invalid; they carry explicit
  primal-barrier Hessian formulas instead.

The backward pass extends the same way: a direct pair enters the adjoint system
through $D\Pi_{\mathcal{K}_1^*}$, mirroring the slack derivation below.

## Asymmetric-cone scaling and correctors (beyond Clarabel)

Moreau inherits Clarabel's scaling unchanged for every **symmetric** cone —
nonneg, SOC, and PSD. (The PSD `update_scaling` is Clarabel's verbatim: Cholesky
factors $L_1, L_2$ of $S, Z$, an SVD of $L_2^\top L_1$, and the symmetric
Kronecker product $\operatorname{skron}(RR^\top)$ to assemble $H_s$.) The
**asymmetric** cones are where three pieces diverge.

### Primal–dual scaling for the generalized power cone

Upstream Clarabel scales the generalized power cone
$\mathcal{K} = \{(p, w) : \prod_i p_i^{\alpha_i} \ge \lVert w\rVert_2,\ p_i \ge 0\}$
with the **dual barrier Hessian only** — $H_s = \mu\,H_\text{dual}(z)$, a diagonal
block that satisfies *neither* Nesterov–Todd secant condition
(`allows_primal_dual_scaling` returns `false`). Moreau replaces it with a
**Mosek–Tunçel primal–dual scaling**: a low-rank correction of $\mu H_\text{dual}$
chosen so $H_s$ reproduces both secants,

$$
H_s\, z = s, \qquad H_s\, \delta z = \delta s,
$$

with $\delta s, \delta z$ the gradient/step differences along the centrality
direction. For cones of dimension $\le 64$ Moreau forms the full dense $H_s$
meeting both conditions; for larger cones it uses a **rank-6 sparse expansion**

$$
H_s = \mu H_\text{dual} + \sum_{k=0}^{5} \pm\, c_k\, a_k a_k^\top,
$$

four axes $a_k$ spanning the cone's rank-structured projector (from a
QR/eigendecomposition) and two enforcing the secants, with coefficients
$c_k \ge 0$ bounded so the $LDL^\top$ pivots cannot explode. If the construction
is numerically unsafe Moreau falls back cleanly to Clarabel's dual scaling
$H_s = \mu H_\text{dual}$.

One subtlety the primal–dual form forces: the secant orthogonality
$\langle \delta s, z\rangle = 0$ follows from log-homogeneity
($\langle \nabla F^*(z), z\rangle = -\nu$) **only** when the *local* duality
measure $\mu_\text{local} = \langle s, z\rangle / \nu$ is used. The IPM hands
`update_scaling` the embedding's $\mu = (\langle s, z\rangle + \tau\kappa)/(\nu+1)$,
off by roughly $1/(\nu+1)$ — enough to drive the secant residual to
$\sim 10^{-3}$ on dimension-100+ cones. Moreau recomputes $\mu_\text{local}$ for
the secant axes (the $\mu H$ filler keeps the IPM's $\mu$, since that term
annihilates $z$ regardless).

### A third-order corrector for the generalized power cone

Clarabel leaves the generalized power cone's higher-order Mehrotra term
**unimplemented** — with no primal–dual scaling there is nothing to correct.
Because Moreau scales it primal–dually, it also computes the matching
**third-order corrector**

$$
\eta = \tfrac12\, D^3 F^*(z)[u, v, \cdot], \qquad H_\text{dual}\, u = \Delta s, \quad v = \Delta z,
$$

the cubic Taylor remainder of the dual barrier along the affine step. The solve
$H_\text{dual}\, u = \Delta s$ is done in $O(\dim)$ by Sherman–Morrison on the
rank-structured dual Hessian rather than an $O(\dim^3)$ dense solve, so the
corrector is nearly free. Clarabel's 3D exponential and power cones already carry
this term; Moreau matches them and extends it to the $N$-dimensional cone.

### A primal-direct corrector for the direct asymmetric cones

The exponential and power cones in **direct** mode have no upstream analog —
Clarabel has no direct conic constraints at all. With $F \ne F^*$ the primal↔dual
self-duality used for symmetric cones is unavailable, so Moreau
scales these with the **dual-only primal barrier**
$H_{s,x} = \mu\,\nabla^2 F_\text{primal}(x)$ and supplies a **primal-direct**
third-order corrector

$$
\eta = \tfrac12\, D^3 F_\text{primal}(x)[\Delta x, \Delta x],
$$

both contraction slots taking the affine $x$-step (there is no separate slack
step to pair it with). The corrector is $\mu$-scaled and $\infty$-norm-capped
against the step to suppress early-iteration overshoot. For the power cone Moreau
also uses an exact **closed-form** primal-barrier Hessian here,
$F(s) = -\log(\varphi - s_2^2) - (1-\alpha)\log s_0 - \alpha\log s_1$ with
$\varphi = s_0^{2\alpha} s_1^{2(1-\alpha)}$, in place of the approximate
conjugate-dual primal Hessian (Karimi–Tunçel) Clarabel carries for the power
cone — whose $(2,2)$ and off-diagonal entries are only approximate and stall the
IPM once several direct power cones share the augmented $(1,1)$ block.

## KKT linear solve (inside the IPM)

After eliminating $\Delta s$ and the cone rows, each Newton step reduces to one
**symmetric quasidefinite** KKT system in $(\Delta x, \Delta z)$,

$$
\begin{bmatrix} P + D_x + \varepsilon I & A^\top \\ A & -(H_s + \varepsilon I) \end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta z \end{bmatrix} =
\begin{bmatrix} r_x \\ r_z \end{bmatrix},
$$

where $H_s \succ 0$ is the slack-cone scaling in the $(2,2)$ block,
$D_x = \sum_J E_J^\top H_{s,x} E_J$ is the direct augmentation of the $(1,1)$
block (from the previous section), and $\varepsilon$ is the static
regularization below. This solve is the dominant per-iteration cost. Moreau
selects a backend by problem structure; `direct_solve_method='auto'` chooses for
you.

### General sparse backends

These factor the system above as-is, by sparse $LDL^\top$ on an AMD-reordered
matrix.

- `faer` (CPU, default) — supernodal $LDL^\top$ (the external `faer` crate).
  `auto` runs it single-threaded: below ~20k KKT rows the parallel fan-out costs
  more than it saves.
- `qdldl` (CPU) — vendored simplicial $LDL^\top$; lighter-weight, no threading.
- `cudss` (GPU, default) — NVIDIA cuDSS batched sparse direct solve; the GPU
  fallback whenever no structured backend applies.

### Structure-exploiting backends (GPU)

When the problem has special structure, factoring the full sparse KKT is
wasteful. Two GPU backends exploit it instead, working on the Schur complement
$M = P + A^\top H_s^{-1} A$ of the $(2,2)$ block:

- `riccati` — for problems whose $M$ is **block-tridiagonal**, the signature of a
  time-stepped optimal-control problem (MPC / LQR / MHE: a horizon of stages,
  each coupled only to its neighbours through the dynamics). Detection is a pure
  host computation — it requires zero+nonneg cones only (so $H_s$ is diagonal),
  $n \ge 6$, and at least one zero cone (the dynamics); it builds the sparsity of
  $M = P + A^\top A$ and greedily searches for a block partition in which every
  column couples only to adjacent blocks. If the variables are **not** already in
  stage order that search yields no usable banding — so a bandwidth-reducing
  **Reverse Cuthill–McKee** reorder is tried and recovers the band for any column
  order that admits one. The solve then runs in the permuted column space, with
  the RHS reads and solution writes gathering/scattering through the reorder at
  the boundaries while the inner loop is untouched; truly dense coupling rows,
  which no reorder can band, fall back to `cudss`. The backward pass mirrors the
  same reorder — its adjoint block assembly is built in the identical band order —
  so differentiation works through a scrambled problem too. The solve itself is a
  **block-tridiagonal Cholesky
  (Riccati recursion)**: a sequential sweep over the $T$ blocks, each a dense
  $O(d^3)$ factor/solve, batched across problems with cuBLAS/cuSOLVER. This is
  $O(T d^3)$ rather than the $O((T d)^3)$ of treating $M$ as dense.
- `woodbury` — for **diagonal $P$ plus low-rank $A$** (portfolio / factor-model
  QPs: a diagonal risk term, a handful of dense factor-exposure rows, and many
  one-entry bound rows). It applies the **Sherman–Morrison–Woodbury identity** to
  collapse the $(n+m)$ system to a small $k_\text{total} \times k_\text{total}$
  Schur complement $S = C^{-1} + F^\top D^{-1} F$, where $D$ is diagonal (the
  diagonal $P$, the static $\varepsilon$, and the rank-1 contributions of the
  sparse bound rows — all inverted elementwise) and $F$ holds only the
  $k_\text{total}$ dense / equality coupling rows. Factoring $S$ by Cholesky costs
  $O(k_\text{total}^3)$ with $k_\text{total} \ll n$. It is gated to zero+nonneg
  cones with a strictly diagonal $P$ and $k_\text{total} < n$, and requires every
  column to be "covered" by a diagonal-$P$, sparse-bound, or direct entry —
  the conditioning guard on the inverted blocks, spelled out under
  [Conditioning](#conditioning) below. GPU-only, opt-in (never auto-selected).

These are linear-algebra strategies for the *same* KKT system, not different
optimization algorithms.

### Conditioning

The KKT matrix is indefinite and, near the cone boundary, badly scaled. Moreau
inherits Clarabel's [1] conditioning stack:

- **Equilibration (Ruiz).** Before the solve, $P, A, q, b$ are symmetrically
  rescaled by diagonals $d, e$ — up to 10 Ruiz sweeps equalizing the row/column
  $\infty$-norms, with the cumulative scaling clipped to $[10^{-4}, 10^{4}]$ —
  which sharply cuts the condition number. The rescaling is **cone-aware**: a cone
  that needs one shared scale across a slice (such as SOC or PSD) gets
  the geometric mean of $d$ over that slice, so the scaling commutes with cone
  membership and the unscaled iterate stays feasible.
- **Static regularization + quasidefiniteness.** The reduced KKT is symmetric
  **quasidefinite** — $(1,1)$ block positive, $(2,2)$ block negative — which
  admits a sign-stable $LDL^\top$ with *no pivoting*, on a fixed sign pattern
  ($+1$ on the first $n$ pivots, $-1$ on the rest). A static shift
  $\varepsilon = \varepsilon_\text{abs} + \varepsilon_\text{rel}\,\lVert\operatorname{diag}K\rVert_\infty$
  (defaults $10^{-8}$ and $\epsilon_\text{mach}^2$) is added *with the block sign*
  — $+\varepsilon$ on the $(1,1)$ diagonal, $-\varepsilon$ on the $(2,2)$ —
  guaranteeing a nonzero pivot everywhere.
- **Bounded conditioning of the reduced systems.** The structure-exploiting
  backends invert their *own* reduced blocks — Woodbury's diagonal $D$ and the
  small Schur complement $S$, Riccati's per-stage blocks — and cap each block's
  condition number by construction. The static $\varepsilon$ floors every
  diagonal of $D$ and $S$ from below; on top of that, Woodbury runs only when
  **every** column is "covered" (by a diagonal-$P$, sparse-bound, or direct
  entry), which holds each $D_j$ at its true scale instead of collapsing to
  $\varepsilon$. An uncovered column would give $D_j \approx \varepsilon$, a
  $\sim 1/\varepsilon$ entry in $D^{-1}$, and a Schur complement poisoned by
  catastrophic cancellation — so rather than invert it, Woodbury **declines and
  falls back to cuDSS**. The cap is a structural gate: Moreau refuses the
  ill-conditioned reduction, it does not estimate $\kappa$ at runtime.
- **Dynamic regularization.** As a backstop inside the factorization, any pivot
  still falling below $10^{-13}$ in magnitude is bumped to $\pm 2\times10^{-7}$
  (sign preserved), so a near-singular iterate cannot produce a NaN.
- **Iterative refinement.** Both regularizations perturb the system, so the
  result is refined against the *unregularized* $K$: form the residual
  $e = b - K\xi$, solve $K\,\delta = e$ with the factorization already in hand,
  accept $\xi + \delta$, and repeat (up to 10 times) while the residual keeps
  dropping by at least $5\times$. This buys back the accuracy the regularization
  traded for stability.

## Differentiation (backward pass)

- **Exact.** Implicit differentiation of the embedding's KKT conditions; the key
  blocks are the dual-cone projection derivatives $D\Pi_{\mathcal{K}^*}(\cdot)$
  for both $\mathcal{K}_1$ (on $x$) and $\mathcal{K}_2$ (on $s$). This is
  **diffqcp** [4] (the QP-cone case of cone-program differentiation [3]). Exact
  but non-smooth at cone boundaries.
- **Smoothed.** Replaces the projection Jacobian with its central-path
  (proximal, parameter $\mu$) smoothing for a continuous derivative. See the
  [smoothed differentiation](smoothed-differentiation.md) guide.

## References

1. Goulart & Chen. *Clarabel: An interior-point solver for conic programs with quadratic objectives.* arXiv:2405.12762, 2024.
2. Chen, Tse, Nobel, Goulart & Boyd. *CuClarabel: GPU Acceleration for a Conic Optimization Solver.* ACM TOMS (to appear). doi:10.1145/3815420.
3. Agrawal, Barratt, Boyd, Busseti & Moursi. *Differentiating through a Cone Program.* J. Appl. Numer. Optim., 1(2):107–115, 2019.
4. Healey, Nobel & Boyd. *Differentiating Through a Quadratic Cone Program.* Optimization Letters (to appear). arXiv:2508.17522, 2025.
5. Arnström, Bemporad & Axehill. *A Dual Active-Set Solver for Embedded Quadratic Programming Using Recursive LDLᵀ Updates.* IEEE TAC, 67(8):4362–4369, 2022. arXiv:2103.16236.
