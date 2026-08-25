/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for solver operations
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

/**
 * @brief Update solver information metrics
 *
 * Computes convergence metrics including:
 * - Primal/dual objective values
 * - Primal/dual residual norms
 * - Infeasibility measures
 * - Duality gaps
 * - κ/τ ratio
 *
 * @param cost_primal Output: primal objective [batchSize]
 * @param cost_dual Output: dual objective [batchSize]
 * @param res_primal Output: primal residual norm [batchSize]
 * @param res_dual Output: dual residual norm [batchSize]
 * @param res_primal_inf Output: primal infeasibility [batchSize]
 * @param res_dual_inf Output: dual infeasibility [batchSize]
 * @param gap_abs Output: absolute duality gap [batchSize]
 * @param gap_rel Output: relative duality gap [batchSize]
 * @param ktratio Output: κ/τ ratio [batchSize]
 * @param rx Primal residual vector [n * batchSize]
 * @param rz Dual residual vector [m * batchSize]
 * @param rtau Gap residual [batchSize]
 * @param dot_qx q'x [batchSize]
 * @param dot_bz b'z [batchSize]
 * @param tau τ values [batchSize]
 * @param kappa κ values [batchSize]
 * @param dot_xPx x'Px [batchSize]
 * @param rx_inf Infeasibility primal residual [n * batchSize]
 * @param rz_inf Infeasibility dual residual [m * batchSize]
 * @param Px Product P*x [n * batchSize]
 * @param x Primal variables [n * batchSize]
 * @param z Dual variables [m * batchSize]
 * @param s Slack variables [m * batchSize]
 * @param b Constraint RHS [m * batchSize]
 * @param q Cost vector [n * batchSize]
 * @param d Equilibration scaling [n * batchSize]
 * @param dinv Inverse equilibration scaling [n * batchSize]
 * @param e Equilibration scaling [m * batchSize]
 * @param einv Inverse equilibration scaling [m * batchSize]
 * @param c Cost scaling [batchSize]
 * @param n Number of primal variables
 * @param m Number of constraints
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void update_info_kernel(
    double* cost_primal,
    double* cost_dual,
    double* res_primal,
    double* res_dual,
    double* res_primal_inf,
    double* res_dual_inf,
    double* gap_abs,
    double* gap_rel,
    double* ktratio,
    const double* rx,
    const double* rz,
    const double* rtau,
    const double* dot_qx,
    const double* dot_bz,
    const double* tau,
    const double* kappa,
    const double* dot_xPx,
    const double* rx_inf,
    const double* rz_inf,
    const double* Px,
    const double* x,
    const double* z,
    const double* s,
    const double* z_x,                 // [totalXConeNumel * batchSize] or nullptr
    const int64_t* d_xcone_indices,    // [totalXConeNumel] or nullptr
    int64_t totalXConeNumel,           // 0 when no direct-x cones
    const double* normb_cached,
    const double* normq_cached,
    const double* d,
    const double* dinv,
    const double* e,
    const double* einv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute centering parameter: σ = (1 - α)³
 *
 * @param sigma Output: centering parameter [batchSize]
 * @param alpha Input: step length [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void calc_centering_parameter_kernel(
    double* sigma,
    const double* alpha,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute Mehrotra correction factor: m = (iter > 1) ? 1.0 : alpha
 *
 * @param m Output: Mehrotra correction factor [batchSize]
 * @param alpha Input: affine step length [batchSize]
 * @param iter Current iteration number
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void calc_mehrotra_correction_kernel(
    double* m,
    const double* alpha,
    int64_t iter,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief FUSED kernel: Compute centering parameter and Mehrotra correction (2→1)
 *
 * Computes both sigma = (1 - α)³ and m = (iter > 1) ? 1.0 : alpha in single kernel.
 *
 * @param sigma Output: centering parameter [batchSize]
 * @param m_out Output: Mehrotra correction factor [batchSize]
 * @param alpha Input: affine step length [batchSize]
 * @param iter Current iteration number
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void calc_solver_parameters_kernel(
    double* sigma,
    double* m_out,
    const double* alpha,
    int64_t iter,
    int64_t batchSize,
    cudaStream_t stream = 0
);

void compute_cached_norms_kernel(
    double* normb,
    double* normq,
    const double* b,
    const double* q,
    const double* einv,
    const double* dinv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Check if any batch has insufficient progress status
 *
 * Returns 1 if any batch has status == InsufficientProgress (10), 0 otherwise.
 *
 * @param status Device status array [batchSize]
 * @param result Output: 1 if any batch has insufficient progress, 0 otherwise [1]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void check_any_insufficient_progress_kernel(
    const int32_t* status,
    int32_t* result,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Project vectors onto nonnegative orthant: v = max(v, eps)
 *
 * @param v Input/output vector [n * batchSize]
 * @param eps Minimum value (for staying strictly in interior)
 * @param n Dimension of vector
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void project_nonneg_kernel(
    double* v,
    double eps,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief GPU-based backtracking line search for all batches in parallel
 *
 * Performs backtracking for each batch independently on the GPU.
 * Each batch computes its barrier function and reduces alpha until barrier < 1.0.
 *
 * @param alpha Output: final step length per batch [batchSize]
 * @param alpha_init Input: initial step length per batch [batchSize]
 * @param barrier_work Workspace for barrier computation [batchSize]
 * @param s Current slack variables [m * batchSize]
 * @param z Current dual variables [m * batchSize]
 * @param tau Current tau [batchSize]
 * @param kappa Current kappa [batchSize]
 * @param ds Step slack variables [m * batchSize]
 * @param dz Step dual variables [m * batchSize]
 * @param dtau Step tau [batchSize]
 * @param dkappa Step kappa [batchSize]
 * @param sz_dot Workspace for s'z dot product [batchSize]
 * @param backtrack_factor Backtracking reduction factor (e.g., 0.99)
 * @param degree Cone degree (m)
 * @param m Number of constraints
 * @param batchSize Number of batches
 * @param max_iters Maximum backtracking iterations (e.g., 50)
 * @param stream CUDA stream
 */
void backtrack_line_search_kernel(
    double* alpha,
    const double* alpha_init,
    double* barrier_work,
    const double* s,
    const double* z,
    const double* tau,
    const double* kappa,
    const double* ds,
    const double* dz,
    const double* dtau,
    const double* dkappa,
    double* sz_dot,
    double backtrack_factor,
    int64_t degree,
    int64_t m,
    int64_t batchSize,
    int max_iters,
    int64_t numZeroCones = 0,
    int64_t numNonnegCones = 0,
    int64_t numSocCones = 0,
    const int64_t* socConeDims = nullptr,
    cudaStream_t stream = 0
);

/**
 * @brief Save solution for terminated batches
 *
 * For each batch where status[b] != 0 (terminated), copies the current
 * variables (x, s, z, τ, κ) to the solution structure if not already saved.
 *
 * @param x_src Source primal variables [n * batchSize]
 * @param s_src Source slack variables [m * batchSize]
 * @param z_src Source dual variables [m * batchSize]
 * @param tau_src Source τ [batchSize]
 * @param kappa_src Source κ [batchSize]
 * @param x_dst Destination primal variables [n * batchSize]
 * @param s_dst Destination slack variables [m * batchSize]
 * @param z_dst Destination dual variables [m * batchSize]
 * @param solution_saved Flag: 1 if batch solution already saved [batchSize]
 * @param status Solver status [batchSize]
 * @param n Number of primal variables
 * @param m Number of constraints
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void save_terminated_solutions_kernel(
    const double* x_src,
    const double* s_src,
    const double* z_src,
    const double* tau_src,
    const double* kappa_src,
    const double* cost_primal_src,
    const double* cost_dual_src,
    double* x_dst,
    double* s_dst,
    double* z_dst,
    double* tau_dst,
    double* kappa_dst,
    double* cost_primal_dst,
    double* cost_dual_dst,
    int32_t* solution_saved,
    int32_t* should_save,
    const int32_t* status,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Restore working variables for terminated batches from saved solution.
 *
 * Prevents converged batches from accumulating drift in s/z that would
 * corrupt the UBATCH KKT factorization for still-active batches.
 */
void restore_terminated_variables_kernel(
    double* x_work,
    double* s_work,
    double* z_work,
    double* tau_work,
    double* kappa_work,
    const double* x_saved,
    const double* s_saved,
    const double* z_saved,
    const double* tau_saved,
    const double* kappa_saved,
    const int32_t* solution_saved,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Snapshot the current iterate as the "best seen so far" when a batch
 * is still running (status==0) and its metrics satisfy the reduced tolerances.
 *
 * Writes current variables into the solution.*_raw buffers and current metrics
 * into the info.best_* buffers (latest-wins). Sets solution_saved[b]=1 so that
 * save_terminated_solutions_kernel's atomicCAS will not later overwrite the
 * snapshot with a degraded terminal iterate. Sets d_best_saved[b]=1 so the
 * post-loop restore can distinguish an in-zone snapshot from a certificate-
 * of-infeasibility snapshot.
 */
void save_best_iterate_kernel(
    const double* x_src,
    const double* s_src,
    const double* z_src,
    const double* tau_src,
    const double* kappa_src,
    const double* cost_primal_src,
    const double* cost_dual_src,
    const double* res_primal_src,
    const double* res_dual_src,
    const double* gap_abs_src,
    const double* gap_rel_src,
    const double* ktratio_src,
    double* x_dst,
    double* s_dst,
    double* z_dst,
    double* tau_dst,
    double* kappa_dst,
    double* cost_primal_dst,
    double* cost_dual_dst,
    double* best_res_primal,
    double* best_res_dual,
    double* best_gap_abs,
    double* best_gap_rel,
    double* best_ktratio,
    int32_t* solution_saved,
    int32_t* should_save,
    int32_t* d_best_saved,
    const int32_t* status,
    double tolGapAbs,
    double tolFeas,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolKtRatio,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Promote per-batch status to AlmostSolved for non-convergent
 * terminations that have a best-iterate snapshot, and restore info metrics
 * from the snapshot. Device buffers are updated in-place; the host-side
 * status vector is NOT touched here — call sync_status_to_host afterwards.
 */
void restore_best_iterate_kernel(
    double* res_primal,
    double* res_dual,
    double* gap_abs,
    double* gap_rel,
    double* ktratio,
    double* cost_primal,
    double* cost_dual,
    const double* best_res_primal,
    const double* best_res_dual,
    const double* best_gap_abs,
    const double* best_gap_rel,
    const double* best_ktratio,
    const double* best_cost_primal,
    const double* best_cost_dual,
    int32_t* status,
    const int32_t* d_best_saved,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Pack separate x and z arrays into interleaved format for cuDSS
 *
 * Transforms from layout:
 *   x: [batch0_x(n), batch1_x(n), ...]
 *   z: [batch0_z(m), batch1_z(m), ...]
 * To interleaved layout:
 *   out: [batch0_x(n), batch0_z(m), batch0_zeros(p), batch1_x(n), ...]
 *
 * @param out Output interleaved array [(n+m+p) * batchSize]
 * @param x Input x array [n * batchSize]
 * @param z Input z array [m * batchSize]
 * @param n Dimension of x per batch
 * @param m Dimension of z per batch
 * @param p Extra expansion dimension (0 if no sparse SOC)
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void pack_interleaved_kernel(
    double* out,
    const double* x,
    const double* z,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused waxpby + pack interleaved: x copied, z = a*z_a + b*z_b
 *
 * Combines a linear combination on the z-part with interleaved packing
 * into a single kernel, eliminating the separate waxpby kernel launch.
 */
void waxpby_and_pack_interleaved_kernel(
    double* out,
    const double* x_src,
    const double* z_a,
    const double* z_b,
    double a,
    double b,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Unpack interleaved format into separate x and z arrays
 *
 * Transforms from interleaved layout:
 *   in: [batch0_x(n), batch0_z(m), batch0_exp(p), batch1_x(n), ...]
 * To separate layout:
 *   x: [batch0_x(n), batch1_x(n), ...]
 *   z: [batch0_z(m), batch1_z(m), ...]
 * (expansion entries are discarded)
 *
 * @param x Output x array [n * batchSize]
 * @param z Output z array [m * batchSize]
 * @param in Input interleaved array [(n+m+p) * batchSize]
 * @param n Dimension of x per batch
 * @param m Dimension of z per batch
 * @param p Extra expansion dimension (0 if no sparse SOC)
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void unpack_interleaved_kernel(
    double* x,
    double* z,
    const double* in,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Check termination without synchronization (for pinned memory)
 *
 * Same as check_termination_kernel but writes to pinned memory for zero-copy access.
 * Does NOT perform cudaStreamSynchronize - caller must use pinned memory polling.
 *
 * @param h_any_done Host-mapped pinned memory pointer for termination count
 * @param ... (same as check_termination_kernel)
 */
void check_termination_async_kernel(
    int32_t* status,
    int32_t* d_any_done,           // Device pointer (may be mapped to h_any_done)
    int32_t* h_any_done,           // Host pointer to poll (pinned, mapped)
    int32_t* d_iterations_per_batch,
    const double* gap_abs,
    const double* gap_rel,
    const double* res_primal,
    const double* res_dual,
    const double* ktratio,
    const double* res_primal_inf,
    const double* res_dual_inf,
    const double* dot_qx,
    const double* dot_bz,
    const double* prev_res_primal,
    const double* prev_res_dual,
    const double* prev_gap_abs,
    const double* prev_gap_rel,
    double tolGapAbs,
    double tolGapRel,
    double tolFeas,
    double tolInfeasAbs,
    double tolInfeasRel,
    double tolKtRatio,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolInfeasAbs,
    double reducedTolInfeasRel,
    double reducedTolKtRatio,
    int64_t iter,
    int64_t maxIter,
    double solve_time_seconds,
    double timeLimit,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Set all batch statuses to a given value
 *
 * Used to mark all batches with NumericalError when KKT factorization fails.
 *
 * @param status Output: solver status per batch [batchSize]
 * @param value Status value to set (cast from SolverStatus enum)
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void set_all_status_kernel(
    int32_t* status,
    int32_t value,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Set status only for batches whose solution has NOT been saved.
 *
 * Already-converged batches (solution_saved[b] == 1) keep their status.
 */
void set_unsaved_status_kernel(
    int32_t* status,
    const int32_t* solution_saved,
    int32_t value,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Set status for batches where scaling failed AND solution not yet saved.
 *
 * Only sets status for batch b where solution_saved[b] == 0 AND
 * scaling_success[b] == 0.  scaling_success points to mapped pinned memory.
 */
void set_unsaved_status_kernel(
    int32_t* status,
    const int32_t* solution_saved,
    int32_t value,
    int64_t batchSize,
    const volatile int32_t* scaling_success,
    cudaStream_t stream = 0
);

/**
 * @brief Zero step length alpha for already-terminated batch elements.
 *
 * Prevents variable drift for converged elements, which otherwise could corrupt
 * cone scaling or KKT factorization for still-running elements in the batch.
 *
 * @param alpha Step length per batch [batchSize] — zeroed where status != 0
 * @param status Solver status per batch [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void zero_alpha_for_terminated_kernel(
    double* alpha,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/// Per-batch PrimalDual→Dual fallback. CPU mirror:
/// `core/solver.rs::strategy_checkpoint_small_step`.
void switch_scaling_on_small_step_kernel(
    double* alpha,
    int8_t* pd_enabled_per_batch,
    int64_t batchSize,
    double min_switch_step_length,
    cudaStream_t stream = 0
);

/// Set all batchSize bytes of `pd_enabled_per_batch` to `value`.
void init_pd_enabled_per_batch_kernel(
    int8_t* pd_enabled_per_batch,
    int64_t batchSize,
    int8_t value,
    cudaStream_t stream = 0
);

/**
 * @brief Set status to InsufficientProgress for batches whose combined-step
 *        length collapsed below `minTerminateStepLength`. Mirrors the CPU
 *        `strategy_checkpoint_small_step` mechanism: when cone backtracking
 *        forces α toward 0, the search direction is exhausted and the IPM
 *        cannot make further progress. Skip already-terminated batches.
 *
 *        Downstream post_process promotes InsufficientProgress to AlmostSolved
 *        if the saved/current iterate meets reduced tolerances, matching CPU
 *        behavior on near-boundary problems.
 *
 * @param alpha Combined step length per batch [batchSize]
 * @param status Solver status per batch [batchSize] (modified in place)
 * @param iterations_per_batch Iteration count per batch (written on terminate)
 * @param minTerminateStepLength Threshold; batches with alpha ≤ this value terminate
 * @param iter Current iteration (recorded on terminate)
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void small_step_terminate_kernel(
    const double* alpha,
    int32_t* status,
    int32_t* iterations_per_batch,
    double minTerminateStepLength,
    int64_t iter,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief YOLO mode: per-batch snapshot variables to raw buffers if no NaN detected.
 *
 * For each batch independently, checks x, s, z, τ, κ for NaN. If all valid
 * for that batch, copies them to the destination (raw) buffers. If any NaN,
 * skips the copy for that batch, preserving its previous valid snapshot.
 * Zero host sync — all on-device.
 *
 * @param x Source primal variables [n * batchSize]
 * @param s Source slack variables [m * batchSize]
 * @param z Source dual variables [m * batchSize]
 * @param tau Source τ [batchSize]
 * @param kappa Source κ [batchSize]
 * @param x_dst Destination raw primal [n * batchSize]
 * @param s_dst Destination raw slack [m * batchSize]
 * @param z_dst Destination raw dual [m * batchSize]
 * @param tau_dst Destination raw τ [batchSize]
 * @param kappa_dst Destination raw κ [batchSize]
 * @param d_has_nan Per-batch NaN flags [batchSize int32_t], preallocated
 * @param n Number of primal variables
 * @param m Number of constraints
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void yolo_snapshot_if_valid_kernel(
    const double* x, const double* s, const double* z,
    const double* tau, const double* kappa,
    const double* z_x,
    double* x_dst, double* s_dst, double* z_dst,
    double* tau_dst, double* kappa_dst,
    double* z_x_dst,
    int32_t* d_has_nan,
    int64_t n, int64_t m, int64_t total_xn, int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute per-batch shift alpha for cone initialization
 *
 * For each batch, computes the shift amount needed to move variables into cone interior:
 *   target = max(1.0, pos_margin * 0.1 / degree)
 *   if min_margin <= 0: alpha = -min_margin + target
 *   elif min_margin < target: alpha = target - min_margin
 *   else: alpha = 0
 *
 * @param alpha Output: per-batch shift amounts [batchSize]
 * @param min_margin Per-batch minimum margins [batchSize]
 * @param pos_margin Per-batch positive margin sums [batchSize]
 * @param degree Total cone degree (number of cones)
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void compute_init_shift_alpha_kernel(
    double* alpha,
    const double* min_margin,
    const double* pos_margin,
    double degree,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute warmness parameter mu from residuals and gap measures
 *
 * mu[i] = max(res_primal[i], res_dual[i], min(gap_abs[i], gap_rel[i]))
 * with a floor of 1e-6.
 *
 * @param mu_out Output: per-batch warmness mu [batchSize]
 * @param res_primal Primal residual norm [batchSize]
 * @param res_dual Dual residual norm [batchSize]
 * @param gap_abs Absolute duality gap [batchSize]
 * @param gap_rel Relative duality gap [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void compute_warmness_mu(
    double* mu_out,
    const double* res_primal,
    const double* res_dual,
    const double* gap_abs,
    const double* gap_rel,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused tau numerator base: ξ=x/τ, tau_num = (rτ - rκ/τ) + q'x1 + b'z1 (3 → 1 kernel)
 *
 * Replaces: div_per_batch(ξ) + sub_div_scalar(tau_num) + multi_dot_add_2(tau_num)
 */
void fused_tau_numerator_base_kernel(
    double* xi_out,
    double* tau_num,
    const double* x_vec,
    const double* tau_vec,
    const double* rtau,
    const double* rkappa,
    const double* q,
    const double* x1,
    const double* b_vec,
    const double* z1,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused tau denominator base: tau_den = κ/τ - q'x2 - b'z2 (2 → 1 kernel)
 *
 * Replaces: elementwise_div(tau_den) + multi_dot_add_2(tau_den)
 */
void fused_tau_denominator_base_kernel(
    double* tau_den,
    const double* kappa,
    const double* tau_vec,
    const double* q,
    const double* x2,
    const double* b_vec,
    const double* z2,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused double quad form: tau_den += (ξ-x2)'P(ξ-x2) - x2'Px2 (4 → 1 kernel)
 *
 * Replaces: waxpby(ξ_minus_x2) + quad_form(quad1) + quad_form(quad2) + axpby2_scalar(tau_den)
 * Eliminates ξ_minus_x2 temporary vector.
 */
void fused_double_quad_form_kernel(
    double* tau_den,
    const int64_t* P_rowOffsets,
    const int64_t* P_colIndices,
    const double* P_values,
    const double* xi_vec,
    const double* x2_vec,
    int64_t n,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Sanitize inf entries in b and detect infeasibility.
 *
 * For nonneg cone rows where b[i] = +inf: zeros the A row, sets b = 1.
 * For nonneg rows where b[i] = -inf: sets infeasible_flags[batch] = 1.
 * For zero cone rows where b[i] = ±inf: sets infeasible_flags[batch] = 1.
 *
 * @param b          Constraint RHS vector [batchSize * m] (modified in-place)
 * @param A_values   CSR values for A matrix [batchSize * nnzA] (modified in-place)
 * @param A_rowOffsets CSR row offsets for A [m+1] (shared across batch)
 * @param infeasible_flags Per-batch flag [batchSize], set to 1 if infeasible (must be zeroed before call)
 * @param zero_row_start  First zero cone row index
 * @param zero_row_end    One past last zero cone row index
 * @param nonneg_row_start First nonneg cone row index
 * @param nonneg_row_end   One past last nonneg cone row index
 * @param m          Number of constraints
 * @param nnzA       Number of nonzeros in A
 * @param batchSize  Number of problems in batch
 */
void sanitize_inf_b_kernel(
    double* b,
    double* A_values,
    const int64_t* A_rowOffsets,
    int* infeasible_flags,
    int64_t zero_row_start,
    int64_t zero_row_end,
    int64_t nonneg_row_start,
    int64_t nonneg_row_end,
    int64_t m,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream = 0
);

} // namespace moreau
