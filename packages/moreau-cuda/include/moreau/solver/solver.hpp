/**
 * @file solver.hpp
 * @brief Main solver for standard-form conic optimization
 *
 * This module defines two solver classes:
 * - CompiledSolver: Three-step API (construct, setup, solve) for batched/repeated solves
 * - Solver: Single-problem solver that takes all data at construction
 *
 * CompiledSolver is the core implementation that orchestrates equilibration,
 * KKT system construction, and the interior-point method iterations.
 */

#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <memory>
#include <mutex>
#include "moreau/vector/vector.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/equilibration/equilibration.hpp"
#include "moreau/variables/variables.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/status.hpp"

// Include these after status is defined
#include "moreau/solution/solution.hpp"
#include "moreau/solver/data.hpp"
#include "moreau/residuals/residuals.hpp"
#include "moreau/kkt/kkt.hpp"
#include "moreau/kkt/kkt_solver.hpp"
#include "moreau/solver/info.hpp"
#include "moreau/diff/diff_kkt.hpp"
#include "moreau/diff/diff_woodbury.hpp"
#include "moreau/diff/diff.hpp"

namespace moreau {

/**
 * @brief Scaling strategy for cone scaling
 */
enum class ScalingStrategy {
    PrimalDual,
    Dual
};

/**
 * @brief Compiled conic optimization solver with three-step API
 *
 * Three-step API pattern:
 * 1. Construct with structure (dimensions, sparsity patterns, cones)
 * 2. setup(P_values, A_values) - Set matrix values (stored on GPU)
 * 3. solve(q, b) - Solve with linear cost and constraint RHS
 *
 * Solves problems of the form:
 *   minimize    (1/2)x'Px + q'x
 *   subject to  Ax + s = b
 *               x in K1,  s in K2
 * K2 constrains the slack s; K1 constrains x directly (direct-x cones).
 * Each is a product of cones.
 */
struct CompiledSolver {
    // Device ID for multi-GPU support (-1 = use current device)
    // NOTE: Must be first member to ensure cudaSetDevice is called before other members initialize
    int device_id_;

    // Problem data
    SolverData data;

    // GPU library handles (must be cleaned up in destructor)
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;

    // Solver components
    std::unique_ptr<KKTSolver> kkt;
    Residuals residuals;

    // Primal-dual variables
    Variables variables;

    // Step direction variables
    Variables step_rhs;  // Right-hand side for step direction
    Variables step_lhs;  // Left-hand side (solution) for step direction

    // Work vectors for step computation
    BatchedVector x1;      // workspace for (x1, z1) solve
    BatchedVector z1;      // workspace for (x1, z1) solve
    BatchedVector Δs_const_term;  // workspace for Δs constant term (size m)

    // Scalar workspaces for τ computations (1 per batch)
    BatchedVector tau_num_work;    // workspace for τ numerator components
    BatchedVector tau_den_work;    // workspace for τ denominator components
    BatchedVector ξ;               // workspace: x/τ (size n)

    // Settings (status lives on solution.status as int32_t; this field used to be
    // a redundant BatchedVector of doubles but would receive only the low 4 bytes
    // of the int32 status, producing garbage when read as double. Removed.)
    Settings settings;

    // Solution from last solve
    Solution solution;

    // Solver information and statistics
    Info info;

    // other misc
    BatchedVector sigma;
    BatchedVector alpha;
    BatchedVector mu;
    BatchedVector m;  // Mehrotra correction factor

    // Work vectors for KKT solve
    BatchedVector workx;   // workspace for RHS (n)
    BatchedVector workz;   // workspace for RHS (m)
    BatchedVector x2;      // workspace for solution (n)
    BatchedVector z2;      // workspace for solution (m)

    // Work vectors for combined step
    BatchedVector work_m_1;       // workspace for cone operations (m)
    BatchedVector work_m_2;       // workspace for cone operations (m)
    BatchedVector alpha_combined; // step length for combined step (1)

    // Work vectors for solveKKT
    BatchedVector kkt_rhs;        // KKT RHS vector (n+m+p)
    BatchedVector kkt_sol;        // KKT solution vector (n+m+p)
    BatchedVector kkt_rhs2;       // KKT RHS for 2-RHS batched solve (n+m+p)*2
    BatchedVector kkt_sol2;       // KKT solution for 2-RHS batched solve (n+m+p)*2
    BatchedVector inv_tau;        // 1/τ for scaling (1)
    BatchedVector temp_scalar;    // temporary scalar workspace (1)
    BatchedVector q_dot_x1;       // q'x1 dot product (1)
    BatchedVector b_dot_z1;       // b'z1 dot product (1)
    BatchedVector ξ_P_x1;         // ξ'P*x1 quadratic form (1)
    BatchedVector q_dot_x2;       // q'x2 dot product (1)
    BatchedVector b_dot_z2;       // b'z2 dot product (1)
    BatchedVector x2_scaled;      // τ * x2 (n)
    BatchedVector z2_scaled;      // τ * z2 (m)

    // Work vectors for getStepLength
    BatchedVector step_work1;     // step length workspace (1)
    BatchedVector step_work2;     // step length workspace (1)
    BatchedVector alpha_init;     // initial step length for backtracking (1)

    // Work vectors for warm start
    BatchedVector warm_work;      // workspace for warm start: work = z - s (m)

    // Work vectors for default_start
    BatchedVector default_rhs;    // default start RHS vector (n+m+p)
    BatchedVector default_sol;    // default start solution vector (n+m+p)
    BatchedVector neg_q;          // -q vector (n)
    BatchedVector workz_zero;     // workspace for LP init zero vector (m)
    BatchedVector init_min_margin;   // per-batch minimum margin for initialization (1)
    BatchedVector init_pos_margin;   // per-batch positive margin sum for initialization (1)
    BatchedVector init_shift_alpha;  // per-batch shift amount for initialization (1)

    // Work vectors for refineSmoothingIterate centering loop
    BatchedVector smoothing_mu_step;  // per-batch mu step target (1)
    BatchedVector smoothing_m_zero;   // per-batch zero vector (1)

    // Work vectors for quad_form_batched (to avoid allocation in hot path)
    BatchedVector quad_form_temp;  // temporary for P*y computation (n)
    void* quad_form_buffer;        // cuSPARSE SpMV buffer
    size_t quad_form_buffer_size;  // size of cuSPARSE buffer
    void* quad_form_aligned_y;     // aligned buffer for y (for odd n, multi-batch)
    void* quad_form_aligned_temp;  // aligned buffer for temp (for odd n, multi-batch)
    size_t quad_form_aligned_size; // size of aligned buffers

    // YOLO mode: preallocated per-batch NaN flags for NaN-gated snapshots
    // Size batchSize when yolo=true, empty otherwise. Avoids cudaMalloc in solve loop.
    device_unique_ptr<int32_t> d_yolo_has_nan_;

    // YOLO mode: last-NaN-free snapshot of the direct-x dual `z_x`. The
    // x/s/z/τ/κ snapshot lands in `solution.*_raw`, but z_x has no raw
    // buffer there — this holds it. Sized (totalXConeNumel, batchSize)
    // when yolo=true and direct-x cones are present; (1,1) placeholder
    // otherwise.
    BatchedVector yolo_zx_raw_;

    // Per-batch PrimalDual gate for direct-x GenPower (1=PrimalDual, 0=Dual).
    // Mirror of CPU `core/solver.rs::strategy_checkpoint_small_step`. Allocated
    // lazily when the problem contains direct-x GenPower; otherwise null.
    device_unique_ptr<int8_t> pd_enabled_per_batch_;

    // DiffKKT for HSDE backward differentiation (lazily initialized)
    std::unique_ptr<DiffKKT> diff_kkt_;

    // DiffWoodbury for Woodbury-specialized backward pass (lazily initialized)
    std::unique_ptr<DiffWoodbury> diff_woodbury_;

    // Preallocated workspace for backward pass (lazily initialized)
    // Avoids cudaMalloc calls during backward differentiation
    std::unique_ptr<BackwardWorkspace> backward_workspace_;

    // DiffState for backward differentiation (created when settings.enableGrad=true)
    // Owned by the solver so caching happens automatically after each solve().
    std::unique_ptr<DiffState> diff_state_;

    /**
     * @brief Access the DiffState for backward differentiation
     * @return Pointer to DiffState, or nullptr if enableGrad is false
     */
    DiffState* diff_state() { return diff_state_.get(); }

    // Track if setup() has been called (for three-step API)
    bool is_setup_ = false;

    // Track if P and A are already equilibrated (for subsequent solve() calls)
    // When true, we only need to equilibrate q and b, not recompute d, e from scratch
    bool matrices_equilibrated_ = false;

private:
    // Helper to set CUDA device before member initialization
    static int init_device(int device_id) {
        if (device_id >= 0) {
            cudaSetDevice(device_id);
        }
        return device_id;
    }

public:
    /**
     * @brief Construct solver for given problem structure
     * @param n_ Number of primal variables
     * @param m_ Number of constraints
     * @param batchSize_ Number of problems to solve in parallel
     * @param P_ro Row offsets for P matrix (host pointer)
     * @param P_ci Column indices for P matrix (host pointer)
     * @param nnzP Number of nonzeros in P
     * @param A_ro Row offsets for A matrix (host pointer)
     * @param A_ci Column indices for A matrix (host pointer)
     * @param nnzA Number of nonzeros in A
     * @param cones Cone structure specification
     * @param settings_ Solver settings (includes deviceId for multi-GPU)
     */
    CompiledSolver(
        int64_t n_, int64_t m_, int64_t batchSize_,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const Cones& cones_,
        const Settings& settings_ = Settings{}
    )
        : device_id_(init_device(settings_.deviceId)),
          data(n_, m_, batchSize_, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA, cones_),
          cusparse_handle_(nullptr),
          cublas_handle_(nullptr),
          kkt(make_kkt_solver(n_, m_, batchSize_, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA, data.cones, settings_)),
          residuals(n_, m_, batchSize_),
          // Direct-x totals aren't populated until Cones::initialize()
          // runs (see data.cones below), so we read from the user-
          // supplied cones_.x_cones directly here — numXCones mirrors
          // x_cones.size() + each cone's dim accumulates into a total.
          variables(n_, m_, batchSize_,
                    [&]{ int64_t t = 0;
                         for (const auto& xc : cones_.x_cones)
                             t += static_cast<int64_t>(xc.indices.size());
                         return t; }()),
          step_rhs(n_, m_, batchSize_,
                   [&]{ int64_t t = 0;
                        for (const auto& xc : cones_.x_cones)
                            t += static_cast<int64_t>(xc.indices.size());
                        return t; }()),
          step_lhs(n_, m_, batchSize_,
                   [&]{ int64_t t = 0;
                        for (const auto& xc : cones_.x_cones)
                            t += static_cast<int64_t>(xc.indices.size());
                        return t; }()),
          x1(n_, batchSize_),
          z1(m_, batchSize_),
          Δs_const_term(m_, batchSize_),
          tau_num_work(1, batchSize_),
          tau_den_work(1, batchSize_),
          ξ(n_, batchSize_),
          settings(settings_),
          solution(n_, m_, batchSize_),
          info(batchSize_),
          sigma(1, batchSize_),
          alpha(1, batchSize_),
          mu(1, batchSize_),
          m(1, batchSize_),
          workx(n_, batchSize_),
          workz(m_, batchSize_),
          x2(n_, batchSize_),
          z2(m_, batchSize_),
          work_m_1(m_, batchSize_),
          work_m_2(m_, batchSize_),
          alpha_combined(1, batchSize_),
          kkt_rhs(n_ + m_ + 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, batchSize_),
          kkt_sol(n_ + m_ + 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, batchSize_),
          kkt_rhs2((n_ + m_ + 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow) * 2, batchSize_),
          kkt_sol2((n_ + m_ + 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow) * 2, batchSize_),
          inv_tau(1, batchSize_),
          temp_scalar(1, batchSize_),
          q_dot_x1(1, batchSize_),
          b_dot_z1(1, batchSize_),
          ξ_P_x1(1, batchSize_),
          q_dot_x2(1, batchSize_),
          b_dot_z2(1, batchSize_),
          x2_scaled(n_, batchSize_),
          z2_scaled(m_, batchSize_),
          step_work1(1, batchSize_),
          step_work2(1, batchSize_),
          alpha_init(1, batchSize_),
          warm_work(m_, batchSize_),
          default_rhs(n_ + m_ + 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, batchSize_),
          default_sol(n_ + m_ + 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, batchSize_),
          neg_q(n_, batchSize_),
          workz_zero(m_, batchSize_),
          init_min_margin(1, batchSize_),
          init_pos_margin(1, batchSize_),
          init_shift_alpha(1, batchSize_),
          quad_form_temp(n_, batchSize_),
          quad_form_buffer(nullptr),
          quad_form_buffer_size(0),
          quad_form_aligned_y(nullptr),
          quad_form_aligned_temp(nullptr),
          quad_form_aligned_size(0),
          smoothing_mu_step(1, batchSize_),
          smoothing_m_zero(1, batchSize_),
          yolo_zx_raw_(1, 1)
    {
        if (batchSize_ > 65536) {
            throw std::invalid_argument("Batch size must be less than or equal to 65536");
        }

        // Direct-x cone index bounds check. Python's
        // `Cones.validate_x_cone_indices` enforces idx ∈ [0, n) at the unified
        // API, but FFI entry points (JAX, torch) build SupportedXConeT
        // straight from device-side metadata and bypass that validator. An
        // out-of-range index later indexes `xsoc_row_v_col` (sized n) inside
        // KKTData's structural assembly with `vec[(size_t)idx]` — a heap
        // buffer overflow that corrupts surrounding allocations.
        //
        // Also enforce disjointness across all direct-x cones (matches
        // Python's `validate_x_cones_disjoint`). KKTData only catches the
        // overlap when two *sparse SOC* cones collide; Nonneg + SOC,
        // Nonneg + Nonneg, dense SOC + dense SOC, or any combination
        // involving Exp/Power/PSD/GenPow indices share the same primal
        // slot silently — Hs writes overwrite each other and backward
        // gradients double-count.
        std::vector<int64_t> xcone_seen_by(static_cast<size_t>(n_), int64_t{-1});
        for (size_t ci = 0; ci < cones_.x_cones.size(); ++ci) {
            const auto& xc = cones_.x_cones[ci];
            for (int64_t idx : xc.indices) {
                if (idx < 0 || idx >= n_) {
                    throw std::invalid_argument(
                        "CompiledSolver: x_cones[" + std::to_string(ci) +
                        "] index " + std::to_string(idx) +
                        " out of range [0, " + std::to_string(n_) + ")");
                }
                int64_t& slot = xcone_seen_by[static_cast<size_t>(idx)];
                if (slot != -1) {
                    throw std::invalid_argument(
                        "CompiledSolver: x_cones index " + std::to_string(idx) +
                        " appears in both x_cones[" + std::to_string(slot) +
                        "] and x_cones[" + std::to_string(ci) +
                        "]; indices must be disjoint across all direct-x cones");
                }
                slot = static_cast<int64_t>(ci);
            }
        }

        // Initialize GPU library handles
        cusparseCreate(&cusparse_handle_);
        cublasCreate(&cublas_handle_);

        // Pass cuBLAS handle to KKT solver (needed for Riccati batched ops)
        kkt->setCublasHandle(cublas_handle_);

        // Preallocate device buffer for inf-in-b infeasible flags
        cudaMalloc(&d_infeasible_flags_, sizeof(int) * batchSize_);
        inf_b_infeasible_flags_.resize(batchSize_, 0);

        // Create DiffState if gradients are enabled. Pass through the
        // direct-x total numel so rhs/sol/z_x are sized for the augmented
        // HSDE system (n + 2m + xn + 1).
        if (settings.enableGrad) {
            int64_t total_xn = 0;
            for (const auto& xc : cones_.x_cones) {
                total_xn += static_cast<int64_t>(xc.indices.size());
            }
            diff_state_ = std::make_unique<DiffState>(
                n_, m_, batchSize_, nnzP, nnzA, nullptr, total_xn
            );
        }

        // Preallocate per-batch NaN flags for YOLO mode, plus the direct-x
        // dual snapshot buffer. The YOLO snapshot kernel NaN-checks and
        // preserves x/s/z/τ/κ into `solution.*_raw` and z_x into
        // `yolo_zx_raw_`; the post-loop restore copies z_x back so a NaN
        // in the direct-x dual can't reach the user-visible solution.
        if (settings.yolo) {
            int32_t* p = nullptr;
            cudaMalloc(&p, sizeof(int32_t) * batchSize_);
            d_yolo_has_nan_.reset(p);

            int64_t total_xn = 0;
            for (const auto& xc : cones_.x_cones) {
                total_xn += static_cast<int64_t>(xc.indices.size());
            }
            if (total_xn > 0) {
                yolo_zx_raw_ = BatchedVector(total_xn, batchSize_);
            }
        }

        // Preallocate per-batch PrimalDual gate for any GenPower cone
        // (direct-x or slack). Sized always (cheap: batchSize bytes); init
        // kernel runs at solve start. The same buffer is read by both the
        // direct-x and slack PD-axes kernels — all batches flip on the same
        // α threshold, so one shared gate is correct.
        bool any_genpow = data.cones.numGenPowerCones > 0;
        if (!any_genpow) {
            for (const auto& xc : data.cones.x_cones) {
                if (xc.kind == XConeKind::GenPower) { any_genpow = true; break; }
            }
        }
        if (any_genpow) {
            int8_t* p = nullptr;
            cudaMalloc(&p, sizeof(int8_t) * batchSize_);
            pd_enabled_per_batch_.reset(p);
        }
    }

    // =========================================================================
    // Three-step API (preferred for batched/repeated solves)
    // =========================================================================

    /**
     * @brief Set P and A matrix values (Step 2 of three-step API)
     *
     * Must be called before solve(q, b). Can be called multiple times
     * to update values for repeated solves with the same structure.
     *
     * @par P symmetry contract
     * d_P_values must satisfy P[i,j] == P[j,i] for every (i,j) in the
     * sparsity pattern. The solver does NOT check this — caller's
     * responsibility. Asymmetric values produce undefined behavior
     * (silent non-convergence on CUDA). The pattern itself is validated
     * for structural symmetry at construction (PR #162); only values
     * are unchecked here. Python wrappers validate values on the way in;
     * raw C++/CUDA/Julia callers do not.
     *
     * @param d_P_values P matrix values (device pointer, size nnzP * batchSize). Must be symmetric.
     * @param d_A_values A matrix values (device pointer, size nnzA * batchSize)
     * @param stream CUDA stream for async operations
     */
    void setup(
        const double* d_P_values,
        const double* d_A_values,
        cudaStream_t stream = 0
    );

    /**
     * @brief Load all problem data and equilibrate, without solving.
     *
     * Used by the backward pass to restore the solver's equilibrated state
     * from saved residuals, ensuring correctness for chained solves where
     * a subsequent forward pass may have overwritten the internal state.
     *
     * Equivalent to: setup(P, A) + copyPerSolveData(q, b) + doEquilibration().
     *
     * @param d_P_values P matrix values (device pointer, size nnzP * batchSize)
     * @param d_A_values A matrix values (device pointer, size nnzA * batchSize)
     * @param d_q Linear cost vector (device pointer, size n * batchSize)
     * @param d_b Constraint RHS (device pointer, size m * batchSize)
     * @param stream CUDA stream for async operations
     */
    void loadDataForBackward(
        const double* d_P_values,
        const double* d_A_values,
        const double* d_q,
        const double* d_b,
        cudaStream_t stream = 0
    );

    /**
     * @brief Solve with pre-set P/A matrices (Step 3 of three-step API)
     *
     * Requires setup() to have been called first.
     *
     * @param d_q Linear cost vector (device pointer, size n * batchSize)
     * @param d_b Constraint RHS (device pointer, size m * batchSize)
     * @param stream CUDA stream for async operations
     */
    void solve(
        const double* d_q,
        const double* d_b,
        cudaStream_t stream = 0
    );

    /**
     * @brief Solve with warm start (Step 3 of three-step API)
     *
     * Like solve(q, b) but uses provided (x, z, s) as warm start point.
     * The warm start is projected onto the mu-central path using cone smoothing.
     *
     * @param d_q Linear cost vector (device pointer, size n * batchSize)
     * @param d_b Constraint RHS (device pointer, size m * batchSize)
     * @param d_warm_x Warm start primal variables (device pointer, size n * batchSize)
     * @param d_warm_z Warm start dual variables (device pointer, size m * batchSize)
     * @param d_warm_s Warm start slack variables (device pointer, size m * batchSize)
     * @param stream CUDA stream for async operations
     */
    void solve(
        const double* d_q,
        const double* d_b,
        const double* d_warm_x,
        const double* d_warm_z,
        const double* d_warm_s,
        cudaStream_t stream = 0,
        const double* d_warm_z_x = nullptr  // direct-x dual warm start; ignored when no x_cones
    );

    /**
     * @brief Convenience: setup + solve in one call
     *
     * Equivalent to:
     *   setup(d_P_values, d_A_values);
     *   solve(d_q, d_b);
     */
    void solveAll(
        const double* d_P_values,
        const double* d_A_values,
        const double* d_q,
        const double* d_b,
        cudaStream_t stream = 0
    );

    // Cold-initialise the direct-x cone block: primal x[J] shifted/seeded
    // into the cone interior and dual z_x at the canonical interior point.
    // Shared by default_start and the warmStart no-warm-z_x fallback.
    void init_xcone_start_point(cudaStream_t stream = 0);

    void default_start(
        cudaStream_t stream = 0
    );

    /**
     * @brief Solve KKT system for step direction
     *
     * Implements the full KKT solve logic from Clarabel.rs:
     * 1. Solves for (x1, z1) from variable RHS
     * 2. Computes Δτ from numerator/denominator expressions
     * 3. Combines solutions: Δx = x1 + τ*x2, Δz = z1 + τ*z2
     * 4. Computes Δs from HsΔz
     * 5. Computes Δκ
     *
     * @param lhs Output: step direction variables
     * @param rhs Input: right-hand side variables
     * @param step_direction Whether this is affine or combined step
     * @param stream CUDA stream for async operations
     * @return true if solve succeeded, false otherwise
     */
    bool solveKKT(
        Variables& lhs,
        const Variables& rhs,
        bool is_affine_step,
        cudaStream_t stream = 0
    );

    /**
     * @brief Solve KKT system using precomputed x1, z1, x2, z2
     *
     * Same as solveKKT but skips step 1 (assumes x1, z1, x2, z2 are already computed).
     * This is used when batching constant + affine solves together with solve2().
     *
     * @param lhs Output: step direction variables
     * @param rhs Input: right-hand side variables
     * @param is_affine_step Whether this is affine or combined step
     * @param stream CUDA stream for async operations
     * @return true if solve succeeded, false otherwise
     */
    bool solveKKTFromPrecomputed(
        Variables& lhs,
        const Variables& rhs,
        bool is_affine_step,
        cudaStream_t stream = 0
    );

    /**
     * @brief Calculate step length to stay within cones
     *
     * Computes maximum α such that variables + α * step stays feasible.
     * For combined steps, applies max_step_fraction. For asymmetric cones
     * with dual scaling, performs barrier backtracking.
     *
     * @param step Step direction variables
     * @param is_affine_step True for affine step, false for combined step
     * @param alpha Output: step length [batchSize][1]
     * @param stream CUDA stream for async operations
     */
    void getStepLength(
        const Variables& step,
        bool is_affine_step,
        BatchedVector& alpha,
        cudaStream_t stream = 0
    );

    /**
     * @brief Backtrack step length using barrier function
     *
     * Reduces step length until barrier(variables + α*step) < 1.
     * Used for asymmetric cones with dual scaling strategy.
     *
     * @param step Step direction variables
     * @param alpha_init Initial step length [batchSize][1]
     * @param alpha Output: backtracked step length [batchSize][1]
     * @param stream CUDA stream for async operations
     */
    void backtrackStepToBarrier(
        const Variables& step,
        const BatchedVector& alpha_init,
        BatchedVector& alpha,
        cudaStream_t stream = 0
    );

    // No copy
    CompiledSolver(const CompiledSolver&) = delete;
    CompiledSolver& operator=(const CompiledSolver&) = delete;

    // Move constructor - must null source's raw pointers to prevent double-free
    CompiledSolver(CompiledSolver&& other) noexcept
        : device_id_(other.device_id_),
          data(std::move(other.data)),
          cusparse_handle_(other.cusparse_handle_),
          cublas_handle_(other.cublas_handle_),
          kkt(std::move(other.kkt)),
          residuals(std::move(other.residuals)),
          variables(std::move(other.variables)),
          step_rhs(std::move(other.step_rhs)),
          step_lhs(std::move(other.step_lhs)),
          x1(std::move(other.x1)),
          z1(std::move(other.z1)),
          Δs_const_term(std::move(other.Δs_const_term)),
          tau_num_work(std::move(other.tau_num_work)),
          tau_den_work(std::move(other.tau_den_work)),
          ξ(std::move(other.ξ)),
          settings(std::move(other.settings)),
          solution(std::move(other.solution)),
          info(std::move(other.info)),
          sigma(std::move(other.sigma)),
          alpha(std::move(other.alpha)),
          mu(std::move(other.mu)),
          m(std::move(other.m)),
          workx(std::move(other.workx)),
          workz(std::move(other.workz)),
          x2(std::move(other.x2)),
          z2(std::move(other.z2)),
          work_m_1(std::move(other.work_m_1)),
          work_m_2(std::move(other.work_m_2)),
          alpha_combined(std::move(other.alpha_combined)),
          kkt_rhs(std::move(other.kkt_rhs)),
          kkt_sol(std::move(other.kkt_sol)),
          kkt_rhs2(std::move(other.kkt_rhs2)),
          kkt_sol2(std::move(other.kkt_sol2)),
          inv_tau(std::move(other.inv_tau)),
          temp_scalar(std::move(other.temp_scalar)),
          q_dot_x1(std::move(other.q_dot_x1)),
          b_dot_z1(std::move(other.b_dot_z1)),
          ξ_P_x1(std::move(other.ξ_P_x1)),
          q_dot_x2(std::move(other.q_dot_x2)),
          b_dot_z2(std::move(other.b_dot_z2)),
          x2_scaled(std::move(other.x2_scaled)),
          z2_scaled(std::move(other.z2_scaled)),
          step_work1(std::move(other.step_work1)),
          step_work2(std::move(other.step_work2)),
          alpha_init(std::move(other.alpha_init)),
          warm_work(std::move(other.warm_work)),
          default_rhs(std::move(other.default_rhs)),
          default_sol(std::move(other.default_sol)),
          neg_q(std::move(other.neg_q)),
          workz_zero(std::move(other.workz_zero)),
          init_min_margin(std::move(other.init_min_margin)),
          init_pos_margin(std::move(other.init_pos_margin)),
          init_shift_alpha(std::move(other.init_shift_alpha)),
          quad_form_temp(std::move(other.quad_form_temp)),
          quad_form_buffer(other.quad_form_buffer),
          quad_form_buffer_size(other.quad_form_buffer_size),
          quad_form_aligned_y(other.quad_form_aligned_y),
          quad_form_aligned_temp(other.quad_form_aligned_temp),
          quad_form_aligned_size(other.quad_form_aligned_size),
          smoothing_mu_step(std::move(other.smoothing_mu_step)),
          smoothing_m_zero(std::move(other.smoothing_m_zero)),
          yolo_zx_raw_(std::move(other.yolo_zx_raw_)),
          diff_kkt_(std::move(other.diff_kkt_)),
          diff_woodbury_(std::move(other.diff_woodbury_)),
          backward_workspace_(std::move(other.backward_workspace_)),
          diff_state_(std::move(other.diff_state_)),
          is_setup_(other.is_setup_),
          matrices_equilibrated_(other.matrices_equilibrated_),
          inf_b_infeasible_flags_(std::move(other.inf_b_infeasible_flags_)),
          d_infeasible_flags_(other.d_infeasible_flags_)
    {
        // Null source's raw pointers to prevent double-free in destructor
        other.cusparse_handle_ = nullptr;
        other.cublas_handle_ = nullptr;
        other.quad_form_buffer = nullptr;
        other.quad_form_aligned_y = nullptr;
        other.quad_form_aligned_temp = nullptr;
        other.d_infeasible_flags_ = nullptr;
    }

    // Move assignment - must null source's raw pointers to prevent double-free
    // Move assignment intentionally deleted (PR #171, fixes #167).
    // Hand-rolling a correct move-assign over ~30 owned workspace fields is
    // a footgun — every existing caller uses `unique_ptr<CompiledSolver>`,
    // which only needs the move CONSTRUCTOR (preserved above). The contract
    // is pinned by static_asserts in tests/cpp/test_compiled_solver_move_semantics.cpp.
    CompiledSolver& operator=(CompiledSolver&& other) noexcept = delete;

    ~CompiledSolver() {
        // Ensure we're on the correct device for cleanup
        if (device_id_ >= 0) {
            cudaSetDevice(device_id_);
        }

        // Ensure all GPU work completes before handle cleanup
        cudaDeviceSynchronize();

        // Destroy GPU library handles to prevent state pollution
        if (cusparse_handle_) {
            cusparseDestroy(cusparse_handle_);
        }
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }

        // Free inf-in-b device buffer
        if (d_infeasible_flags_) {
            cudaFree(d_infeasible_flags_);
        }

        // Free quad_form buffers
        if (quad_form_buffer) {
            cudaFree(quad_form_buffer);
        }
        if (quad_form_aligned_y) {
            cudaFree(quad_form_aligned_y);
        }
        if (quad_form_aligned_temp) {
            cudaFree(quad_form_aligned_temp);
        }
    }

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept {
        return data.memoryUsage() +
               kkt->memoryUsage() + residuals.memoryUsage() +
               variables.memoryUsage();
    }

    /**
     * @brief Reset all solver state to prepare for a new solve
     *
     * Clears workspace vectors and resets state. Called at the start of each solve().
     */
    void resetState(cudaStream_t stream = 0);

    /**
     * @brief Walk converged iterate up the central path for smoothed differentiation
     *
     * After solve() converges at μ≈tol_gap_abs, this centering loop increases μ
     * to diffSmoothingMu. The smoothed iterate is cached into DiffState for use
     * by the backward pass, producing continuous gradients at cone boundaries.
     *
     * Only runs when diffMethod == Smoothed. No-op otherwise.
     *
     * @param state DiffState to cache the smoothed iterate into
     * @param stream CUDA stream
     */
    void refineSmoothingIterate(DiffState& state, cudaStream_t stream = 0);

    /**
     * @brief Run the interior-point method iteration loop
     *
     * Shared IPM loop used by solve().
     * Assumes problem data, equilibration, and initial variables are already set up.
     *
     * @param stream CUDA stream
     */
    void runIPMLoop(cudaStream_t stream = 0);

private:
    /**
     * @brief Copy problem data from device pointers into internal structures
     */
    void copyProblemData(
        const double* d_P_values,
        const double* d_A_values,
        const double* d_q,
        const double* d_b,
        cudaStream_t stream
    );

    /**
     * @brief Perform matrix equilibration
     */
    void doEquilibration(cudaStream_t stream);

    /**
     * @brief Compute cached norms for residual computation
     */
    void computeCachedNorms(cudaStream_t stream);

    /**
     * @brief Save solutions for terminated batches
     */
    void save_terminated_solutions(
        const Variables& variables,
        Solution& solution,
        const Info& info,
        const int32_t* status,
        int64_t batchSize,
        cudaStream_t stream
    );

    /**
     * @brief Snapshot the current iterate for any still-active batch whose
     * metrics satisfy the reduced tolerances. Paired with
     * restore_best_iterate to promote non-convergent terminations back to
     * AlmostSolved when a reduced-tolerance iterate was observed.
     */
    void save_best_iterate(
        const Variables& variables,
        Solution& solution,
        Info& info,
        int64_t batchSize,
        cudaStream_t stream
    );

    /**
     * @brief Promote non-convergent per-batch statuses to AlmostSolved when a
     * best-iterate snapshot exists, and copy snapshot metrics back into Info
     * so post_process sees the restored iterate.
     */
    void restore_best_iterate(Info& info, int64_t batchSize, cudaStream_t stream);

    /**
     * @brief Sanitize inf entries in b and detect infeasibility.
     *
     * For nonneg cone rows where b[i] = +inf, zeros the A row and sets b=1.
     * Detects infeasible cases (±inf in zero cone, -inf in nonneg cone).
     * Stores per-batch infeasible flags in inf_b_infeasible_flags_.
     * Must be called after copyPerSolveData but before doEquilibration.
     */
    void sanitizeInfB(cudaStream_t stream);

    /**
     * @brief Override status to PrimalInfeasible for batches flagged by sanitizeInfB.
     */
    void overrideInfeasibleStatus();

    /// Per-batch infeasible flags from sanitizeInfB (host-side)
    std::vector<int> inf_b_infeasible_flags_;

    /// Device buffer for per-batch infeasible flags (preallocated)
    int* d_infeasible_flags_ = nullptr;

    /**
     * @brief Copy per-solve data (q, b) into internal structures
     */
    void copyPerSolveData(
        const double* d_q,
        const double* d_b,
        cudaStream_t stream
    );

    /**
     * @brief Internal solve implementation for three-step API
     *
     * Called after setup() has been called. Only copies q, b (not P, A).
     */
    void solveInternalWithSetup(
        const double* d_q,
        const double* d_b,
        cudaStream_t stream
    );

    /**
     * @brief Internal solve implementation with warm start
     *
     * Same as solveInternalWithSetup but replaces default_start() with warm start logic.
     * Pass d_warm_z_x = nullptr when the problem has no direct-x cones.
     */
    void solveInternalWithWarmStart(
        const double* d_q,
        const double* d_b,
        const double* d_warm_x,
        const double* d_warm_z,
        const double* d_warm_s,
        const double* d_warm_z_x,
        cudaStream_t stream
    );

    /**
     * @brief Apply warm start: scale, compute mu, smooth, set variables
     *
     * Given user-provided (x, z, s, z_x) in original space:
     * 1. Copy into variables, set tau=1, kappa=1
     * 2. Apply forward equilibration scaling (incl. z_x → equilibrated frame)
     * 3. Compute residuals and info (for res_primal, res_dual, gap)
     * 4. Compute warmness mu
     * 5. Compute work = z - s
     * 6. Apply cone smoothing on z
     * 7. Recover s = z - work
     * 8. Set kappa = mu
     *
     * `d_warm_z_x` is the user-frame direct-x dual (length total_xcone_numel
     * per batch). Pass nullptr if the problem has no direct-x cones; the
     * placeholder z_x stays zero in that case.
     */
    void warmStart(
        const double* d_warm_x,
        const double* d_warm_z,
        const double* d_warm_s,
        const double* d_warm_z_x,
        cudaStream_t stream
    );
};

/**
 * @brief Single-problem solver that takes all data at construction
 *
 * This is a convenience wrapper around CompiledSolver for single problems
 * where all data (P, q, A, b) is known upfront.
 *
 * Usage:
 *   Solver solver(P, q, A, b, cones, settings);
 *   solver.solve();
 *   // Access solver.solution() for results
 */
class Solver {
private:
    std::unique_ptr<CompiledSolver> impl_;

    // GPU buffers for problem data
    double* d_q_;
    double* d_b_;
    int64_t n_, m_;

public:
    /**
     * @brief Construct solver with full problem data
     *
     * @param P Sparse P matrix (CSR format: row_offsets, col_indices, values)
     * @param q Linear cost vector (device pointer, size n)
     * @param A Sparse A matrix (CSR format: row_offsets, col_indices, values)
     * @param b Constraint RHS (device pointer, size m)
     * @param cones Cone structure specification
     * @param settings Solver settings
     */
    Solver(
        int64_t n, int64_t m,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP, const double* d_P_values,
        const double* d_q,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA, const double* d_A_values,
        const double* d_b,
        const Cones& cones,
        const Settings& settings = Settings{}
    );

    ~Solver();

    // No copy
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    // Move ok
    Solver(Solver&&) noexcept = default;
    Solver& operator=(Solver&&) noexcept = default;

    /**
     * @brief Solve the optimization problem
     * @param stream CUDA stream for async operations
     */
    void solve(cudaStream_t stream = 0);

    /**
     * @brief Access the solution
     */
    const Solution& solution() const { return impl_->solution; }
    Solution& solution() { return impl_->solution; }

    /**
     * @brief Access solver info/statistics
     */
    const Info& info() const { return impl_->info; }
    Info& info() { return impl_->info; }

    /**
     * @brief Get memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept { return impl_->memoryUsage(); }
};

// Backwards compatibility alias
using BatchSolver = CompiledSolver;

} // namespace moreau
