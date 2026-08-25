/**
 * @file solver.hpp
 * @brief ActiveSetSolver — CPU-only batched QP solver using dual active-set method
 *
 * The active-set core is derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * Solves problems in moreau's conic form:
 *   minimize    (1/2)x'Px + q'x
 *   subject to  Ax + s = b,  s ∈ K
 *
 * Restricted to K = ZeroCone × NonnegCone (pure QP).
 * Internally converts to LDP form and runs the dual active-set method.
 *
 * Three-step API:
 *   1. Construct with structure (dimensions, sparsity, cones)
 *   2. setup(P_values, A_values) — set matrix values
 *   3. solve(q, b) — solve with given RHS
 *
 * Supports backward differentiation via KKT implicit differentiation
 * and warm starting from a previous solution.
 */

#pragma once

#include "moreau/solver/active_set/types.hpp"
#include "moreau/solver/active_set/backward.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/solver/status.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>

namespace moreau {

class ActiveSetSolver {
public:
    ActiveSetSolver(
        int64_t n, int64_t m, int64_t batchSize,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const Cones& cones,
        const ActiveSetSettings& settings = ActiveSetSettings{},
        bool enable_grad = false,
        bool verbose = false
    );

    /**
     * @brief Set P and A matrix values (Step 2)
     */
    void setup(const double* P_values, const double* A_values, bool shared = true);

    /**
     * @brief Solve with given RHS (Step 3)
     */
    void solve(const double* q, const double* b);

    /**
     * @brief Solve with warm start from previous solution
     *
     * Uses the active set from warm_z/warm_s to initialize the iteration.
     * Falls back to cold start if warm start fails.
     */
    void solve_warm_start(const double* q, const double* b,
                          const double* warm_x, const double* warm_z,
                          const double* warm_s);

    /**
     * @brief Backward differentiation via KKT implicit differentiation
     *
     * Given upstream gradients (dx, dz, ds), computes gradients w.r.t.
     * problem data (P_values, q, A_values, b) in sparse CSR format.
     *
     * Must be called after solve(). Requires enable_grad=true at construction.
     */
    void backward(const double* dx, const double* dz, const double* ds);

    /**
     * @brief Backward differentiation from explicit problem data and saved solution.
     *
     * Reconstructs the necessary solver state from the provided forward inputs
     * and iterates, then computes backward gradients without depending on this
     * instance's prior solve() state.
     */
    void backward_with_data(
        const double* dx, const double* dz, const double* ds,
        const double* P_values, const double* A_values, bool shared_matrices,
        const double* q, const double* b,
        const double* x, const double* z, const double* s,
        const double* state_rinv,
        const double* state_rinv_diag,
        const int32_t* state_use_rinv_diag,
        const int32_t* state_n_active,
        const int32_t* state_ws,
        const int32_t* state_sense,
        const double* state_lam_star
    );

    const ActiveSetBackwardState& backward_state(int64_t batch_idx) const;

    // ========================================================================
    // Forward results (host memory, populated after solve())
    // ========================================================================

    std::vector<double> x_sol;       // [batchSize * n]
    std::vector<double> s_sol;       // [batchSize * m]
    std::vector<double> z_sol;       // [batchSize * m]
    std::vector<int32_t> status_vec; // [batchSize]
    std::vector<double> obj_val;     // [batchSize]
    std::vector<int32_t> iters;      // [batchSize]
    double solve_time = 0.0;

    // ========================================================================
    // Backward results (host memory, populated after backward())
    // ========================================================================

    std::vector<double> dP_values;   // [batchSize * nnzP]
    std::vector<double> dq;          // [batchSize * n]
    std::vector<double> dA_values;   // [batchSize * nnzA]
    std::vector<double> db;          // [batchSize * m]

    // Dimensions
    int64_t n() const { return n_; }
    int64_t m() const { return m_; }
    int64_t batchSize() const { return batchSize_; }
    int64_t nnzP() const { return nnzP_; }
    int64_t nnzA() const { return nnzA_; }
    bool enable_grad() const { return enable_grad_; }

private:
    int64_t n_, m_, batchSize_;
    int64_t nnzP_, nnzA_;
    int64_t numZeroCones_, numNonnegCones_;
    ActiveSetSettings settings_;
    bool enable_grad_ = false;
    bool verbose_ = false;
    bool is_setup_ = false;
    bool has_solved_ = false;

    // Cached sparsity structure
    std::vector<int64_t> P_ro_, P_ci_;
    std::vector<int64_t> A_ro_, A_ci_;

    // Dense host buffers for matrix values (set by setup())
    std::vector<double> H_dense_;   // [n*n] (or [batchSize*n*n] if per-batch)
    std::vector<double> A_dense_;   // [m*n] (or [batchSize*m*n] if per-batch)
    bool matrices_shared_ = true;

    // Persisted workspaces (one per batch element, kept for backward pass)
    // unique_ptr because DaqpWorkspace has raw pointer aliasing into its own buffers
    std::vector<std::unique_ptr<DaqpWorkspace>> workspaces_;
    std::vector<ActiveSetBackwardState> backward_states_;

    // Per-thread backward workspaces (allocated once, reused)
    std::vector<ActiveSetBackwardWorkspace> bw_workspaces_;

    // Per-thread dense gradient buffers (avoid allocation in hot loop)
    std::vector<std::vector<double>> dH_dense_bufs_;  // [nthreads][n*n]
    std::vector<std::vector<double>> dA_dense_bufs_;  // [nthreads][m*n]

    // Cached RHS (needed for backward)
    std::vector<double> q_cached_;  // [batchSize * n]
    std::vector<double> b_cached_;  // [batchSize * m]

    void solveSingle(int64_t batch_idx,
                     const double* H, const double* q,
                     const double* A, const double* b,
                     bool warm_start = false);

    void buildBounds(const double* b, double* bupper, double* blower,
                     int* sense) const;

    void extractSolution(int64_t batch_idx, const DaqpWorkspace& work,
                         int exitflag, const double* A, const double* b);

    void cache_backward_state(int64_t batch_idx, const DaqpWorkspace& work);
};

} // namespace moreau
