/**
 * @file solution.hpp
 * @brief Solution structure for conic optimization solver
 *
 * This module defines the solution structure containing primal/dual
 * solutions, objective values, residuals, and solver statistics.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "moreau/vector/vector.hpp"
#include "moreau/solver/status.hpp"
#include "moreau/cuda/utils.hpp"

namespace moreau {

// Forward declarations
struct EquilibrationData;
class Variables;

/**
 * @brief Solution information returned by solver
 *
 * Contains primal solution (x), dual solution (z), slack variables (s),
 * objective values, residuals, and solve statistics.
 */
struct Solution {
    // Solution vectors
    BatchedVector x;     // Primal solution [batchSize][n]
    BatchedVector z;     // Dual solution [batchSize][m]
    BatchedVector s;     // Slack variables [batchSize][m]

    // Raw variables before post-processing (used for incremental saving)
    BatchedVector x_raw;    // Raw primal [batchSize][n]
    BatchedVector z_raw;    // Raw dual [batchSize][m]
    BatchedVector s_raw;    // Raw slack [batchSize][m]
    BatchedVector τ_raw;    // Raw τ [batchSize][1]
    BatchedVector κ_raw;    // Raw κ [batchSize][1]
    BatchedVector cost_primal_raw;  // Saved primal cost at convergence [batchSize][1]
    BatchedVector cost_dual_raw;    // Saved dual cost at convergence [batchSize][1]

    // Flag to track which batches have been saved (int32_t, properly typed)
    device_unique_ptr<int32_t> solution_saved;  // [batchSize], 1 if saved, 0 otherwise

    // Temporary buffer for two-kernel save approach
    device_unique_ptr<int32_t> should_save;     // [batchSize], set by mark kernel, read by copy kernel

    int64_t batchSize_ = 0;  // Stored for reset

    // Solver status (per batch) - int32_t on device
    device_unique_ptr<int32_t> status;  // Termination status per batch [batchSize]

    // Objective values (per batch)
    BatchedVector obj_val;       // Primal objective value [batchSize][1]
    BatchedVector obj_val_dual;  // Dual objective value [batchSize][1]

    // Solve statistics
    double solve_time;        // Total solve time in seconds
    int64_t iterations;       // Number of iterations performed

    // Residuals (per batch)
    BatchedVector r_prim;  // Primal residual [batchSize][1]
    BatchedVector r_dual;  // Dual residual [batchSize][1]

    /**
     * @brief Default constructor
     */
    Solution()
        : x(1, 1),
          z(1, 1),
          s(1, 1),
          x_raw(1, 1),
          z_raw(1, 1),
          s_raw(1, 1),
          τ_raw(1, 1),
          κ_raw(1, 1),
          cost_primal_raw(1, 1),
          cost_dual_raw(1, 1),
          batchSize_(1),
          obj_val(1, 1),
          obj_val_dual(1, 1),
          solve_time(0.0),
          iterations(0),
          r_prim(1, 1),
          r_dual(1, 1)
    {
        int32_t* p_saved = nullptr;
        int32_t* p_should = nullptr;
        int32_t* p_status = nullptr;
        cudaMalloc(&p_saved, sizeof(int32_t));
        cudaMalloc(&p_should, sizeof(int32_t));
        cudaMalloc(&p_status, sizeof(int32_t));
        cudaMemset(p_saved, 0, sizeof(int32_t));
        cudaMemset(p_status, 0, sizeof(int32_t));
        solution_saved.reset(p_saved);
        should_save.reset(p_should);
        status.reset(p_status);
    }

    /**
     * @brief Construct a solution with given problem dimensions
     */
    Solution(int64_t n, int64_t m, int64_t batchSize)
        : x(n, batchSize),
          z(m, batchSize),
          s(m, batchSize),
          x_raw(n, batchSize),
          z_raw(m, batchSize),
          s_raw(m, batchSize),
          τ_raw(1, batchSize),
          κ_raw(1, batchSize),
          cost_primal_raw(1, batchSize),
          cost_dual_raw(1, batchSize),
          batchSize_(batchSize),
          obj_val(1, batchSize),
          obj_val_dual(1, batchSize),
          solve_time(0.0),
          iterations(0),
          r_prim(1, batchSize),
          r_dual(1, batchSize)
    {
        int32_t* p_saved = nullptr;
        int32_t* p_should = nullptr;
        int32_t* p_status = nullptr;
        cudaMalloc(&p_saved, sizeof(int32_t) * batchSize);
        cudaMalloc(&p_should, sizeof(int32_t) * batchSize);
        cudaMalloc(&p_status, sizeof(int32_t) * batchSize);
        cudaMemset(p_saved, 0, sizeof(int32_t) * batchSize);
        cudaMemset(p_status, 0, sizeof(int32_t) * batchSize);
        solution_saved.reset(p_saved);
        should_save.reset(p_should);
        status.reset(p_status);
    }

    /**
     * @brief Reset solution_saved flags to 0 for a new solve
     */
    void resetSolutionSaved(cudaStream_t stream = 0) {
        cudaMemsetAsync(solution_saved.get(), 0, sizeof(int32_t) * batchSize_, stream);
    }

    // No copying (BatchedVector owns device memory)
    Solution(const Solution&) = delete;
    Solution& operator=(const Solution&) = delete;

    // Moves ok
    Solution(Solution&&) noexcept = default;
    Solution& operator=(Solution&&) noexcept = default;

    ~Solution() = default;

    /**
     * @brief Post-process solution to unscale variables
     *
     * Unscales the primal and dual variables to obtain solutions to the
     * original problem (before equilibration). This should be called after
     * the solver completes.
     */
    void post_process(
        int64_t n,
        int64_t m,
        int64_t batchSize,
        const struct EquilibrationData& equilibration,
        class Variables& variables,
        const struct Info& info,
        bool equilibration_enabled,
        cudaStream_t stream = 0
    );

    /**
     * @brief Finalize solution with information from Info
     *
     * Copies solve time and other final information from Info to Solution.
     *
     * @param info Solver info object
     */
    void finalize(const struct Info& info);
};

} // namespace moreau
