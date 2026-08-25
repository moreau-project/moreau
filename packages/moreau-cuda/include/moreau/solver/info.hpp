/**
 * @file info.hpp
 * @brief Solver information and statistics
 *
 * This module defines the Info structure for tracking solver progress,
 * convergence metrics, and iteration statistics.
 */

#pragma once

#include "moreau/vector/vector.hpp"
#include "moreau/solver/status.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace moreau {

// Forward declarations
struct SolverData;
struct Variables;
struct Residuals;
struct Settings;
struct Cones;
enum class ScalingStrategy;
enum class KKTSolverType;

/**
 * @brief Standard-form solver information and statistics
 *
 * Contains iteration statistics, convergence metrics, and solver status.
 * All metrics are tracked per batch.
 */
struct Info {
    // Current iteration metrics
    BatchedVector mu;                 // Interior point path parameter μ [batchSize]
    BatchedVector sigma;              // Path parameter reduction ratio σ [batchSize]
    BatchedVector step_length;        // Step length for current iteration [batchSize]

    int32_t iterations;               // Number of iterations (max across batches)
    std::vector<int32_t> iterations_per_batch;  // Iteration at which each batch converged [batchSize]
    int32_t* d_iterations_per_batch;  // Device array for per-batch iterations [batchSize]

    BatchedVector cost_primal;        // Primal objective value [batchSize]
    BatchedVector cost_dual;          // Dual objective value [batchSize]

    BatchedVector res_primal;         // Primal residual [batchSize]
    BatchedVector res_dual;           // Dual residual [batchSize]
    BatchedVector res_primal_inf;     // Primal infeasibility residual [batchSize]
    BatchedVector res_dual_inf;       // Dual infeasibility residual [batchSize]

    BatchedVector gap_abs;            // Absolute duality gap [batchSize]
    BatchedVector gap_rel;            // Relative duality gap [batchSize]

    BatchedVector ktratio;            // κ/τ ratio [batchSize]

    // Previous iteration metrics
    BatchedVector prev_cost_primal;   // Previous primal objective [batchSize]
    BatchedVector prev_cost_dual;     // Previous dual objective [batchSize]
    BatchedVector prev_res_primal;    // Previous primal residual [batchSize]
    BatchedVector prev_res_dual;      // Previous dual residual [batchSize]
    BatchedVector prev_gap_abs;       // Previous absolute gap [batchSize]
    BatchedVector prev_gap_rel;       // Previous relative gap [batchSize]

    // Best-iterate snapshot (reduced-tolerance fallback). Written by
    // save_best_iterate_kernel whenever a batch is still Unsolved and its
    // current metrics satisfy the reduced tolerances. On non-convergent
    // termination (InsufficientProgress / NumericalError / MaxIter /
    // MaxTime) the snapshot is restored and status promoted to AlmostSolved.
    BatchedVector best_cost_primal;
    BatchedVector best_cost_dual;
    BatchedVector best_res_primal;
    BatchedVector best_res_dual;
    BatchedVector best_gap_abs;
    BatchedVector best_gap_rel;
    BatchedVector best_ktratio;
    int32_t* d_best_saved;            // Per-batch flag: 1 if best iterate was saved [batchSize]

    // Timing and status
    double construction_time;         // Construction/allocation time in seconds
    double setup_time;                // Setup/matrix values time in seconds
    double solve_time;                // IPM iteration time in seconds
    std::vector<SolverStatus> status; // Solver status per batch [batchSize] (host)
    int32_t* status_device;           // Device status array as int32_t [batchSize]
    int32_t* d_any_done;              // Device pointer (mapped to pinned host memory)
    int32_t* h_any_done_pinned;       // Pinned host memory for zero-copy termination check
    int64_t batchSize;                // Number of problems in batch

    /**
     * @brief Construct Info for given batch size
     * @param batchSize Number of problems to solve in parallel
     */
    Info(int64_t batchSize)
        : mu(1, batchSize),
          sigma(1, batchSize),
          step_length(1, batchSize),
          iterations(0),
          iterations_per_batch(batchSize, 0),
          d_iterations_per_batch(nullptr),
          cost_primal(1, batchSize),
          cost_dual(1, batchSize),
          res_primal(1, batchSize),
          res_dual(1, batchSize),
          res_primal_inf(1, batchSize),
          res_dual_inf(1, batchSize),
          gap_abs(1, batchSize),
          gap_rel(1, batchSize),
          ktratio(1, batchSize),
          prev_cost_primal(1, batchSize),
          prev_cost_dual(1, batchSize),
          prev_res_primal(1, batchSize),
          prev_res_dual(1, batchSize),
          prev_gap_abs(1, batchSize),
          prev_gap_rel(1, batchSize),
          best_cost_primal(1, batchSize),
          best_cost_dual(1, batchSize),
          best_res_primal(1, batchSize),
          best_res_dual(1, batchSize),
          best_gap_abs(1, batchSize),
          best_gap_rel(1, batchSize),
          best_ktratio(1, batchSize),
          d_best_saved(nullptr),
          construction_time(0.0),
          setup_time(0.0),
          solve_time(0.0),
          status(batchSize, SolverStatus::Unsolved),
          status_device(nullptr),
          d_any_done(nullptr),
          h_any_done_pinned(nullptr),
          batchSize(batchSize)
    {
        // Allocate and initialize status_device
        cudaMalloc(&status_device, sizeof(int32_t) * batchSize);
        cudaMemset(status_device, 0, sizeof(int32_t) * batchSize);

        // Allocate pinned memory with device mapping for zero-copy termination check.
        // This allows the GPU to write directly to host-visible memory, avoiding explicit D2H copy.
        // NOTE: Do NOT use cudaHostAllocWriteCombined — it's optimized for GPU writes but makes
        // CPU reads extremely slow (uncached). We need fast CPU polling reads here.
        cudaError_t err = cudaHostAlloc(&h_any_done_pinned, sizeof(int32_t),
                                        cudaHostAllocMapped);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaHostAlloc failed for termination flag — mapped pinned memory is required");
        }
        err = cudaHostGetDevicePointer(&d_any_done, h_any_done_pinned, 0);
        if (err != cudaSuccess) {
            cudaFreeHost(h_any_done_pinned);
            throw std::runtime_error("cudaHostGetDevicePointer failed for termination flag");
        }
        *h_any_done_pinned = 0;

        // Allocate per-batch iteration counters on device
        cudaMalloc(&d_iterations_per_batch, sizeof(int32_t) * batchSize);
        cudaMemset(d_iterations_per_batch, 0, sizeof(int32_t) * batchSize);

        // Allocate and zero-initialize the best-iterate saved flag array.
        cudaMalloc(&d_best_saved, sizeof(int32_t) * batchSize);
        cudaMemset(d_best_saved, 0, sizeof(int32_t) * batchSize);
    }

    // No copying (large device buffers)
    Info(const Info&) = delete;
    Info& operator=(const Info&) = delete;

    // Moves ok
    Info(Info&& other) noexcept
        : mu(std::move(other.mu)),
          sigma(std::move(other.sigma)),
          step_length(std::move(other.step_length)),
          iterations(other.iterations),
          iterations_per_batch(std::move(other.iterations_per_batch)),
          d_iterations_per_batch(other.d_iterations_per_batch),
          cost_primal(std::move(other.cost_primal)),
          cost_dual(std::move(other.cost_dual)),
          res_primal(std::move(other.res_primal)),
          res_dual(std::move(other.res_dual)),
          res_primal_inf(std::move(other.res_primal_inf)),
          res_dual_inf(std::move(other.res_dual_inf)),
          gap_abs(std::move(other.gap_abs)),
          gap_rel(std::move(other.gap_rel)),
          ktratio(std::move(other.ktratio)),
          prev_cost_primal(std::move(other.prev_cost_primal)),
          prev_cost_dual(std::move(other.prev_cost_dual)),
          prev_res_primal(std::move(other.prev_res_primal)),
          prev_res_dual(std::move(other.prev_res_dual)),
          prev_gap_abs(std::move(other.prev_gap_abs)),
          prev_gap_rel(std::move(other.prev_gap_rel)),
          best_cost_primal(std::move(other.best_cost_primal)),
          best_cost_dual(std::move(other.best_cost_dual)),
          best_res_primal(std::move(other.best_res_primal)),
          best_res_dual(std::move(other.best_res_dual)),
          best_gap_abs(std::move(other.best_gap_abs)),
          best_gap_rel(std::move(other.best_gap_rel)),
          best_ktratio(std::move(other.best_ktratio)),
          d_best_saved(other.d_best_saved),
          construction_time(other.construction_time),
          setup_time(other.setup_time),
          solve_time(other.solve_time),
          status(std::move(other.status)),
          status_device(other.status_device),
          d_any_done(other.d_any_done),
          h_any_done_pinned(other.h_any_done_pinned),
          batchSize(other.batchSize)
    {
        other.status_device = nullptr;
        other.d_any_done = nullptr;
        other.h_any_done_pinned = nullptr;
        other.d_iterations_per_batch = nullptr;
        other.d_best_saved = nullptr;
    }

    Info& operator=(Info&& other) noexcept {
        if (this != &other) {
            if (status_device) cudaFree(status_device);
            if (h_any_done_pinned) cudaFreeHost(h_any_done_pinned);
            if (d_iterations_per_batch) cudaFree(d_iterations_per_batch);
            if (d_best_saved) cudaFree(d_best_saved);

            mu = std::move(other.mu);
            sigma = std::move(other.sigma);
            step_length = std::move(other.step_length);
            iterations = other.iterations;
            iterations_per_batch = std::move(other.iterations_per_batch);
            d_iterations_per_batch = other.d_iterations_per_batch;
            cost_primal = std::move(other.cost_primal);
            cost_dual = std::move(other.cost_dual);
            res_primal = std::move(other.res_primal);
            res_dual = std::move(other.res_dual);
            res_primal_inf = std::move(other.res_primal_inf);
            res_dual_inf = std::move(other.res_dual_inf);
            gap_abs = std::move(other.gap_abs);
            gap_rel = std::move(other.gap_rel);
            ktratio = std::move(other.ktratio);
            prev_cost_primal = std::move(other.prev_cost_primal);
            prev_cost_dual = std::move(other.prev_cost_dual);
            prev_res_primal = std::move(other.prev_res_primal);
            prev_res_dual = std::move(other.prev_res_dual);
            prev_gap_abs = std::move(other.prev_gap_abs);
            prev_gap_rel = std::move(other.prev_gap_rel);
            best_cost_primal = std::move(other.best_cost_primal);
            best_cost_dual = std::move(other.best_cost_dual);
            best_res_primal = std::move(other.best_res_primal);
            best_res_dual = std::move(other.best_res_dual);
            best_gap_abs = std::move(other.best_gap_abs);
            best_gap_rel = std::move(other.best_gap_rel);
            best_ktratio = std::move(other.best_ktratio);
            d_best_saved = other.d_best_saved;
            construction_time = other.construction_time;
            setup_time = other.setup_time;
            solve_time = other.solve_time;
            status = std::move(other.status);
            status_device = other.status_device;
            d_any_done = other.d_any_done;
            h_any_done_pinned = other.h_any_done_pinned;
            batchSize = other.batchSize;

            other.status_device = nullptr;
            other.d_any_done = nullptr;
            other.h_any_done_pinned = nullptr;
            other.d_iterations_per_batch = nullptr;
            other.d_best_saved = nullptr;
        }
        return *this;
    }

    ~Info() {
        if (status_device) {
            cudaFree(status_device);
        }
        if (h_any_done_pinned) {
            // d_any_done is a device-mapped alias of h_any_done_pinned, no separate free needed
            cudaFreeHost(h_any_done_pinned);
        }
        if (d_iterations_per_batch) {
            cudaFree(d_iterations_per_batch);
        }
        if (d_best_saved) {
            cudaFree(d_best_saved);
        }
    }

    /**
     * @brief Update solver information based on current iteration
     *
     * Computes convergence metrics, objective values, and residual norms
     * based on current variables and residuals.
     *
     * @param data Problem data
     * @param variables Current iterate
     * @param residuals Current residuals
     * @param stream CUDA stream for async operations
     */
    void update(const SolverData& data, const Variables& variables, const Residuals& residuals, cudaStream_t stream = 0);

    /**
     * @brief Save scalar iteration values
     *
     * @param mu_vals Complementarity measure values [batchSize]
     * @param alpha_vals Step length values [batchSize]
     * @param sigma_vals Centering parameter values [batchSize]
     * @param iter Iteration number
     * @param stream CUDA stream for async operations
     */
    void save_scalars(const double* mu_vals, const double* alpha_vals, const double* sigma_vals, int32_t iter, cudaStream_t stream = 0);

    /**
     * @brief Print solver configuration
     *
     * Prints problem dimensions, cone structure, and solver settings.
     *
     * @param settings Solver settings
     * @param data Problem data
     * @param cones Cone structure
     * @param actualKktSolverType The actual KKT solver type being used (after any fallbacks)
     */
    void print_configuration(const Settings& settings, const SolverData& data, const Cones& cones, KKTSolverType actualKktSolverType);

    /**
     * @brief Print status header line
     *
     * Prints the header for iteration status output.
     *
     * @param settings Solver settings (checks verbose flag)
     */
    void print_status_header(const Settings& settings);

    /**
     * @brief Print iteration status line
     *
     * Prints a single line of iteration statistics including:
     * iter, cost_primal, cost_dual, gap, res_primal, res_dual, ktratio, mu, step_length
     *
     * @param settings Solver settings (checks verbose flag)
     * @param stream CUDA stream for async operations
     */
    void print_status(const Settings& settings, cudaStream_t stream = 0);

    /**
     * @brief Print solver footer with termination status
     *
     * Prints final status and solve time for all batches.
     *
     * @param settings Solver settings (checks verbose flag)
     * @param stream CUDA stream for async operations
     */
    void print_footer(const Settings& settings, cudaStream_t stream = 0);

    /**
     * @brief Check convergence status (optimality or infeasibility)
     *
     * Updates the solver status based on convergence criteria.
     * Checks for: Solved, PrimalInfeasible, DualInfeasible, and reduced accuracy variants.
     *
     * @param residuals Current residuals
     * @param settings Solver settings (tolerances)
     * @param stream CUDA stream for async operations
     */
    void check_convergence(const Residuals& residuals, const Settings& settings, cudaStream_t stream = 0);

    /**
     * @brief Check termination conditions (synchronous)
     *
     * Checks all termination criteria: convergence, insufficient progress, max iterations, time limit.
     * Returns true if all batches have terminated (status != Unsolved).
     *
     * @param residuals Current residuals
     * @param settings Solver settings
     * @param iter Current iteration number
     * @param solve_time_seconds Elapsed solve time
     * @param stream CUDA stream for async operations
     * @return true if all batches have terminated, false otherwise
     */
    bool are_all_done(const Residuals& residuals, const Settings& settings, int64_t iter, double solve_time_seconds, cudaStream_t stream = 0);

    /**
     * @brief Launch async termination check (no sync)
     *
     * Launches the termination check kernel but does NOT synchronize.
     * Call wait_for_termination() to block until complete.
     */
    void check_termination_async(const Residuals& residuals, const Settings& settings, int64_t iter, double solve_time_seconds, cudaStream_t stream = 0);

    /**
     * @brief Wait for termination check to complete and return result
     *
     * Synchronizes the stream and returns the termination result.
     *
     * @param stream CUDA stream to synchronize
     * @return true if all batches have terminated, false otherwise
     */
    bool wait_for_termination(cudaStream_t stream = 0);

    /**
     * @brief Reset termination counter for next iteration
     *
     * Resets h_any_done_pinned to 0 for the next termination check.
     */
    void reset_termination_counter() {
        *h_any_done_pinned = 0;
    }

    /**
     * @brief Post-process info after solver termination
     *
     * Checks for "almost" convergence when solver terminates without full convergence.
     * If status is MaxIterations, MaxTime, or an error status, checks if problem
     * meets reduced accuracy criteria.
     *
     * @param residuals Current residuals
     * @param settings Solver settings
     * @param stream CUDA stream for async operations
     */
    void post_process(const Residuals& residuals, const Settings& settings, cudaStream_t stream = 0);

    /**
     * @brief Sync status from device to host
     *
     * Copies the status array from GPU to CPU.
     *
     * @param stream CUDA stream for async operations
     */
    void sync_status_to_host(cudaStream_t stream = 0);

    /**
     * @brief Finalize solver info with solve time
     *
     * @param solve_time_seconds Total solve time in seconds
     */
    void finalize(double solve_time_seconds);

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept {
        return mu.memoryUsage() + sigma.memoryUsage() + step_length.memoryUsage() +
               cost_primal.memoryUsage() + cost_dual.memoryUsage() +
               res_primal.memoryUsage() + res_dual.memoryUsage() +
               res_primal_inf.memoryUsage() + res_dual_inf.memoryUsage() +
               gap_abs.memoryUsage() + gap_rel.memoryUsage() + ktratio.memoryUsage() +
               prev_cost_primal.memoryUsage() + prev_cost_dual.memoryUsage() +
               prev_res_primal.memoryUsage() + prev_res_dual.memoryUsage() +
               prev_gap_abs.memoryUsage() + prev_gap_rel.memoryUsage();
    }
};

} // namespace moreau
