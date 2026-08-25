/**
 * @file settings.hpp
 * @brief Solver settings and configuration
 *
 * This module defines all settings structures for the solver,
 * including solver parameters and tolerances.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include "moreau/equilibration/equilibration.hpp"

#define MOREAU_EPSILON std::numeric_limits<double>::epsilon()
#define MOREAU_INFINITY std::numeric_limits<double>::infinity()

namespace moreau {

/**
 * @brief Solver algorithm type
 *
 * - IPM: Interior Point Method (supports all cones)
 */
enum class SolverType {
    IPM = 0,
};

/**
 * @brief KKT linear system solver type
 *
 * Selects which algorithm to use for solving the KKT linear systems
 * that arise in each IPM iteration.
 */
enum class KKTSolverType {
    Auto = 0,       // Auto-detect: Riccati for block-tridiagonal, Woodbury for diagonal P + low-rank A, CuDSS otherwise
    CuDSS,          // Sparse LDL via cuDSS (works for all problems)
    Riccati,         // Block-tridiagonal Cholesky (MPC/MHE problems only)
    Woodbury         // Woodbury identity (diagonal P + low-rank/sparse A, e.g. portfolio)
};

/**
 * @brief cuDSS batching strategy for batched KKT solves
 *
 * Uses cuDSS uniform batching (UBATCH) to solve all batch problems in parallel.
 */
enum class CuDSSStrategy {
    Auto = 0,       // Use UBatch (uniform batching)
    UBatch          // Uniform batch mode: single cuDSS handle, SYMMETRIC matrix
};

/**
 * @brief Differentiation method selection
 *
 * Controls how gradients are computed in the backward pass.
 * - Auto: Uses Exact (default)
 * - Exact: Standard DΠ_K* derivative (discontinuous at cone boundaries)
 * - Smoothed: Central-path smoothing for continuous gradients
 */
enum class DiffMethod {
    Auto = 0,
    Exact,
    Smoothed
};

/**
 * @brief Interior Point Method specific settings
 *
 * These settings control the behavior of the IPM solver algorithm,
 * including convergence tolerances, equilibration, and linear solver options.
 */
struct IPMSettings {
    // maximum interior point step length
    double maxStepFraction = 0.99;

    // absolute duality gap tolerance
    double tolGapAbs = 1e-8;

    // relative duality gap tolerance
    double tolGapRel = 1e-8;

    // feasibility check tolerance (primal and dual)
    double tolFeas = 1e-8;

    // absolute infeasibility tolerance (primal and dual)
    double tolInfeasAbs = 1e-8;

    // relative infeasibility tolerance (primal and dual)
    double tolInfeasRel = 1e-8;

    // kappa/tau tolerance
    double tolKtRatio = 1e-6;

    // reduced absolute duality gap tolerance
    double reducedTolGapAbs = 5e-5;

    // reduced relative duality gap tolerance
    double reducedTolGapRel = 5e-5;

    // reduced feasibility check tolerance (primal and dual)
    double reducedTolFeas = 1e-4;

    // reduced absolute infeasibility tolerance (primal and dual)
    double reducedTolInfeasAbs = 5e-12;

    // reduced relative infeasibility tolerance (primal and dual)
    double reducedTolInfeasRel = 5e-5;

    // reduced kappa/tau tolerance
    double reducedTolKtRatio = 1e-4;

    // equilibration settings
    EquilibrationSettings equilibrationSettings;

    // line search backtracking
    double linesearchBacktrackStep = 0.8;

    // minimum step size allowed for asymmetric cones with PrimalDual scaling
    double minSwitchStepLength = 1e-1;

    // minimum step size allowed for symmetric cones & asymmetric cones with Dual scaling
    double minTerminateStepLength = 1e-4;

    // enable KKT static regularization
    bool staticRegularizationEnable = true;

    // KKT static regularization parameter
    double staticRegularizationConstant = 1e-8;

    // additional regularization parameter w.r.t. the maximum abs diagonal term
    double staticRegularizationProportional = MOREAU_EPSILON * MOREAU_EPSILON;

    // enable KKT dynamic regularization
    bool dynamicRegularizationEnable = true;

    // KKT dynamic regularization threshold
    double dynamicRegularizationEps = 1e-13;

    // KKT dynamic regularization shift
    double dynamicRegularizationDelta = 2e-7;

    // KKT linear solver type
    KKTSolverType kktSolverType = KKTSolverType::Auto;

    // cuDSS batching strategy (only applies when kktSolverType is CuDSS)
    CuDSSStrategy cudssStrategy = CuDSSStrategy::Auto;

    // Differentiation method (Auto resolves to Exact)
    DiffMethod diffMethod = DiffMethod::Auto;

    // Target smoothing parameter for smoothed differentiation
    double diffSmoothingMu = 1e-4;

    // Step factor for central-path walk-up (used directly as per-iteration multiplier)
    double diffSmoothingStepFactor = 30.0;

    // cuDSS max LU nonzeros: upper limit on fill-in during factorization.
    // Default -1 means cuDSS uses 100*nnz which can OOM for dense PSD problems.
    // Set to a smaller value (e.g. 0 to disable the limit, or a specific byte
    // budget based on available GPU memory) to prevent OOM.
    // Only relevant for non-symmetric matrices with ALG_1/ALG_2 reordering.
    int64_t maxLuNnz = -1;

    // cuDSS iterative refinement steps (only applies when kktSolverType is CuDSS)
    int cudssIrSteps = 2;

    // cuDSS pivoting (only applies when kktSolverType is CuDSS)
    // false = CUDSS_PIVOT_NONE (quasi-definite, matching Clarabel)
    // true  = pivoting on (for numerically challenging problems)
    bool cudssPivotEnable = false;

    // Chordal decomposition merge strategy for sparse PSD cones.
    // Must match the CPU default to keep CPU/CUDA on the same reformulation
    // for the same problem. Accepted values:
    //   "clique_graph"  — Garstka et al. 2019 (default; also CPU default)
    //   "parent_child"  — fill_in-thresholded parent-child merging
    //   "none"          — no merging beyond the symbolic supernode tree
    std::string chordalDecompositionMergeMethod = "clique_graph";
};

/**
 * @brief Main solver settings with nested IPM settings
 *
 * Contains common solver settings (max_iter, time_limit, verbose)
 * plus nested IPM-specific settings in the `ipm` field.
 */
struct Settings {
    // CUDA device ID (-1 = use current device)
    int deviceId = -1;

    // Solver algorithm type (currently only IPM supported)
    SolverType solver = SolverType::IPM;

    // maximum number of iterations
    uint32_t maxIter = 200;

    // maximum run time (seconds)
    double timeLimit = MOREAU_INFINITY;

    // verbose printing
    //
    // PERFORMANCE NOTE: When verbose=true, the solver must synchronize CPU/GPU
    // every iteration to copy residual values for printing. This adds latency
    // that can significantly impact performance for small-to-medium problems.
    // Set verbose=false for production/benchmarking to avoid this overhead.
    bool verbose = true;

    // Enable gradient computation (backward pass)
    // When true, the solver owns a DiffState and automatically caches the
    // solution after each solve() for use in backward differentiation.
    bool enableGrad = false;

    // YOLO mode: run exactly yoloNumIters iterations with zero GPU-host sync.
    // Skips all convergence checking. Returns MaxIterations status for all batches.
    // Incompatible with enableGrad (backward pass needs convergence data).
    bool yolo = false;
    uint32_t yoloNumIters = 15;

    // IPM-specific settings (nested)
    IPMSettings ipm;
};

} // namespace moreau
