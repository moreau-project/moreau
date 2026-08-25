/**
 * @file solver.cpp
 * @brief Implementation of the main solver
 */

#include "moreau/solver/solver.hpp"
#include "moreau/solver/solver_kernels.cuh"
#include "moreau/equilibration/equilibration.hpp"
#include "moreau/equilibration/equilibration_kernels.cuh"
#include "moreau/cones/cone_kernels.cuh"
#include "moreau/cones/psd_kernels.cuh"
#include "moreau/vector/vector_kernels.cuh"
#include "moreau/variables/variables_kernels.cuh"
#include "moreau/kkt/kkt_kernels.cuh"
#include "moreau/diff/diff_kernels.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include "moreau/debug.hpp"

namespace moreau {

// Version string
constexpr const char* VERSION = "0.3.3";

#ifdef MOREAU_DEBUG
// Full per-iteration debug: prints every element of every vector (matches CPU format)
// Enable at runtime with MOREAU_ITER_DEBUG=1 (requires compile with -DMOREAU_DEBUG)
static bool iterDebugEnabled() {
    static std::once_flag flag;
    static bool enabled = false;
    std::call_once(flag, []() {
        const char* env = std::getenv("MOREAU_ITER_DEBUG");
        enabled = (env && std::string(env) != "0");
    });
    return enabled;
}

// Print all elements of a device vector to stderr (matches CPU eprintln format)
static void debugPrintVec(const char* name, const double* d_data, int64_t len, int64_t batchSize, cudaStream_t stream, int64_t batch_idx = 0) {
    cudaStreamSynchronize(stream);
    std::vector<double> h(len * batchSize);
    cudaMemcpy(h.data(), d_data, sizeof(double) * len * batchSize, cudaMemcpyDeviceToHost);
    std::cerr << "GPU: " << name << "=[";
    for (int64_t i = 0; i < len; ++i) {
        if (i > 0) std::cerr << ", ";
        char buf[32];
        snprintf(buf, sizeof(buf), "%.16e", h[batch_idx * len + i]);
        std::cerr << buf;
    }
    std::cerr << "]\n";
}

// Print a scalar from device memory
static void debugPrintScalar(const char* name, const double* d_data, cudaStream_t stream, int64_t batch_idx = 0) {
    double val;
    cudaMemcpy(&val, d_data + batch_idx, sizeof(double), cudaMemcpyDeviceToHost);
    char buf[64];
    snprintf(buf, sizeof(buf), "GPU: %s=%.16e", name, val);
    std::cerr << buf << "\n";
}
#endif // MOREAU_DEBUG


// Banner is now printed in Info::print_configuration()

// ============================================================================
// Three-step API implementation
// ============================================================================

// Step 2: Set P and A matrix values
void CompiledSolver::setup(
    const double* d_P_values,
    const double* d_A_values,
    cudaStream_t stream
) {
    // Ensure we're on the correct device for multi-GPU support
    if (device_id_ >= 0) {
        cudaSetDevice(device_id_);
    }

    auto setup_start = std::chrono::high_resolution_clock::now();

    // Copy P and A values to internal structures
    cudaMemcpyAsync(data.P.values(), d_P_values,
                    sizeof(double) * data.P.nnz() * data.batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(data.A.values(), d_A_values,
                    sizeof(double) * data.A.nnz() * data.batchSize,
                    cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);

    auto setup_end = std::chrono::high_resolution_clock::now();
    info.setup_time = std::chrono::duration<double>(setup_end - setup_start).count();

    is_setup_ = true;
    matrices_equilibrated_ = false;  // Reset so first solve() does full equilibration
}

// Load all problem data and equilibrate without solving (for backward pass)
void CompiledSolver::loadDataForBackward(
    const double* d_P_values,
    const double* d_A_values,
    const double* d_q,
    const double* d_b,
    cudaStream_t stream
) {
    if (device_id_ >= 0) {
        cudaSetDevice(device_id_);
    }

    // Copy P and A values
    cudaMemcpyAsync(data.P.values(), d_P_values,
                    sizeof(double) * data.P.nnz() * data.batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(data.A.values(), d_A_values,
                    sizeof(double) * data.A.nnz() * data.batchSize,
                    cudaMemcpyDeviceToDevice, stream);

    // Copy q and b
    copyPerSolveData(d_q, d_b, stream);

    // Sanitize inf entries in b before equilibration (matches solve path)
    sanitizeInfB(stream);

    // Equilibrate all data (P, A, q, b)
    matrices_equilibrated_ = false;
    if (settings.ipm.equilibrationSettings.enable) {
        doEquilibration(stream);
    }

    cudaStreamSynchronize(stream);
}

// Step 3: Solve with pre-set P/A matrices
void CompiledSolver::solve(
    const double* d_q,
    const double* d_b,
    cudaStream_t stream
) {
    if (!is_setup_) {
        throw std::runtime_error("setup() must be called before solve()");
    }

    // Copy q and b, then run solver
    solveInternalWithSetup(d_q, d_b, stream);
}

// Step 3 (warm start): Solve with pre-set P/A matrices and warm start
void CompiledSolver::solve(
    const double* d_q,
    const double* d_b,
    const double* d_warm_x,
    const double* d_warm_z,
    const double* d_warm_s,
    cudaStream_t stream,
    const double* d_warm_z_x
) {
    if (!is_setup_) {
        throw std::runtime_error("setup() must be called before solve()");
    }

    solveInternalWithWarmStart(d_q, d_b, d_warm_x, d_warm_z, d_warm_s, d_warm_z_x, stream);
}

// Convenience: setup + solve in one call
void CompiledSolver::solveAll(
    const double* d_P_values,
    const double* d_A_values,
    const double* d_q,
    const double* d_b,
    cudaStream_t stream
) {
    // Ensure we're on the correct device for multi-GPU support
    if (device_id_ >= 0) {
        cudaSetDevice(device_id_);
    }

    // Set up P and A
    setup(d_P_values, d_A_values, stream);

    // Solve
    solveInternalWithSetup(d_q, d_b, stream);
}

// Internal solve implementation for three-step API (P/A already set via setup())
void CompiledSolver::solveInternalWithSetup(
    const double* d_q,
    const double* d_b,
    cudaStream_t stream
) {

    // Ensure we're on the correct device for multi-GPU support
    if (device_id_ >= 0) {
        cudaSetDevice(device_id_);
    }

    // CRITICAL: Reset all solver state to prevent pollution from previous solves
    resetState(stream);

    // Print banner, configuration, and status header
    info.print_configuration(settings, data, data.cones, kkt->actualSolverType());
    info.print_status_header(settings);

    // Copy only q and b (P and A already set via setup())
    copyPerSolveData(d_q, d_b, stream);

    // Debug: print q values per batch
    if (isDebugEnabled() && data.batchSize > 1) {
        std::vector<double> q_host(data.n * data.batchSize);
        cudaMemcpy(q_host.data(), data.q.data(), sizeof(double) * data.n * data.batchSize, cudaMemcpyDeviceToHost);
        std::cout << "[DEBUG] q after copyPerSolveData:\n";
        for (int64_t b = 0; b < data.batchSize; b++) {
            std::cout << "  batch " << b << ": q = [";
            for (int64_t i = 0; i < std::min((int64_t)3, data.n); i++) {
                std::cout << q_host[b * data.n + i] << (i < std::min((int64_t)2, data.n - 1) ? ", " : "");
            }
            std::cout << "...]\n";
        }
    }

    // Sanitize inf entries in b before equilibration.
    sanitizeInfB(stream);


    // Step 1: Equilibration
    if (settings.ipm.equilibrationSettings.enable) {
        doEquilibration(stream);
    }

    // Initial KKT matrix population with P and A values
    kkt->populate(data.P, data.A, stream);

    // Direct-x cones: snapshot P's contribution at x-cone KKT slots so
    // per-iteration refresh_xcone_hs can keep P static and only update
    // the Hs delta. No-op when there are no x-cones. Must run AFTER
    // populate() (so the slots hold P's values) and BEFORE default_start
    // (which calls refresh_xcone_hs using this baseline).
    kkt->init_xcone_px_baseline(data.cones, stream);

    // Compute cached norms for residual computation
    computeCachedNorms(stream);

    // Perform default start
    default_start(stream);

    // Run IPM loop
    runIPMLoop(stream);

    // Override status for batches detected as infeasible by sanitizeInfB
    overrideInfeasibleStatus();

    // Auto-cache solution for backward if DiffState is owned
    if (diff_state_) {
        cache_solution_for_backward(*diff_state_, *this, stream);
    }
}

// Internal solve implementation with warm start
void CompiledSolver::solveInternalWithWarmStart(
    const double* d_q,
    const double* d_b,
    const double* d_warm_x,
    const double* d_warm_z,
    const double* d_warm_s,
    const double* d_warm_z_x,
    cudaStream_t stream
) {

    // Ensure we're on the correct device for multi-GPU support
    if (device_id_ >= 0) {
        cudaSetDevice(device_id_);
    }

    // Reset all solver state
    resetState(stream);

    // Print banner, configuration, and status header
    info.print_configuration(settings, data, data.cones, kkt->actualSolverType());
    info.print_status_header(settings);

    // Copy only q and b (P and A already set via setup())
    copyPerSolveData(d_q, d_b, stream);

    // Sanitize inf entries in b for nonneg cone rows before equilibration.
    sanitizeInfB(stream);

    // Step 1: Equilibration
    if (settings.ipm.equilibrationSettings.enable) {
        doEquilibration(stream);
    }

    // Initial KKT matrix population with P and A values
    kkt->populate(data.P, data.A, stream);

    // Direct-x cones: snapshot P's contribution at x-cone KKT slots so
    // per-iteration refresh_xcone_hs can keep P static and only update
    // the Hs delta. No-op when there are no x-cones.
    kkt->init_xcone_px_baseline(data.cones, stream);

    // Compute cached norms for residual computation
    computeCachedNorms(stream);

    // Warm start instead of default_start
    warmStart(d_warm_x, d_warm_z, d_warm_s, d_warm_z_x, stream);

    // Run IPM loop
    runIPMLoop(stream);

    // Override status for batches detected as infeasible by sanitizeInfB
    overrideInfeasibleStatus();

    // Auto-cache solution for backward if DiffState is owned
    if (diff_state_) {
        cache_solution_for_backward(*diff_state_, *this, stream);
    }
}

void CompiledSolver::warmStart(
    const double* d_warm_x,
    const double* d_warm_z,
    const double* d_warm_s,
    const double* d_warm_z_x,
    cudaStream_t stream
) {
    // Follows the algorithm from arXiv:2512.00693 and Clarabel.jl
    // yc/warmstart branch: smoothing is done directly in equilibrated
    // space (valid because rectify_equilibration ensures uniform
    // scaling within each cone block).
    int64_t n = data.n;
    int64_t m = data.m;
    int64_t batchSize = data.batchSize;
    const int64_t total_xn = variables.totalXConeNumel();

    // Step 1: Copy warm values and scale into equilibrated space
    // so we can compute residuals/info for the warmness ratio.
    cudaMemcpyAsync(variables.x.data(), d_warm_x,
                    sizeof(double) * n * batchSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(variables.z.data(), d_warm_z,
                    sizeof(double) * m * batchSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(variables.s.data(), d_warm_s,
                    sizeof(double) * m * batchSize, cudaMemcpyDeviceToDevice, stream);
    variables.τ.setToConstant(1.0, stream);
    variables.κ.setToConstant(1.0, stream);

    fused_warm_start_scaling(variables.x, variables.z, variables.s,
                             data.equilibration.dinv, data.equilibration.einv,
                             data.equilibration.e, data.equilibration.c, stream);

    // Direct-x: convert user-frame z_x to the equilibrated frame the IPM
    // operates in. `z_x_eq[b,k] = z_x_user[b,k] * c[b] / d[J[k]]`. Inverse
    // of the user-facing unscale. When warm_z_x is omitted we must fall
    // back to the same unit-init point `default_start` uses — otherwise
    // z_x carries stale values from a prior solve (resetState clears only
    // status flags, not variable data), producing wildly wrong directions
    // on iter 0.
    if (total_xn > 0) {
        if (d_warm_z_x != nullptr) {
            equilibrate_z_x(
                variables.z_x.data(), d_warm_z_x,
                data.equilibration.dinv.data(),
                data.equilibration.c.data(),
                data.cones.d_xcone_indices,
                n, total_xn, batchSize, stream);
        } else {
            // No warm z_x supplied: cold-init the whole direct-x cone
            // block, primal x[J] included. Resetting only z_x would pair
            // a default dual with a boundary-valued warm x (e.g. an
            // optimal x from a prior solve), starting the IPM on the cone
            // face — from which it cannot converge.
            init_xcone_start_point(stream);
        }
    }

    // Step 2: Compute residuals and info at the scaled warm point.
    residuals.update(variables, data, cusparse_handle_, cublas_handle_, stream);
    info.update(data, variables, residuals, stream);

    // Step 3: Compute warmness mu = max(rp, rd, min(ga, gr)), floor 1e-6.
    compute_warmness_mu(
        mu.data(),
        info.res_primal.data(),
        info.res_dual.data(),
        info.gap_abs.data(),
        info.gap_rel.data(),
        batchSize,
        stream
    );

    // Step 4: Set τ=1, κ=mu
    variables.τ.setToConstant(1.0, stream);
    cudaMemcpyAsync(variables.κ.data(), mu.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);

    // Step 5: Smooth z_eq, s_eq directly in equilibrated space.
    waxpby(warm_work, 1.0, variables.z, -1.0, variables.s, stream);
    data.cones.smoothing(variables.z, warm_work, mu, stream);
    waxpby(variables.s, 1.0, variables.z, -1.0, warm_work, stream);

    // Direct-x dual: we use the user-supplied `warm_z_x` as-is (after the
    // equilibration scaling at step 1). A naïve central-path projection
    // `z_x = -μ·∇F(x)` would move *away* from the user's z_x for warm
    // points near optimal — there ∇F(x) blows up at the boundary while
    // the user's z_x is small, so the projection produces a much worse
    // initial dual than the user supplied. The slack smoothing avoids
    // this trap because it preserves the (z − s) gap; the direct-x form
    // has no analogous preservation invariant, so the safest default is
    // to trust the user's z_x and let the IPM correct it on iter 0.
}

void CompiledSolver::init_xcone_start_point(cudaStream_t stream) {
    // Cold interior start for the direct-x cone block. Extracted from the
    // two (formerly duplicated) default_start branches so the warmStart
    // no-warm-z_x fallback can reuse it: when a warm start omits z_x, the
    // whole cone block must be cold-started — a default z_x paired with a
    // boundary-valued warm x leaves the IPM stuck on the cone face.
    if (data.cones.numXCones == 0) {
        return;
    }
    const double total_degree = static_cast<double>(data.cones.degree());
    init_xcone_z_x(
        variables.z_x.data(),
        data.cones.d_xcone_kinds,
        data.cones.d_xcone_dims,
        data.cones.d_xcone_numel_offsets,
        data.batchSize, data.cones.numXCones,
        data.cones.totalXConeNumel, stream);
    shift_x_into_nonneg_interior(
        variables.x.data(),
        data.cones.d_xcone_kinds,
        data.cones.d_xcone_dims,
        data.cones.d_xcone_numel_offsets,
        data.cones.d_xcone_indices,
        data.batchSize, data.n, data.cones.numXCones, stream);
    shift_x_into_soc_interior(
        variables.x.data(),
        data.cones.d_xcone_kinds,
        data.cones.d_xcone_dims,
        data.cones.d_xcone_numel_offsets,
        data.cones.d_xcone_indices,
        data.batchSize, data.n, data.cones.numXCones, stream);
    if (data.cones.numXExpCones > 0) {
        init_xcone_x_exp(
            variables.x.data(),
            data.cones.d_xcone_kinds,
            data.cones.d_xcone_dims,
            data.cones.d_xcone_numel_offsets,
            data.cones.d_xcone_indices,
            data.batchSize, data.n, data.cones.numXCones, stream);
    }
    if (data.cones.numXPowerCones > 0) {
        // Initialize x[J] for Power cones to (sqrt(1+α), sqrt(2-α), 0).
        init_xcone_x_pow(
            variables.x.data(),
            data.cones.d_xcone_kinds,
            data.cones.d_xcone_dims,
            data.cones.d_xcone_numel_offsets,
            data.cones.d_xcone_indices,
            data.cones.d_xcone_pow_idx, data.cones.d_xcone_pow_alpha,
            data.batchSize, data.n, data.cones.numXCones, stream);
        // Fix z_x for Power cones: init_xcone_z_x wrote the wrong Exp
        // unit point for kind==4; overwrite with the correct Power point.
        init_xcone_z_x_pow(
            variables.z_x.data(),
            data.cones.d_xcone_kinds,
            data.cones.d_xcone_dims,
            data.cones.d_xcone_numel_offsets,
            data.cones.d_xcone_pow_idx, data.cones.d_xcone_pow_alpha,
            data.batchSize, data.cones.numXCones,
            data.cones.totalXConeNumel, stream);
    }
    if (data.cones.numXPsdCones > 0) {
        shift_x_into_psd_interior(
            data.cones, variables.x.data(), data.n,
            total_degree, stream);
    }
    if (data.cones.numXGenPowerCones > 0) {
        init_xcone_x_genpow(
            variables.x.data(),
            data.cones.d_xcone_kinds,
            data.cones.d_xcone_dims,
            data.cones.d_xcone_numel_offsets,
            data.cones.d_xcone_indices,
            data.cones.d_xcone_genpow_idx,
            data.cones.d_xcone_genpow_dim1s,
            data.cones.d_xcone_genpow_alpha_offsets,
            data.cones.d_xcone_genpow_alphas,
            data.batchSize, data.n, data.cones.numXCones, stream);
        init_xcone_z_x_genpow(
            variables.z_x.data(),
            data.cones.d_xcone_kinds,
            data.cones.d_xcone_dims,
            data.cones.d_xcone_numel_offsets,
            data.cones.d_xcone_genpow_idx,
            data.cones.d_xcone_genpow_dim1s,
            data.cones.d_xcone_genpow_alpha_offsets,
            data.cones.d_xcone_genpow_alphas,
            data.batchSize, data.cones.numXCones,
            data.cones.totalXConeNumel, stream);
    }
}


void CompiledSolver::default_start(
    cudaStream_t stream
) {
    // Initialize all variables (x, s, z, τ, κ) to default starting values
    variables.initialize(stream);

    if (data.cones.isSymmetric()) {
        // Step 1: Set all scalings to identity (or zero for the zero cone)
        data.cones.set_identity_scaling(stream);

        // Direct-x identity scaling: seed each cone's Hs block with the
        // correct identity Hs (nonneg diag=1; dense SOC packed-tri
        // identity; rank-2 sparse SOC diag=[0.5, 1, 1, ...]). Primes the
        // KKT (1,1) slots with a well-conditioned initial factor so the
        // first linear solve produces a reasonable initial iterate.
        // setToConstant(1.0) is WRONG for dense SOC: all-ones packed tri
        // decodes to a rank-1 all-ones matrix, blowing up initial μ by
        // ~200× and costing 10+ extra IPM iterations at dim ≤ 4.
        // eta MUST be set to 1 so refresh_xcone_sparse_expansion doesn't
        // write zero to the expansion-col diag (would be singular).
        if (data.cones.numXCones > 0) {
            seed_xcone_Hs_identity(
                data.cones.xcone_Hs.data(),
                data.cones.d_xcone_kinds,
                data.cones.d_xcone_dims,
                data.cones.d_xcone_hs_offsets,
                data.cones.xcone_Hs.batchSize(),
                data.cones.numXCones,
                data.cones.totalXConeHsEntries,
                stream);
            data.cones.xcone_w.setToConstant(1.0, stream);
            data.cones.xcone_lambda.setToConstant(1.0, stream);
            data.cones.xcone_eta.setToConstant(1.0, stream);
        }

        // Step 2: Update KKT system with identity scaling
        // We've already set identity scaling on cones, now update the H block
        // and refactor the KKT system
        kkt->update_H(data.cones, nullptr, stream);
        // Blend the identity x-cone Hs into KKT.values on top of
        // init_xcone_px_baseline-snapshotted P values.
        if (data.cones.numXCones > 0) {
            kkt->refresh_xcone_hs(data.cones, stream);
        }

        // Refactor with regularization
        bool is_kkt_refactor_success = kkt->regularize_and_refactor(
            settings.ipm.staticRegularizationEnable,
            settings.ipm.staticRegularizationConstant,
            settings.ipm.staticRegularizationProportional,
            stream
        );
        if (!is_kkt_refactor_success) {
            if (settings.verbose) {
                std::cout << "KKT factorization failed during default start (symmetric cones)" << std::endl;
                std::cout << "This likely indicates the problem is poorly conditioned or infeasible." << std::endl;
            }
            throw std::runtime_error("KKT factorization failed during default start");
        }

        // Step 3: Solve for primal/dual initial points via KKT
        // The approach depends on whether this is an LP (P.nnz() == 0) or QP

        if (data.P.nnz() == 0) {
            // LP initialization (following Clarabel's solve_initial_point)
            // Solve 1: RHS = [0; b] to get (x, -s), then negate s
            neg_q.setToConstant(0.0, stream);  // workx = 0

            pack_interleaved_kernel(default_rhs.data(), neg_q.data(), data.b.data(),
                                   data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);
            kkt->solve(default_rhs.data(), default_sol.data(), stream);
            unpack_interleaved_kernel(variables.x.data(), variables.s.data(), default_sol.data(),
                                     data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

            // s = -s (negate)
            waxpby(variables.s, -1.0, variables.s, 0.0, variables.s, stream);

            // Solve 2: RHS = [-q; 0] to get z
            waxpby(neg_q, -1.0, data.q, 0.0, neg_q, stream);  // neg_q = -q
            // workz = 0 (for the m-part of RHS) - use preallocated workz_zero
            workz_zero.setToConstant(0.0, stream);

            pack_interleaved_kernel(default_rhs.data(), neg_q.data(), workz_zero.data(),
                                   data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);
            kkt->solve(default_rhs.data(), default_sol.data(), stream);
            // Only extract z (we ignore the x part)
            unpack_interleaved_kernel(neg_q.data(), variables.z.data(), default_sol.data(),
                                     data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

            if (isDebugEnabled()) {
                std::vector<double> z_host(data.m * data.batchSize);
                cudaMemcpy(z_host.data(), variables.z.data(), sizeof(double) * data.m * data.batchSize, cudaMemcpyDeviceToHost);
                std::cout << "LP init solve 2: z after solve = [";
                for (int i = 0; i < data.m; i++) std::cout << z_host[i] << (i < data.m-1 ? ", " : "");
                std::cout << "]" << std::endl;
            }
        } else {
            // QP initialization
            // Solve: [P + schur A'; A -H] [x; z] = [-q; b]
            // This gives us initial (x, z) that satisfy KKT conditions

            neg_q.setToConstant(0.0, stream);
            waxpby(neg_q, -1.0, data.q, 0.0, neg_q, stream);

            pack_interleaved_kernel(default_rhs.data(), neg_q.data(), data.b.data(),
                                   data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);
            kkt->solve(default_rhs.data(), default_sol.data(), stream);
            unpack_interleaved_kernel(variables.x.data(), variables.z.data(), default_sol.data(),
                                     data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

            // For QP, set s = -z
            waxpby(variables.s, -1.0, variables.z, 0.0, variables.s, stream);
        }

        // Shift s and z into cone interior using smart margin-based approach
        // This uses per-batch margins and shifts to prevent cross-batch coupling
        // which is critical for correct gradient computation.

        double total_degree = static_cast<double>(data.cones.degree());

        // Fused: margins + alpha + shift for s and z (6→2 kernels)
        data.cones.fused_margins_and_shift(variables.s, true, total_degree, stream);
        data.cones.fused_margins_and_shift(variables.z, false, total_degree, stream);

        // Direct-x variables: cone-interior initialization per kind.
        // z_x: 1.0 for nonneg (positive orthant interior); e_0 for SOC
        //      ([1,0,0,...]; minimal SOC interior point).
        // x[J]: margin-based shift into the cone interior, mirror of CPU
        //      `_shift_single_cone_to_interior`. Nonneg and SOC each have
        //      their own shift formulas (margins() differ).
        init_xcone_start_point(stream);

        // PSD cones: apply eigendecomp-based margin+shift matching CPU.
        // Applied for both LP (P=0) and QP problems — without this shift,
        // LP+PSD problems have near-zero initial residuals and the solver
        // falsely reports convergence at iteration 0.
        if (data.cones.numPsdCones > 0) {
            data.cones.psd_shift_to_interior(variables.s, total_degree, stream);
            data.cones.psd_shift_to_interior(variables.z, total_degree, stream);
        }

    } else {
        // Asymmetric cones: unit_initialization on z, s; x[J]/z_x for direct-x.
        data.cones.unit_initialization(variables.s, variables.z, stream);

        init_xcone_start_point(stream);

        mu.setToConstant(1.0, stream);

        ScalingStrategy init_scaling = data.cones.allows_primal_dual_scaling() ?
            ScalingStrategy::PrimalDual : ScalingStrategy::Dual;

        // Seed direct-x scaling state to an identity-like baseline before the
        // legacy `kkt->update` (which only refreshes slack scaling, leaving
        // xcone_eta/w/lambda untouched). Without this, the asymmetric default
        // start factored a KKT whose direct-x sparse-SOC expansion diagonals
        // still held populate-time zeros — singular factor on problems mixing
        // slack-asymmetric cones with direct-x sparse SOC (dim > 4). The
        // symmetric branch above does the same reset for the same reason.
        if (data.cones.numXCones > 0) {
            seed_xcone_Hs_identity(
                data.cones.xcone_Hs.data(),
                data.cones.d_xcone_kinds,
                data.cones.d_xcone_dims,
                data.cones.d_xcone_hs_offsets,
                data.cones.xcone_Hs.batchSize(),
                data.cones.numXCones,
                data.cones.totalXConeHsEntries,
                stream);
            data.cones.xcone_w.setToConstant(1.0, stream);
            data.cones.xcone_lambda.setToConstant(1.0, stream);
            data.cones.xcone_eta.setToConstant(1.0, stream);
            kkt->refresh_xcone_hs(data.cones, stream);
        }

        kkt->update(
            data.cones, variables.s, variables.z, mu, init_scaling,
            settings.ipm.staticRegularizationEnable,
            settings.ipm.staticRegularizationConstant,
            settings.ipm.staticRegularizationProportional,
            stream
        );

    }

    // No sync needed - next operations are GPU work on same stream
}



void CompiledSolver::copyProblemData(
    const double* d_P_values,
    const double* d_A_values,
    const double* d_q,
    const double* d_b,
    cudaStream_t stream
) {
    // Only copy P if it has nonzeros (support for LP where P is empty)
    if (data.P.nnz() > 0) {
        cudaMemcpyAsync(data.P.values(), d_P_values, sizeof(double) * data.P.nnz() * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (data.A.nnz() > 0) {
        cudaMemcpyAsync(data.A.values(), d_A_values, sizeof(double) * data.A.nnz() * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (data.n > 0) {
        cudaMemcpyAsync(data.q.data(), d_q, sizeof(double) * data.n * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (data.m > 0) {
        cudaMemcpyAsync(data.b.data(), d_b, sizeof(double) * data.m * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
    // Reset flag so next solve does full equilibration with new P/A
    matrices_equilibrated_ = false;
}


void CompiledSolver::copyPerSolveData(
    const double* d_q,
    const double* d_b,
    cudaStream_t stream
) {
    // Copy q and b (P and A already set via setup() and stay equilibrated)
    if (data.n > 0) {
        cudaMemcpyAsync(data.q.data(), d_q, sizeof(double) * data.n * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (data.m > 0) {
        cudaMemcpyAsync(data.b.data(), d_b, sizeof(double) * data.m * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

// NOTE: A rows zeroed here for +inf b entries are NOT restored on subsequent
// solves. If the inf pattern in b changes between solves (e.g. b[i] goes from
// +inf to finite), call setup() again to reload original A values.
void CompiledSolver::sanitizeInfB(cudaStream_t stream) {
    int64_t zero_start = 0;
    int64_t zero_end = data.cones.numZeroCones;
    int64_t nonneg_start = zero_end;
    int64_t nonneg_end = nonneg_start + data.cones.numNonnegCones;

    int64_t num_rows = (zero_end - zero_start) + (nonneg_end - nonneg_start);
    std::fill(inf_b_infeasible_flags_.begin(), inf_b_infeasible_flags_.end(), 0);
    if (num_rows == 0) return;

    moreau::sanitize_inf_b_kernel(
        data.b.data(),
        data.A.values(),
        data.A.rowOffsets(),
        d_infeasible_flags_,
        zero_start, zero_end,
        nonneg_start, nonneg_end,
        data.m,
        data.A.nnz(),
        data.batchSize,
        stream
    );

    // Copy flags to host for post-solve status override
    cudaMemcpyAsync(inf_b_infeasible_flags_.data(), d_infeasible_flags_,
                    sizeof(int) * data.batchSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void CompiledSolver::overrideInfeasibleStatus() {
    for (int64_t b = 0; b < data.batchSize; b++) {
        if (inf_b_infeasible_flags_[b]) {
            info.status[b] = SolverStatus::PrimalInfeasible;
            int32_t pi_status = static_cast<int32_t>(SolverStatus::PrimalInfeasible);
            cudaMemcpy(info.status_device + b, &pi_status, sizeof(int32_t), cudaMemcpyHostToDevice);
        }
    }
}

void CompiledSolver::doEquilibration(cudaStream_t stream) {
    if (!matrices_equilibrated_) {
        // First solve: run full equilibration on P, A, q, b
        // This computes d, e, c and caches mean_P_row_norm for subsequent solves
        moreau::equilibration(
            data.equilibration,
            data.P, data.matrixPRowOf,
            data.A, data.matrixARowOf,
            data.q, data.b,
            settings.ipm.equilibrationSettings,
            data.cones,
            stream
        );
        matrices_equilibrated_ = true;
    } else {
        // Subsequent solves: P and A are already equilibrated, only equilibrate q and b
        // Uses stored d, e and cached mean_P_row_norm to compute cost scaling
        moreau::equilibrate_vectors_only(
            data.equilibration,
            data.P, data.matrixPRowOf,
            data.q, data.b,
            settings.ipm.equilibrationSettings,
            stream
        );
    }
}

void CompiledSolver::computeCachedNorms(cudaStream_t stream) {
    // Compute ||b||_inf with equilibration scaling: norm_inf(b .* einv)
    // Compute ||q||_inf with equilibration scaling: norm_inf(q .* dinv) * cinv
    compute_cached_norms_kernel(
        data.normb.data(),
        data.normq.data(),
        data.b.data(),
        data.q.data(),
        data.equilibration.einv.data(),
        data.equilibration.dinv.data(),
        data.equilibration.c.data(),
        data.n,
        data.m,
        data.batchSize,
        stream
    );
}

bool CompiledSolver::solveKKT(
    Variables& lhs,
    const Variables& rhs,
    bool is_affine_step,
    cudaStream_t stream
) {
    // Implementation based on Clarabel.rs KKTSystem::solve

    // Step 1: Solve for (x1, z1)
    // ---------------------------------------------------
    // workx = rhs.x (already contains the RHS from affine_step_rhs or combined_step_rhs)
    cudaMemcpyAsync(workx.data(), rhs.x.data(),
                    sizeof(double) * data.n * data.batchSize,
                    cudaMemcpyDeviceToDevice, stream);

    // Compute Δs_const_term
    if (is_affine_step) {
        // Affine case: Δs_const_term = variables.s
        cudaMemcpyAsync(Δs_const_term.data(), variables.s.data(),
                        sizeof(double) * data.m * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        // Combined case: Δs_const_term = Δs_from_Δz_offset(rhs.s, work, variables.z)
        data.cones.Δs_from_Δz_offset(Δs_const_term, rhs.s, workz, variables.z, stream);
    }

    // Direct-x KKT RHS correction. The reduced (1,1) x-gradient row
    // includes an extra term `workx -= Σ_J E_J' c_J` where c_J
    // depends on the step direction:
    //   Affine    : c_J = variables.z_x[J]   (kind-agnostic)
    //   Combined  : Nonneg: c_J = rhs.z_x[J] / variables.x[J]
    //               SOC:    c_J = Δs_from_Δz_offset(rhs.z_x, x[J])
    // No-op when there are no direct-x cones.
    const int64_t total_xcone_numel = variables.totalXConeNumel();
    if (total_xcone_numel > 0 && data.cones.d_xcone_indices != nullptr) {
        if (is_affine_step) {
            subtract_xcone_affine_from_workx(
                workx.data(), variables.z_x.data(),
                data.cones.d_xcone_indices,
                data.batchSize, data.n, total_xcone_numel, stream);
        } else {
            subtract_xcone_combined_from_workx(
                workx.data(), rhs.z_x.data(), variables.x.data(),
                data.cones.d_xcone_indices,
                data.cones.d_xcone_kind_per_entry,
                data.batchSize, data.n, total_xcone_numel, stream);
            subtract_xcone_combined_from_workx_soc(
                workx.data(), rhs.z_x.data(), variables.x.data(),
                data.cones.xcone_w.data(), data.cones.xcone_lambda.data(),
                data.cones.xcone_eta.data(),
                data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                data.cones.d_xcone_numel_offsets,
                data.cones.d_xcone_indices,
                data.batchSize, data.n, data.cones.numXCones,
                total_xcone_numel, stream);
            if (data.cones.numXExpCones > 0) {
                subtract_xcone_combined_from_workx_exp(
                    workx.data(), rhs.z_x.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel, stream);
            }
            if (data.cones.numXPowerCones > 0) {
                subtract_xcone_combined_from_workx_pow(
                    workx.data(), rhs.z_x.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel, stream);
            }
            if (data.cones.numXPsdCones > 0) {
                subtract_xcone_combined_from_workx_psd(
                    data.cones, workx.data(), rhs.z_x.data(),
                    data.n, stream);
            }
            if (data.cones.numXGenPowerCones > 0) {
                subtract_xcone_combined_from_workx_genpow(
                    workx.data(), rhs.z_x.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel, stream);
            }
        }
    }

    // Fused: z_part = Δs_const_term - rhs.z, pack interleaved with workx  (2→1 kernel)
    waxpby_and_pack_interleaved_kernel(kkt_rhs.data(), workx.data(),
                                       Δs_const_term.data(), rhs.z.data(),
                                       1.0, -1.0,
                                       data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

    // Solve
    kkt->solve(kkt_rhs.data(), kkt_sol.data(), stream);

    // Extract x1 and z1 from interleaved solution
    // kkt_sol layout: [batch0_full(n+m), batch1_full(n+m), ...]
    // Unpack into separate x1 and z1 arrays using GPU kernel
    unpack_interleaved_kernel(x1.data(), z1.data(), kkt_sol.data(),
                             data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

    // Step 2: Solve for Δτ
    // ---------------------------------------------------
    // Fused numerator: ξ=x/τ, tau_num = (rτ - rκ/τ) + q'x1 + b'z1  (3 → 1 kernel)
    fused_tau_numerator_base_kernel(
        ξ.data(), tau_num_work.data(),
        variables.x.data(), variables.τ.data(),
        rhs.τ.data(), rhs.κ.data(),
        data.q.data(), x1.data(), data.b.data(), z1.data(),
        data.n, data.m, data.batchSize, stream);

    // tau_num_work += 2 * ξ'P*x1 (QP only)
    if (data.P.nnz() > 0) {
        quad_form_symmetric_batched(ξ_P_x1, &data.P, ξ, x1, quad_form_temp, stream);
        axpby(tau_num_work, 2.0, ξ_P_x1, 1.0, stream);
    }

    // Fused denominator: tau_den = κ/τ - q'x2 - b'z2  (2 → 1 kernel)
    fused_tau_denominator_base_kernel(
        tau_den_work.data(),
        variables.κ.data(), variables.τ.data(),
        data.q.data(), x2.data(), data.b.data(), z2.data(),
        data.n, data.m, data.batchSize, stream);

    // Fused double quad form: tau_den += (ξ-x2)'P(ξ-x2) - x2'Px2  (4 → 1 kernel, QP only)
    if (data.P.nnz() > 0) {
        fused_double_quad_form_kernel(
            tau_den_work.data(),
            data.P.rowOffsets(), data.P.colIndices(), data.P.values(),
            ξ.data(), x2.data(),
            data.n, data.P.nnz(), data.batchSize, stream);
    }

    // lhs.τ = tau_num / tau_den (with safeguard for near-zero tau_den)
    safe_elementwise_div(lhs.τ, tau_num_work, tau_den_work, 1e-14, stream);

    // Step 3: Solve for (Δx, Δz)
    // ---------------------------------------------------
    axpby_scaled(lhs.x, x1, lhs.τ, x2, stream);
    axpby_scaled(lhs.z, z1, lhs.τ, z2, stream);

    // Step 4: Solve for Δs
    // ---------------------------------------------------
    data.cones.mul_Hs(lhs.s, lhs.z, workz, stream);
    waxpby(lhs.s, -1.0, lhs.s, -1.0, Δs_const_term, stream);

    // Step 5: Solve for Δκ
    // ---------------------------------------------------
    compute_delta_kappa_kernel(
        lhs.κ.data(),
        rhs.κ.data(),
        variables.κ.data(),
        lhs.τ.data(),
        variables.τ.data(),
        data.batchSize,
        stream
    );

    // Direct-x Δz_x recovery (CPU mirror: recover_direct_x_dual).
    //   Affine  : Δz_J = -Hs_J · Δx[J] - variables.z_x[k]
    //   Combined: Δz_J = -Hs_J · Δx[J] - c_J (nonneg: rhs/x; SOC: Δs_from_Δz)
    // Hs_J for nonneg is z/x (scalar), for SOC is the packed upper-tri
    // η²(2ww'-J). Both already in cones.xcone_Hs after scale_cones.
    if (total_xcone_numel > 0 && data.cones.d_xcone_indices != nullptr) {
        if (is_affine_step) {
            recover_dz_x_affine_nonneg(
                lhs.z_x.data(), lhs.x.data(),
                variables.z_x.data(),
                data.cones.xcone_Hs.data(),
                data.cones.d_xcone_indices,
                data.cones.d_xcone_kind_per_entry,
                data.batchSize, data.n, total_xcone_numel,
                data.cones.totalXConeHsEntries, stream);
            recover_dz_x_affine_soc(
                lhs.z_x.data(), lhs.x.data(),
                variables.z_x.data(),
                data.cones.xcone_w.data(),
                data.cones.xcone_eta.data(),
                data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                data.cones.d_xcone_numel_offsets,
                data.cones.d_xcone_indices,
                data.batchSize, data.n, data.cones.numXCones,
                total_xcone_numel, stream);
            if (data.cones.numXExpCones > 0) {
                recover_dz_x_affine_exp(
                    lhs.z_x.data(), lhs.x.data(),
                    variables.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_hs_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel,
                    data.cones.totalXConeHsEntries, stream);
            }
            if (data.cones.numXPowerCones > 0) {
                recover_dz_x_affine_pow(
                    lhs.z_x.data(), lhs.x.data(),
                    variables.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_hs_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel,
                    data.cones.totalXConeHsEntries, stream);
            }
            if (data.cones.numXPsdCones > 0) {
                recover_dz_x_affine_psd(
                    data.cones, lhs.z_x.data(), lhs.x.data(),
                    variables.z_x.data(), data.n, stream);
            }
            if (data.cones.numXGenPowerCones > 0) {
                recover_dz_x_affine_genpow(
                    lhs.z_x.data(), lhs.x.data(),
                    variables.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.xcone_genpow_p.data(),
                    data.cones.xcone_genpow_q.data(),
                    data.cones.xcone_genpow_r.data(),
                    data.cones.xcone_genpow_pd_axes.data(),
                    data.cones.xcone_genpow_pd_coefs.data(),
                    data.cones.xcone_genpow_pd_signs.data(),
                    data.cones.xcone_genpow_pd_active.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_hs_offsets,
                    data.cones.d_xcone_indices,
                    data.cones.d_xcone_genpow_idx,
                    data.cones.d_xcone_genpow_dim1s,
                    data.cones.d_xcone_genpow_dim2s,
                    data.cones.d_xcone_genpow_alpha_offsets,
                    data.cones.d_xcone_genpow_dim_offsets,
                    data.cones.d_xcone_genpow_sparse_idx,
                    data.cones.d_xcone_genpow_sparse_offsets,
                    data.cones.xcone_genpow_pd_workspace.data(),
                    data.batchSize, data.n, data.cones.numXCones,
                    data.cones.numXGenPowerCones,
                    total_xcone_numel,
                    data.cones.totalXConeHsEntries,
                    data.cones.totalXGenPowerDim,
                    data.cones.totalXGenPowerAlphas,
                    data.cones.totalXGenPowerDim2,
                    data.cones.numSparseXGenPow, stream);
            }
        } else {
            recover_dz_x_combined_nonneg(
                lhs.z_x.data(), lhs.x.data(),
                rhs.z_x.data(), variables.x.data(),
                data.cones.xcone_Hs.data(),
                data.cones.d_xcone_indices,
                data.cones.d_xcone_kind_per_entry,
                data.batchSize, data.n, total_xcone_numel,
                data.cones.totalXConeHsEntries, stream);
            recover_dz_x_combined_soc(
                lhs.z_x.data(), lhs.x.data(),
                rhs.z_x.data(), variables.x.data(),
                data.cones.xcone_w.data(), data.cones.xcone_lambda.data(),
                data.cones.xcone_eta.data(),
                data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                data.cones.d_xcone_numel_offsets,
                data.cones.d_xcone_indices,
                data.batchSize, data.n, data.cones.numXCones,
                total_xcone_numel, stream);
            if (data.cones.numXExpCones > 0) {
                recover_dz_x_combined_exp(
                    lhs.z_x.data(), lhs.x.data(),
                    rhs.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_hs_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel,
                    data.cones.totalXConeHsEntries, stream);
            }
            if (data.cones.numXPowerCones > 0) {
                recover_dz_x_combined_pow(
                    lhs.z_x.data(), lhs.x.data(),
                    rhs.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_hs_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel,
                    data.cones.totalXConeHsEntries, stream);
            }
            if (data.cones.numXPsdCones > 0) {
                recover_dz_x_combined_psd(
                    data.cones, lhs.z_x.data(), lhs.x.data(),
                    rhs.z_x.data(), data.n, stream);
            }
            if (data.cones.numXGenPowerCones > 0) {
                recover_dz_x_combined_genpow(
                    lhs.z_x.data(), lhs.x.data(),
                    rhs.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.xcone_genpow_p.data(),
                    data.cones.xcone_genpow_q.data(),
                    data.cones.xcone_genpow_r.data(),
                    data.cones.xcone_genpow_pd_axes.data(),
                    data.cones.xcone_genpow_pd_coefs.data(),
                    data.cones.xcone_genpow_pd_signs.data(),
                    data.cones.xcone_genpow_pd_active.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_hs_offsets,
                    data.cones.d_xcone_indices,
                    data.cones.d_xcone_genpow_idx,
                    data.cones.d_xcone_genpow_dim1s,
                    data.cones.d_xcone_genpow_dim2s,
                    data.cones.d_xcone_genpow_alpha_offsets,
                    data.cones.d_xcone_genpow_dim_offsets,
                    data.cones.d_xcone_genpow_sparse_idx,
                    data.cones.d_xcone_genpow_sparse_offsets,
                    data.cones.xcone_genpow_pd_workspace.data(),
                    data.batchSize, data.n, data.cones.numXCones,
                    data.cones.numXGenPowerCones,
                    total_xcone_numel,
                    data.cones.totalXConeHsEntries,
                    data.cones.totalXGenPowerDim,
                    data.cones.totalXGenPowerAlphas,
                    data.cones.totalXGenPowerDim2,
                    data.cones.numSparseXGenPow, stream);
            }
        }
    }

    return true;
}

bool CompiledSolver::solveKKTFromPrecomputed(
    Variables& lhs,
    const Variables& rhs,
    bool is_affine_step,
    cudaStream_t stream
) {
    // Same as solveKKT but assumes x1, z1, x2, z2 are already computed
    // This is used when batching constant + affine solves together

    // Compute Δs_const_term (needed for step 4)
    if (is_affine_step) {
        // Affine case: Δs_const_term = variables.s
        cudaMemcpyAsync(Δs_const_term.data(), variables.s.data(),
                        sizeof(double) * data.m * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        // Combined case: Δs_const_term = Δs_from_Δz_offset(rhs.s, work, variables.z)
        data.cones.Δs_from_Δz_offset(Δs_const_term, rhs.s, workz, variables.z, stream);
    }

    // Step 2: Solve for Δτ (x1, z1, x2, z2 are precomputed)
    // ---------------------------------------------------
    // Fused numerator: ξ=x/τ, tau_num = (rτ - rκ/τ) + q'x1 + b'z1  (3 → 1 kernel)
    fused_tau_numerator_base_kernel(
        ξ.data(), tau_num_work.data(),
        variables.x.data(), variables.τ.data(),
        rhs.τ.data(), rhs.κ.data(),
        data.q.data(), x1.data(), data.b.data(), z1.data(),
        data.n, data.m, data.batchSize, stream);

    // tau_num_work += 2 * ξ'P*x1 (QP only)
    if (data.P.nnz() > 0) {
        quad_form_symmetric_batched(ξ_P_x1, &data.P, ξ, x1, quad_form_temp, stream);
        axpby(tau_num_work, 2.0, ξ_P_x1, 1.0, stream);
    }

    // Fused denominator: tau_den = κ/τ - q'x2 - b'z2  (2 → 1 kernel)
    fused_tau_denominator_base_kernel(
        tau_den_work.data(),
        variables.κ.data(), variables.τ.data(),
        data.q.data(), x2.data(), data.b.data(), z2.data(),
        data.n, data.m, data.batchSize, stream);

    // Fused double quad form: tau_den += (ξ-x2)'P(ξ-x2) - x2'Px2  (4 → 1 kernel, QP only)
    if (data.P.nnz() > 0) {
        fused_double_quad_form_kernel(
            tau_den_work.data(),
            data.P.rowOffsets(), data.P.colIndices(), data.P.values(),
            ξ.data(), x2.data(),
            data.n, data.P.nnz(), data.batchSize, stream);
    }

    // lhs.τ = tau_num / tau_den (with safeguard for near-zero tau_den)
    safe_elementwise_div(lhs.τ, tau_num_work, tau_den_work, 1e-14, stream);

    // Step 3: Solve for (Δx, Δz)
    // ---------------------------------------------------
    axpby_scaled(lhs.x, x1, lhs.τ, x2, stream);
    axpby_scaled(lhs.z, z1, lhs.τ, z2, stream);

    // Step 4: Solve for Δs
    // ---------------------------------------------------
    data.cones.mul_Hs(lhs.s, lhs.z, workz, stream);
    waxpby(lhs.s, -1.0, lhs.s, -1.0, Δs_const_term, stream);

    // Step 5: Solve for Δκ
    // ---------------------------------------------------
    compute_delta_kappa_kernel(
        lhs.κ.data(),
        rhs.κ.data(),
        variables.κ.data(),
        lhs.τ.data(),
        variables.τ.data(),
        data.batchSize,
        stream
    );

    // Direct-x Δz_x recovery (CPU mirror: recover_direct_x_dual), for
    // whichever step direction solveKKTFromPrecomputed is assembling.
    {
        const int64_t total_xcone_numel = variables.totalXConeNumel();
        if (total_xcone_numel > 0 && data.cones.d_xcone_indices != nullptr) {
            if (is_affine_step) {
                recover_dz_x_affine_nonneg(
                    lhs.z_x.data(), lhs.x.data(),
                    variables.z_x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.d_xcone_indices,
                    data.cones.d_xcone_kind_per_entry,
                    data.batchSize, data.n, total_xcone_numel,
                    data.cones.totalXConeHsEntries, stream);
                recover_dz_x_affine_soc(
                    lhs.z_x.data(), lhs.x.data(),
                    variables.z_x.data(),
                    data.cones.xcone_w.data(),
                    data.cones.xcone_eta.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel, stream);
                if (data.cones.numXExpCones > 0) {
                    recover_dz_x_affine_exp(
                        lhs.z_x.data(), lhs.x.data(),
                        variables.z_x.data(),
                        data.cones.xcone_Hs.data(),
                        data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                        data.cones.d_xcone_numel_offsets,
                        data.cones.d_xcone_hs_offsets,
                        data.cones.d_xcone_indices,
                        data.batchSize, data.n, data.cones.numXCones,
                        total_xcone_numel,
                        data.cones.totalXConeHsEntries, stream);
                }
                if (data.cones.numXPowerCones > 0) {
                    recover_dz_x_affine_pow(
                        lhs.z_x.data(), lhs.x.data(),
                        variables.z_x.data(),
                        data.cones.xcone_Hs.data(),
                        data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                        data.cones.d_xcone_numel_offsets,
                        data.cones.d_xcone_hs_offsets,
                        data.cones.d_xcone_indices,
                        data.batchSize, data.n, data.cones.numXCones,
                        total_xcone_numel,
                        data.cones.totalXConeHsEntries, stream);
                }
                if (data.cones.numXPsdCones > 0) {
                    recover_dz_x_affine_psd(
                        data.cones, lhs.z_x.data(), lhs.x.data(),
                        variables.z_x.data(), data.n, stream);
                }
                if (data.cones.numXGenPowerCones > 0) {
                    recover_dz_x_affine_genpow(
                        lhs.z_x.data(), lhs.x.data(),
                        variables.z_x.data(),
                        data.cones.xcone_Hs.data(),
                        data.cones.xcone_genpow_p.data(),
                        data.cones.xcone_genpow_q.data(),
                        data.cones.xcone_genpow_r.data(),
                        data.cones.xcone_genpow_pd_axes.data(),
                        data.cones.xcone_genpow_pd_coefs.data(),
                        data.cones.xcone_genpow_pd_signs.data(),
                        data.cones.xcone_genpow_pd_active.data(),
                        data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                        data.cones.d_xcone_numel_offsets,
                        data.cones.d_xcone_hs_offsets,
                        data.cones.d_xcone_indices,
                        data.cones.d_xcone_genpow_idx,
                        data.cones.d_xcone_genpow_dim1s,
                        data.cones.d_xcone_genpow_dim2s,
                        data.cones.d_xcone_genpow_alpha_offsets,
                        data.cones.d_xcone_genpow_dim_offsets,
                        data.cones.d_xcone_genpow_sparse_idx,
                        data.cones.d_xcone_genpow_sparse_offsets,
                        data.cones.xcone_genpow_pd_workspace.data(),
                        data.batchSize, data.n, data.cones.numXCones,
                        data.cones.numXGenPowerCones,
                        total_xcone_numel,
                        data.cones.totalXConeHsEntries,
                        data.cones.totalXGenPowerDim,
                        data.cones.totalXGenPowerAlphas,
                        data.cones.totalXGenPowerDim2,
                        data.cones.numSparseXGenPow, stream);
                }
            } else {
                recover_dz_x_combined_nonneg(
                    lhs.z_x.data(), lhs.x.data(),
                    rhs.z_x.data(), variables.x.data(),
                    data.cones.xcone_Hs.data(),
                    data.cones.d_xcone_indices,
                    data.cones.d_xcone_kind_per_entry,
                    data.batchSize, data.n, total_xcone_numel,
                    data.cones.totalXConeHsEntries, stream);
                recover_dz_x_combined_soc(
                    lhs.z_x.data(), lhs.x.data(),
                    rhs.z_x.data(), variables.x.data(),
                    data.cones.xcone_w.data(), data.cones.xcone_lambda.data(),
                    data.cones.xcone_eta.data(),
                    data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                    data.cones.d_xcone_numel_offsets,
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, data.cones.numXCones,
                    total_xcone_numel, stream);
                if (data.cones.numXExpCones > 0) {
                    recover_dz_x_combined_exp(
                        lhs.z_x.data(), lhs.x.data(),
                        rhs.z_x.data(),
                        data.cones.xcone_Hs.data(),
                        data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                        data.cones.d_xcone_numel_offsets,
                        data.cones.d_xcone_hs_offsets,
                        data.cones.d_xcone_indices,
                        data.batchSize, data.n, data.cones.numXCones,
                        total_xcone_numel,
                        data.cones.totalXConeHsEntries, stream);
                }
                if (data.cones.numXPowerCones > 0) {
                    recover_dz_x_combined_pow(
                        lhs.z_x.data(), lhs.x.data(),
                        rhs.z_x.data(),
                        data.cones.xcone_Hs.data(),
                        data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                        data.cones.d_xcone_numel_offsets,
                        data.cones.d_xcone_hs_offsets,
                        data.cones.d_xcone_indices,
                        data.batchSize, data.n, data.cones.numXCones,
                        total_xcone_numel,
                        data.cones.totalXConeHsEntries, stream);
                }
                if (data.cones.numXPsdCones > 0) {
                    recover_dz_x_combined_psd(
                        data.cones, lhs.z_x.data(), lhs.x.data(),
                        rhs.z_x.data(), data.n, stream);
                }
                if (data.cones.numXGenPowerCones > 0) {
                    recover_dz_x_combined_genpow(
                        lhs.z_x.data(), lhs.x.data(),
                        rhs.z_x.data(),
                        data.cones.xcone_Hs.data(),
                        data.cones.xcone_genpow_p.data(),
                        data.cones.xcone_genpow_q.data(),
                        data.cones.xcone_genpow_r.data(),
                        data.cones.xcone_genpow_pd_axes.data(),
                        data.cones.xcone_genpow_pd_coefs.data(),
                        data.cones.xcone_genpow_pd_signs.data(),
                        data.cones.xcone_genpow_pd_active.data(),
                        data.cones.d_xcone_kinds, data.cones.d_xcone_dims,
                        data.cones.d_xcone_numel_offsets,
                        data.cones.d_xcone_hs_offsets,
                        data.cones.d_xcone_indices,
                        data.cones.d_xcone_genpow_idx,
                        data.cones.d_xcone_genpow_dim1s,
                        data.cones.d_xcone_genpow_dim2s,
                        data.cones.d_xcone_genpow_alpha_offsets,
                        data.cones.d_xcone_genpow_dim_offsets,
                        data.cones.d_xcone_genpow_sparse_idx,
                        data.cones.d_xcone_genpow_sparse_offsets,
                        data.cones.xcone_genpow_pd_workspace.data(),
                        data.batchSize, data.n, data.cones.numXCones,
                        data.cones.numXGenPowerCones,
                        total_xcone_numel,
                        data.cones.totalXConeHsEntries,
                        data.cones.totalXGenPowerDim,
                        data.cones.totalXGenPowerAlphas,
                        data.cones.totalXGenPowerDim2,
                        data.cones.numSparseXGenPow, stream);
                }
            }
        }
    }

    return true;
}

void CompiledSolver::getStepLength(
    const Variables& step,
    bool is_affine_step,
    BatchedVector& alpha,
    cudaStream_t stream
) {
    // Use member variables step_work1 and step_work2

    // Calculate step length using Variables method
    variables.calc_step_length(
        step,
        data.cones,
        settings.ipm.maxStepFraction,
        !is_affine_step,  // is_combined_step = !is_affine_step
        alpha,
        step_work1,
        step_work2,
        settings.ipm.linesearchBacktrackStep,
        settings.ipm.minTerminateStepLength,
        stream
    );

    // Additional barrier backtracking for asymmetric cones with dual scaling.
    // Clarabel backtracks with Dual scaling + Combined step + nonsymmetric cones.
    // We use PrimalDual scaling for all nonsymmetric cones (exp, power, GenPowerCone),
    // so the combined-step barrier backtracking path is never triggered.
    // GenPowerCone step feasibility is enforced via step_length_asymmetric_kernel
    // (backtrack_search_genpow) and barrier is computed in compute_barriers_all_cones_kernel.
    bool needs_backtrack = false;
    if (needs_backtrack) {
        // Use member variable alpha_init
        cudaMemcpyAsync(alpha_init.data(), alpha.data(),
                        sizeof(double) * data.batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        backtrackStepToBarrier(step, alpha_init, alpha, stream);
    }

    // No sync needed - alpha only used in GPU operations on same stream
}

void CompiledSolver::backtrackStepToBarrier(
    const Variables& step,
    const BatchedVector& alpha_init,
    BatchedVector& alpha,
    cudaStream_t stream
) {
    // GPU-based backtracking line search for all batches in parallel
    // Each batch performs independent backtracking

    double backtrack_step = settings.ipm.linesearchBacktrackStep;
    const int max_iters = 50;
    int64_t degree = data.cones.degree();

    // Call GPU kernel to perform backtracking for all batches
    backtrack_line_search_kernel(
        alpha.data(),
        alpha_init.data(),
        nullptr,  // barrier_work (not currently used)
        variables.s.data(),
        variables.z.data(),
        variables.τ.data(),
        variables.κ.data(),
        step.s.data(),
        step.z.data(),
        step.τ.data(),
        step.κ.data(),
        variables.sz_dot.data(),
        backtrack_step,
        degree,
        data.m,
        data.batchSize,
        max_iters,
        data.cones.numZeroCones,
        data.cones.numNonnegCones,
        data.cones.numSocCones,
        data.cones.d_soc_dims,
        stream
    );

    // No sync needed - alpha only used in GPU operations on same stream
}

void CompiledSolver::save_best_iterate(
    const Variables& variables,
    Solution& solution,
    Info& info,
    int64_t batchSize,
    cudaStream_t stream
) {
    save_best_iterate_kernel(
        variables.x.data(),
        variables.s.data(),
        variables.z.data(),
        variables.τ.data(),
        variables.κ.data(),
        info.cost_primal.data(),
        info.cost_dual.data(),
        info.res_primal.data(),
        info.res_dual.data(),
        info.gap_abs.data(),
        info.gap_rel.data(),
        info.ktratio.data(),
        solution.x_raw.data(),
        solution.s_raw.data(),
        solution.z_raw.data(),
        solution.τ_raw.data(),
        solution.κ_raw.data(),
        solution.cost_primal_raw.data(),
        solution.cost_dual_raw.data(),
        info.best_res_primal.data(),
        info.best_res_dual.data(),
        info.best_gap_abs.data(),
        info.best_gap_rel.data(),
        info.best_ktratio.data(),
        solution.solution_saved.get(),
        solution.should_save.get(),
        info.d_best_saved,
        info.status_device,
        settings.ipm.tolGapAbs,
        settings.ipm.tolFeas,
        settings.ipm.reducedTolGapAbs,
        settings.ipm.reducedTolGapRel,
        settings.ipm.reducedTolFeas,
        settings.ipm.reducedTolKtRatio,
        data.n,
        data.m,
        batchSize,
        stream
    );
    // Also mirror the cost snapshot into the best_cost_* buffers so
    // restore_best_iterate has a complete metric picture (save_best_iterate
    // writes cost into solution.*_raw; restore_best_iterate reads from Info).
    cudaMemcpyAsync(info.best_cost_primal.data(), solution.cost_primal_raw.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(info.best_cost_dual.data(), solution.cost_dual_raw.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);
}

void CompiledSolver::restore_best_iterate(Info& info, int64_t batchSize, cudaStream_t stream) {
    restore_best_iterate_kernel(
        info.res_primal.data(),
        info.res_dual.data(),
        info.gap_abs.data(),
        info.gap_rel.data(),
        info.ktratio.data(),
        info.cost_primal.data(),
        info.cost_dual.data(),
        info.best_res_primal.data(),
        info.best_res_dual.data(),
        info.best_gap_abs.data(),
        info.best_gap_rel.data(),
        info.best_ktratio.data(),
        info.best_cost_primal.data(),
        info.best_cost_dual.data(),
        info.status_device,
        info.d_best_saved,
        batchSize,
        stream
    );
}

void CompiledSolver::save_terminated_solutions(
    const Variables& variables,
    Solution& solution,
    const Info& info,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream
) {
    save_terminated_solutions_kernel(
        variables.x.data(),
        variables.s.data(),
        variables.z.data(),
        variables.τ.data(),
        variables.κ.data(),
        info.cost_primal.data(),
        info.cost_dual.data(),
        solution.x_raw.data(),
        solution.s_raw.data(),
        solution.z_raw.data(),
        solution.τ_raw.data(),
        solution.κ_raw.data(),
        solution.cost_primal_raw.data(),
        solution.cost_dual_raw.data(),
        solution.solution_saved.get(),
        solution.should_save.get(),
        status,
        data.n,
        data.m,
        batchSize,
        stream
    );
}

void CompiledSolver::resetState(cudaStream_t stream) {
    // Clear any previous CUDA errors
    cudaGetLastError();

    // Reset status to unsolved (info.status_device, removed local `status`).
    cudaMemsetAsync(info.status_device, 0,
                    sizeof(int32_t) * data.batchSize, stream);

    // Reset solution_saved flag so solutions are saved on next convergence
    solution.resetSolutionSaved(stream);

    // Reset per-batch best-iterate flag so a stale snapshot from the previous
    // solve doesn't cause the post-loop restore to promote this solve.
    cudaMemsetAsync(info.d_best_saved, 0, sizeof(int32_t) * data.batchSize, stream);

    // Reset info status_device to 0 (unsolved) so IPM iterates again
    cudaMemsetAsync(info.status_device, 0, sizeof(int32_t) * data.batchSize, stream);
}

// ============================================================================
// Smoothed Differentiation: Central-path centering loop
// ============================================================================

void CompiledSolver::refineSmoothingIterate(DiffState& state, cudaStream_t stream) {
    // Guard: only run if smoothed differentiation is requested
    if (settings.ipm.diffMethod != DiffMethod::Smoothed) {
        state.smoothing_cached = false;
        return;
    }

    // Guard: zero-cone-only problems have degree=0, which would cause
    // division by zero when computing μ. These problems use the exact
    // backward path that doesn't need the smoothing iterate.
    if (data.cones.degree() == 0) {
        state.smoothing_cached = false;
        return;
    }

    int64_t n = data.n;
    int64_t m_dim = data.m;
    int64_t batchSize = data.batchSize;

    double target_mu = settings.ipm.diffSmoothingMu;
    double tol_gap_abs = settings.ipm.tolGapAbs;

    // Skip refinement if μ_target is already at or below convergence tolerance
    // (the iterate is already past the target on the central path).
    if (target_mu <= tol_gap_abs) {
        state.smoothing_cached = false;
        return;
    }

    // Per-iteration multiplier for the central-path walk-up.
    // Used directly (no sqrt) — matches the CPU (Rust) convention.
    double eff_factor = settings.ipm.diffSmoothingStepFactor;

    // Precompute fixed iteration count: enough steps to reach target_mu from
    // tol_gap_abs (the convergence tolerance, which bounds initial μ).
    // With step_factor=30: ceil(log(1e-4/1e-8)/log(30))*2+2 = 8.
    // Extra iterations after convergence are nearly free (step ≈ 0, α ≈ 1).
    double ratio = target_mu / tol_gap_abs;
    int n_iters = std::min(50, (int)(std::ceil(
        std::log(ratio) / std::log(eff_factor)) * 2) + 2);

    // Step 1: Restore variables from the saved converged solution.
    // In batched mode, different batch elements converge at different iterations.
    // After an element converges, its solution is saved to solution.*_raw, but
    // variables continue to be modified by subsequent IPM iterations for other
    // elements. We must restore from solution.*_raw to get the correct starting
    // point for each batch element.
    cudaMemcpyAsync(variables.x.data(), solution.x_raw.data(),
                    sizeof(double) * n * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(variables.s.data(), solution.s_raw.data(),
                    sizeof(double) * m_dim * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(variables.z.data(), solution.z_raw.data(),
                    sizeof(double) * m_dim * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(variables.τ.data(), solution.τ_raw.data(),
                    sizeof(double) * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(variables.κ.data(), solution.κ_raw.data(),
                    sizeof(double) * batchSize,
                    cudaMemcpyDeviceToDevice, stream);

    // Normalize τ=1: divide x, s, z, κ by τ, then set τ=1.
    div_per_batch(variables.x, variables.x, variables.τ, stream);
    div_per_batch(variables.s, variables.s, variables.τ, stream);
    div_per_batch(variables.z, variables.z, variables.τ, stream);
    elementwise_div(variables.κ, variables.κ, variables.τ, stream);
    variables.τ.setToConstant(1.0, stream);

    ScalingStrategy scaling = data.cones.allows_primal_dual_scaling() ?
        ScalingStrategy::PrimalDual : ScalingStrategy::Dual;

    // Pre-compute neg_q = -q for batched KKT solves
    waxpby(neg_q, -1.0, data.q, 0.0, neg_q, stream);

    // Use pre-allocated workspace for centering loop
    BatchedVector& mu_step = smoothing_mu_step;
    BatchedVector& m_zero = smoothing_m_zero;
    m_zero.setToConstant(0.0, stream);

    // Step 2: Centering loop — fully device-side, zero GPU→CPU syncs.
    // mu_step is computed per-batch on the GPU: min(mu[b] * step_factor, target_mu).
    // Alpha is sanitized on the GPU (NaN/negative → 0), making add_step safe.
    // Fixed iteration count; after convergence, extra iters are nearly free
    // (step ≈ 0, α ≈ 1, variables unchanged).

    for (int iter = 0; iter < n_iters; ++iter) {
        // a. Compute residuals (rx, rz, rτ, dot_sz)
        residuals.update(variables, data, cusparse_handle_, cublas_handle_, stream);

        // b. Compute current μ and adaptive centering target on device
        variables.calc_mu(mu, residuals, data.cones, stream);
        compute_mu_step(mu_step, mu, eff_factor, target_mu, stream);

        // c. Update cone scaling with current μ. The fused scatter path
        // (`scale_cones(..., kkt.get())`) is currently disabled. Its
        // historical regression was driven by the single-thread scaling
        // kernel — now block-parallel — so the fusion is worth
        // re-benchmarking before declaring it dead.
        bool scale_ok = variables.scale_cones(data.cones, mu, scaling, stream);
        if (!scale_ok) {
            break;
        }

        // d. Factor KKT system (also solves constant RHS → x2, z2)
        bool kkt_ok = kkt->updateFactorOnly(
            data.cones, variables.s, variables.z, mu, scaling,
            settings.ipm.staticRegularizationEnable,
            settings.ipm.staticRegularizationConstant,
            settings.ipm.staticRegularizationProportional,
            data.q, data.b,
            workx, kkt_rhs, kkt_sol, x2, z2,
            stream
        );
        if (!kkt_ok) {
            break;
        }

        // e. Compute affine step RHS (sets step_rhs.s = affine_ds, etc.)
        step_rhs.affine_step_rhs(residuals, variables, data.cones, stream);

        // f. Compute centering direction
        sigma.setToConstant(1.0, stream);
        step_rhs.combined_step_rhs(residuals, variables, data.cones,
                                    step_lhs, sigma, mu_step, m_zero, stream);

        // Centering KKT solve: skip HSDE τ/κ coupling
        data.cones.Δs_from_Δz_offset(Δs_const_term, step_rhs.s, workz, variables.z, stream);
        waxpby(workz, 1.0, Δs_const_term, -1.0, step_rhs.z, stream);

        pack_interleaved_kernel(kkt_rhs.data(), step_rhs.x.data(), workz.data(),
                                n, m_dim, 0, batchSize, stream);
        // Use separate buffers for RHS and solution — cuDSS IR needs original RHS
        kkt->solve(kkt_rhs.data(), kkt_sol.data(), stream);
        unpack_interleaved_kernel(x1.data(), z1.data(), kkt_sol.data(),
                                  n, m_dim, 0, batchSize, stream);

        cudaMemcpyAsync(step_lhs.x.data(), x1.data(),
                        sizeof(double) * n * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(step_lhs.z.data(), z1.data(),
                        sizeof(double) * m_dim * batchSize,
                        cudaMemcpyDeviceToDevice, stream);

        data.cones.mul_Hs(step_lhs.s, step_lhs.z, workz, stream);
        waxpby(step_lhs.s, -1.0, step_lhs.s, -1.0, Δs_const_term, stream);
        step_lhs.τ.setToConstant(0.0, stream);
        step_lhs.κ.setToConstant(0.0, stream);

        // g. Compute step length and sanitize on device (NaN/negative → 0)
        // Use is_affine_step=false so max_step_fraction (0.99) is applied,
        // matching CPU which applies it for Combined and Centering directions.
        getStepLength(step_lhs, false, alpha, stream);
        sanitize_alpha(alpha, stream);

        // Zero any NaN step directions to prevent NaN propagation.
        zero_nan_step(step_lhs.x, step_lhs.z, step_lhs.s, n, m_dim, stream);

        // The centering KKT solve above (pack/solve/unpack) writes step_lhs.x
        // and step_lhs.z but NOT step_lhs.z_x — the smoothing loop skips the
        // `recover_dz_x_combined_*` chain that runIPMLoop normally runs. Without
        // this reset, `add_step` below adds whatever stale value step_lhs.z_x
        // last held (e.g. runIPMLoop's last combined-step recovery) into
        // variables.z_x, corrupting the direct-x dual at every centering
        // iteration. A proper future fix would recover dz_x for the centering
        // direction; meanwhile, holding z_x at the converged value during
        // smoothing keeps the backward cache consistent.
        if (variables.totalXConeNumel() > 0) {
            step_lhs.z_x.setToConstant(0.0, stream);
        }

        // h. Take the step (α=0 batches are unchanged)
        variables.add_step(step_lhs, alpha, stream);
    }

    // Step 3: Compute final μ on device
    residuals.update(variables, data, cusparse_handle_, cublas_handle_, stream);
    variables.calc_mu(mu, residuals, data.cones, stream);

    // Step 4: Cache the smoothed iterate into DiffState
    // These are already in equilibrated space (τ=1), which is what backward() needs
    cudaMemcpyAsync(state.smoothing_x.data(), variables.x.data(),
                    sizeof(double) * n * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.smoothing_z.data(), variables.z.data(),
                    sizeof(double) * m_dim * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.smoothing_s.data(), variables.s.data(),
                    sizeof(double) * m_dim * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(state.smoothing_mu.data(), mu.data(),
                    sizeof(double) * batchSize,
                    cudaMemcpyDeviceToDevice, stream);
    state.smoothing_cached = true;
}

void CompiledSolver::runIPMLoop(cudaStream_t stream) {
    // Initialize step control parameters
    int64_t iter = 0;
    sigma.setToConstant(1.0, stream);
    alpha.setToConstant(0.0, stream);

    // Pre-compute neg_q = -q for use in batched KKT solves
    waxpby(neg_q, -1.0, data.q, 0.0, neg_q, stream);

    // Per-batch strategy override for any GenPower cone (direct-x or slack).
    // Mirrors CPU `strategy_checkpoint_small_step`: a batch starts in
    // PrimalDual and switches to Dual when its α drops below
    // `min_switch_step_length`. The device array `pd_enabled_per_batch_`
    // (1=PrimalDual, 0=Dual) is mutated by `switch_scaling_on_small_step_kernel`
    // after each step length compute, and read by
    // `compute_xgenpow_pd_axes_kernel` (direct-x) and
    // `try_compute_pd_axes_genpow_kernel` (slack).
    bool has_xcone_genpow = false;
    for (const auto& xc : data.cones.x_cones) {
        if (xc.kind == XConeKind::GenPower) { has_xcone_genpow = true; break; }
    }
    bool has_slack_genpow = data.cones.numGenPowerCones > 0;
    bool has_any_genpow = has_xcone_genpow || has_slack_genpow;
    ScalingStrategy scaling = data.cones.allows_primal_dual_scaling() ?
        ScalingStrategy::PrimalDual : ScalingStrategy::Dual;
    int8_t* pd_enabled_per_batch = nullptr;
    if (has_any_genpow && data.cones.allows_primal_dual_scaling()) {
        pd_enabled_per_batch = pd_enabled_per_batch_.get();
        init_pd_enabled_per_batch_kernel(
            pd_enabled_per_batch, data.batchSize, /*value=*/1, stream);
    }

    auto solve_start_time = std::chrono::high_resolution_clock::now();
    bool are_all_done = false;

#ifdef MOREAU_ENABLE_PROFILING
    // Timing accumulators (in milliseconds) - used for profiling when enabled
    float time_residuals = 0, time_convergence = 0, time_scaling = 0;
    float time_kkt_factor = 0, time_kkt_solve2 = 0, time_combined_solve = 0;
    float time_step_length = 0, time_add_step = 0, time_rhs_prep = 0;

    cudaEvent_t t_start, t_end;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);

    auto record_time = [&](float& accum) {
        cudaEventRecord(t_end, stream);
        cudaEventSynchronize(t_end);
        float ms;
        cudaEventElapsedTime(&ms, t_start, t_end);
        accum += ms;
    };
#define PROFILE_START() cudaEventRecord(t_start, stream)
#define PROFILE_END(accum) record_time(accum)
#else
#define PROFILE_START() ((void)0)
#define PROFILE_END(accum) ((void)0)
#endif

    kkt->setupL2Persistence(stream);

    // Initialize per-batch scaling success flags for deferred check pattern
    if (data.cones.h_scaling_success_pinned) {
        std::fill_n(data.cones.h_scaling_success_pinned, data.batchSize, 1);
    }

    const bool yolo_mode = settings.yolo;
    const uint32_t yolo_max = settings.yoloNumIters;

    // YOLO mode: seed snapshot buffers from initial variables so that a
    // first-iteration NaN restores the actual starting point, not stale data.
    if (yolo_mode) {
        yolo_snapshot_if_valid_kernel(
            variables.x.data(), variables.s.data(), variables.z.data(),
            variables.τ.data(), variables.κ.data(), variables.z_x.data(),
            solution.x_raw.data(), solution.s_raw.data(), solution.z_raw.data(),
            solution.τ_raw.data(), solution.κ_raw.data(), yolo_zx_raw_.data(),
            d_yolo_has_nan_.get(),
            data.n, data.m, variables.totalXConeNumel(), data.batchSize, stream);
    }

    while (true) {
        // Update residuals
        PROFILE_START();
        residuals.update(variables, data, cusparse_handle_, cublas_handle_, stream);
        PROFILE_END(time_residuals);
#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            std::cerr << "\n========== GPU ITER " << iter << " ==========\n";
            debugPrintVec("x", variables.x.data(), data.n, data.batchSize, stream);
            debugPrintVec("s", variables.s.data(), data.m, data.batchSize, stream);
            debugPrintVec("z", variables.z.data(), data.m, data.batchSize, stream);
            debugPrintScalar("τ", variables.τ.data(), stream);
            debugPrintScalar("κ", variables.κ.data(), stream);
            debugPrintVec("residuals.rx", residuals.rx.data(), data.n, data.batchSize, stream);
            debugPrintVec("residuals.rz", residuals.rz.data(), data.m, data.batchSize, stream);
            debugPrintScalar("residuals.rτ", residuals.rτ.data(), stream);
            debugPrintScalar("residuals.dot_sz", residuals.dot_sz.data(), stream);
        }
#endif

        // Calculate duality gap (scaled)
        variables.calc_mu(mu, residuals, data.cones, stream);

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            debugPrintScalar("μ", mu.data(), stream);
        }
#endif

        if (!yolo_mode) {
            // Record scalar values from most recent iteration
            info.save_scalars(mu.data(), alpha.data(), sigma.data(), iter, stream);

            // Convergence check and printing
            PROFILE_START();
            info.update(data, variables, residuals, stream);
            info.print_status(settings, stream);

            // Best-iterate snapshot: for any batch still Unsolved whose
            // metrics satisfy the reduced tolerances, copy current variables
            // and metrics into solution.*_raw and info.best_*. Sets
            // solution_saved[b]=1 so the downstream save_terminated call
            // does not clobber the snapshot with a bad terminal iterate.
            save_best_iterate(variables, solution, info, data.batchSize, stream);

            // Termination checks
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - solve_start_time;
            double solve_time_seconds = elapsed.count();

            are_all_done = info.are_all_done(residuals, settings, iter, solve_time_seconds, stream);
            PROFILE_END(time_convergence);

            // Save solution for newly terminated batches
            save_terminated_solutions(variables, solution, info, info.status_device, data.batchSize, stream);

            if (are_all_done) {
                break;
            }

            // Restore working variables for terminated batches from saved solution.
            // This prevents converged batches from drifting in s/z (via scaling and
            // step updates), which would corrupt the UBATCH KKT factorization for
            // still-active batches.
            if (data.batchSize > 1) {
                restore_terminated_variables_kernel(
                    variables.x.data(), variables.s.data(), variables.z.data(),
                    variables.τ.data(), variables.κ.data(),
                    solution.x_raw.data(), solution.s_raw.data(), solution.z_raw.data(),
                    solution.τ_raw.data(), solution.κ_raw.data(),
                    solution.solution_saved.get(),
                    data.n, data.m, data.batchSize, stream);
            }

        // Check deferred per-batch scaling success from previous iteration.
            // By now the scaling kernel has completed (an entire iteration of GPU work elapsed).
            // Only set NumericalError for still-active batches whose scaling failed.
            // Already-converged batches may have s/z at the cone boundary (restored by
            // restore_terminated_variables), which legitimately fails scaling but they
            // already have valid saved solutions.
            if (data.cones.h_scaling_success_pinned) {
                bool any_failure = false;
                for (int64_t b = 0; b < data.batchSize; b++) {
                    if (data.cones.h_scaling_success_pinned[b] == 0) {
                        any_failure = true;
                        break;
                    }
                }
                if (any_failure) {
                    set_unsaved_status_kernel(info.status_device,
                        solution.solution_saved.get(),
                        static_cast<int32_t>(SolverStatus::NumericalError),
                        data.batchSize, data.cones.h_scaling_success_pinned,
                        stream);
                }
                // Reset all flags for next iteration
                std::fill_n(data.cones.h_scaling_success_pinned, data.batchSize, 1);
            }
        } else {
            // YOLO mode: skip all convergence checking, break at fixed iteration count
            if (iter >= yolo_max) break;
        }

        // Update the cone scalings (non-blocking: result checked at start of next iteration)
        PROFILE_START();
        // NOTE: scale_cones can take `kkt.get()` as the last arg to fuse
        // the x-cone Hs / expansion scatter into the scaling kernel.
        // Historically this fusion regressed at k=5000 because the SOC
        // scaling kernel was single-thread per cone, so the extra ~15k
        // scatter writes serialized on that thread. Both the scaling
        // kernel and the refresh kernels are now block-parallel
        // (`SOC_PARALLEL_BLOCK_SIZE` threads / cone), so the fusion is
        // worth re-benchmarking before declaring it dead.
        variables.scale_cones(data.cones, mu, scaling, stream,
                              /*kkt_solver=*/nullptr,
                              pd_enabled_per_batch);
        PROFILE_END(time_scaling);

#ifndef NDEBUG
        cudaError_t err_after = cudaGetLastError();
        if (err_after != cudaSuccess && settings.verbose) {
            cudaStreamSynchronize(stream);
            std::cout << "CUDA error AFTER scale_cones: " << cudaGetErrorString(err_after) << "\n";
        }
#endif

        iter += 1;

        // Direct-x cones: write current xcone_Hs values into KKT.values
        // at the x-cone slots (on top of P's baseline) so updateFactorOnly
        // factors a KKT with the correct Σ E_J' H_J E_J contribution.
        kkt->refresh_xcone_hs(data.cones, stream);

        // Update the KKT system: factor + solve constant RHS [-q; b] → (x2, z2)
        PROFILE_START();
        bool is_kkt_update_success = kkt->updateFactorOnly(
            data.cones, variables.s, variables.z, mu, scaling,
            settings.ipm.staticRegularizationEnable,
            settings.ipm.staticRegularizationConstant,
            settings.ipm.staticRegularizationProportional,
            data.q, data.b,
            workx, kkt_rhs, kkt_sol, x2, z2,
            stream
        );
        PROFILE_END(time_kkt_factor);

        // Check for KKT factorization failure
        if (!yolo_mode && !is_kkt_update_success) {
            if (settings.verbose) std::cerr << "NumericalError: KKT factor failure at iter " << iter << std::endl;
            set_unsaved_status_kernel(info.status_device,
                solution.solution_saved.get(),
                static_cast<int32_t>(SolverStatus::NumericalError),
                data.batchSize, stream);
            break;
        }

        // Calculate the affine step RHS
        step_rhs.affine_step_rhs(residuals, variables, data.cones, stream);

        // Direct-x KKT RHS correction for the affine step. Mirror of
        // solveKKT()'s is_affine_step branch, but applied here because
        // the affine solve is done via solve_combined below (outside
        // solveKKT). Subtract c_J = variables.z_x[k] from step_rhs.x at
        // each x-cone's indices BEFORE the pack so the reduced KKT RHS
        // carries the correct `rx - Σ E' c_J` value.
        {
            const int64_t total_xcone_numel = variables.totalXConeNumel();
            if (total_xcone_numel > 0 && data.cones.d_xcone_indices != nullptr) {
                subtract_xcone_affine_from_workx(
                    step_rhs.x.data(), variables.z_x.data(),
                    data.cones.d_xcone_indices,
                    data.batchSize, data.n, total_xcone_numel, stream);
            }
        }

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            std::cerr << "--- GPU affine_step_rhs ---\n";
            debugPrintVec("step_rhs.x", step_rhs.x.data(), data.n, data.batchSize, stream);
            debugPrintVec("step_rhs.s", step_rhs.s.data(), data.m, data.batchSize, stream);
            debugPrintVec("step_rhs.z", step_rhs.z.data(), data.m, data.batchSize, stream);
            debugPrintScalar("step_rhs.τ", step_rhs.τ.data(), stream);
            debugPrintScalar("step_rhs.κ", step_rhs.κ.data(), stream);
        }
#endif

        // Fused: z_part = s - step_rhs.z, pack interleaved with step_rhs.x  (2→1 kernel)
        waxpby_and_pack_interleaved_kernel(kkt_rhs.data(), step_rhs.x.data(),
                                           variables.s.data(), step_rhs.z.data(),
                                           1.0, -1.0,
                                           data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

        PROFILE_START();
        // Virtual dispatch: Riccati fuses affine + constant solve,
        // CuDSS/Woodbury default to just solving the affine RHS.
        kkt->solve_combined(kkt_rhs.data(), kkt_sol.data(),
                           x2.data(), z2.data(), stream);
        PROFILE_END(time_kkt_solve2);

        // NaN from cuDSS KKT solves is detected per-batch by the GPU convergence
        // check kernel (check_termination_kernel) which inspects residuals and sets
        // NumericalError for individual affected batches without poisoning others.

        // Unpack affine solution → (x1, z1)
        unpack_interleaved_kernel(x1.data(), z1.data(), kkt_sol.data(),
                                 data.n, data.m, 2 * data.cones.numSparseSoc + 9 * data.cones.numSparseGenPow + 2 * data.cones.numSparseXSoc + 9 * data.cones.numSparseXGenPow, data.batchSize, stream);

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            std::cerr << "--- GPU KKT intermediate (x1,z1,x2,z2) ---\n";
            debugPrintVec("x1", x1.data(), data.n, data.batchSize, stream);
            debugPrintVec("z1", z1.data(), data.m, data.batchSize, stream);
            debugPrintVec("x2", x2.data(), data.n, data.batchSize, stream);
            debugPrintVec("z2", z2.data(), data.m, data.batchSize, stream);
        }
#endif

        // Complete KKT solve for affine step
        bool is_kkt_solve_success = solveKKTFromPrecomputed(step_lhs, step_rhs, true, stream);

        // Check for KKT solve failure
        if (!yolo_mode && !is_kkt_solve_success) {
            if (settings.verbose) std::cerr << "NumericalError: KKT affine solve failure at iter " << iter << std::endl;
            set_unsaved_status_kernel(info.status_device,
                solution.solution_saved.get(),
                static_cast<int32_t>(SolverStatus::NumericalError),
                data.batchSize, stream);
            break;
        }

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            std::cerr << "--- GPU step_lhs (affine) ---\n";
            debugPrintVec("step_lhs.x", step_lhs.x.data(), data.n, data.batchSize, stream);
            debugPrintVec("step_lhs.s", step_lhs.s.data(), data.m, data.batchSize, stream);
            debugPrintVec("step_lhs.z", step_lhs.z.data(), data.m, data.batchSize, stream);
            debugPrintScalar("step_lhs.τ", step_lhs.τ.data(), stream);
            debugPrintScalar("step_lhs.κ", step_lhs.κ.data(), stream);
        }
#endif

        // Calculate step length
        getStepLength(step_lhs, true, alpha, stream);

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            debugPrintScalar("α (affine)", alpha.data(), stream);
        }
#endif

        // Calculate centering parameter and Mehrotra correction
        calc_solver_parameters_kernel(sigma.data(), m.data(), alpha.data(), iter, data.batchSize, stream);

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            debugPrintScalar("σ", sigma.data(), stream);
            debugPrintScalar("m (Mehrotra)", m.data(), stream);
        }
#endif

        // Calculate the combined step RHS
        PROFILE_START();
        step_rhs.combined_step_rhs(residuals, variables, data.cones, step_lhs, sigma, mu, m, stream);
        PROFILE_END(time_rhs_prep);

        // NaN in combined step RHS is caught per-batch by the GPU convergence
        // check kernel at the top of the next iteration.

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            std::cerr << "--- GPU combined_step_rhs ---\n";
            debugPrintVec("step_rhs.x", step_rhs.x.data(), data.n, data.batchSize, stream);
            debugPrintVec("step_rhs.s", step_rhs.s.data(), data.m, data.batchSize, stream);
            debugPrintVec("step_rhs.z", step_rhs.z.data(), data.m, data.batchSize, stream);
            debugPrintScalar("step_rhs.τ", step_rhs.τ.data(), stream);
            debugPrintScalar("step_rhs.κ", step_rhs.κ.data(), stream);
        }
#endif

        // Solve KKT for combined step
        PROFILE_START();
        is_kkt_solve_success = solveKKT(step_lhs, step_rhs, false, stream);
        PROFILE_END(time_combined_solve);

        // Check for combined KKT solve failure
        if (!yolo_mode && !is_kkt_solve_success) {
            if (settings.verbose) std::cerr << "NumericalError: KKT combined solve failure at iter " << iter << std::endl;
            set_unsaved_status_kernel(info.status_device,
                solution.solution_saved.get(),
                static_cast<int32_t>(SolverStatus::NumericalError),
                data.batchSize, stream);
            break;
        }

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            std::cerr << "--- GPU step_lhs (combined) ---\n";
            debugPrintVec("step_lhs.x", step_lhs.x.data(), data.n, data.batchSize, stream);
            debugPrintVec("step_lhs.s", step_lhs.s.data(), data.m, data.batchSize, stream);
            debugPrintVec("step_lhs.z", step_lhs.z.data(), data.m, data.batchSize, stream);
            debugPrintScalar("step_lhs.τ", step_lhs.τ.data(), stream);
            debugPrintScalar("step_lhs.κ", step_lhs.κ.data(), stream);
        }
#endif

        // Compute final step length
        PROFILE_START();
        getStepLength(step_lhs, false, alpha, stream);
        PROFILE_END(time_step_length);

#ifdef MOREAU_DEBUG
        if (iterDebugEnabled()) {
            debugPrintScalar("α (combined)", alpha.data(), stream);
        }
#endif

        // Small-step termination: if cone backtracking collapsed α below the
        // termination threshold, the Newton direction is exhausted. Mark the
        // batch InsufficientProgress so it stops cycling; post_process promotes
        // to AlmostSolved if reduced tolerances are met. Matches CPU's
        // strategy_checkpoint_small_step.
        if (!yolo_mode) {
            small_step_terminate_kernel(alpha.data(), info.status_device,
                                        info.d_iterations_per_batch,
                                        settings.ipm.minTerminateStepLength,
                                        iter, data.batchSize, stream);
        }

        // Zero alpha for already-converged batch elements to prevent variable drift.
        // Without this, converged elements keep getting stepped, eventually causing
        // cone scaling or KKT factorization failure that aborts the entire batch.
        zero_alpha_for_terminated_kernel(alpha.data(), info.status_device, data.batchSize, stream);

        // Per-batch PrimalDual→Dual fallback (CPU `strategy_checkpoint_small_step`).
        // Mutates `alpha[b] = 0` and `pd_enabled_per_batch[b] = 0` for any batch
        // where α dropped below `min_switch_step_length` while still in PrimalDual.
        // Next iter's scale_cones reads the updated `pd_enabled_per_batch`.
        if (pd_enabled_per_batch != nullptr) {
            switch_scaling_on_small_step_kernel(
                alpha.data(), pd_enabled_per_batch, data.batchSize,
                settings.ipm.minSwitchStepLength, stream);
        }

        // Apply the combined step
        PROFILE_START();
        variables.add_step(step_lhs, alpha, stream);
        PROFILE_END(time_add_step);

        // YOLO mode: snapshot variables to raw buffers if all valid (no NaN).
        // Runs every iteration, fully async, zero host sync.
        if (yolo_mode) {
            yolo_snapshot_if_valid_kernel(
                variables.x.data(), variables.s.data(), variables.z.data(),
                variables.τ.data(), variables.κ.data(), variables.z_x.data(),
                solution.x_raw.data(), solution.s_raw.data(), solution.z_raw.data(),
                solution.τ_raw.data(), solution.κ_raw.data(), yolo_zx_raw_.data(),
                d_yolo_has_nan_.get(),
                data.n, data.m, variables.totalXConeNumel(), data.batchSize, stream);
        }

    }

    // Tear down L2 cache persistence
    kkt->resetL2Persistence(stream);

#ifdef MOREAU_ENABLE_PROFILING
    // Env-gated per-phase dump so every solve() does not spam stderr when
    // profiling-enabled builds are used for normal work. Set
    // MOREAU_PROFILE_ENABLED=1 to activate.
    if (const char* env = std::getenv("MOREAU_PROFILE_ENABLED");
        env != nullptr && env[0] != '0' && env[0] != '\0') {
        float total = time_residuals + time_convergence + time_scaling
                    + time_kkt_factor + time_kkt_solve2 + time_combined_solve
                    + time_step_length + time_add_step + time_rhs_prep;
        std::fprintf(stderr,
            "[moreau-profile] iters=%ld total=%.3fms "
            "residuals=%.3f convergence=%.3f scaling=%.3f "
            "kkt_factor=%.3f kkt_solve2=%.3f combined_solve=%.3f "
            "step_length=%.3f add_step=%.3f rhs_prep=%.3f\n",
            (long)iter, total,
            time_residuals, time_convergence, time_scaling,
            time_kkt_factor, time_kkt_solve2, time_combined_solve,
            time_step_length, time_add_step, time_rhs_prep);
    }
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
#endif
#undef PROFILE_START
#undef PROFILE_END

    if (!yolo_mode) {
        // Save current iterate for any batches that never had their solution saved
        // (e.g., still-iterating batches when a scaling failure aborts the loop).
        // Without this, post_process reads stale/uninitialized x_raw/z_raw/τ_raw,
        // producing NaN in the output.
        save_terminated_solutions(variables, solution, info, info.status_device, data.batchSize, stream);
    } else {
        // YOLO mode: set status, iteration count, and zero stale info fields.
        // Raw variable buffers already contain the last NaN-free snapshot from
        // yolo_snapshot_if_valid_kernel (called every iteration inside the loop).
        set_all_status_kernel(info.status_device,
            static_cast<int32_t>(SolverStatus::MaxIterations),
            data.batchSize, stream);

        // Restore the direct-x dual from its last NaN-free snapshot. The
        // x/s/z/τ/κ snapshot is consumed from solution.*_raw by
        // solution.post_process; z_x has no raw slot there and is read
        // straight from variables.z_x by the result bindings, so copy the
        // snapshot back over it. No-op when there are no direct-x cones.
        if (variables.totalXConeNumel() > 0) {
            cudaMemcpyAsync(variables.z_x.data(), yolo_zx_raw_.data(),
                            sizeof(double) * variables.totalXConeNumel()
                                * data.batchSize,
                            cudaMemcpyDeviceToDevice, stream);
        }

        // info.update was never called, so set iteration count explicitly
        info.iterations = static_cast<int32_t>(iter);

        // info.update was never called, so cost/residual values are stale — zero them
        cudaMemsetAsync(solution.cost_primal_raw.data(), 0, sizeof(double) * data.batchSize, stream);
        cudaMemsetAsync(solution.cost_dual_raw.data(), 0, sizeof(double) * data.batchSize, stream);
        cudaMemsetAsync(info.res_primal.data(), 0, sizeof(double) * data.batchSize, stream);
        cudaMemsetAsync(info.res_dual.data(), 0, sizeof(double) * data.batchSize, stream);
    }

    // Post-process: check for "almost" convergence and extract solution
    if (!yolo_mode) {
        // Best-iterate fallback: any batch with a saved reduced-tolerance
        // snapshot that terminated in InsufficientProgress / NumericalError /
        // MaxIterations / MaxTime gets promoted to AlmostSolved with the
        // snapshot's metrics substituted into Info. solution.*_raw already
        // holds the snapshot variables (not clobbered by save_terminated due
        // to the solution_saved flag set by save_best_iterate).
        restore_best_iterate(info, data.batchSize, stream);
        info.post_process(residuals, settings, stream);
    } else {
        info.sync_status_to_host(stream);
    }

    solution.post_process(
        data.n, data.m, data.batchSize,
        data.equilibration,
        variables,
        info,
        settings.ipm.equilibrationSettings.enable,
        stream
    );

    // Update solve time
    auto end_time = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end_time - solve_start_time).count();
    info.finalize(solve_time);

    solution.finalize(info);

    // Sync solver status with info status — no-op now (status field
    // removed; `solution.finalize(info)` above already mirrors
    // info.status into solution).
    cudaStreamSynchronize(stream);

    // Print footer
    info.print_footer(settings, stream);
}

// ============================================================================
// Solver wrapper implementation (single-problem convenience API)
// ============================================================================

Solver::Solver(
    int64_t n, int64_t m,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP, const double* d_P_values,
    const double* d_q,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA, const double* d_A_values,
    const double* d_b,
    const Cones& cones,
    const Settings& settings
) : n_(n), m_(m), d_q_(nullptr), d_b_(nullptr) {
    // Create CompiledSolver with batch_size=1
    impl_ = std::make_unique<CompiledSolver>(
        n, m, 1,
        P_ro, P_ci, nnzP,
        A_ro, A_ci, nnzA,
        cones, settings
    );

    // Allocate GPU buffers for q, b with exception-safe cleanup
    auto cleanup = [this]() {
        if (d_q_) { cudaFree(d_q_); d_q_ = nullptr; }
        if (d_b_) { cudaFree(d_b_); d_b_ = nullptr; }
    };

    try {
        cudaError_t err;
        err = cudaMalloc(&d_q_, sizeof(double) * n);
        if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_q_ failed");

        err = cudaMalloc(&d_b_, sizeof(double) * m);
        if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_b_ failed");

        // Copy input data
        cudaMemcpy(d_q_, d_q, sizeof(double) * n, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_b_, d_b, sizeof(double) * m, cudaMemcpyDeviceToDevice);

        // Call setup with P and A values
        impl_->setup(d_P_values, d_A_values);
    } catch (...) {
        cleanup();
        throw;
    }
}

Solver::~Solver() {
    if (d_q_) cudaFree(d_q_);
    if (d_b_) cudaFree(d_b_);
}

void Solver::solve(cudaStream_t stream) {
    impl_->solve(d_q_, d_b_, stream);
}

} // namespace moreau
