/**
 * @file info.cpp
 * @brief Implementation of solver information updates
 */

#include "moreau/solver/info.hpp"
#include "moreau/solver/data.hpp"
#include "moreau/solver/solver.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/kkt/kkt_solver.hpp"
#include "moreau/variables/variables.hpp"
#include "moreau/residuals/residuals.hpp"
#include "moreau/solver/solver_kernels.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <map>

namespace moreau {

// Version string
static constexpr const char* MOREAU_VERSION = "0.4.0-beta.1";

// Helper to format scientific notation to match Rust's LowerExp format
static std::string format_exp(double val) {
    if (!std::isfinite(val)) {
        std::ostringstream oss;
        oss << std::setw(10) << val;
        return oss.str();
    }

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2) << val;
    std::string s = oss.str();

    // Find the exponent position
    size_t e_pos = s.find('e');
    if (e_pos != std::string::npos) {
        // Ensure sign after e
        if (s[e_pos + 1] != '-' && s[e_pos + 1] != '+') {
            s.insert(e_pos + 1, "+");
        }
        // Ensure at least 2 digits in exponent
        size_t sign_pos = e_pos + 1;
        size_t exp_start = sign_pos + 1;
        if (s.length() - exp_start == 1) {
            s.insert(exp_start, "0");
        }
    }
    return s;
}

static std::string format_exp_signed(double val) {
    if (!std::isfinite(val)) {
        std::ostringstream oss;
        oss << std::setw(12) << std::showpos << val;
        return oss.str();
    }

    std::ostringstream oss;
    oss << std::scientific << std::showpos << std::setprecision(4) << val;
    std::string s = oss.str();

    // Find the exponent position
    size_t e_pos = s.find('e');
    if (e_pos != std::string::npos) {
        // Ensure sign after e
        if (s[e_pos + 1] != '-' && s[e_pos + 1] != '+') {
            s.insert(e_pos + 1, "+");
        }
        // Ensure at least 2 digits in exponent
        size_t sign_pos = e_pos + 1;
        size_t exp_start = sign_pos + 1;
        if (s.length() - exp_start == 1) {
            s.insert(exp_start, "0");
        }
    }
    return s;
}

static void print_banner(bool verbose) {
    if (!verbose) return;

    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│             Moreau v" << MOREAU_VERSION << "  ─  Conic Solver (CUDA)           │\n";
#ifndef NDEBUG
    std::cout << "│                    ⚠ debug build                            │\n";
#else
    std::cout << "│                                                             │\n";
#endif
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";
}

void Info::update(const SolverData& data, const Variables& variables, const Residuals& residuals, cudaStream_t stream) {
    // Store previous iteration values
    cudaMemcpyAsync(prev_cost_primal.data(), cost_primal.data(), sizeof(double) * cost_primal.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(prev_cost_dual.data(), cost_dual.data(), sizeof(double) * cost_dual.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(prev_res_primal.data(), res_primal.data(), sizeof(double) * res_primal.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(prev_res_dual.data(), res_dual.data(), sizeof(double) * res_dual.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(prev_gap_abs.data(), gap_abs.data(), sizeof(double) * gap_abs.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(prev_gap_rel.data(), gap_rel.data(), sizeof(double) * gap_rel.batchSize(), cudaMemcpyDeviceToDevice, stream);

    // Direct-x metadata for the primal-inf certificate denominator.
    // Both pointers are nullptr (and totalXConeNumel == 0) on slack-only runs.
    const int64_t total_xcone_numel = variables.totalXConeNumel();
    const double* z_x_ptr = (total_xcone_numel > 0) ? variables.z_x.data() : nullptr;
    const int64_t* xcone_idx_ptr = (total_xcone_numel > 0) ? data.cones.d_xcone_indices : nullptr;

    // Update convergence metrics using kernel
    update_info_kernel(
        cost_primal.data(),
        cost_dual.data(),
        res_primal.data(),
        res_dual.data(),
        res_primal_inf.data(),
        res_dual_inf.data(),
        gap_abs.data(),
        gap_rel.data(),
        ktratio.data(),
        residuals.rx.data(),
        residuals.rz.data(),
        residuals.rτ.data(),
        residuals.dot_qx.data(),
        residuals.dot_bz.data(),
        variables.τ.data(),
        variables.κ.data(),
        residuals.dot_xPx.data(),
        residuals.rx_inf.data(),
        residuals.rz_inf.data(),
        residuals.Px.data(),
        variables.x.data(),
        variables.z.data(),
        variables.s.data(),
        z_x_ptr,
        xcone_idx_ptr,
        total_xcone_numel,
        data.normb.data(),
        data.normq.data(),
        data.equilibration.d.data(),
        data.equilibration.dinv.data(),
        data.equilibration.e.data(),
        data.equilibration.einv.data(),
        data.equilibration.c.data(),
        data.n,
        data.m,
        data.batchSize,
        stream
    );
}

void Info::save_scalars(const double* mu_vals, const double* alpha_vals, const double* sigma_vals, int32_t iter, cudaStream_t stream) {
    // Copy mu, step_length (alpha), and sigma from device pointers
    cudaMemcpyAsync(mu.data(), mu_vals, sizeof(double) * mu.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(step_length.data(), alpha_vals, sizeof(double) * step_length.batchSize(), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(sigma.data(), sigma_vals, sizeof(double) * sigma.batchSize(), cudaMemcpyDeviceToDevice, stream);

    // Update iteration count
    iterations = iter;
}

static const char* kktSolverTypeName(KKTSolverType type) {
    switch (type) {
        case KKTSolverType::Auto: return "Auto";
        case KKTSolverType::CuDSS: return "cuDSS";
        case KKTSolverType::Riccati: return "Riccati";
        default: return "Unknown";
    }
}

void Info::print_configuration(const Settings& settings, const SolverData& data, const Cones& cones, KKTSolverType actualKktSolverType) {
    if (!settings.verbose) {
        return;
    }

    // Print banner first
    print_banner(settings.verbose);

    // Print batch size info if batching
    if (batchSize > 1) {
        std::cout << "\nbatch size: " << batchSize << "\n";
    }

    std::cout << "\nproblem:\n";
    std::cout << "  variables     = " << data.n << "\n";
    std::cout << "  constraints   = " << data.m << "\n";
    std::cout << "  nnz(P)        = " << data.P.nnz() << "\n";
    std::cout << "  nnz(A)        = " << data.A.nnz() << "\n";

    int64_t total_cones = cones.numZeroCones + cones.numNonnegCones + cones.numSocCones +
                          cones.numExpCones + cones.numPowerCones + cones.numPsdCones;
    std::cout << "  cones (total) = " << total_cones << "\n";

    // Print cone breakdown with aligned formatting
    if (cones.numZeroCones > 0) {
        std::cout << "          Zero : " << cones.numZeroCones << ",  numel = " << cones.numZeroCones << "\n";
    }
    if (cones.numNonnegCones > 0) {
        std::cout << "    Nonnegative : " << cones.numNonnegCones << ",  numel = " << cones.numNonnegCones << "\n";
    }
    if (cones.numSocCones > 0) {
        std::cout << "   SecondOrder : " << cones.numSocCones << ",  numel = " << cones.totalSocDim << "\n";
    }
    if (cones.numExpCones > 0) {
        std::cout << "   Exponential : " << cones.numExpCones << ",  numel = " << (cones.numExpCones * 3) << "\n";
    }
    if (cones.numPowerCones > 0) {
        std::cout << "         Power : " << cones.numPowerCones << ",  numel = " << (cones.numPowerCones * 3) << "\n";
    }
    if (cones.numPsdCones > 0) {
        std::cout << "   PSDTriangle : " << cones.numPsdCones << ",  numel = " << cones.totalPsdSvecDim << "\n";
    }

    std::cout << "\nsettings:\n";
    std::cout << "  linear algebra: direct / " << kktSolverTypeName(actualKktSolverType) << "\n";
    std::cout << "  max iter = " << settings.maxIter << ", time limit = ";
    if (std::isinf(settings.timeLimit)) {
        std::cout << "Inf";
    } else {
        std::cout << std::fixed << std::setprecision(1) << settings.timeLimit << "s";
    }
    std::cout << ", max step = " << std::fixed << std::setprecision(3) << settings.ipm.maxStepFraction << "\n";
    std::cout << std::scientific << std::setprecision(1);
    std::cout << "  tol_feas = " << settings.ipm.tolFeas
              << ", tol_gap_abs = " << settings.ipm.tolGapAbs
              << ", tol_gap_rel = " << settings.ipm.tolGapRel << "\n";

    std::cout << "\n";
}

void Info::print_status_header(const Settings& settings) {
    if (!settings.verbose) {
        return;
    }

    if (batchSize == 1) {
        // Single problem header
        std::cout << "iter    ";
        std::cout << "pcost        ";
        std::cout << "dcost       ";
        std::cout << "gap       ";
        std::cout << "pres      ";
        std::cout << "dres      ";
        std::cout << "k/t       ";
        std::cout << " μ       ";
        std::cout << "step      ";
        std::cout << "\n";
        std::cout << "─────────────────────────────────────────────────────────────────────────────────────────────\n";
    } else {
        // Batched header - show converged count and gap range
        std::cout << "iter    ";
        std::cout << "done        ";
        std::cout << "gap_min   ";
        std::cout << "gap_max   ";
        std::cout << "pres      ";
        std::cout << "dres      ";
        std::cout << "k/t       ";
        std::cout << " μ       ";
        std::cout << "step      ";
        std::cout << "\n";
        std::cout << "─────────────────────────────────────────────────────────────────────────────────────────────\n";
    }
    std::cout << std::flush;
}

void Info::print_status(const Settings& settings, cudaStream_t stream) {
    if (!settings.verbose) {
        return;
    }

    if (batchSize == 1) {
        // Single problem: show exact values
        double mu_val, step_length_val, cost_primal_val, cost_dual_val;
        double res_primal_val, res_dual_val, gap_abs_val, gap_rel_val, ktratio_val;

        cudaMemcpyAsync(&mu_val, mu.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&step_length_val, step_length.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&cost_primal_val, cost_primal.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&cost_dual_val, cost_dual.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&res_primal_val, res_primal.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&res_dual_val, res_dual.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&gap_abs_val, gap_abs.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&gap_rel_val, gap_rel.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&ktratio_val, ktratio.data(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Format and print
        std::cout << std::right << std::setw(3) << iterations << "  ";
        std::cout << format_exp_signed(cost_primal_val) << "  ";
        std::cout << format_exp_signed(cost_dual_val) << "  ";
        double gap_print = std::min(gap_abs_val, gap_rel_val);
        std::cout << format_exp(gap_print) << "  ";
        std::cout << format_exp(res_primal_val) << "  ";
        std::cout << format_exp(res_dual_val) << "  ";
        std::cout << format_exp(ktratio_val) << "  ";
        std::cout << format_exp(mu_val) << "  ";

        if (iterations > 0) {
            std::cout << format_exp(step_length_val) << "  ";
        } else {
            std::cout << " ------   ";
        }
        std::cout << "\n";
    } else {
        // Batched: show statistics across all problems
        std::vector<double> mu_host(batchSize);
        std::vector<double> step_host(batchSize);
        std::vector<double> res_primal_host(batchSize);
        std::vector<double> res_dual_host(batchSize);
        std::vector<double> gap_abs_host(batchSize);
        std::vector<double> gap_rel_host(batchSize);
        std::vector<double> ktratio_host(batchSize);
        std::vector<int32_t> status_host(batchSize);

        cudaMemcpyAsync(mu_host.data(), mu.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(step_host.data(), step_length.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(res_primal_host.data(), res_primal.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(res_dual_host.data(), res_dual.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(gap_abs_host.data(), gap_abs.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(gap_rel_host.data(), gap_rel.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(ktratio_host.data(), ktratio.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(status_host.data(), status_device, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Compute per-problem gaps and count converged/running
        std::vector<double> gaps(batchSize);
        int64_t num_converged = 0;
        int64_t num_running = 0;
        for (int64_t b = 0; b < batchSize; b++) {
            gaps[b] = std::min(gap_abs_host[b], gap_rel_host[b]);
            if (status_host[b] == 0) {
                num_running++;  // Unsolved = still running
            } else {
                num_converged++;
            }
        }

        // Compute min/max values across batch
        double min_gap = *std::min_element(gaps.begin(), gaps.end());
        double max_gap = *std::max_element(gaps.begin(), gaps.end());
        double max_res_primal = *std::max_element(res_primal_host.begin(), res_primal_host.end());
        double max_res_dual = *std::max_element(res_dual_host.begin(), res_dual_host.end());
        double max_ktratio = *std::max_element(ktratio_host.begin(), ktratio_host.end());
        double max_mu = *std::max_element(mu_host.begin(), mu_host.end());
        double min_step = *std::min_element(step_host.begin(), step_host.end());

        // Format and print with convergence count
        std::cout << std::right << std::setw(3) << iterations << "  ";
        // Show converged/total instead of pcost/dcost (more useful for batch)
        std::cout << std::right << std::setw(4) << num_converged << "/" << std::left << std::setw(4) << batchSize << "   ";
        // Show gap range [min, max] for better insight
        std::cout << format_exp(min_gap) << "  ";
        std::cout << format_exp(max_gap) << "  ";
        std::cout << format_exp(max_res_primal) << "  ";
        std::cout << format_exp(max_res_dual) << "  ";
        std::cout << format_exp(max_ktratio) << "  ";
        std::cout << format_exp(max_mu) << "  ";

        if (iterations > 0) {
            std::cout << format_exp(min_step) << "  ";
        } else {
            std::cout << " ------   ";
        }
        std::cout << "\n";
    }
}

void Info::check_convergence(const Residuals& residuals, const Settings& settings, cudaStream_t stream) {
    // Use the kernel to check convergence on GPU
    // This is called within check_termination, so we don't call the kernel directly here
    // The kernel handles convergence checking as part of termination checking
}

bool Info::are_all_done(const Residuals& residuals, const Settings& settings, int64_t iter, double solve_time_seconds, cudaStream_t stream) {
    check_termination_async(residuals, settings, iter, solve_time_seconds, stream);
    return wait_for_termination(stream);
}

void Info::check_termination_async(const Residuals& residuals, const Settings& settings, int64_t iter, double solve_time_seconds, cudaStream_t stream) {
    // Reset termination counter
    reset_termination_counter();

    // Launch async termination check (no sync)
    check_termination_async_kernel(
        status_device,
        d_any_done,
        h_any_done_pinned,
        d_iterations_per_batch,
        gap_abs.data(),
        gap_rel.data(),
        res_primal.data(),
        res_dual.data(),
        ktratio.data(),
        res_primal_inf.data(),
        res_dual_inf.data(),
        residuals.dot_qx.data(),
        residuals.dot_bz.data(),
        prev_res_primal.data(),
        prev_res_dual.data(),
        prev_gap_abs.data(),
        prev_gap_rel.data(),
        settings.ipm.tolGapAbs,
        settings.ipm.tolGapRel,
        settings.ipm.tolFeas,
        settings.ipm.tolInfeasAbs,
        settings.ipm.tolInfeasRel,
        settings.ipm.tolKtRatio,
        settings.ipm.reducedTolGapAbs,
        settings.ipm.reducedTolGapRel,
        settings.ipm.reducedTolFeas,
        settings.ipm.reducedTolInfeasAbs,
        settings.ipm.reducedTolInfeasRel,
        settings.ipm.reducedTolKtRatio,
        iter,
        static_cast<int64_t>(settings.maxIter),
        solve_time_seconds,
        settings.timeLimit,
        batchSize,
        stream
    );
}

bool Info::wait_for_termination(cudaStream_t stream) {
    // Spin-poll on mapped memory — GPU writes directly to host-visible address.
    // volatile prevents the compiler from caching the read.
    volatile int32_t* flag = h_any_done_pinned;
    constexpr int SPIN_LIMIT = 100000;
    for (int i = 0; i < SPIN_LIMIT; ++i) {
        int32_t val = *flag;
        if (val >= batchSize) return true;
        // Yield occasionally to avoid starving other threads
        if ((i & 0xFF) == 0xFF) {
            cudaError_t err = cudaStreamQuery(stream);
            if (err == cudaSuccess) {
                // Stream completed — read final value
                return *flag >= batchSize;
            }
        }
    }
    // Fallback: full sync if spin-poll didn't converge
    cudaStreamSynchronize(stream);
    return *flag >= batchSize;
}

void Info::sync_status_to_host(cudaStream_t stream) {
    // Copy status from device to host
    std::vector<int32_t> status_int(batchSize);
    cudaMemcpyAsync(status_int.data(), status_device, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost, stream);

    // Copy per-batch iterations from device to host
    cudaMemcpyAsync(iterations_per_batch.data(), d_iterations_per_batch, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Convert int32_t to SolverStatus enum
    for (int64_t b = 0; b < batchSize; b++) {
        status[b] = static_cast<SolverStatus>(status_int[b]);
    }
}

void Info::post_process(const Residuals& residuals, const Settings& settings, cudaStream_t stream) {
    // Sync status to host to check which batches need post-processing
    sync_status_to_host(stream);

    // Batch copy all metrics for all batches at once (instead of per-batch copies)
    std::vector<double> gap_abs_host(batchSize);
    std::vector<double> gap_rel_host(batchSize);
    std::vector<double> res_primal_host(batchSize);
    std::vector<double> res_dual_host(batchSize);
    std::vector<double> ktratio_host(batchSize);
    std::vector<double> res_primal_inf_host(batchSize);
    std::vector<double> res_dual_inf_host(batchSize);
    std::vector<double> dot_qx_host(batchSize);
    std::vector<double> dot_bz_host(batchSize);

    cudaMemcpyAsync(gap_abs_host.data(), gap_abs.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(gap_rel_host.data(), gap_rel.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(res_primal_host.data(), res_primal.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(res_dual_host.data(), res_dual.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ktratio_host.data(), ktratio.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(res_primal_inf_host.data(), res_primal_inf.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(res_dual_inf_host.data(), res_dual_inf.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(dot_qx_host.data(), residuals.dot_qx.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(dot_bz_host.data(), residuals.dot_bz.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // Single sync for all copies

    // For each batch, if solver terminated with an error, max iterations, or max time
    // without converging, check if we meet "almost" convergence criteria (reduced tolerances)
    for (int64_t b = 0; b < batchSize; b++) {
        bool is_errored = (status[b] == SolverStatus::NumericalError ||
                           status[b] == SolverStatus::InsufficientProgress);

        if (is_errored || status[b] == SolverStatus::MaxIterations || status[b] == SolverStatus::MaxTime) {
            // Use batched host data
            double gap_abs_val = gap_abs_host[b];
            double gap_rel_val = gap_rel_host[b];
            double res_primal_val = res_primal_host[b];
            double res_dual_val = res_dual_host[b];
            double ktratio_val = ktratio_host[b];
            double res_primal_inf_val = res_primal_inf_host[b];
            double res_dual_inf_val = res_dual_inf_host[b];
            double dot_qx_val = dot_qx_host[b];
            double dot_bz_val = dot_bz_host[b];

            // Check for optimality with reduced tolerances
            bool gap_converged = (gap_abs_val < settings.ipm.reducedTolGapAbs) || (gap_rel_val < settings.ipm.reducedTolGapRel);
            bool feasible = (res_primal_val < settings.ipm.reducedTolFeas) && (res_dual_val < settings.ipm.reducedTolFeas);
            bool interior = (ktratio_val <= 1.0);

            if (gap_converged && feasible && interior) {
                status[b] = SolverStatus::AlmostSolved;
                continue;
            }

            // Check for primal infeasibility with reduced tolerances
            double ktratio_threshold = 1000.0 / settings.ipm.reducedTolKtRatio;
            if (ktratio_val > ktratio_threshold && dot_bz_val < -settings.ipm.reducedTolInfeasAbs &&
                res_primal_inf_val < -settings.ipm.reducedTolInfeasRel * dot_bz_val) {
                status[b] = SolverStatus::AlmostPrimalInfeasible;
                continue;
            }

            // Check for dual infeasibility with reduced tolerances
            if (ktratio_val > ktratio_threshold && dot_qx_val < -settings.ipm.reducedTolInfeasAbs &&
                res_dual_inf_val < -settings.ipm.reducedTolInfeasRel * dot_qx_val) {
                status[b] = SolverStatus::AlmostDualInfeasible;
                continue;
            }
        }
    }

    // Copy updated status back to device
    std::vector<int32_t> status_int(batchSize);
    for (int64_t b = 0; b < batchSize; b++) {
        status_int[b] = static_cast<int32_t>(status[b]);
    }
    cudaMemcpy(status_device, status_int.data(), sizeof(int32_t) * batchSize, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);
}

void Info::finalize(double solve_time_seconds) {
    solve_time = solve_time_seconds;
}

static const char* status_indicator(SolverStatus status) {
    switch (status) {
        case SolverStatus::Solved: return "✓";
        case SolverStatus::AlmostSolved: return "≈";
        case SolverStatus::PrimalInfeasible:
        case SolverStatus::DualInfeasible:
        case SolverStatus::AlmostPrimalInfeasible:
        case SolverStatus::AlmostDualInfeasible: return "∅";
        case SolverStatus::MaxIterations:
        case SolverStatus::MaxTime: return "⏱";
        case SolverStatus::NumericalError:
        case SolverStatus::InsufficientProgress: return "✗";
        default: return "•";
    }
}

static const char* status_name(SolverStatus status) {
    switch (status) {
        case SolverStatus::Solved: return "Solved";
        case SolverStatus::AlmostSolved: return "AlmostSolved";
        case SolverStatus::PrimalInfeasible: return "PrimalInfeasible";
        case SolverStatus::AlmostPrimalInfeasible: return "AlmostPrimalInfeasible";
        case SolverStatus::DualInfeasible: return "DualInfeasible";
        case SolverStatus::AlmostDualInfeasible: return "AlmostDualInfeasible";
        case SolverStatus::MaxIterations: return "MaxIterations";
        case SolverStatus::MaxTime: return "MaxTime";
        case SolverStatus::NumericalError: return "NumericalError";
        case SolverStatus::InsufficientProgress: return "InsufficientProgress";
        case SolverStatus::Unsolved: return "Unsolved";
        default: return "Unknown";
    }
}

// Helper to format time in ms or s
static std::string format_time(double secs) {
    std::ostringstream oss;
    if (secs < 1.0) {
        oss << std::fixed << std::setprecision(3) << (secs * 1000.0) << "ms";
    } else {
        oss << std::fixed << std::setprecision(3) << secs << "s";
    }
    return oss.str();
}

void Info::print_footer(const Settings& settings, cudaStream_t stream) {
    if (!settings.verbose) {
        return;
    }

    // Sync status to host first (also syncs iterations_per_batch)
    sync_status_to_host(stream);

    std::cout << "─────────────────────────────────────────────────────────────────────────────────────────────\n";

    if (batchSize == 1) {
        // Single problem: show single status with indicator
        std::cout << status_indicator(status[0]) << " Terminated with status = " << status_name(status[0]) << "\n";
        // Show timing breakdown: total = construction + setup + solve
        double total_time = construction_time + setup_time + solve_time;
        std::cout << "time = " << format_time(total_time)
                  << " (construction: " << format_time(construction_time)
                  << ", setup: " << format_time(setup_time)
                  << ", solve: " << format_time(solve_time) << ")\n";
    } else {
        // Batched: show summary with per-batch status breakdown
        // Count statuses
        std::map<SolverStatus, int> status_counts;
        for (int64_t b = 0; b < batchSize; b++) {
            status_counts[status[b]]++;
        }

        // Compute iteration statistics
        int32_t min_iter = *std::min_element(iterations_per_batch.begin(), iterations_per_batch.end());
        int32_t max_iter = *std::max_element(iterations_per_batch.begin(), iterations_per_batch.end());
        double avg_iter = 0.0;
        for (int64_t b = 0; b < batchSize; b++) {
            avg_iter += iterations_per_batch[b];
        }
        avg_iter /= batchSize;

        // Print summary
        std::cout << "Batch summary (" << batchSize << " problems):\n";
        for (const auto& [stat, count] : status_counts) {
            std::cout << "  " << status_indicator(stat) << " " << status_name(stat) << ": " << count << "\n";
        }
        std::cout << "iterations: min=" << min_iter << ", max=" << max_iter
                  << ", avg=" << std::fixed << std::setprecision(1) << avg_iter << "\n";

        double secs = solve_time;
        if (secs < 1.0) {
            std::cout << "solve time = " << std::fixed << std::setprecision(3) << (secs * 1000.0) << "ms";
        } else {
            std::cout << "solve time = " << std::fixed << std::setprecision(3) << secs << "s";
        }
        std::cout << " (" << std::fixed << std::setprecision(3) << (secs * 1000.0 / batchSize) << "ms/problem)\n";
    }
    std::cout << std::flush;
}

} // namespace moreau
