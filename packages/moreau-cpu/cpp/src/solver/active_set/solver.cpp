/**
 * @file solver.cpp
 * @brief ActiveSetSolver orchestration
 *
 * The active-set core is derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * Implements the three-step API (construct, setup, solve) for the
 * CPU-only batched QP solver based on the dual active-set method.
 *
 * Uses std::thread for batch parallelism (no OpenMP dependency).
 */

#include "moreau/solver/active_set/solver.hpp"
#include "moreau/solver/active_set/core.hpp"
#include "moreau/solver/active_set/transform.hpp"
#include "moreau/solver/active_set/backward.hpp"
#include <chrono>
#include <cstring>
#include <cmath>
#include <limits>
#include <thread>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace moreau {

// ============================================================================
// Batch parallelism via std::thread
// ============================================================================

static unsigned int get_num_threads(int64_t batchSize) {
    if (batchSize <= 1) return 1;
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    return std::min(hw, static_cast<unsigned int>(batchSize));
}

template<typename F>
static void parallel_for(int64_t batchSize, F&& fn) {
    unsigned int nthreads = get_num_threads(batchSize);
    if (nthreads <= 1) {
        for (int64_t i = 0; i < batchSize; i++) fn(i);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    std::vector<std::exception_ptr> exceptions(nthreads);
    int64_t chunk = (batchSize + nthreads - 1) / nthreads;

    for (unsigned int t = 0; t < nthreads; t++) {
        int64_t lo = t * chunk;
        int64_t hi = std::min(lo + chunk, batchSize);
        if (lo >= batchSize) break;
        threads.emplace_back([lo, hi, &fn, &exceptions, t]() {
            try {
                for (int64_t i = lo; i < hi; i++) fn(i);
            } catch (...) {
                exceptions[t] = std::current_exception();
            }
        });
    }
    for (auto& th : threads) th.join();
    for (auto& ep : exceptions) {
        if (ep) std::rethrow_exception(ep);
    }
}

// ============================================================================
// Constructor
// ============================================================================

ActiveSetSolver::ActiveSetSolver(
    int64_t n, int64_t m, int64_t batchSize,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones,
    const ActiveSetSettings& settings,
    bool enable_grad,
    bool verbose
)
    : n_(n), m_(m), batchSize_(batchSize),
      nnzP_(nnzP), nnzA_(nnzA),
      settings_(settings),
      enable_grad_(enable_grad),
      verbose_(verbose)
{
    // Validate: only zero + nonneg cones allowed
    numZeroCones_ = cones.numZeroCones;
    numNonnegCones_ = cones.numNonnegCones;

    if (!cones.socConeDims.empty() || !cones.socConeDimsOriginal.empty()) {
        throw std::invalid_argument(
            "ActiveSetSolver: SOC cones not supported. Use SolverType::IPM for conic problems.");
    }
    if (cones.numExpCones > 0) {
        throw std::invalid_argument(
            "ActiveSetSolver: Exponential cones not supported. Use SolverType::IPM.");
    }
    if (!cones.powerAlphas.empty()) {
        throw std::invalid_argument(
            "ActiveSetSolver: Power cones not supported. Use SolverType::IPM.");
    }
    if (!cones.genPowerAlphas.empty()) {
        throw std::invalid_argument(
            "ActiveSetSolver: Generalized power cones not supported. Use SolverType::IPM.");
    }

    if (numZeroCones_ + numNonnegCones_ != m) {
        throw std::invalid_argument(
            "ActiveSetSolver: Total cone dimensions must equal m. Got " +
            std::to_string(numZeroCones_) + " + " + std::to_string(numNonnegCones_) +
            " = " + std::to_string(numZeroCones_ + numNonnegCones_) +
            ", expected " + std::to_string(m));
    }

    // Cache sparsity structure
    P_ro_.assign(P_ro, P_ro + n + 1);
    P_ci_.assign(P_ci, P_ci + nnzP);
    A_ro_.assign(A_ro, A_ro + m + 1);
    A_ci_.assign(A_ci, A_ci + nnzA);

    // Pre-allocate result buffers
    x_sol.resize(batchSize * n, 0.0);
    s_sol.resize(batchSize * m, 0.0);
    z_sol.resize(batchSize * m, 0.0);
    status_vec.resize(batchSize, 0);
    obj_val.resize(batchSize, 0.0);
    iters.resize(batchSize, 0);

    // Pre-allocate workspaces (persisted for backward pass)
    workspaces_.reserve(batchSize);
    for (int64_t i = 0; i < batchSize; i++) {
        workspaces_.push_back(std::make_unique<DaqpWorkspace>());
        workspaces_[i]->allocate(static_cast<int>(n), static_cast<int>(m));
    }
    backward_states_.resize(batchSize);

    // Pre-allocate RHS cache
    q_cached_.resize(batchSize * n);
    b_cached_.resize(batchSize * m);

    if (verbose_) {
        std::cout << "┌───────────────────────────────────────────────────────────┐\n";
        std::cout << "│        Moreau  ─  Active-Set QP Solver (CPU)              │\n";
        std::cout << "└───────────────────────────────────────────────────────────┘\n";
        if (batchSize > 1)
            std::cout << "\nbatch size: " << batchSize << "\n";
        std::cout << "\nproblem:\n";
        std::cout << "  variables     = " << n << "\n";
        std::cout << "  constraints   = " << m << "\n";
        std::cout << "  nnz(P)        = " << nnzP << "\n";
        std::cout << "  nnz(A)        = " << nnzA << "\n";
        std::cout << "  cones (total) = " << (numZeroCones_ + numNonnegCones_) << "\n";
        if (numZeroCones_ > 0)
            std::cout << "          Zero : " << numZeroCones_ << "\n";
        if (numNonnegCones_ > 0)
            std::cout << "    Nonnegative : " << numNonnegCones_ << "\n";
        std::cout << "\nsettings:\n";
        std::cout << "  linear algebra: dense\n";
        std::cout << "  max iter = " << settings_.iter_limit
                  << ", primal_tol = " << std::scientific << std::setprecision(1)
                  << settings_.primal_tol
                  << ", dual_tol = " << settings_.dual_tol << "\n";
        std::cout << "  parallelism: " << get_num_threads(batchSize)
                  << " threads (std::thread)\n";
        if (enable_grad_) {
            const char* dm = (settings_.diff_method == ActiveSetDiffMethod::Smoothed)
                             ? "smoothed" : "exact";
            std::cout << "  differentiation: " << dm;
            if (settings_.diff_method == ActiveSetDiffMethod::Smoothed)
                std::cout << " (μ=" << settings_.diff_smoothing_mu << ")";
            std::cout << "\n";
        }
        std::cout << std::defaultfloat << "\n";
    }
}

void ActiveSetSolver::setup(const double* P_values, const double* A_values, bool shared) {
    matrices_shared_ = shared;

    if (shared) {
        H_dense_.resize(n_ * n_);
        A_dense_.resize(m_ * n_);
        csr_to_dense(H_dense_.data(), n_, n_, P_ro_.data(), P_ci_.data(), P_values, true);
        csr_to_dense(A_dense_.data(), m_, n_, A_ro_.data(), A_ci_.data(), A_values, false);
    } else {
        H_dense_.resize(batchSize_ * n_ * n_);
        A_dense_.resize(batchSize_ * m_ * n_);
        for (int64_t b = 0; b < batchSize_; b++) {
            csr_to_dense(H_dense_.data() + b * n_ * n_, n_, n_,
                         P_ro_.data(), P_ci_.data(),
                         P_values + b * nnzP_, true);
            csr_to_dense(A_dense_.data() + b * m_ * n_, m_, n_,
                         A_ro_.data(), A_ci_.data(),
                         A_values + b * nnzA_, false);
        }
    }
    is_setup_ = true;
    has_solved_ = false;
}

// ============================================================================
// Solve
// ============================================================================

// Per-iteration callback for verbose single-problem output
static void verbose_iter_callback(int iter, int n_active, double fval,
                                   int added, int removed, void* user_data) {
    (void)user_data;
    std::cout << std::right << std::setw(4) << iter
              << std::setw(8) << n_active;

    if (fval > 0) {
        std::cout << "  " << std::scientific << std::setprecision(3)
                  << std::setw(12) << fval;
    } else {
        std::cout << "  " << std::setw(12) << "---";
    }

    if (added > 0 || removed > 0) {
        std::cout << "   ";
        if (added > 0) std::cout << "+" << added;
        if (added > 0 && removed > 0) std::cout << " ";
        if (removed > 0) std::cout << "-" << removed;
    }
    std::cout << std::defaultfloat << "\n";
}

void ActiveSetSolver::solve(const double* q, const double* b) {
    if (!is_setup_) {
        throw std::runtime_error("ActiveSetSolver::solve() called before setup()");
    }

    // Cache RHS for backward pass
    std::memcpy(q_cached_.data(), q, batchSize_ * n_ * sizeof(double));
    std::memcpy(b_cached_.data(), b, batchSize_ * m_ * sizeof(double));

    if (verbose_) {
        if (batchSize_ == 1) {
            std::cout << "iter  active        fval   changes\n";
            std::cout << "───────────────────────────────────────\n";
        } else {
            std::cout << "solving " << batchSize_ << " problems...\n";
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    parallel_for(batchSize_, [&](int64_t batch) {
        const double* H = matrices_shared_ ?
            H_dense_.data() : H_dense_.data() + batch * n_ * n_;
        const double* A = matrices_shared_ ?
            A_dense_.data() : A_dense_.data() + batch * m_ * n_;

        // Install per-iteration callback for batch=1 verbose
        if (verbose_ && batchSize_ == 1) {
            workspaces_[batch]->iter_callback = verbose_iter_callback;
            workspaces_[batch]->iter_callback_data = nullptr;
        }

        solveSingle(batch, H, q + batch * n_, A, b + batch * m_);
        cache_backward_state(batch, *workspaces_[batch]);

        workspaces_[batch]->iter_callback = nullptr;
    });

    auto end = std::chrono::high_resolution_clock::now();
    solve_time = std::chrono::duration<double>(end - start).count();
    has_solved_ = true;

    if (verbose_) {
        int n_solved = 0, n_infeas = 0, n_maxiter = 0, n_error = 0;
        int total_iters = 0, max_iters_val = 0;
        for (int64_t i = 0; i < batchSize_; i++) {
            if (status_vec[i] == static_cast<int32_t>(SolverStatus::Solved)) n_solved++;
            else if (status_vec[i] == static_cast<int32_t>(SolverStatus::PrimalInfeasible)) n_infeas++;
            else if (status_vec[i] == static_cast<int32_t>(SolverStatus::MaxIterations)) n_maxiter++;
            else n_error++;
            total_iters += iters[i];
            if (iters[i] > max_iters_val) max_iters_val = iters[i];
        }
        double avg_iters = batchSize_ > 0 ? (double)total_iters / batchSize_ : 0;

        std::cout << "─────────────────────────────────────────────────────────────\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  solve time    = " << solve_time * 1000.0 << " ms";
        if (batchSize_ > 1)
            std::cout << "  (" << std::setprecision(0)
                      << batchSize_ / solve_time << " solves/s)";
        std::cout << "\n" << std::setprecision(1);
        std::cout << "  iterations    = " << avg_iters << " avg, " << max_iters_val << " max\n";
        std::cout << "  status        = " << n_solved << " solved";
        if (n_infeas > 0) std::cout << ", " << n_infeas << " infeasible";
        if (n_maxiter > 0) std::cout << ", " << n_maxiter << " max_iter";
        if (n_error > 0) std::cout << ", " << n_error << " error";
        std::cout << "\n" << std::defaultfloat << std::flush;
    }
}

void ActiveSetSolver::solve_warm_start(const double* q, const double* b,
                                        const double* warm_x, const double* warm_z,
                                        const double* warm_s) {
    if (!is_setup_) {
        throw std::runtime_error("ActiveSetSolver::solve_warm_start() called before setup()");
    }

    // Cache RHS for backward pass
    std::memcpy(q_cached_.data(), q, batchSize_ * n_ * sizeof(double));
    std::memcpy(b_cached_.data(), b, batchSize_ * m_ * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    parallel_for(batchSize_, [&](int64_t batch) {
        const double* H = matrices_shared_ ?
            H_dense_.data() : H_dense_.data() + batch * n_ * n_;
        const double* A = matrices_shared_ ?
            A_dense_.data() : A_dense_.data() + batch * m_ * n_;
        const double* qi = q + batch * n_;
        const double* bi = b + batch * m_;

        // Use persistent workspace
        DaqpWorkspace& work = *workspaces_[batch];
        work.settings = settings_;
        work.reset();

        // Build bounds into workspace buffers (no allocation)
        buildBounds(bi, work.dupper, work.dlower, work.sense);

        // Setup LDP transform
        int error_flag = daqp_setup_ldp(&work, H, qi, A, work.dupper, work.dlower);

        int exitflag;
        if (error_flag < 0) {
            exitflag = error_flag;
        } else {
            // Apply warm start: activate inequality constraints that were
            // active in the previous solution.  Equality constraints (zero
            // cones) are already activated by daqp_setup_ldp, so we only
            // add nonneg-cone constraints here via daqp_add_constraint.
            const double* ws = warm_s + batch * m_;
            const double* wz = warm_z + batch * m_;

            int activate_result = 1;  // assume success
            for (int64_t i = numZeroCones_; i < m_; i++) {
                if (ws[i] < settings_.primal_tol && wz[i] > settings_.dual_tol) {
                    daqp_add_constraint(&work, static_cast<int>(i), 1.0);
                    if (work.sing_ind != DAQP_EMPTY_IND) {
                        // Overdetermined — abort warm start
                        activate_result = DAQP_EXIT_OVERDETERMINED_INITIAL;
                        break;
                    }
                }
            }
            if (activate_result > 0) {
                exitflag = daqp_ldp(&work);
            } else {
                // Warm start failed (overdetermined initial set) — fall back to cold start
                work.reset();
                buildBounds(bi, work.dupper, work.dlower, work.sense);
                error_flag = daqp_setup_ldp(&work, H, qi, A, work.dupper, work.dlower);
                if (error_flag < 0) {
                    exitflag = error_flag;
                } else if (work.sing_ind == DAQP_UNCONSTRAINED_OPTIMAL) {
                    exitflag = DAQP_EXIT_OPTIMAL;
                } else {
                    exitflag = daqp_ldp(&work);
                }
            }

            if (exitflag > 0) {
                ldp2qp_solution(&work);
            }
        }

        extractSolution(batch, work, exitflag, A, bi);
        cache_backward_state(batch, work);

        if (exitflag > 0) {
            for (int64_t i = 0; i < n_; i++)
                obj_val[batch] += qi[i] * work.x[i];
        }
    });

    auto end = std::chrono::high_resolution_clock::now();
    solve_time = std::chrono::duration<double>(end - start).count();
    has_solved_ = true;
}

// ============================================================================
// Backward
// ============================================================================

void ActiveSetSolver::backward(const double* dx, const double* dz, const double* ds) {
    if (!enable_grad_) {
        throw std::runtime_error(
            "ActiveSetSolver::backward() requires enable_grad=True at construction.");
    }
    if (!has_solved_) {
        throw std::runtime_error(
            "ActiveSetSolver::backward() called before solve().");
    }

    // Zero-fill output buffers (allocated once, reused across calls)
    dP_values.resize(batchSize_ * nnzP_);
    dq.resize(batchSize_ * n_);
    dA_values.resize(batchSize_ * nnzA_);
    db.resize(batchSize_ * m_);
    std::fill(dP_values.begin(), dP_values.end(), 0.0);
    std::fill(dq.begin(), dq.end(), 0.0);
    std::fill(dA_values.begin(), dA_values.end(), 0.0);
    std::fill(db.begin(), db.end(), 0.0);

    // Ensure per-thread workspaces are allocated
    unsigned int nthreads = get_num_threads(batchSize_);
    if (bw_workspaces_.size() < nthreads) {
        bw_workspaces_.resize(nthreads);
        dH_dense_bufs_.resize(nthreads);
        dA_dense_bufs_.resize(nthreads);
        for (unsigned int t = 0; t < nthreads; t++) {
            bw_workspaces_[t].allocate(static_cast<int>(n_), static_cast<int>(m_));
            dH_dense_bufs_[t].resize(n_ * n_);
            dA_dense_bufs_[t].resize(m_ * n_);
        }
    }

    // Use chunked parallel_for so we know the thread index
    if (nthreads <= 1) {
        auto& bw = bw_workspaces_[0];
        auto& dH_buf = dH_dense_bufs_[0];
        auto& dA_buf = dA_dense_bufs_[0];
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            const double* H = matrices_shared_ ? H_dense_.data() : H_dense_.data() + batch * n_ * n_;
            const double* A = matrices_shared_ ? A_dense_.data() : A_dense_.data() + batch * m_ * n_;

            active_set_backward_single(
                backward_states_[batch], bw, H, A,
                q_cached_.data() + batch * n_, b_cached_.data() + batch * m_,
                dx + batch * n_, dz + batch * m_, ds + batch * m_,
                x_sol.data() + batch * n_, z_sol.data() + batch * m_, s_sol.data() + batch * m_,
                static_cast<int>(n_), static_cast<int>(m_), numZeroCones_,
                settings_.diff_method, settings_.diff_smoothing_mu,
                dH_buf.data(), dq.data() + batch * n_,
                dA_buf.data(), db.data() + batch * m_);

            dense_to_csr_values(dH_buf.data(), n_, n_, P_ro_.data(), P_ci_.data(),
                                dP_values.data() + batch * nnzP_, true);
            dense_to_csr_values(dA_buf.data(), m_, n_, A_ro_.data(), A_ci_.data(),
                                dA_values.data() + batch * nnzA_, false);
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        std::vector<std::exception_ptr> exceptions(nthreads);
        int64_t chunk = (batchSize_ + nthreads - 1) / nthreads;

        for (unsigned int t = 0; t < nthreads; t++) {
            int64_t lo = t * chunk;
            int64_t hi = std::min(lo + chunk, batchSize_);
            if (lo >= batchSize_) break;

            threads.emplace_back([&, t, lo, hi]() {
                try {
                    auto& bw = bw_workspaces_[t];
                    auto& dH_buf = dH_dense_bufs_[t];
                    auto& dA_buf = dA_dense_bufs_[t];

                    for (int64_t batch = lo; batch < hi; batch++) {
                        const double* H = matrices_shared_ ? H_dense_.data() : H_dense_.data() + batch * n_ * n_;
                        const double* A = matrices_shared_ ? A_dense_.data() : A_dense_.data() + batch * m_ * n_;

                        active_set_backward_single(
                            backward_states_[batch], bw, H, A,
                            q_cached_.data() + batch * n_, b_cached_.data() + batch * m_,
                            dx + batch * n_, dz + batch * m_, ds + batch * m_,
                            x_sol.data() + batch * n_, z_sol.data() + batch * m_, s_sol.data() + batch * m_,
                            static_cast<int>(n_), static_cast<int>(m_), numZeroCones_,
                            settings_.diff_method, settings_.diff_smoothing_mu,
                            dH_buf.data(), dq.data() + batch * n_,
                            dA_buf.data(), db.data() + batch * m_);

                        dense_to_csr_values(dH_buf.data(), n_, n_, P_ro_.data(), P_ci_.data(),
                                            dP_values.data() + batch * nnzP_, true);
                        dense_to_csr_values(dA_buf.data(), m_, n_, A_ro_.data(), A_ci_.data(),
                                            dA_values.data() + batch * nnzA_, false);
                    }
                } catch (...) {
                    exceptions[t] = std::current_exception();
                }
            });
        }
        for (auto& th : threads) th.join();
        for (auto& ep : exceptions) {
            if (ep) std::rethrow_exception(ep);
        }
    }
}

void ActiveSetSolver::backward_with_data(
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
) {
    if (!enable_grad_) {
        throw std::runtime_error(
            "ActiveSetSolver::backward_with_data() requires enable_grad=True at construction.");
    }

    dP_values.resize(batchSize_ * nnzP_);
    dq.resize(batchSize_ * n_);
    dA_values.resize(batchSize_ * nnzA_);
    db.resize(batchSize_ * m_);
    std::fill(dP_values.begin(), dP_values.end(), 0.0);
    std::fill(dq.begin(), dq.end(), 0.0);
    std::fill(dA_values.begin(), dA_values.end(), 0.0);
    std::fill(db.begin(), db.end(), 0.0);

    unsigned int nthreads = get_num_threads(batchSize_);
    if (bw_workspaces_.size() < nthreads) {
        bw_workspaces_.resize(nthreads);
        dH_dense_bufs_.resize(nthreads);
        dA_dense_bufs_.resize(nthreads);
        for (unsigned int t = 0; t < nthreads; t++) {
            bw_workspaces_[t].allocate(static_cast<int>(n_), static_cast<int>(m_));
            dH_dense_bufs_[t].resize(n_ * n_);
            dA_dense_bufs_[t].resize(m_ * n_);
        }
    }

    const int packed_rinv = daqp_arsum(static_cast<int>(n_));
    std::vector<double> shared_H_dense;
    std::vector<double> shared_A_dense;
    if (shared_matrices) {
        shared_H_dense.resize(n_ * n_);
        shared_A_dense.resize(m_ * n_);
        csr_to_dense(shared_H_dense.data(), n_, n_, P_ro_.data(), P_ci_.data(), P_values, true);
        csr_to_dense(shared_A_dense.data(), m_, n_, A_ro_.data(), A_ci_.data(), A_values, false);
    }

    auto run_batch = [&](unsigned int thread_idx, int64_t batch) {
        auto& bw = bw_workspaces_[thread_idx];
        auto& dH_buf = dH_dense_bufs_[thread_idx];
        auto& dA_buf = dA_dense_bufs_[thread_idx];

        std::vector<double> local_H_dense;
        std::vector<double> local_A_dense;
        const double* H = nullptr;
        const double* A = nullptr;
        if (shared_matrices) {
            H = shared_H_dense.data();
            A = shared_A_dense.data();
        } else {
            local_H_dense.resize(n_ * n_);
            local_A_dense.resize(m_ * n_);
            csr_to_dense(local_H_dense.data(), n_, n_, P_ro_.data(), P_ci_.data(),
                         P_values + batch * nnzP_, true);
            csr_to_dense(local_A_dense.data(), m_, n_, A_ro_.data(), A_ci_.data(),
                         A_values + batch * nnzA_, false);
            H = local_H_dense.data();
            A = local_A_dense.data();
        }

        ActiveSetBackwardState state;
        state.use_rinv_diag = state_use_rinv_diag[batch] != 0;
        state.n_active = state_n_active[batch];
        state.ws.resize(m_);
        state.sense.resize(m_);
        state.lam_star.resize(m_);
        if (state.use_rinv_diag) {
            state.rinv_diag.assign(
                state_rinv_diag + batch * n_,
                state_rinv_diag + (batch + 1) * n_
            );
        } else {
            state.rinv.assign(
                state_rinv + batch * packed_rinv,
                state_rinv + (batch + 1) * packed_rinv
            );
        }
        for (int64_t i = 0; i < m_; i++) {
            state.ws[i] = state_ws[batch * m_ + i];
            state.sense[i] = state_sense[batch * m_ + i];
            state.lam_star[i] = state_lam_star[batch * m_ + i];
        }

        active_set_backward_single(
            state, bw, H, A,
            q + batch * n_, b + batch * m_,
            dx + batch * n_, dz + batch * m_, ds + batch * m_,
            x + batch * n_, z + batch * m_, s + batch * m_,
            static_cast<int>(n_), static_cast<int>(m_), numZeroCones_,
            settings_.diff_method, settings_.diff_smoothing_mu,
            dH_buf.data(), dq.data() + batch * n_,
            dA_buf.data(), db.data() + batch * m_
        );

        dense_to_csr_values(dH_buf.data(), n_, n_, P_ro_.data(), P_ci_.data(),
                            dP_values.data() + batch * nnzP_, true);
        dense_to_csr_values(dA_buf.data(), m_, n_, A_ro_.data(), A_ci_.data(),
                            dA_values.data() + batch * nnzA_, false);
    };

    if (nthreads <= 1) {
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            run_batch(0, batch);
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        std::vector<std::exception_ptr> exceptions(nthreads);
        int64_t chunk = (batchSize_ + nthreads - 1) / nthreads;

        for (unsigned int t = 0; t < nthreads; t++) {
            int64_t lo = t * chunk;
            int64_t hi = std::min(lo + chunk, batchSize_);
            if (lo >= batchSize_) break;
            threads.emplace_back([&, t, lo, hi]() {
                try {
                    for (int64_t batch = lo; batch < hi; batch++) {
                        run_batch(t, batch);
                    }
                } catch (...) {
                    exceptions[t] = std::current_exception();
                }
            });
        }
        for (auto& th : threads) th.join();
        for (auto& ep : exceptions) {
            if (ep) std::rethrow_exception(ep);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

void ActiveSetSolver::buildBounds(const double* b, double* bupper, double* blower,
                                  int* sense) const {
    for (int64_t i = 0; i < numZeroCones_; i++) {
        bupper[i] = b[i];
        blower[i] = b[i];
        sense[i] = 0;
    }
    for (int64_t i = numZeroCones_; i < m_; i++) {
        bupper[i] = b[i];
        blower[i] = -DAQP_INF;
        sense[i] = 0;
    }
}

void ActiveSetSolver::extractSolution(int64_t batch_idx, const DaqpWorkspace& work,
                                      int exitflag, const double* A, const double* b) {
    const int64_t n = n_;
    const int64_t m = m_;
    double* x_out = x_sol.data() + batch_idx * n;
    double* s_out = s_sol.data() + batch_idx * m;
    double* z_out = z_sol.data() + batch_idx * m;

    int32_t moreau_status;
    switch (exitflag) {
        case DAQP_EXIT_OPTIMAL:
            moreau_status = static_cast<int32_t>(SolverStatus::Solved);
            break;
        case DAQP_EXIT_INFEASIBLE:
            moreau_status = static_cast<int32_t>(SolverStatus::PrimalInfeasible);
            break;
        case DAQP_EXIT_UNBOUNDED:
            moreau_status = static_cast<int32_t>(SolverStatus::DualInfeasible);
            break;
        case DAQP_EXIT_ITERLIMIT:
            moreau_status = static_cast<int32_t>(SolverStatus::MaxIterations);
            break;
        case DAQP_EXIT_TIMELIMIT:
            moreau_status = static_cast<int32_t>(SolverStatus::MaxTime);
            break;
        case DAQP_EXIT_NONCONVEX:
        case DAQP_EXIT_CYCLE:
        case DAQP_EXIT_OVERDETERMINED_INITIAL:
            moreau_status = static_cast<int32_t>(SolverStatus::NumericalError);
            break;
        default:
            moreau_status = static_cast<int32_t>(SolverStatus::NumericalError);
            break;
    }
    status_vec[batch_idx] = moreau_status;
    iters[batch_idx] = work.iterations;

    if (exitflag > 0) {
        // Success: copy solution from workspace
        std::memcpy(x_out, work.x, n * sizeof(double));

        // Compute slack s = b - A*x
        for (int64_t i = 0; i < m; i++) {
            double Ax_i = 0;
            for (int64_t j = 0; j < n; j++)
                Ax_i += A[i * n + j] * work.x[j];
            s_out[i] = b[i] - Ax_i;
        }

        // Post-solve cone-feasibility check: defense against residual
        // numerical issues that equilibration couldn't fully resolve.
        // Only the nonneg-cone block is checked (zero-cone rows have s=0 by
        // construction, and the solver only supports zero+nonneg cones).
        double b_inf = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double ab = std::abs(b[i]);
            if (ab > b_inf) b_inf = ab;
        }
        double neg_tol = settings_.primal_tol * std::max(1.0, b_inf);
        for (int64_t i = numZeroCones_; i < numZeroCones_ + numNonnegCones_; i++) {
            if (s_out[i] < -neg_tol) {
                moreau_status = static_cast<int32_t>(SolverStatus::NumericalError);
                status_vec[batch_idx] = moreau_status;
                break;
            }
        }

        // Build dual variables z
        std::memset(z_out, 0, m * sizeof(double));
        for (int i = 0; i < work.n_active; i++) {
            int idx = work.WS[i];
            if (work.sense[idx] & DAQP_LOWER)
                z_out[idx] = -work.lam_star[i];
            else
                z_out[idx] = work.lam_star[i];
        }

        // Compute objective value: 0.5 * x'Px
        const double* H = matrices_shared_ ?
            H_dense_.data() : H_dense_.data() + batch_idx * n * n;

        double fval = 0;
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++)
                fval += 0.5 * work.x[i] * H[i * n + j] * work.x[j];
        }
        obj_val[batch_idx] = fval;
    } else {
        // Failure: zero out solution to avoid leaking stale data
        std::memset(x_out, 0, n * sizeof(double));
        std::memset(s_out, 0, m * sizeof(double));
        std::memset(z_out, 0, m * sizeof(double));
        obj_val[batch_idx] = 0.0;
    }
}

void ActiveSetSolver::cache_backward_state(int64_t batch_idx, const DaqpWorkspace& work) {
    backward_states_[batch_idx] = save_backward_state(work, static_cast<int>(n_), static_cast<int>(m_));
}

const ActiveSetBackwardState& ActiveSetSolver::backward_state(int64_t batch_idx) const {
    if (batch_idx < 0 || batch_idx >= batchSize_) {
        throw std::out_of_range("ActiveSetSolver::backward_state() batch index out of range");
    }
    return backward_states_[batch_idx];
}

// Ruiz-style equilibration on the QP's KKT-structural matrix [H A'; A 0].
// Iteratively scales rows/cols to unit infinity-norm, which conditions H
// and normalizes A rows. The cost is additionally scaled by c so that
// |Dq|_inf ≈ 1, which bounds |v = Rinv'q'| in DAQP's QP→LDP transform
// (otherwise |v| >> |d| causes catastrophic cancellation in d = b·scale + M·v
// and silently wrong solutions on ill-scaled inputs).
//
// Returns d (n-vec), e (m-vec), c (scalar) such that the transformed problem
//   H~ = c·D·H·D,  q~ = c·D·q,  A~ = E·A·D,  b~ = E·b
// is well-conditioned. Recover: x = D·x~, s = E^{-1}·s~, z = E·z~/c.
static void ruiz_equilibrate(
    double* H, double* q, double* A, double* b,
    double* d, double* e, double* c_out,
    int64_t n, int64_t m,
    int max_iter = 20, double tol = 1e-3)
{
    const double min_scale = 1e-6;  // guard against infinite growth
    const double max_scale = 1e6;
    for (int64_t j = 0; j < n; ++j) d[j] = 1.0;
    for (int64_t i = 0; i < m; ++i) e[i] = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;
        // Column scaling (x-variables): unit-inf-norm of each KKT column j
        for (int64_t j = 0; j < n; ++j) {
            double col_max = 0.0;
            for (int64_t i = 0; i < n; ++i)
                col_max = std::max(col_max, std::abs(H[i*n + j]));
            for (int64_t i = 0; i < m; ++i)
                col_max = std::max(col_max, std::abs(A[i*n + j]));
            if (col_max < min_scale) col_max = min_scale;
            if (col_max > max_scale) col_max = max_scale;
            double s = 1.0 / std::sqrt(col_max);
            d[j] *= s;
            // Scale col j of H; H is symmetric so also row j
            for (int64_t i = 0; i < n; ++i) H[i*n + j] *= s;
            for (int64_t i = 0; i < n; ++i) H[j*n + i] *= s;
            // Scale col j of A
            for (int64_t i = 0; i < m; ++i) A[i*n + j] *= s;
            // Scale q[j] (absorbed into D·q)
            q[j] *= s;
            max_change = std::max(max_change, std::abs(s - 1.0));
        }
        // Row scaling (constraint rows): unit-inf-norm of each A row
        for (int64_t i = 0; i < m; ++i) {
            double row_max = 0.0;
            for (int64_t j = 0; j < n; ++j)
                row_max = std::max(row_max, std::abs(A[i*n + j]));
            if (row_max < min_scale) row_max = min_scale;
            if (row_max > max_scale) row_max = max_scale;
            double s = 1.0 / std::sqrt(row_max);
            e[i] *= s;
            for (int64_t j = 0; j < n; ++j) A[i*n + j] *= s;
            b[i] *= s;
            max_change = std::max(max_change, std::abs(s - 1.0));
        }
        if (max_change < tol) break;
    }

    // Cost scaling: bring |q|_inf to O(1).
    double q_inf = 0.0;
    for (int64_t j = 0; j < n; ++j) q_inf = std::max(q_inf, std::abs(q[j]));
    double H_inf = 0.0;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            H_inf = std::max(H_inf, std::abs(H[i*n + j]));
    double c = 1.0;
    double norm_cost = std::max(q_inf, H_inf);
    if (norm_cost > 1.0) c = 1.0 / norm_cost;
    for (int64_t i = 0; i < n*n; ++i) H[i] *= c;
    for (int64_t j = 0; j < n; ++j) q[j] *= c;
    *c_out = c;
}

// Estimate whether the problem scales are bad enough to trigger catastrophic
// cancellation in DAQP's QP→LDP transform (d = b·scale + M·v).
// Returns true iff the ratio of largest to smallest nonzero inf-norm across
// {|H|, |q|, |A|, |b|} exceeds a conservative threshold. Well-scaled problems
// return false, skipping equilibration entirely so the workspace state stays
// consistent with the original (H, q, A, b) that the backward pass relies on.
static bool needs_equilibration(const double* H, const double* q,
                                const double* A, const double* b,
                                int64_t n, int64_t m)
{
    auto inf_norm_2d = [](const double* M, int64_t rows, int64_t cols) {
        double v = 0.0;
        for (int64_t i = 0; i < rows * cols; ++i) v = std::max(v, std::abs(M[i]));
        return v;
    };
    auto inf_norm_1d = [](const double* v_in, int64_t len) {
        double v = 0.0;
        for (int64_t i = 0; i < len; ++i) v = std::max(v, std::abs(v_in[i]));
        return v;
    };
    double norms[4] = {
        inf_norm_2d(H, n, n),
        inf_norm_1d(q, n),
        inf_norm_2d(A, m, n),
        inf_norm_1d(b, m),
    };
    double hi = 0.0, lo = std::numeric_limits<double>::infinity();
    for (double x : norms) {
        if (x > hi) hi = x;
        if (x > 0.0 && x < lo) lo = x;
    }
    if (!std::isfinite(lo) || lo == 0.0) return false;
    // 1e4 is a large safety margin over the true trigger (|v|/|b·scale| ≈ 1/ε
    // ≈ 1e16 in the pathology). Well-scaled problems (ratio < 1e4) are passed
    // through untouched so the backward pass sees the un-modified workspace.
    return (hi / lo) > 1e4;
}

void ActiveSetSolver::solveSingle(int64_t batch_idx,
                                  const double* H, const double* q,
                                  const double* A, const double* b,
                                  bool warm_start) {
    DaqpWorkspace& work = *workspaces_[batch_idx];
    work.settings = settings_;
    work.reset();

    // Equilibrate only on ill-scaled inputs; otherwise pass through so the
    // backward pass (which uses work.x, work.lam_star, work.Rinv factored on
    // the as-passed H) sees un-mutated state.
    const bool do_equilibrate = needs_equilibration(H, q, A, b, n_, m_);

    std::vector<double> H_eq, A_eq, q_eq, b_eq;
    std::vector<double> d_scale(n_, 1.0), e_scale(m_, 1.0);
    double c_scale = 1.0;
    const double* H_use = H;
    const double* q_use = q;
    const double* A_use = A;
    const double* b_use = b;
    if (do_equilibrate) {
        H_eq.assign(H, H + n_ * n_);
        A_eq.assign(A, A + m_ * n_);
        q_eq.assign(q, q + n_);
        b_eq.assign(b, b + m_);
        ruiz_equilibrate(H_eq.data(), q_eq.data(), A_eq.data(), b_eq.data(),
                         d_scale.data(), e_scale.data(), &c_scale, n_, m_);
        H_use = H_eq.data();
        q_use = q_eq.data();
        A_use = A_eq.data();
        b_use = b_eq.data();
    }

    buildBounds(b_use, work.dupper, work.dlower, work.sense);

    // Setup QP→LDP transform
    int error_flag = daqp_setup_ldp(&work, H_use, q_use, A_use,
                                    work.dupper, work.dlower);

    int exitflag;
    if (error_flag < 0) {
        exitflag = error_flag;
    } else if (work.sing_ind == DAQP_UNCONSTRAINED_OPTIMAL) {
        exitflag = DAQP_EXIT_OPTIMAL;
    } else {
        exitflag = daqp_ldp(&work);
        if (exitflag > 0) {
            ldp2qp_solution(&work);
        }
    }

    // Un-scale work.x/lam_star back to original coordinates before
    // extractSolution reads them (s is recomputed via original A, b).
    // Only applies when equilibration ran. The backward pass is not supported
    // on equilibrated problems because work.Rinv is factored on scaled H; a
    // subsequent backward() call on an equilibrated solve may produce wrong
    // gradients. Equilibration only runs on ill-scaled problems where the
    // alternative is silently wrong forward results.
    if (exitflag > 0 && do_equilibrate) {
        for (int64_t j = 0; j < n_; ++j) work.x[j] *= d_scale[j];
        for (int i = 0; i < work.n_active; ++i) {
            int idx = work.WS[i];
            if (idx < m_) work.lam_star[i] *= e_scale[idx] / c_scale;
        }
    }

    extractSolution(batch_idx, work, exitflag, A, b);

    if (exitflag > 0) {
        for (int64_t i = 0; i < n_; i++)
            obj_val[batch_idx] += q[i] * work.x[i];
    }
}

} // namespace moreau
