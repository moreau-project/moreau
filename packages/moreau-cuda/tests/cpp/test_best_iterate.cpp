// test_best_iterate.cpp
// Tests the best-iterate safety net that promotes non-convergent terminations
// (MaxIterations / MaxTime / InsufficientProgress / NumericalError) to
// AlmostSolved when the iterate passed through the reduced-tolerance zone
// earlier in the solve.
//
// The CPU solver has the same behavior; see
// packages/moreau-cpu/tests/best_iterate_integration.rs for the CPU mirror.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>

using namespace moreau;

namespace {

// Minimal QP: min x0 + x1 s.t. x0+x1 = 1, x0 >= 0, x1 >= 0.
// Picks P=0 so the problem is linear — convergence path is short and the
// iterate reliably passes through the reduced zone on the way to tight.
struct BestIterateFixture {
    int64_t n = 2;
    int64_t m = 3;
    std::vector<int64_t> P_ro{0, 0, 0};  // empty P
    std::vector<int64_t> P_ci{};
    std::vector<double>  P_val{};
    // A = [[1,1],[-1,0],[0,-1]]  in CSR
    std::vector<int64_t> A_ro{0, 2, 3, 4};
    std::vector<int64_t> A_ci{0, 1, 0, 1};
    std::vector<double>  A_val{1.0, 1.0, -1.0, -1.0};
    std::vector<double>  q{1.0, 1.0};
    std::vector<double>  b{1.0, 0.0, 0.0};
    Cones cones;
    Settings settings;

    BestIterateFixture(int64_t batch = 1) {
        cones.numZeroCones = 1;
        cones.numNonnegCones = 2;
        settings.verbose = false;
        // Tight tolerance unreachable; reduced tolerance trivially reachable.
        settings.ipm.tolGapAbs = 1e-15;
        settings.ipm.tolGapRel = 1e-15;
        settings.ipm.tolFeas = 1e-15;

        // Replicate values across batches.
        if (batch > 1) {
            std::vector<double> P_val_batched, A_val_batched, q_batched, b_batched;
            for (int64_t b = 0; b < batch; ++b) {
                P_val_batched.insert(P_val_batched.end(), P_val.begin(), P_val.end());
                A_val_batched.insert(A_val_batched.end(), A_val.begin(), A_val.end());
                q_batched.insert(q_batched.end(), q.begin(), q.end());
                b_batched.insert(b_batched.end(), this->b.begin(), this->b.end());
            }
            P_val = std::move(P_val_batched);
            A_val = std::move(A_val_batched);
            q = std::move(q_batched);
            this->b = std::move(b_batched);
        }
    }
};

// Runs the solver with the given max_iter cap, returns the per-batch
// status vector and d_best_saved flags.
struct RunResult {
    std::vector<int32_t> status;
    std::vector<int32_t> best_saved;
    std::vector<double> gap_abs;
    std::vector<double> res_primal;
};

RunResult run_with_max_iter(BestIterateFixture& fx, int max_iter, int64_t batch = 1) {
    fx.settings.maxIter = max_iter;
    int64_t nnzP = 0;
    int64_t nnzA = fx.A_val.size() / batch;

    CompiledSolver solver(fx.n, fx.m, batch,
                          fx.P_ro.data(), fx.P_ci.data(), nnzP,
                          fx.A_ro.data(), fx.A_ci.data(), nnzA,
                          fx.cones, fx.settings);

    double *d_P = nullptr, *d_A = nullptr, *d_q = nullptr, *d_b = nullptr;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * fx.A_val.size());
    cudaMalloc(&d_q, sizeof(double) * fx.q.size());
    cudaMalloc(&d_b, sizeof(double) * fx.b.size());
    cudaMemcpy(d_A, fx.A_val.data(), sizeof(double) * fx.A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, fx.q.data(), sizeof(double) * fx.q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, fx.b.data(), sizeof(double) * fx.b.size(), cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    RunResult r;
    r.status.resize(batch);
    r.best_saved.resize(batch);
    r.gap_abs.resize(batch);
    r.res_primal.resize(batch);
    cudaMemcpy(r.status.data(), solver.solution.status.get(),
               sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.best_saved.data(), solver.info.d_best_saved,
               sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);
    solver.info.gap_abs.gpuToCpu(r.gap_abs.data());
    solver.info.res_primal.gpuToCpu(r.res_primal.data());

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    return r;
}

}  // namespace

// A normal run reaches tight tolerance and records a snapshot along the way
// (every in-zone iterate triggers save_best_iterate). d_best_saved must be
// set, but the status stays Solved since the solve converged normally.
TEST(BestIterateTest, SavedOnNormalSolve) {
    BestIterateFixture fx;
    fx.settings.ipm.tolGapAbs = 1e-8;
    fx.settings.ipm.tolGapRel = 1e-8;
    fx.settings.ipm.tolFeas = 1e-8;
    auto r = run_with_max_iter(fx, 200);
    EXPECT_EQ(r.status[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_EQ(r.best_saved[0], 1);
}

// With tight tol = 1e-15 (unreachable) and maxIter chosen to stop after the
// iterate enters the reduced zone, the restore path must fire: status ends
// up AlmostSolved, d_best_saved=1, and the restored metrics satisfy the
// reduced feasibility tolerance.
TEST(BestIterateTest, MaxIterPromotedToAlmostSolved) {
    bool promoted_at_any_cap = false;
    for (int cap = 3; cap <= 12 && !promoted_at_any_cap; ++cap) {
        BestIterateFixture fx;
        auto r = run_with_max_iter(fx, cap);
        if (r.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved) &&
            r.best_saved[0] == 1) {
            promoted_at_any_cap = true;
            EXPECT_LT(r.res_primal[0], 1e-4);
            EXPECT_LT(r.gap_abs[0], 5e-5);
        }
    }
    EXPECT_TRUE(promoted_at_any_cap)
        << "expected best-iterate restore to fire for some max_iter in [3, 12]";
}

// Batched solve with identical problems: every batch must take the same
// promotion path. Checks that per-batch flags aren't cross-contaminated.
TEST(BestIterateTest, BatchedPromotionPerBatch) {
    constexpr int64_t batch = 4;
    bool promoted_at_any_cap = false;
    for (int cap = 3; cap <= 12 && !promoted_at_any_cap; ++cap) {
        BestIterateFixture fx(batch);
        auto r = run_with_max_iter(fx, cap, batch);
        bool all_promoted = true;
        for (int64_t i = 0; i < batch; ++i) {
            if (r.status[i] != static_cast<int32_t>(SolverStatus::AlmostSolved) ||
                r.best_saved[i] != 1) {
                all_promoted = false;
                break;
            }
        }
        if (all_promoted) {
            promoted_at_any_cap = true;
            for (int64_t i = 0; i < batch; ++i) {
                EXPECT_LT(r.res_primal[i], 1e-4);
                EXPECT_LT(r.gap_abs[i], 5e-5);
            }
        }
    }
    EXPECT_TRUE(promoted_at_any_cap)
        << "expected all batches to be promoted to AlmostSolved at some cap";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
