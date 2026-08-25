// test_yolo.cpp
// Tests YOLO mode's documented contract: run exactly yoloNumIters IPM
// iterations with no convergence checking, return MaxIterations for all
// batches, and preserve the last non-NaN iterate when overshooting.
//
// The QP below converges to its unique optimum in ~30 iterations; running far
// past convergence drives the duality gap to zero, after which the unchecked
// IPM produces NaN working iterates. yolo_snapshot_if_valid_kernel must keep
// the last finite snapshot so the returned solution is still the optimum.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <cmath>
#include <vector>

using namespace moreau;

namespace {

// QP: min 0.5*(x0^2 + x1^2) - x0  s.t. x0+x1 = 1, x0 >= 0, x1 >= 0.
// Substituting x1 = 1-x0 gives a strictly convex objective minimized at the
// unique optimum (1, 0). A = [[1,1],[-1,0],[0,-1]] (CSR); b = [1,0,0]; first
// row zero cone, rest nonneg.
struct YoloFixture {
    int64_t n = 2;
    int64_t m = 3;
    std::vector<int64_t> P_ro{0, 1, 2};
    std::vector<int64_t> P_ci{0, 1};
    std::vector<double>  P_val{1.0, 1.0};
    std::vector<int64_t> A_ro{0, 2, 3, 4};
    std::vector<int64_t> A_ci{0, 1, 0, 1};
    std::vector<double>  A_val{1.0, 1.0, -1.0, -1.0};
    std::vector<double>  q{-1.0, 0.0};
    std::vector<double>  b{1.0, 0.0, 0.0};
    Cones cones;
    Settings settings;

    explicit YoloFixture(int64_t batch, uint32_t yolo_iters) {
        cones.numZeroCones = 1;
        cones.numNonnegCones = 2;
        settings.verbose = false;
        settings.yolo = true;
        settings.yoloNumIters = yolo_iters;

        if (batch > 1) {
            std::vector<double> P_b, A_b, q_b, b_b;
            for (int64_t k = 0; k < batch; ++k) {
                P_b.insert(P_b.end(), P_val.begin(), P_val.end());
                A_b.insert(A_b.end(), A_val.begin(), A_val.end());
                q_b.insert(q_b.end(), q.begin(), q.end());
                b_b.insert(b_b.end(), b.begin(), b.end());
            }
            P_val = std::move(P_b);
            A_val = std::move(A_b);
            q = std::move(q_b);
            b = std::move(b_b);
        }
    }
};

struct YoloResult {
    std::vector<int32_t> status;
    std::vector<double> sol_x;  // post-processed solution: batch * n
    std::vector<double> raw_x;  // working iterate at loop exit: batch * n
};

YoloResult run_yolo(YoloFixture& fx, int64_t batch) {
    int64_t nnzP = fx.P_val.size() / batch;
    int64_t nnzA = fx.A_val.size() / batch;
    CompiledSolver solver(fx.n, fx.m, batch,
                          fx.P_ro.data(), fx.P_ci.data(), nnzP,
                          fx.A_ro.data(), fx.A_ci.data(), nnzA,
                          fx.cones, fx.settings);

    double *d_P = nullptr, *d_A = nullptr, *d_q = nullptr, *d_b = nullptr;
    cudaMalloc(&d_P, sizeof(double) * fx.P_val.size());
    cudaMalloc(&d_A, sizeof(double) * fx.A_val.size());
    cudaMalloc(&d_q, sizeof(double) * fx.q.size());
    cudaMalloc(&d_b, sizeof(double) * fx.b.size());
    cudaMemcpy(d_P, fx.P_val.data(), sizeof(double) * fx.P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, fx.A_val.data(), sizeof(double) * fx.A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, fx.q.data(), sizeof(double) * fx.q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, fx.b.data(), sizeof(double) * fx.b.size(), cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    YoloResult r;
    r.status.resize(batch);
    r.sol_x.resize(batch * fx.n);
    r.raw_x.resize(batch * fx.n);
    cudaMemcpy(r.status.data(), solver.solution.status.get(),
               sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);
    solver.solution.x.gpuToCpu(r.sol_x.data());   // batch-major
    solver.variables.x.gpuToCpu(r.raw_x.data());  // batch-major
    cudaDeviceSynchronize();

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    return r;
}

}  // namespace

// Overshooting to 200 iterations drives the working iterate to NaN, but the
// returned solution must remain the finite optimum (1, 0). Asserting the raw
// iterate is NaN keeps the test honest: it confirms the preservation path is
// actually exercised, not bypassed by a benign solve that never overshoots.
TEST(YoloTest, OvershootPreservesLastFiniteIterate) {
    constexpr int64_t batch = 1;
    YoloFixture fx(batch, /*yolo_iters=*/200);
    auto r = run_yolo(fx, batch);

    EXPECT_EQ(r.status[0], static_cast<int32_t>(SolverStatus::MaxIterations));
    EXPECT_FALSE(std::isfinite(r.raw_x[0]) && std::isfinite(r.raw_x[1]))
        << "expected the overshot working iterate to be NaN";
    EXPECT_TRUE(std::isfinite(r.sol_x[0])) << "returned x0 must be finite";
    EXPECT_TRUE(std::isfinite(r.sol_x[1])) << "returned x1 must be finite";
    EXPECT_NEAR(r.sol_x[0], 1.0, 1e-4) << "x0";
    EXPECT_NEAR(r.sol_x[1], 0.0, 1e-4) << "x1";
}

// Batched overshoot: each batch independently returns MaxIterations with a
// finite, correct iterate. Guards against per-batch snapshot contamination.
TEST(YoloTest, BatchedOvershootAllMaxIterFinite) {
    constexpr int64_t batch = 4;
    YoloFixture fx(batch, /*yolo_iters=*/200);
    auto r = run_yolo(fx, batch);

    for (int64_t bi = 0; bi < batch; ++bi) {
        EXPECT_EQ(r.status[bi], static_cast<int32_t>(SolverStatus::MaxIterations))
            << "batch " << bi;
        EXPECT_TRUE(std::isfinite(r.sol_x[bi * 2])) << "batch " << bi << " x0";
        EXPECT_TRUE(std::isfinite(r.sol_x[bi * 2 + 1])) << "batch " << bi << " x1";
        EXPECT_NEAR(r.sol_x[bi * 2], 1.0, 1e-4) << "batch " << bi << " x0";
        EXPECT_NEAR(r.sol_x[bi * 2 + 1], 0.0, 1e-4) << "batch " << bi << " x1";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
