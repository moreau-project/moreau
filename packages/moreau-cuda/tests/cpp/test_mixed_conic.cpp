// test_mixed_conic.cpp
// Tests for problems with multiple cone types
// Equivalent to packages/moreau-cpu/tests/rust/mixed_conic.rs

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <limits>

using namespace moreau;

class MixedConicTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test mixed zero and nonnegative cones
// min 0.5 * ||x||^2 + x'q
// s.t. x1 + x2 = 1 (equality)
//      x1 >= 0.2 (inequality via nonneg cone)
//      x2 >= 0.3 (inequality via nonneg cone)
TEST_F(MixedConicTest, ZeroAndNonneg) {
    const int64_t n = 2;
    const int64_t m = 3;  // 1 equality + 2 inequality
    const int64_t batchSize = 1;

    // P = I
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q pushes solution toward (0, 0), but constraints force x1 + x2 = 1
    std::vector<double> q = {0.0, 0.0};

    // A = [1 1; -1 0; 0 -1], b = [1, -0.2, -0.3]
    // Row 0: x1 + x2 = 1 (zero cone)
    // Row 1: -x1 >= -0.2 => x1 <= 0.2... wait, that's wrong
    // For x1 >= 0.2: we need -x1 + s = -0.2 with s >= 0 => x1 = 0.2 + s >= 0.2
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;
    std::vector<double> A_values = {1.0, 1.0, -1.0, -1.0};
    std::vector<double> b = {1.0, -0.2, -0.3};

    std::vector<double> l(n, -std::numeric_limits<double>::infinity());
    std::vector<double> u(n, std::numeric_limits<double>::infinity());

    Cones cones;
    cones.numZeroCones = 1;    // equality
    cones.numNonnegCones = 2;  // two inequality constraints

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_l, sizeof(double) * n);
    cudaMalloc(&d_u, sizeof(double) * n);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Check constraints
    EXPECT_NEAR(x[0] + x[1], 1.0, 1e-5);  // equality
    EXPECT_GE(x[0], 0.2 - 1e-5);          // x1 >= 0.2
    EXPECT_GE(x[1], 0.3 - 1e-5);          // x2 >= 0.3

    // With q=0, unconstrained solution is (0,0), but constraints force
    // solution to be on the line x1+x2=1 closest to origin while satisfying bounds
    // Expected: (0.5, 0.5) if constraints don't bind, else on boundary

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

// Test zero + SOC cones (simpler mixed problem)
// min 0.5 ||x||^2 - x0 (encourage large x0)
// s.t. x0 + x1 + x2 = 2 (zero cone equality)
//      ||[x1, x2]|| <= x0 (SOC)
// Solution should be x0 = 2, x1 = x2 = 0 (maximizes x0 while satisfying constraints)
TEST_F(MixedConicTest, ZeroAndSOC) {
    const int64_t n = 3;
    const int64_t m = 4;  // 1 zero + 1 SOC(3)
    const int64_t batchSize = 1;

    // P = I
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;
    std::vector<double> P_values = {1.0, 1.0, 1.0};

    // q = [-1, 0, 0] to encourage large x0
    std::vector<double> q = {-1.0, 0.0, 0.0};

    // A matrix: 4 rows x 3 cols
    // Row 0: x0 + x1 + x2 = 2 (zero cone)
    // Rows 1-3: s = x for SOC => ||[x1,x2]|| <= x0
    std::vector<int64_t> A_ro = {0, 3, 4, 5, 6};  // 5 row offsets for 4 rows
    std::vector<int64_t> A_ci = {0, 1, 2, 0, 1, 2};
    int64_t nnzA = 6;
    // Row 0: [1, 1, 1], Rows 1-3: [-1, 0, 0], [0, -1, 0], [0, 0, -1]
    std::vector<double> A_values = {1.0, 1.0, 1.0, -1.0, -1.0, -1.0};
    std::vector<double> b = {2.0, 0.0, 0.0, 0.0};

    std::vector<double> l(n, -std::numeric_limits<double>::infinity());
    std::vector<double> u(n, std::numeric_limits<double>::infinity());

    Cones cones;
    cones.numZeroCones = 1;
    cones.socConeDims = {3};
    cones.numSocCones = 1;  // one SOC of dimension 3

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_l, sizeof(double) * n);
    cudaMalloc(&d_u, sizeof(double) * n);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Verify equality constraint
    EXPECT_NEAR(x[0] + x[1] + x[2], 2.0, 1e-4);

    // Verify SOC constraint: ||[x1,x2]|| <= x0
    double norm_tail = std::sqrt(x[1]*x[1] + x[2]*x[2]);
    EXPECT_LE(norm_tail, x[0] + 1e-4);

    // Solution should be valid (not NaN)
    for (int i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(x[i])) << "x[" << i << "] is NaN";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

// Test batched mixed cones
TEST_F(MixedConicTest, BatchedMixed) {
    const int64_t n = 2;
    const int64_t m = 2;  // 1 zero + 1 nonneg
    const int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    // A: row 0 = [1, 1] (equality), row 1 = [-1, 0] (x >= 0)
    std::vector<int64_t> A_ro = {0, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 0};
    int64_t nnzA = 3;

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 1;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    // P = I for all
    std::vector<double> P_batch(nnzP * batchSize);
    std::vector<double> A_batch(nnzA * batchSize);
    for (int64_t i = 0; i < batchSize; i++) {
        P_batch[i * nnzP + 0] = 1.0;
        P_batch[i * nnzP + 1] = 1.0;
        A_batch[i * nnzA + 0] = 1.0;
        A_batch[i * nnzA + 1] = 1.0;
        A_batch[i * nnzA + 2] = -1.0;
    }

    // Different q for each, same constraints
    // x1 + x2 = 1, x1 >= 0
    std::vector<double> q_batch = {
        1.0, -1.0,   // pushes to (0, 1)
        -1.0, 1.0,   // pushes to (1, 0), but x1 >= 0 satisfied
        0.0, 0.0,    // center (0.5, 0.5)
        2.0, 2.0,    // pushes negative but bounded
    };

    std::vector<double> b_batch(m * batchSize);
    for (int64_t i = 0; i < batchSize; i++) {
        b_batch[i * m + 0] = 1.0;  // x1 + x2 = 1
        b_batch[i * m + 1] = 0.0;  // -x1 >= 0 => x1 >= 0
    }

    std::vector<double> l_batch(n * batchSize, -std::numeric_limits<double>::infinity());
    std::vector<double> u_batch(n * batchSize, std::numeric_limits<double>::infinity());

    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);
    cudaMalloc(&d_l, sizeof(double) * n * batchSize);
    cudaMalloc(&d_u, sizeof(double) * n * batchSize);

    cudaMemcpy(d_P, P_batch.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_batch.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_batch(n * batchSize);
    cudaMemcpy(x_batch.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);

    // Verify all satisfy constraints
    for (int64_t i = 0; i < batchSize; i++) {
        double x0 = x_batch[i * n + 0];
        double x1 = x_batch[i * n + 1];

        EXPECT_NEAR(x0 + x1, 1.0, 1e-4) << "Problem " << i << " violates equality";
        EXPECT_GE(x0, -1e-4) << "Problem " << i << " violates x0 >= 0";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
