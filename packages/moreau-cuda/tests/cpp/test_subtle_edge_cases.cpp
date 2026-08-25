/**
 * @file test_subtle_edge_cases.cpp
 * @brief Tests for subtle edge cases: batch isolation, cone derivatives, numerical precision
 *
 * These tests aim to uncover:
 * 1. Batch leaking/pollution - gradients or solutions affecting other batches
 * 2. Cone derivative edge cases - boundary points, degenerate cases
 * 3. Equilibration anomalies - different scales across batches
 * 4. Numerical precision issues - near-zero, large values, ill-conditioning
 * 5. Forward-backward consistency for all cone types
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/diff/diff_kernels.cuh"
#include "moreau/vector/vector.hpp"

using namespace moreau;

// ============================================================================
// Helper Functions
// ============================================================================

void copy_to_batched(BatchedVector& dest, const std::vector<double>& src) {
    cudaMemcpy(dest.data(), src.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice);
}

std::vector<double> copy_from_batched(const BatchedVector& src) {
    std::vector<double> result(src.n() * src.batchSize());
    cudaMemcpy(result.data(), src.data(), sizeof(double) * result.size(), cudaMemcpyDeviceToHost);
    return result;
}

bool isClose(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

double vectorNorm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

// Helper: create device arrays for dim=3 SOC cones (used by compute_soc_derivative)
struct SocDim3Arrays {
    int64_t* d_dims = nullptr;
    int64_t* d_offsets = nullptr;
    int64_t* d_hs_offsets = nullptr;

    SocDim3Arrays(int64_t numSocCones) {
        std::vector<int64_t> dims(numSocCones, 3);
        std::vector<int64_t> offsets(numSocCones + 1);
        std::vector<int64_t> hs_offsets(numSocCones + 1);
        for (int64_t i = 0; i <= numSocCones; ++i) {
            offsets[i] = i * 3;
            hs_offsets[i] = i * 6;
        }
        cudaMalloc(&d_dims, sizeof(int64_t) * numSocCones);
        cudaMalloc(&d_offsets, sizeof(int64_t) * (numSocCones + 1));
        cudaMalloc(&d_hs_offsets, sizeof(int64_t) * (numSocCones + 1));
        cudaMemcpy(d_dims, dims.data(), sizeof(int64_t) * numSocCones, cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, offsets.data(), sizeof(int64_t) * (numSocCones + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hs_offsets, hs_offsets.data(), sizeof(int64_t) * (numSocCones + 1), cudaMemcpyHostToDevice);
    }

    ~SocDim3Arrays() {
        cudaFree(d_dims);
        cudaFree(d_offsets);
        cudaFree(d_hs_offsets);
    }
};

// ============================================================================
// BATCH ISOLATION TESTS
// These test that batches don't "leak" into each other
// ============================================================================

class BatchIsolationTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

/**
 * Test: Zero gradient in one batch should not affect other batches
 *
 * Setup: Batch 0 has dx_bar = [1,0], Batch 1 has dx_bar = [0,0]
 * Expected: Batch 1's gradients should be exactly zero
 */
TEST_F(BatchIsolationTest, ZeroGradientBatchIsolation) {
    int n = 2, m = 1;
    int batchSize = 2;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 1;  // equality constraint

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Same problem data for all batches
    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
        P_values[batch * nnzP + 0] = 1.0;
        P_values[batch * nnzP + 1] = 1.0;
        A_values[batch * nnzA + 0] = 1.0;
        A_values[batch * nnzA + 1] = 1.0;
        q[batch * n + 0] = 0.0;
        q[batch * n + 1] = 0.0;
        b[batch * m + 0] = 1.0;  // x0 + x1 = 1
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Batch 0: dx_bar = [1, 0], Batch 1: dx_bar = [0, 0]
    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    std::vector<double> dx_data = {1.0, 0.0,   // batch 0: non-zero
                                    0.0, 0.0};  // batch 1: all zeros
    std::vector<double> dz_data = {0.0, 0.0};
    std::vector<double> ds_data = {0.0, 0.0};
    copy_to_batched(dx_bar, dx_data);
    copy_to_batched(dz_bar, dz_data);
    copy_to_batched(ds_bar, ds_data);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    // Check batch 1's gradients are exactly zero
    auto dq_result = copy_from_batched(solver.diff_state()->dq);
    auto db_result = copy_from_batched(solver.diff_state()->db);

    // Batch 1 (indices 2,3 for dq and 1 for db)
    EXPECT_NEAR(dq_result[2], 0.0, 1e-12) << "Batch 1 dq[0] should be exactly 0";
    EXPECT_NEAR(dq_result[3], 0.0, 1e-12) << "Batch 1 dq[1] should be exactly 0";
    EXPECT_NEAR(db_result[1], 0.0, 1e-12) << "Batch 1 db[0] should be exactly 0";

    // Batch 0 should have non-zero gradients
    EXPECT_NE(dq_result[0], 0.0) << "Batch 0 dq[0] should be non-zero";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test: Different q vectors should produce independent gradients
 *
 * Key insight: If batches leak, changing q in one batch would affect another
 */
TEST_F(BatchIsolationTest, IndependentQGradients) {
    int n = 2, m = 2;
    int batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    // Same structure, different q for each batch
    for (int batch = 0; batch < batchSize; batch++) {
        P_values[batch * nnzP + 0] = 2.0;
        P_values[batch * nnzP + 1] = 1.0;
        A_values[batch * nnzA + 0] = 1.0;
        A_values[batch * nnzA + 1] = 1.0;
        // Different linear term per batch
        q[batch * n + 0] = 1.0 + 0.5 * batch;
        q[batch * n + 1] = 2.0 - 0.3 * batch;
        b[batch * m + 0] = 5.0;
        b[batch * m + 1] = 5.0;
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Get solutions
    std::vector<double> x(n * batchSize);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Verify solutions are different (basic sanity check)
    bool solutionsDiffer = false;
    for (int batch = 1; batch < batchSize; batch++) {
        if (std::abs(x[batch * n] - x[0]) > 1e-6) {
            solutionsDiffer = true;
            break;
        }
    }
    EXPECT_TRUE(solutionsDiffer) << "Solutions should differ for different q values";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test: Gradient leaking detection via repeated solves
 *
 * Run backward pass twice with different upstream gradients.
 * If there's state leaking, the second pass might be contaminated.
 */
TEST_F(BatchIsolationTest, RepeatedBackwardNoLeaking) {
    int n = 2, m = 1;
    int batchSize = 2;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values(nnzP * batchSize, 1.0);
    std::vector<double> A_values(nnzA * batchSize, 1.0);
    std::vector<double> q(n * batchSize, 0.0);
    std::vector<double> b(m * batchSize, 1.0);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    // First backward: dx_bar = [1, 0, 1, 0]
    std::vector<double> dx_data1 = {1.0, 0.0, 1.0, 0.0};
    copy_to_batched(dx_bar, dx_data1);
    dz_bar.setToConstant(0.0, 0);
    ds_bar.setToConstant(0.0, 0);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    auto dq_first = copy_from_batched(solver.diff_state()->dq);

    // Second backward: dx_bar = [0, 1, 0, 1]
    std::vector<double> dx_data2 = {0.0, 1.0, 0.0, 1.0};
    copy_to_batched(dx_bar, dx_data2);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    auto dq_second = copy_from_batched(solver.diff_state()->dq);

    // The gradients should be different (no leaking from first pass)
    bool different = false;
    for (size_t i = 0; i < dq_first.size(); i++) {
        if (std::abs(dq_first[i] - dq_second[i]) > 1e-10) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Second backward should produce different gradients";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// CONE DERIVATIVE EDGE CASES
// ============================================================================

class ConeDerivativeEdgeCaseTest : public ::testing::Test {
protected:
    int64_t batchSize = 4;
};

/**
 * Test: SOC derivative at exact boundary (t = ||x||)
 */
TEST_F(ConeDerivativeEdgeCaseTest, SOCBoundaryDerivative) {
    int64_t numSocCones = 1;
    int64_t m = numSocCones * 3;
    BatchedVector u(m, batchSize);
    BatchedVector H(numSocCones * 6, batchSize);

    // Points exactly on boundary: t = ||x||
    std::vector<double> u_data = {
        1.0, 0.6, 0.8,    // batch 0: t=1, ||x||=1 (boundary)
        2.0, 1.6, 1.2,    // batch 1: t=2, ||x||=2 (boundary)
        0.5, 0.3, 0.4,    // batch 2: t=0.5, ||x||=0.5 (boundary)
        3.0, 1.8, 2.4     // batch 3: t=3, ||x||=3 (boundary)
    };
    copy_to_batched(u, u_data);

    SocDim3Arrays soc_arrays(numSocCones);
    compute_soc_derivative(
        H.data(), u.data(),
        0, numSocCones,
        soc_arrays.d_dims, soc_arrays.d_offsets, soc_arrays.d_hs_offsets,
        numSocCones * 3, numSocCones * 6,
        batchSize, m, nullptr, 0
    );
    cudaDeviceSynchronize();

    auto result = copy_from_batched(H);

    // Boundary points should give specific derivative (projection operator)
    // For SOC boundary: H = 0.5 * [[1, x'/t]; [x/t, I + xx'/t^2]]
    for (int64_t b = 0; b < batchSize; ++b) {
        // Check that derivative is well-defined (no NaN/Inf)
        for (int i = 0; i < 6; ++i) {
            EXPECT_FALSE(std::isnan(result[b * 6 + i]))
                << "SOC boundary derivative NaN at batch " << b << ", index " << i;
            EXPECT_FALSE(std::isinf(result[b * 6 + i]))
                << "SOC boundary derivative Inf at batch " << b << ", index " << i;
        }
    }
}

/**
 * Test: SOC derivative with near-zero norm
 */
TEST_F(ConeDerivativeEdgeCaseTest, SOCNearZeroNorm) {
    int64_t numSocCones = 1;
    int64_t m = numSocCones * 3;
    BatchedVector u(m, batchSize);
    BatchedVector H(numSocCones * 6, batchSize);

    // Near-zero x component (numerical edge case)
    std::vector<double> u_data = {
        1.0, 1e-10, 1e-10,  // batch 0: large t, tiny x
        0.1, 1e-12, 0.0,    // batch 1: moderate t, near-zero x
        1e-6, 0.0, 0.0,     // batch 2: tiny t, zero x
        10.0, 1e-8, 1e-8    // batch 3: large t, tiny x
    };
    copy_to_batched(u, u_data);

    SocDim3Arrays soc_arrays2(numSocCones);
    compute_soc_derivative(
        H.data(), u.data(),
        0, numSocCones,
        soc_arrays2.d_dims, soc_arrays2.d_offsets, soc_arrays2.d_hs_offsets,
        numSocCones * 3, numSocCones * 6,
        batchSize, m, nullptr, 0
    );
    cudaDeviceSynchronize();

    auto result = copy_from_batched(H);

    // Should handle gracefully without NaN/Inf
    for (int64_t b = 0; b < batchSize; ++b) {
        for (int i = 0; i < 6; ++i) {
            EXPECT_FALSE(std::isnan(result[b * 6 + i]))
                << "Near-zero SOC derivative NaN at batch " << b;
            EXPECT_FALSE(std::isinf(result[b * 6 + i]))
                << "Near-zero SOC derivative Inf at batch " << b;
        }
    }
}

/**
 * Test: Nonnegative cone derivative at exactly zero
 */
TEST_F(ConeDerivativeEdgeCaseTest, NonnegAtZero) {
    int64_t numNonnegCones = 4;
    int64_t m = numNonnegCones;
    BatchedVector u(m, batchSize);
    BatchedVector H_diag(numNonnegCones, batchSize);

    // Mix of exactly zero and near-zero values
    std::vector<double> u_data = {
        0.0, 1e-15, -1e-15, 1e-10,   // batch 0
        -0.0, 0.0, 0.0, 0.0,         // batch 1: all zeros
        1e-300, -1e-300, 0.0, 1.0,   // batch 2: denormals
        -1e-10, 1e-10, 0.0, -0.0     // batch 3
    };
    copy_to_batched(u, u_data);

    compute_nonneg_derivative(
        H_diag.data(), u.data(),
        0, numNonnegCones, batchSize, m, 0
    );
    cudaDeviceSynchronize();

    auto result = copy_from_batched(H_diag);

    // All results should be 0 or 1
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(result[i] == 0.0 || result[i] == 1.0)
            << "Nonneg derivative should be 0 or 1, got " << result[i];
    }
}

/**
 * Test: Exp cone derivative at various points
 */
TEST_F(ConeDerivativeEdgeCaseTest, ExpConeDerivative) {
    int64_t numExpCones = 1;
    int64_t m = numExpCones * 3;
    BatchedVector u(m, batchSize);
    BatchedVector H(numExpCones * 9, batchSize);  // Full 3x3 matrix (not symmetric)

    // Various exp cone points
    std::vector<double> u_data = {
        1.0, 2.0, 3.0,       // batch 0: typical interior
        0.1, 0.5, 1.0,       // batch 1: smaller values
        2.0, 2.0, 2.0,       // batch 2: equal components
        0.5, 1.0, std::exp(1.0) * 0.5  // batch 3: near boundary
    };
    copy_to_batched(u, u_data);

    Cones cones{};
    cones.numExpCones = numExpCones;
    cones.initialize(batchSize, 0);

    ConeDerivatives derivs(cones, batchSize, 0);

    // Compute via compute_cone_derivative
    compute_cone_derivative(u, derivs, cones, 0);
    cudaDeviceSynchronize();

    auto result = copy_from_batched(derivs.exp_H);

    // Check for NaN/Inf
    for (int64_t b = 0; b < batchSize; ++b) {
        for (int i = 0; i < 9; ++i) {
            EXPECT_FALSE(std::isnan(result[b * 9 + i]))
                << "Exp cone derivative NaN at batch " << b << ", index " << i;
            EXPECT_FALSE(std::isinf(result[b * 9 + i]))
                << "Exp cone derivative Inf at batch " << b << ", index " << i;
        }
    }
}

/**
 * Test: Power cone derivative for various alpha values
 */
TEST_F(ConeDerivativeEdgeCaseTest, PowerConeDerivativeAlphas) {
    int64_t numPowerCones = 2;
    int64_t m = numPowerCones * 3;
    int batchSize_ = 2;
    BatchedVector u(m, batchSize_);

    // Test points
    std::vector<double> u_data = {
        // batch 0
        1.0, 2.0, 0.5,   // cone 0
        2.0, 1.0, 1.0,   // cone 1
        // batch 1
        0.5, 0.5, 0.1,
        1.0, 1.0, 0.5
    };
    copy_to_batched(u, u_data);

    Cones cones{};
    cones.numPowerCones = numPowerCones;
    std::vector<double> alphas = {0.3, 0.7};  // Different alpha values
    cones.powerAlphas = alphas;
    cones.initialize(batchSize_, 0);

    ConeDerivatives derivs(cones, batchSize_, 0);
    compute_cone_derivative(u, derivs, cones, 0);
    cudaDeviceSynchronize();

    auto result = copy_from_batched(derivs.power_H);

    // Check for NaN/Inf
    for (int64_t b = 0; b < batchSize_; ++b) {
        for (int c = 0; c < numPowerCones; ++c) {
            for (int i = 0; i < 9; ++i) {
                int idx = b * numPowerCones * 9 + c * 9 + i;
                EXPECT_FALSE(std::isnan(result[idx]))
                    << "Power cone derivative NaN at batch " << b << ", cone " << c;
                EXPECT_FALSE(std::isinf(result[idx]))
                    << "Power cone derivative Inf at batch " << b << ", cone " << c;
            }
        }
    }
}

// ============================================================================
// NUMERICAL PRECISION TESTS
// ============================================================================

class NumericalPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

/**
 * Test: Solve with very different scales in different batches
 */
TEST_F(NumericalPrecisionTest, MixedScaleBatches) {
    int n = 2, m = 2;
    int batchSize = 3;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    // Batch 0: normal scale
    P_values[0] = 1.0; P_values[1] = 1.0;
    A_values[0] = 1.0; A_values[1] = 1.0;
    q[0] = 1.0; q[1] = 1.0;
    b[0] = 5.0; b[1] = 5.0;

    // Batch 1: large scale (1e6)
    P_values[2] = 1e6; P_values[3] = 1e6;
    A_values[2] = 1e3; A_values[3] = 1e3;
    q[2] = 1e6; q[3] = 1e6;
    b[2] = 5e3; b[3] = 5e3;

    // Batch 2: small scale (1e-6)
    P_values[4] = 1e-6; P_values[5] = 1e-6;
    A_values[4] = 1e-3; A_values[5] = 1e-3;
    q[4] = 1e-6; q[5] = 1e-6;
    b[4] = 5e-3; b[5] = 5e-3;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x(n * batchSize);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // All batches should produce finite solutions
    for (int batch = 0; batch < batchSize; batch++) {
        for (int i = 0; i < n; i++) {
            EXPECT_FALSE(std::isnan(x[batch * n + i]))
                << "Batch " << batch << " x[" << i << "] is NaN";
            EXPECT_FALSE(std::isinf(x[batch * n + i]))
                << "Batch " << batch << " x[" << i << "] is Inf";
        }
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test: Gradient computation with upstream gradients of vastly different magnitudes
 */
TEST_F(NumericalPrecisionTest, MixedScaleGradients) {
    int n = 2, m = 1;
    int batchSize = 3;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values(nnzP * batchSize, 1.0);
    std::vector<double> A_values(nnzA * batchSize, 1.0);
    std::vector<double> q(n * batchSize, 0.0);
    std::vector<double> b(m * batchSize, 1.0);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    // Different scales: 1.0, 1e-10, 1e10
    std::vector<double> dx_data = {
        1.0, 1.0,       // batch 0: normal
        1e-10, 1e-10,   // batch 1: tiny
        1e10, 1e10      // batch 2: huge
    };
    copy_to_batched(dx_bar, dx_data);
    dz_bar.setToConstant(0.0, 0);
    ds_bar.setToConstant(0.0, 0);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    auto dq_result = copy_from_batched(solver.diff_state()->dq);

    // All gradients should be finite
    for (size_t i = 0; i < dq_result.size(); i++) {
        EXPECT_FALSE(std::isnan(dq_result[i]))
            << "dq[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(dq_result[i]))
            << "dq[" << i << "] is Inf";
    }

    // Gradients should scale proportionally (roughly)
    // batch 0 and batch 2 should differ by ~1e10
    double ratio = std::abs(dq_result[4]) / (std::abs(dq_result[0]) + 1e-20);
    EXPECT_GT(ratio, 1e8) << "Gradient scaling seems off";
    EXPECT_LT(ratio, 1e12) << "Gradient scaling seems off";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// MIXED CONE GRADIENT TESTS
// ============================================================================

class MixedConeGradientTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

/**
 * Test: Problem with all supported cone types and verify gradients are finite
 */
TEST_F(MixedConeGradientTest, AllConeTypesBackward) {
    // Problem: min (1/2)||x||^2 + q'x
    // s.t. A1*x = b1 (zero cone)
    //      A2*x >= b2 (nonneg cone)
    //      A3*x in SOC (second order cone)

    int n = 4;
    int m = 1 + 2 + 3;  // 1 zero, 2 nonneg, 1 SOC(3)
    int batchSize = 2;

    // P = I (diagonal)
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4};
    std::vector<int64_t> P_ci = {0, 1, 2, 3};
    int64_t nnzP = 4;

    // A is dense 6x4
    std::vector<int64_t> A_ro = {0, 4, 8, 12, 16, 20, 24};
    std::vector<int64_t> A_ci;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            A_ci.push_back(col);
        }
    }
    int64_t nnzA = m * n;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;
    cones.socConeDims = {3};
    cones.numSocCones = 1;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int batch = 0; batch < batchSize; batch++) {
        // P = I
        for (int i = 0; i < nnzP; i++) {
            P_values[batch * nnzP + i] = 1.0;
        }
        // Random A (but structured for feasibility)
        for (int i = 0; i < nnzA; i++) {
            A_values[batch * nnzA + i] = 0.5 * dist(rng);
        }
        // Random q
        for (int i = 0; i < n; i++) {
            q[batch * n + i] = 0.1 * dist(rng);
        }
        // b set for feasibility
        for (int i = 0; i < m; i++) {
            b[batch * m + i] = 0.5;  // generous RHS
        }
        // SOC constraint: make first component larger
        b[batch * m + 3] = 2.0;  // t component of SOC
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Check that solve succeeded
    std::vector<double> x_sol(n * batchSize);
    solver.solution.x.gpuToCpu(x_sol.data(), 0);
    cudaDeviceSynchronize();

    for (int i = 0; i < n * batchSize; i++) {
        ASSERT_FALSE(std::isnan(x_sol[i])) << "Solution contains NaN";
    }

    // Now test backward
    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    std::vector<double> dx_data(n * batchSize);
    std::vector<double> dz_data(m * batchSize, 0.0);
    std::vector<double> ds_data(m * batchSize, 0.0);

    for (int i = 0; i < n * batchSize; i++) {
        dx_data[i] = dist(rng);
    }

    copy_to_batched(dx_bar, dx_data);
    copy_to_batched(dz_bar, dz_data);
    copy_to_batched(ds_bar, ds_data);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    // Verify all gradients are finite
    auto dq_result = copy_from_batched(solver.diff_state()->dq);
    auto db_result = copy_from_batched(solver.diff_state()->db);
    auto dP_result = copy_from_batched(solver.diff_state()->dP_values);
    auto dA_result = copy_from_batched(solver.diff_state()->dA_values);

    for (size_t i = 0; i < dq_result.size(); i++) {
        EXPECT_FALSE(std::isnan(dq_result[i])) << "dq[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(dq_result[i])) << "dq[" << i << "] is Inf";
    }
    for (size_t i = 0; i < db_result.size(); i++) {
        EXPECT_FALSE(std::isnan(db_result[i])) << "db[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(db_result[i])) << "db[" << i << "] is Inf";
    }
    for (size_t i = 0; i < dP_result.size(); i++) {
        EXPECT_FALSE(std::isnan(dP_result[i])) << "dP[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(dP_result[i])) << "dP[" << i << "] is Inf";
    }
    for (size_t i = 0; i < dA_result.size(); i++) {
        EXPECT_FALSE(std::isnan(dA_result[i])) << "dA[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(dA_result[i])) << "dA[" << i << "] is Inf";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// ADJOINT IDENTITY TESTS
// For backward correctness: <v, backward(u)> = <forward(v), u>
// NOTE: forward() is not yet implemented, so this test is disabled
// ============================================================================

class AdjointIdentityTest : public ::testing::Test {
protected:
    void SetUp() override {}

    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

/**
 * Test: Adjoint identity for QP with equality constraints
 *
 * The backward pass should satisfy the adjoint identity:
 * <dx_bar, dx_fwd> = <dq_bar, dq> + <db_bar, db>
 *
 * where dx_fwd = forward(dq, db) and (dq_bar, db_bar) = backward(dx_bar)
 *
 * DISABLED: forward() not yet implemented
 */
TEST_F(AdjointIdentityTest, DISABLED_QPEqualityAdjoint) {
    int n = 3, m = 1;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;

    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {2.0, 1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0};
    std::vector<double> q = {1.0, 2.0, 0.5};
    std::vector<double> b = {1.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Random perturbations for forward
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 0.1);

    std::vector<double> dq_pert(n), db_pert(m);
    for (int i = 0; i < n; i++) dq_pert[i] = dist(rng);
    for (int i = 0; i < m; i++) db_pert[i] = dist(rng);

    // Random upstream gradients for backward
    std::vector<double> dx_bar_data(n), dz_bar_data(m), ds_bar_data(m);
    for (int i = 0; i < n; i++) dx_bar_data[i] = dist(rng);
    for (int i = 0; i < m; i++) dz_bar_data[i] = dist(rng);
    for (int i = 0; i < m; i++) ds_bar_data[i] = dist(rng);

    // Forward pass
    BatchedVector dP_pert(nnzP, batchSize);
    BatchedVector dq_bv(n, batchSize);
    BatchedVector db_bv(m, batchSize);
    BatchedVector dx_fwd(n, batchSize);
    BatchedVector dz_fwd(m, batchSize);
    BatchedVector ds_fwd(m, batchSize);

    dP_pert.setToConstant(0.0, 0);
    copy_to_batched(dq_bv, dq_pert);
    copy_to_batched(db_bv, db_pert);

    BatchedVector dA_pert(nnzA, batchSize);
    dA_pert.setToConstant(0.0, 0);

    forward(*solver.diff_state(), dP_pert, dq_bv, dA_pert, db_bv, dx_fwd, dz_fwd, ds_fwd, solver, 0);
    cudaDeviceSynchronize();

    auto dx_fwd_result = copy_from_batched(dx_fwd);
    auto dz_fwd_result = copy_from_batched(dz_fwd);
    auto ds_fwd_result = copy_from_batched(ds_fwd);

    // Backward pass
    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    copy_to_batched(dx_bar, dx_bar_data);
    copy_to_batched(dz_bar, dz_bar_data);
    copy_to_batched(ds_bar, ds_bar_data);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    auto dq_bar_result = copy_from_batched(solver.diff_state()->dq);
    auto db_bar_result = copy_from_batched(solver.diff_state()->db);

    // Adjoint identity: <dx_bar, dx_fwd> + <dz_bar, dz_fwd> + <ds_bar, ds_fwd>
    //                 = <dq_bar, dq_pert> + <db_bar, db_pert>
    double lhs = dotProduct(dx_bar_data, dx_fwd_result)
               + dotProduct(dz_bar_data, dz_fwd_result)
               + dotProduct(ds_bar_data, ds_fwd_result);
    double rhs = dotProduct(dq_bar_result, dq_pert)
               + dotProduct(db_bar_result, db_pert);

    EXPECT_NEAR(lhs, rhs, 1e-6)
        << "Adjoint identity violated: " << lhs << " vs " << rhs;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// EQUILIBRATION EDGE CASES
// ============================================================================

class EquilibrationEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

/**
 * Test: Batch with one very poorly scaled problem and one well-scaled
 *
 * The poorly scaled problem shouldn't affect the well-scaled one
 */
TEST_F(EquilibrationEdgeCaseTest, AsymmetricScaling) {
    int n = 2, m = 2;
    int batchSize = 2;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Batch 0: well-scaled (all ~1)
    // Batch 1: poorly scaled (P has entries ~1e-8 and ~1e8)
    std::vector<double> P_values = {
        1.0, 1.0,       // batch 0
        1e-8, 1e8       // batch 1: badly scaled
    };
    std::vector<double> A_values = {
        1.0, 1.0,       // batch 0
        1.0, 1.0        // batch 1
    };
    std::vector<double> q = {
        1.0, 1.0,       // batch 0
        1e-8, 1e8       // batch 1
    };
    std::vector<double> b = {
        5.0, 5.0,       // batch 0
        5.0, 5.0        // batch 1
    };

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x(n * batchSize);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Batch 0 should have a reasonable solution
    EXPECT_FALSE(std::isnan(x[0])) << "Batch 0 x[0] is NaN";
    EXPECT_FALSE(std::isnan(x[1])) << "Batch 0 x[1] is NaN";
    EXPECT_FALSE(std::isinf(x[0])) << "Batch 0 x[0] is Inf";
    EXPECT_FALSE(std::isinf(x[1])) << "Batch 0 x[1] is Inf";

    // Batch 0 solution should be roughly what we expect for a well-conditioned problem
    // With P=I, q=[1,1], A=diag([1,1]), b=[5,5], nonneg constraints:
    // The solution should be around x = [0, 0] (constrained by bounds)
    EXPECT_LT(std::abs(x[0]), 10.0) << "Batch 0 x[0] magnitude unreasonable";
    EXPECT_LT(std::abs(x[1]), 10.0) << "Batch 0 x[1] magnitude unreasonable";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// DETERMINISM TESTS
// ============================================================================

// cuDSS >= 0.8 solves are not bitwise deterministic, and
// CUDSS_CONFIG_DETERMINISTIC_MODE is NOT_SUPPORTED with uniform batching.
// Allow few-ulp noise there; anything larger still indicates state leaking
// between runs. Older cuDSS is bitwise deterministic — require equality.
void expect_repeated_run_identical(double a, double b, const char* what, size_t i) {
#if CUDSS_VERSION >= 800
    EXPECT_NEAR(a, b, 1e-12 * (1.0 + std::abs(a)))
        << what << " beyond cuDSS ulp noise at index " << i
        << ": " << a << " vs " << b;
#else
    EXPECT_EQ(a, b)
        << what << " at index " << i << ": " << a << " vs " << b;
#endif
}

class DeterminismTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

/**
 * Test: Running the same problem twice should give identical results
 */
TEST_F(DeterminismTest, RepeatedSolveDeterministic) {
    int n = 3, m = 2;
    int batchSize = 2;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {0, 1, 1, 2};
    int64_t nnzA = 4;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;

    // Run twice with fresh solver instances
    std::vector<double> x_first(n * batchSize);
    std::vector<double> x_second(n * batchSize);

    for (int run = 0; run < 2; run++) {
        CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                              A_ro.data(), A_ci.data(), nnzA, cones, settings);

        std::vector<double> P_values(nnzP * batchSize);
        std::vector<double> A_values(nnzA * batchSize);
        std::vector<double> q(n * batchSize);
        std::vector<double> b(m * batchSize);

        for (int batch = 0; batch < batchSize; batch++) {
            P_values[batch * nnzP + 0] = 2.0;
            P_values[batch * nnzP + 1] = 1.0;
            P_values[batch * nnzP + 2] = 3.0;
            A_values[batch * nnzA + 0] = 1.0;
            A_values[batch * nnzA + 1] = 1.0;
            A_values[batch * nnzA + 2] = 1.0;
            A_values[batch * nnzA + 3] = 1.0;
            q[batch * n + 0] = 1.0 + batch;
            q[batch * n + 1] = 2.0;
            q[batch * n + 2] = 0.5;
            b[batch * m + 0] = 5.0;
            b[batch * m + 1] = 5.0;
        }

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
        cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);

        cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        if (run == 0) {
            solver.solution.x.gpuToCpu(x_first.data(), 0);
        } else {
            solver.solution.x.gpuToCpu(x_second.data(), 0);
        }
        cudaDeviceSynchronize();

        cudaFree(d_P);
        cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    for (size_t i = 0; i < x_first.size(); i++) {
        expect_repeated_run_identical(x_first[i], x_second[i], "Solve non-determinism", i);
    }
}

/**
 * Test: Backward pass determinism
 */
TEST_F(DeterminismTest, RepeatedBackwardDeterministic) {
    int n = 2, m = 1;
    int batchSize = 2;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values(nnzP * batchSize, 1.0);
    std::vector<double> A_values(nnzA * batchSize, 1.0);
    std::vector<double> q(n * batchSize, 0.0);
    std::vector<double> b(m * batchSize, 1.0);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    std::vector<double> dx_data = {1.0, 0.5, 0.5, 1.0};
    copy_to_batched(dx_bar, dx_data);
    dz_bar.setToConstant(0.0, 0);
    ds_bar.setToConstant(0.0, 0);

    // First backward
    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();
    auto dq_first = copy_from_batched(solver.diff_state()->dq);

    // Second backward (same inputs)
    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();
    auto dq_second = copy_from_batched(solver.diff_state()->dq);

    for (size_t i = 0; i < dq_first.size(); i++) {
        expect_repeated_run_identical(dq_first[i], dq_second[i], "Backward non-determinism", i);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// EQUILIBRATION CONSISTENCY TESTS
// Verify that repeated solves with same solver give consistent results
// ============================================================================

class EquilibrationConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

/**
 * Test: Multiple solves with same q,b on same solver should give identical results
 * This tests the equilibration caching behavior
 */
TEST_F(EquilibrationConsistencyTest, RepeatedSolveSameSolverIdentical) {
    int n = 2, m = 3;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 2, 4};
    std::vector<int64_t> P_ci = {0, 1, 0, 1};
    int64_t nnzP = 4;

    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // P = [[4, 1], [1, 2]] (full symmetric)
    std::vector<double> P_values = {4.0, 1.0, 1.0, 2.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {1.0, 0.7, 0.7};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // First solve
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();
    std::vector<double> x_first(n);
    solver.solution.x.gpuToCpu(x_first.data(), 0);
    cudaDeviceSynchronize();

    // Second solve with same data (tests vectors-only equilibration path)
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();
    std::vector<double> x_second(n);
    solver.solution.x.gpuToCpu(x_second.data(), 0);
    cudaDeviceSynchronize();

    // Third solve
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();
    std::vector<double> x_third(n);
    solver.solution.x.gpuToCpu(x_third.data(), 0);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) {
        expect_repeated_run_identical(x_first[i], x_second[i], "Solve 1 vs 2 mismatch", i);
        expect_repeated_run_identical(x_second[i], x_third[i], "Solve 2 vs 3 mismatch", i);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test: Different q values on same solver should give different (correct) results
 */
TEST_F(EquilibrationConsistencyTest, DifferentQGivesCorrectResults) {
    int n = 2, m = 3;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 2, 4};
    std::vector<int64_t> P_ci = {0, 1, 0, 1};
    int64_t nnzP = 4;

    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;

    std::vector<double> P_values = {4.0, 1.0, 1.0, 2.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> b = {1.0, 0.7, 0.7};

    // Test with multiple q values
    std::vector<std::vector<double>> q_values = {
        {1.0, 1.0},
        {2.0, 1.0},
        {0.5, 2.0},
        {-1.0, -1.0}
    };

    std::vector<std::vector<double>> results;

    for (const auto& q : q_values) {
        // Fresh solver for each q to get reference solution
        CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                              A_ro.data(), A_ci.data(), nnzA, cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP);
        cudaMalloc(&d_A, sizeof(double) * nnzA);
        cudaMalloc(&d_q, sizeof(double) * n);
        cudaMalloc(&d_b, sizeof(double) * m);

        cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x(n);
        solver.solution.x.gpuToCpu(x.data(), 0);
        cudaDeviceSynchronize();

        results.push_back(x);

        cudaFree(d_P);
        cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    // Now test with reused solver - solve all q values sequentially
    CompiledSolver reused_solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                                  A_ro.data(), A_ci.data(), nnzA, cones, settings);

    for (size_t qi = 0; qi < q_values.size(); qi++) {
        const auto& q = q_values[qi];

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP);
        cudaMalloc(&d_A, sizeof(double) * nnzA);
        cudaMalloc(&d_q, sizeof(double) * n);
        cudaMalloc(&d_b, sizeof(double) * m);

        cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

        reused_solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x(n);
        reused_solver.solution.x.gpuToCpu(x.data(), 0);
        cudaDeviceSynchronize();

        // Results should match fresh solver
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(results[qi][i], x[i], 1e-6)
                << "q[" << qi << "]: x[" << i << "] mismatch: fresh="
                << results[qi][i] << " vs reused=" << x[i];
        }

        cudaFree(d_P);
        cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);
    }
}

/**
 * Test: Batch solve should match individual solves
 */
TEST_F(EquilibrationConsistencyTest, BatchMatchesIndividualSolves) {
    int n = 2, m = 3;
    int batchSize = 4;

    std::vector<int64_t> P_ro = {0, 2, 4};
    std::vector<int64_t> P_ci = {0, 1, 0, 1};
    int64_t nnzP = 4;

    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;

    // Different q values for batch
    std::vector<std::vector<double>> q_values = {
        {1.0, 1.0},
        {2.0, 1.0},
        {0.5, 2.0},
        {-1.0, -1.0}
    };

    std::vector<double> P_values = {4.0, 1.0, 1.0, 2.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> b = {1.0, 0.7, 0.7};

    // First, get individual solutions
    std::vector<std::vector<double>> individual_results;
    for (const auto& q : q_values) {
        CompiledSolver solver(n, m, 1, P_ro.data(), P_ci.data(), nnzP,
                              A_ro.data(), A_ci.data(), nnzA, cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP);
        cudaMalloc(&d_A, sizeof(double) * nnzA);
        cudaMalloc(&d_q, sizeof(double) * n);
        cudaMalloc(&d_b, sizeof(double) * m);

        cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x(n);
        solver.solution.x.gpuToCpu(x.data(), 0);
        cudaDeviceSynchronize();
        individual_results.push_back(x);

        cudaFree(d_P);
        cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    // Now batch solve
    CompiledSolver batch_solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                                 A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Prepare batched data
    std::vector<double> P_batch(nnzP * batchSize);
    std::vector<double> A_batch(nnzA * batchSize);
    std::vector<double> q_batch(n * batchSize);
    std::vector<double> b_batch(m * batchSize);

    for (int bi = 0; bi < batchSize; bi++) {
        for (int64_t j = 0; j < nnzP; j++) P_batch[bi * nnzP + j] = P_values[j];
        for (int64_t j = 0; j < nnzA; j++) A_batch[bi * nnzA + j] = A_values[j];
        for (int j = 0; j < n; j++) q_batch[bi * n + j] = q_values[bi][j];
        for (int j = 0; j < m; j++) b_batch[bi * m + j] = b[j];
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_batch.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_batch.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    batch_solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x_batch(n * batchSize);
    batch_solver.solution.x.gpuToCpu(x_batch.data(), 0);
    cudaDeviceSynchronize();

    // Compare batch results with individual results
    for (int bi = 0; bi < batchSize; bi++) {
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(individual_results[bi][i], x_batch[bi * n + i], 1e-6)
                << "Batch " << bi << " x[" << i << "] mismatch: individual="
                << individual_results[bi][i] << " vs batch=" << x_batch[bi * n + i];
        }
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
