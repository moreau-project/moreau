/**
 * @file test_inf_in_b.cpp
 * @brief Tests for inf in b vector handling.
 *
 * Tests:
 * - +inf in nonneg cone: vacuous constraint, should solve correctly
 * - -inf in nonneg cone: infeasible, should return PrimalInfeasible
 * - ±inf in zero cone: infeasible, should return PrimalInfeasible
 * - Mixed cones with inf
 * - Normal problems without inf (regression)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"

using namespace moreau;

class InfInBTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    // Helper: allocate device memory, copy, solve, return host status and x
    struct SolveResult {
        SolverStatus status;
        std::vector<double> x;
        double obj_val;
    };

    SolveResult solveLP(
        int n, int m,
        const std::vector<int64_t>& P_ro,
        const std::vector<int64_t>& P_ci,
        int64_t nnzP,
        const std::vector<int64_t>& A_ro,
        const std::vector<int64_t>& A_ci,
        int64_t nnzA,
        const Cones& cones,
        const std::vector<double>& P_values,
        const std::vector<double>& A_values,
        const std::vector<double>& q_data,
        const std::vector<double>& b_data)
    {
        Settings settings;
        settings.verbose = false;
        int batchSize = 1;

        CompiledSolver solver(
            n, m, batchSize,
            P_ro.data(), P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones, settings
        );

        // Allocate device memory
        double *d_P = nullptr, *d_A = nullptr, *d_q = nullptr, *d_b = nullptr;
        if (nnzP > 0) {
            cudaMalloc(&d_P, sizeof(double) * nnzP);
            cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
        }
        if (nnzA > 0) {
            cudaMalloc(&d_A, sizeof(double) * nnzA);
            cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
        }
        cudaMalloc(&d_q, sizeof(double) * n);
        cudaMalloc(&d_b, sizeof(double) * m);
        cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

        solver.solveAll(d_P, d_A, d_q, d_b);

        // Extract results
        SolveResult result;
        result.status = solver.info.status[0];
        result.x.resize(n);
        cudaMemcpy(result.x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

        // Get obj_val
        std::vector<double> obj(1);
        cudaMemcpy(obj.data(), solver.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
        result.obj_val = obj[0];

        if (d_P) cudaFree(d_P);
        if (d_A) cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);

        return result;
    }
};

// ============================================================================
// +inf in nonneg cone (vacuous, should solve)
// ============================================================================

TEST_F(InfInBTest, NonnegPosInfSimple) {
    // min x s.t. x >= 0, x <= inf → x = 0
    Cones cones{};
    cones.numNonnegCones = 2;

    auto result = solveLP(
        1, 2,
        {0, 0}, {}, 0,
        {0, 1, 2}, {0, 0}, 2,
        cones,
        {}, {-1.0, 1.0},
        {1.0}, {0.0, std::numeric_limits<double>::infinity()}
    );

    EXPECT_TRUE(result.status == SolverStatus::Solved ||
                result.status == SolverStatus::AlmostSolved)
        << "status = " << static_cast<int>(result.status);
    EXPECT_NEAR(result.x[0], 0.0, 1e-6);
}

TEST_F(InfInBTest, NonnegPosInfWithLowerBound) {
    // min x s.t. x >= 0.5, x <= inf → x = 0.5
    Cones cones{};
    cones.numNonnegCones = 2;

    auto result = solveLP(
        1, 2,
        {0, 0}, {}, 0,
        {0, 1, 2}, {0, 0}, 2,
        cones,
        {}, {-1.0, 1.0},
        {1.0}, {-0.5, std::numeric_limits<double>::infinity()}
    );

    EXPECT_TRUE(result.status == SolverStatus::Solved ||
                result.status == SolverStatus::AlmostSolved)
        << "status = " << static_cast<int>(result.status);
    EXPECT_NEAR(result.x[0], 0.5, 1e-5);
}

TEST_F(InfInBTest, NonnegPosInf2D) {
    // min x'Px + q'x s.t. x >= [0.5, 0.3], x <= [inf, inf]
    // P = diag(2,2), q = [1, -1] → x = [0.5, 0.5]
    Cones cones{};
    cones.numNonnegCones = 4;

    auto result = solveLP(
        2, 4,
        {0, 1, 2}, {0, 1}, 2,
        {0, 1, 2, 3, 4}, {0, 1, 0, 1}, 4,
        cones,
        {2.0, 2.0},
        {-1.0, -1.0, 1.0, 1.0},
        {1.0, -1.0},
        {-0.5, -0.3, std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()}
    );

    EXPECT_TRUE(result.status == SolverStatus::Solved ||
                result.status == SolverStatus::AlmostSolved)
        << "status = " << static_cast<int>(result.status);
    EXPECT_NEAR(result.x[0], 0.5, 1e-4);
    EXPECT_NEAR(result.x[1], 0.5, 1e-4);
}

// ============================================================================
// -inf in nonneg cone (infeasible)
// ============================================================================

TEST_F(InfInBTest, NonnegNegInfInfeasible) {
    Cones cones{};
    cones.numNonnegCones = 1;

    auto result = solveLP(
        1, 1,
        {0, 0}, {}, 0,
        {0, 1}, {0}, 1,
        cones,
        {}, {-1.0},
        {1.0}, {-std::numeric_limits<double>::infinity()}
    );

    EXPECT_EQ(result.status, SolverStatus::PrimalInfeasible);
}

TEST_F(InfInBTest, NonnegNegInfMixed) {
    // Two nonneg rows: first finite, second -inf
    Cones cones{};
    cones.numNonnegCones = 2;

    auto result = solveLP(
        1, 2,
        {0, 0}, {}, 0,
        {0, 1, 2}, {0, 0}, 2,
        cones,
        {}, {-1.0, 1.0},
        {1.0}, {0.0, -std::numeric_limits<double>::infinity()}
    );

    EXPECT_EQ(result.status, SolverStatus::PrimalInfeasible);
}

// ============================================================================
// ±inf in zero cone (always infeasible)
// ============================================================================

TEST_F(InfInBTest, ZeroConePosInfInfeasible) {
    Cones cones{};
    cones.numZeroCones = 1;

    auto result = solveLP(
        1, 1,
        {0, 0}, {}, 0,
        {0, 1}, {0}, 1,
        cones,
        {}, {1.0},
        {1.0}, {std::numeric_limits<double>::infinity()}
    );

    EXPECT_EQ(result.status, SolverStatus::PrimalInfeasible);
}

TEST_F(InfInBTest, ZeroConeNegInfInfeasible) {
    Cones cones{};
    cones.numZeroCones = 1;

    auto result = solveLP(
        1, 1,
        {0, 0}, {}, 0,
        {0, 1}, {0}, 1,
        cones,
        {}, {1.0},
        {1.0}, {-std::numeric_limits<double>::infinity()}
    );

    EXPECT_EQ(result.status, SolverStatus::PrimalInfeasible);
}

// ============================================================================
// Mixed cones with inf
// ============================================================================

TEST_F(InfInBTest, MixedZeroNonnegPosInfSolves) {
    // min 0 s.t. x = 1 (zero), x <= inf (nonneg) → x = 1
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 1;

    auto result = solveLP(
        1, 2,
        {0, 0}, {}, 0,
        {0, 1, 2}, {0, 0}, 2,
        cones,
        {}, {1.0, 1.0},
        {0.0}, {1.0, std::numeric_limits<double>::infinity()}
    );

    EXPECT_TRUE(result.status == SolverStatus::Solved ||
                result.status == SolverStatus::AlmostSolved)
        << "status = " << static_cast<int>(result.status);
    EXPECT_NEAR(result.x[0], 1.0, 1e-6);
}

TEST_F(InfInBTest, MixedZeroInfNonnegFiniteInfeasible) {
    // Zero cone has inf → infeasible regardless of nonneg
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 1;

    auto result = solveLP(
        1, 2,
        {0, 0}, {}, 0,
        {0, 1, 2}, {0, 0}, 2,
        cones,
        {}, {1.0, -1.0},
        {1.0}, {std::numeric_limits<double>::infinity(), 0.0}
    );

    EXPECT_EQ(result.status, SolverStatus::PrimalInfeasible);
}

// ============================================================================
// No inf (regression)
// ============================================================================

TEST_F(InfInBTest, NoInfRegression) {
    // Normal QP: min (1/2)(2x^2) + x s.t. x >= 0 → x = 0
    Cones cones{};
    cones.numNonnegCones = 1;

    auto result = solveLP(
        1, 1,
        {0, 1}, {0}, 1,
        {0, 1}, {0}, 1,
        cones,
        {2.0}, {-1.0},
        {1.0}, {0.0}
    );

    EXPECT_TRUE(result.status == SolverStatus::Solved ||
                result.status == SolverStatus::AlmostSolved);
    EXPECT_NEAR(result.x[0], 0.0, 1e-6);
}
