/**
 * @file test_c_api.cpp
 * @brief Test for the Moreau CUDA C API
 *
 * Tests the full lifecycle: create -> setup -> solve -> get_solution -> backward -> destroy
 * Uses a simple QP with device memory.
 *
 * Problem:
 *   P = [[2, 1], [1, 2]]  (full symmetric, positive definite)
 *   q = [2, 1]
 *   A = [[1, 1], [1, 0], [0, 1]]
 *   b = [1, 0.7, 0.7]
 *   cones: 1 zero cone (equality), 2 nonneg cones (inequalities)
 *
 *   Expected solution: x = [0.3, 0.7]
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

// Include the C API header
extern "C" {
#include "moreau.h"
}

// Helper to allocate device memory and copy from host
static double* to_device(const double* host, size_t count) {
    double* d_ptr = nullptr;
    cudaMalloc(&d_ptr, count * sizeof(double));
    cudaMemcpy(d_ptr, host, count * sizeof(double), cudaMemcpyHostToDevice);
    return d_ptr;
}

// Helper to allocate zeroed device memory
static double* device_zeros(size_t count) {
    double* d_ptr = nullptr;
    cudaMalloc(&d_ptr, count * sizeof(double));
    cudaMemset(d_ptr, 0, count * sizeof(double));
    return d_ptr;
}

// Helper to copy from device to host
static void to_host(double* host, const double* device, size_t count) {
    cudaMemcpy(host, device, count * sizeof(double), cudaMemcpyDeviceToHost);
}

class CApiTest : public ::testing::Test {
protected:
    // Problem dimensions
    static constexpr int64_t n = 2;
    static constexpr int64_t m = 3;

    // Structure arrays (host pointers for create)
    // P = [[2, 1], [1, 2]]
    int64_t P_ro[3] = {0, 2, 4};
    int64_t P_ci[4] = {0, 1, 0, 1};
    static constexpr int64_t nnz_P = 4;

    int64_t A_ro[4] = {0, 2, 3, 4};
    int64_t A_ci[4] = {0, 1, 0, 1};
    static constexpr int64_t nnz_A = 4;

    // Values
    double h_P_values[4] = {2.0, 1.0, 1.0, 2.0};
    double h_A_values[4] = {1.0, 1.0, 1.0, 1.0};
    double h_q[2] = {2.0, 1.0};
    double h_b[3] = {1.0, 0.7, 0.7};
};

TEST_F(CApiTest, Version) {
    const char* ver = moreau_version();
    ASSERT_NE(ver, nullptr);
    EXPECT_GT(strlen(ver), 0);
}

TEST_F(CApiTest, SettingsDefault) {
    moreau_settings_t s;
    moreau_settings_default(&s);
    EXPECT_EQ(s.batch_size, 1);
    EXPECT_EQ(s.max_iter, 200u);
    EXPECT_EQ(s.verbose, 1);
    EXPECT_EQ(s.enable_grad, 0);
    EXPECT_DOUBLE_EQ(s.ipm.tol_gap_abs, 1e-8);
}

TEST_F(CApiTest, BasicQP) {
    moreau_cones_t cones = {};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;

    moreau_solver_t* solver = nullptr;
    ASSERT_EQ(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P,
              A_ro, A_ci, nnz_A, &cones, &settings), MOREAU_OK);
    ASSERT_NE(solver, nullptr);

    // Verify dimensions
    int64_t qn, qm, qbs, qnnzP, qnnzA;
    ASSERT_EQ(moreau_solver_get_dims(solver, &qn, &qm, &qbs, &qnnzP, &qnnzA), MOREAU_OK);
    EXPECT_EQ(qn, n);
    EXPECT_EQ(qm, m);
    EXPECT_EQ(qbs, 1);

    // Setup with device pointers
    double* d_P = to_device(h_P_values, nnz_P);
    double* d_A = to_device(h_A_values, nnz_A);
    ASSERT_EQ(moreau_solver_setup(solver, d_P, nnz_P, d_A, nnz_A), MOREAU_OK);

    // Solve with device pointers
    double* d_q = to_device(h_q, n);
    double* d_b = to_device(h_b, m);
    ASSERT_EQ(moreau_solver_solve(solver, d_q, d_b), MOREAU_OK);

    // Check status
    moreau_status_t status = moreau_solver_get_status(solver, 0);
    EXPECT_EQ(status, MOREAU_STATUS_SOLVED);

    // Copy solution to host: expected x = [0.3, 0.7]
    double x[2], z[3], s[3];
    ASSERT_EQ(moreau_solver_copy_solution(solver, 0, x, z, s), MOREAU_OK);

    EXPECT_NEAR(x[0], 0.3, 1e-4);
    EXPECT_NEAR(x[1], 0.7, 1e-4);

    // Get solution (returns device pointers)
    moreau_solution_t sol;
    ASSERT_EQ(moreau_solver_get_solution(solver, 0, &sol), MOREAU_OK);
    EXPECT_EQ(sol.status, MOREAU_STATUS_SOLVED);
    EXPECT_GT(sol.iterations, 0);

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    moreau_solver_destroy(solver);
}

TEST_F(CApiTest, Backward) {
    moreau_cones_t cones = {};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;
    settings.enable_grad = 1;

    moreau_solver_t* solver = nullptr;
    ASSERT_EQ(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P,
              A_ro, A_ci, nnz_A, &cones, &settings), MOREAU_OK);

    double* d_P = to_device(h_P_values, nnz_P);
    double* d_A = to_device(h_A_values, nnz_A);
    ASSERT_EQ(moreau_solver_setup(solver, d_P, nnz_P, d_A, nnz_A), MOREAU_OK);

    double* d_q = to_device(h_q, n);
    double* d_b = to_device(h_b, m);
    ASSERT_EQ(moreau_solver_solve(solver, d_q, d_b), MOREAU_OK);

    EXPECT_EQ(moreau_solver_get_status(solver, 0), MOREAU_STATUS_SOLVED);

    // Backward: dx = [1, 1], dz = NULL, ds = NULL
    double h_dx[] = {1.0, 1.0};
    double* d_dx = to_device(h_dx, n);

    // Allocate output device memory
    double* d_dP = device_zeros(nnz_P);
    double* d_dq = device_zeros(n);
    double* d_dA = device_zeros(nnz_A);
    double* d_db = device_zeros(m);

    ASSERT_EQ(moreau_solver_backward(solver, d_dx, nullptr, nullptr,
              d_dP, d_dA, d_dq, d_db), MOREAU_OK);

    // Copy back and check non-zero
    double h_dq[2], h_db[3], h_dP[4];
    to_host(h_dq, d_dq, n);
    to_host(h_db, d_db, m);
    to_host(h_dP, d_dP, nnz_P);

    double sum = fabs(h_dq[0]) + fabs(h_dq[1]) + fabs(h_db[0]) + fabs(h_db[1]) + fabs(h_db[2]);
    EXPECT_GT(sum, 1e-10) << "All gradients are zero";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_dx);
    cudaFree(d_dP);
    cudaFree(d_dq);
    cudaFree(d_dA);
    cudaFree(d_db);
    moreau_solver_destroy(solver);
}

TEST_F(CApiTest, ErrorCases) {
    moreau_cones_t cones = {};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;

    moreau_solver_t* solver = nullptr;
    ASSERT_EQ(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P,
              A_ro, A_ci, nnz_A, &cones, &settings), MOREAU_OK);

    // Solve before setup should fail
    double* d_q = to_device(h_q, n);
    double* d_b = to_device(h_b, m);
    EXPECT_EQ(moreau_solver_solve(solver, d_q, d_b), MOREAU_ERROR_NOT_SETUP);

    // Backward without enable_grad should fail
    double h_dx[] = {1.0, 1.0};
    double* d_dx = to_device(h_dx, n);
    EXPECT_EQ(moreau_solver_backward(solver, d_dx, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr), MOREAU_ERROR_INVALID_ARGUMENT);

    // Destroy NULL is a no-op
    moreau_solver_destroy(nullptr);

    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_dx);
    moreau_solver_destroy(solver);
}

TEST_F(CApiTest, BatchedQP) {
    /*
     * Batch of 4 QPs with shared P, A, b and per-batch q.
     * P = [[2, 1], [1, 2]], A = [[1,1],[1,0],[0,1]], b = [1, 0.7, 0.7]
     *
     * Expected solutions:
     *   q=[2,1]: x=[0.3, 0.7]
     *   q=[1,2]: x=[0.7, 0.3]
     *   q=[0,0]: x=[0.5, 0.5]
     *   q=[3,3]: x=[0.5, 0.5]
     */
    const int64_t batch_size = 4;

    moreau_cones_t cones = {};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;
    settings.batch_size = batch_size;

    moreau_solver_t* solver = nullptr;
    ASSERT_EQ(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P,
              A_ro, A_ci, nnz_A, &cones, &settings), MOREAU_OK);

    // Verify batch_size in dims
    int64_t qn, qm, qbs, qnnzP, qnnzA;
    ASSERT_EQ(moreau_solver_get_dims(solver, &qn, &qm, &qbs, &qnnzP, &qnnzA), MOREAU_OK);
    EXPECT_EQ(qbs, batch_size);

    // Setup with shared P and A (single copy, P_count = nnz_P)
    double* d_P = to_device(h_P_values, nnz_P);
    double* d_A = to_device(h_A_values, nnz_A);
    ASSERT_EQ(moreau_solver_setup(solver, d_P, nnz_P, d_A, nnz_A), MOREAU_OK);

    // Per-batch q (batch_size * n)
    double h_q_batch[] = {
        2.0, 1.0,   // batch 0
        1.0, 2.0,   // batch 1
        0.0, 0.0,   // batch 2
        3.0, 3.0    // batch 3
    };
    // Shared b replicated for each batch (batch_size * m)
    double h_b_batch[] = {
        1.0, 0.7, 0.7,
        1.0, 0.7, 0.7,
        1.0, 0.7, 0.7,
        1.0, 0.7, 0.7
    };

    double* d_q_batch = to_device(h_q_batch, batch_size * n);
    double* d_b_batch = to_device(h_b_batch, batch_size * m);
    ASSERT_EQ(moreau_solver_solve(solver, d_q_batch, d_b_batch), MOREAU_OK);

    // Expected solutions
    double expected_x[][2] = {
        {0.3, 0.7},
        {0.7, 0.3},
        {0.5, 0.5},
        {0.5, 0.5}
    };

    for (int i = 0; i < batch_size; i++) {
        moreau_status_t status = moreau_solver_get_status(solver, i);
        EXPECT_EQ(status, MOREAU_STATUS_SOLVED) << "batch " << i;

        double x[2], z[3], s[3];
        ASSERT_EQ(moreau_solver_copy_solution(solver, i, x, z, s), MOREAU_OK);
        EXPECT_NEAR(x[0], expected_x[i][0], 1e-4) << "batch " << i << " x[0]";
        EXPECT_NEAR(x[1], expected_x[i][1], 1e-4) << "batch " << i << " x[1]";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q_batch);
    cudaFree(d_b_batch);
    moreau_solver_destroy(solver);
}
