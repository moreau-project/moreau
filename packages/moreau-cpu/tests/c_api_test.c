/**
 * @file c_api_test.c
 * @brief Test for the Moreau CPU C API
 *
 * Tests the full lifecycle: create -> setup -> solve -> get_solution -> backward -> destroy
 * Uses a simple QP: minimize (1/2)x'Px + q'x  s.t.  Ax + s = b, s in K
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "moreau.h"

#define CHECK(expr) do { \
    moreau_error_t _err = (expr); \
    if (_err != MOREAU_OK) { \
        const char* msg = moreau_last_error(); \
        fprintf(stderr, "FAIL: %s returned %d: %s\n", #expr, _err, msg ? msg : "(null)"); \
        return 1; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (tol)) { \
        fprintf(stderr, "FAIL: %s = %.8f, expected %.8f (tol %.1e)\n", #a, _a, _b, (tol)); \
        return 1; \
    } \
} while(0)

int test_basic_qp(void) {
    printf("=== test_basic_qp ===\n");

    /* Problem dimensions */
    int64_t n = 2, m = 3;

    /* P matrix (2x2 with off-diagonals, full symmetric) in CSR format:
     * P = [[2, 1], [1, 2]]
     * row_offsets = [0, 2, 4], col_indices = [0, 1, 0, 1], values = [2, 1, 1, 2]
     */
    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    int64_t nnz_P = 4;

    /* A matrix (3x2) in CSR format:
     * A = [[1, 1], [1, 0], [0, 1]]
     * row_offsets = [0, 2, 3, 4], col_indices = [0, 1, 0, 1], values = [1, 1, 1, 1]
     */
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};
    int64_t nnz_A = 4;

    /* Cones: 1 zero cone (1 equality), 2 nonneg cones (2 inequalities) */
    moreau_cones_t cones = {0};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    /* Settings */
    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;

    /* Create solver */
    moreau_solver_t* solver = NULL;
    CHECK(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P, A_ro, A_ci, nnz_A, &cones, &settings));

    /* Verify dimensions */
    int64_t qn, qm, qbs, qnnzP, qnnzA;
    CHECK(moreau_solver_get_dims(solver, &qn, &qm, &qbs, &qnnzP, &qnnzA));
    if (qn != n || qm != m || qbs != 1 || qnnzP != nnz_P || qnnzA != nnz_A) {
        fprintf(stderr, "FAIL: get_dims returned wrong values\n");
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Setup: P and A values */
    double P_values[] = {2.0, 1.0, 1.0, 2.0};
    double A_values[] = {1.0, 1.0, 1.0, 1.0};
    CHECK(moreau_solver_setup(solver, P_values, nnz_P, A_values, nnz_A));

    /* Solve */
    double q[] = {2.0, 1.0};
    double b[] = {1.0, 0.7, 0.7};
    CHECK(moreau_solver_solve(solver, q, b));

    /* Check status */
    moreau_status_t status = moreau_solver_get_status(solver, 0);
    if (status != MOREAU_STATUS_SOLVED) {
        fprintf(stderr, "FAIL: expected SOLVED, got %d\n", status);
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Get solution */
    moreau_solution_t sol;
    CHECK(moreau_solver_get_solution(solver, 0, &sol));
    if (sol.status != MOREAU_STATUS_SOLVED) {
        fprintf(stderr, "FAIL: solution status = %d\n", sol.status);
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Check solution values: x = [0.3, 0.7] */
    printf("  x = [%.6f, %.6f]\n", sol.x[0], sol.x[1]);
    printf("  obj = %.6f, iterations = %d\n", sol.obj_val, sol.iterations);
    ASSERT_NEAR(sol.x[0], 0.3, 1e-4);
    ASSERT_NEAR(sol.x[1], 0.7, 1e-4);

    /* Also test copy_solution */
    double x_copy[2], z_copy[3], s_copy[3];
    CHECK(moreau_solver_copy_solution(solver, 0, x_copy, z_copy, s_copy));
    ASSERT_NEAR(x_copy[0], sol.x[0], 1e-12);
    ASSERT_NEAR(x_copy[1], sol.x[1], 1e-12);

    /* Destroy */
    moreau_solver_destroy(solver);

    printf("  PASSED\n");
    return 0;
}

int test_backward(void) {
    printf("=== test_backward ===\n");

    /* Same problem but with enable_grad=1 */
    int64_t n = 2, m = 3;
    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    int64_t nnz_P = 4;
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};
    int64_t nnz_A = 4;

    moreau_cones_t cones = {0};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;
    settings.enable_grad = 1;

    moreau_solver_t* solver = NULL;
    CHECK(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P, A_ro, A_ci, nnz_A, &cones, &settings));

    double P_values[] = {2.0, 1.0, 1.0, 2.0};
    double A_values[] = {1.0, 1.0, 1.0, 1.0};
    CHECK(moreau_solver_setup(solver, P_values, nnz_P, A_values, nnz_A));

    double q[] = {2.0, 1.0};
    double b[] = {1.0, 0.7, 0.7};
    CHECK(moreau_solver_solve(solver, q, b));

    moreau_status_t status = moreau_solver_get_status(solver, 0);
    if (status != MOREAU_STATUS_SOLVED) {
        fprintf(stderr, "FAIL: expected SOLVED for backward test, got %d\n", status);
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Backward pass: dx = [1, 1], dz = NULL (zeros), ds = NULL (zeros) */
    double dx[] = {1.0, 1.0};
    double dP_out[4] = {0};
    double dq_out[2] = {0};
    double dA_out[4] = {0};
    double db_out[3] = {0};

    CHECK(moreau_solver_backward(solver, dx, NULL, NULL, dP_out, dA_out, dq_out, db_out));

    printf("  dq = [%.6f, %.6f]\n", dq_out[0], dq_out[1]);
    printf("  db = [%.6f, %.6f, %.6f]\n", db_out[0], db_out[1], db_out[2]);
    printf("  dP = [%.6f, %.6f, %.6f, %.6f]\n", dP_out[0], dP_out[1], dP_out[2], dP_out[3]);

    /* Basic sanity: gradients should not all be zero */
    double sum = fabs(dq_out[0]) + fabs(dq_out[1]) + fabs(db_out[0]) + fabs(db_out[1]) + fabs(db_out[2]);
    if (sum < 1e-10) {
        fprintf(stderr, "FAIL: all gradients are zero\n");
        moreau_solver_destroy(solver);
        return 1;
    }

    moreau_solver_destroy(solver);
    printf("  PASSED\n");
    return 0;
}

int test_error_cases(void) {
    printf("=== test_error_cases ===\n");

    /* Solve before setup should fail */
    int64_t n = 2, m = 3;
    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};

    moreau_cones_t cones = {0};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;

    moreau_solver_t* solver = NULL;
    CHECK(moreau_solver_create(&solver, n, m, P_ro, P_ci, 4, A_ro, A_ci, 4, &cones, &settings));

    double q[] = {1.0, 1.0};
    double b[] = {1.0, 1.0, 1.0};
    moreau_error_t err = moreau_solver_solve(solver, q, b);
    if (err != MOREAU_ERROR_NOT_SETUP) {
        fprintf(stderr, "FAIL: expected NOT_SETUP, got %d\n", err);
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Backward without enable_grad should fail */
    double dx[] = {1.0, 1.0};
    err = moreau_solver_backward(solver, dx, NULL, NULL, NULL, NULL, NULL, NULL);
    if (err != MOREAU_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "FAIL: expected INVALID_ARGUMENT for backward without grad, got %d\n", err);
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Destroy NULL should be no-op */
    moreau_solver_destroy(NULL);

    moreau_solver_destroy(solver);
    printf("  PASSED\n");
    return 0;
}

int test_version(void) {
    printf("=== test_version ===\n");
    const char* ver = moreau_version();
    if (!ver || strlen(ver) == 0) {
        fprintf(stderr, "FAIL: moreau_version() returned null or empty\n");
        return 1;
    }
    printf("  version = %s\n", ver);
    printf("  PASSED\n");
    return 0;
}

int test_batched_qp(void) {
    printf("=== test_batched_qp ===\n");

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
    int64_t n = 2, m = 3;
    int64_t batch_size = 4;

    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    int64_t nnz_P = 4;

    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};
    int64_t nnz_A = 4;

    moreau_cones_t cones = {0};
    cones.num_zero_cones = 1;
    cones.num_nonneg_cones = 2;

    moreau_settings_t settings;
    moreau_settings_default(&settings);
    settings.verbose = 0;
    settings.batch_size = batch_size;

    moreau_solver_t* solver = NULL;
    CHECK(moreau_solver_create(&solver, n, m, P_ro, P_ci, nnz_P, A_ro, A_ci, nnz_A, &cones, &settings));

    /* Verify batch_size in dims */
    int64_t qn, qm, qbs, qnnzP, qnnzA;
    CHECK(moreau_solver_get_dims(solver, &qn, &qm, &qbs, &qnnzP, &qnnzA));
    if (qbs != batch_size) {
        fprintf(stderr, "FAIL: get_dims batch_size = %ld, expected %ld\n", (long)qbs, (long)batch_size);
        moreau_solver_destroy(solver);
        return 1;
    }

    /* Setup with shared P and A */
    double P_values[] = {2.0, 1.0, 1.0, 2.0};
    double A_values[] = {1.0, 1.0, 1.0, 1.0};
    CHECK(moreau_solver_setup(solver, P_values, nnz_P, A_values, nnz_A));

    /* Per-batch q (batch_size * n), shared b (batch_size * m) */
    double q[] = {
        2.0, 1.0,   /* batch 0 */
        1.0, 2.0,   /* batch 1 */
        0.0, 0.0,   /* batch 2 */
        3.0, 3.0    /* batch 3 */
    };
    double b[] = {
        1.0, 0.7, 0.7,   /* batch 0 */
        1.0, 0.7, 0.7,   /* batch 1 */
        1.0, 0.7, 0.7,   /* batch 2 */
        1.0, 0.7, 0.7    /* batch 3 */
    };
    CHECK(moreau_solver_solve(solver, q, b));

    /* Expected solutions */
    double expected_x[][2] = {
        {0.3, 0.7},
        {0.7, 0.3},
        {0.5, 0.5},
        {0.5, 0.5}
    };

    for (int i = 0; i < batch_size; i++) {
        moreau_status_t status = moreau_solver_get_status(solver, i);
        if (status != MOREAU_STATUS_SOLVED) {
            fprintf(stderr, "FAIL: batch %d status = %d, expected SOLVED\n", i, status);
            moreau_solver_destroy(solver);
            return 1;
        }

        double x[2], z[3], s[3];
        CHECK(moreau_solver_copy_solution(solver, i, x, z, s));
        printf("  batch %d: x = [%.6f, %.6f]\n", i, x[0], x[1]);
        ASSERT_NEAR(x[0], expected_x[i][0], 1e-4);
        ASSERT_NEAR(x[1], expected_x[i][1], 1e-4);
    }

    moreau_solver_destroy(solver);
    printf("  PASSED\n");
    return 0;
}


int main(void) {
    int failures = 0;

    failures += test_version();
    failures += test_basic_qp();
    failures += test_backward();
    failures += test_error_cases();
    failures += test_batched_qp();

    printf("\n%s: %d test(s) failed\n", failures ? "FAIL" : "OK", failures);
    return failures ? 1 : 0;
}
