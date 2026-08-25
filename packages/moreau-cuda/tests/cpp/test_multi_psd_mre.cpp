// MRE: Multiple PSD cones produce wrong results on CUDA.
// Problem: SmallestCircle02 from Migarstka SDP Benchmark Problems.
// 6 PSD cones of size 3x3 (6 svec entries each) + 5 nonneg constraints.
// n=9 variables, m=41 constraints.
// Expected objective: ~98.87
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

using namespace moreau;

TEST(MultiPsdMre, SmallestCircle02_NoChordalRaw) {
    // Problem dimensions
    int n = 9, m = 41;
    int batchSize = 1;
    double obj_true = 98.87372912081759;

    // P: 9x9 diagonal with 1e-7 entries (full symmetric = same as upper+lower for diagonal)
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int64_t nnzP = 9;

    // A is 41x9 sparse in CSR format
    // Converted from the Python scipy CSR
    std::vector<double> A_values = {
        -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.414213562373095, -1.414213562373095,
        -1.0, -1.0,
        -1.4704970806639295, -0.08408756461876067, -1.5360276835453306,
        -1.414213562373095, 0.46036521780265105, -1.414213562373095,
        17.501474782327158, 1.0, -98.72181381477408,
        -0.03428214113939981, -0.006553466114588631, -0.02293508426928332,
        -1.414213562373095, -0.32297015663680384, -1.414213562373095,
        -0.13049361700049217, 1.0, -0.6903491227746954,
        -0.04871729706664302, 0.026951701229534462, -0.02677361950669832,
        -1.414213562373095, -0.34361041779902846, -1.414213562373095,
        0.12773902766332884, 1.0, -0.21292218926103867,
        -0.34936010994705113, 0.26459518743856536, -0.48117254554003625,
        -1.414213562373095, 0.40454793521545684, -1.414213562373095,
        -0.16051832281358613, 1.0, 0.7616376517256921,
        -0.2660339628620861, 0.388854830888231, -0.3615608648317895,
        -1.414213562373095, -1.4104191111514466, -1.414213562373095,
        1.8503815675590445, 1.0, -3.73499561221377,
    };
    std::vector<int64_t> A_ci_vals = {
        4, 5, 6, 7, 8,
        1, 2, 0, 3,
        4, 4, 4, 1, 4, 2, 4, 3, 4,
        5, 5, 5, 1, 5, 2, 5, 3, 5,
        6, 6, 6, 1, 6, 2, 6, 3, 6,
        7, 7, 7, 1, 7, 2, 7, 3, 7,
        8, 8, 8, 1, 8, 2, 8, 3, 8,
    };
    std::vector<int64_t> A_ro_vals = {
        0, 1, 2, 3, 4, 5,   // rows 0-4: nonneg (1 entry each), row 5: 0 nnz
        5, 5, 5,             // rows 5-7
        6, 7, 9,             // rows 8-10
        10, 11, 12, 14, 16, 18,  // rows 11-16
        19, 20, 21, 23, 25, 27,  // rows 17-22
        28, 29, 30, 32, 34, 36,  // rows 23-28
        37, 38, 39, 41, 43, 45,  // rows 29-34
        46, 47, 48, 50, 52, 54,  // rows 35-40
    };
    int64_t nnzA = A_values.size();

    // Cones: 5 nonneg + 6 PSD(3x3)
    Cones cones{};
    cones.numNonnegCones = 5;
    cones.psdConeDims = {3, 3, 3, 3, 3, 3};
    cones.numPsdCones = 6;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = true;

    printf("n=%d, m=%d, nnzP=%ld, nnzA=%ld\n", n, m, nnzP, nnzA);
    printf("A_ro size=%zu, A_ci size=%zu, A_val size=%zu\n",
           A_ro_vals.size(), A_ci_vals.size(), A_values.size());

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro_vals.data(), A_ci_vals.data(), nnzA,
                          cones, settings);

    // P and q values
    std::vector<double> P_values(9, 1e-7);
    std::vector<double> q = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> b = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
    };

    // Allocate GPU memory
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.setup(d_P, d_A);
    solver.solve(d_q, d_b);

    // Get solution
    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    int32_t status;
    cudaMemcpy(&status, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    printf("Status: %d\n", status);
    printf("x = [");
    for (int i = 0; i < n; i++) printf(" %.6f", x[i]);
    printf(" ]\n");

    // Get objective
    double obj = 0.0;
    for (int i = 0; i < n; i++) {
        obj += 0.5 * P_values[i] * x[i] * x[i]; // P is diagonal
        obj += q[i] * x[i];
    }
    printf("Objective: %.6f (expected: %.6f)\n", obj, obj_true);
    printf("Error: %.2e\n", std::abs(obj - obj_true));

    // Check status
    EXPECT_EQ(status, 1) << "Expected Solved status";
    EXPECT_NEAR(obj, obj_true, 1.0) << "Objective should be close to expected";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
