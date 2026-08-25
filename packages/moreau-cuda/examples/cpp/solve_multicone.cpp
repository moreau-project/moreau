#include "moreau/solver/solver.hpp"
#include <iostream>
#include <vector>

using namespace moreau;

int main() {
    // 5 variables, 13 constraints
    int n = 5;
    int m = 13;
    int batchSize = 1;

    // P matrix: 5x5 diagonal [1, 2, 3, 4, 5]
    std::vector<int64_t> P_rowOffsets = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_colIndices = {0, 1, 2, 3, 4};
    std::vector<double> P_values = {1.0, 2.0, 3.0, 4.0, 5.0};
    int64_t nnzP = 5;

    // q vector
    std::vector<double> q_values = {-1.0, -1.0, 0.0, 0.0, 0.0};

    // A matrix (13x5):
    // Row 0-1: Zero cone (2 rows)
    // Row 2-3: Nonnegative cone (2 rows)
    // Row 4-6: Exponential cone (3 rows)
    // Row 7-9: SOC (3 rows)
    // Row 10-12: Power cone (3 rows)
    std::vector<int64_t> A_rowOffsets = {
        0,   // Row 0: [1, 0, 0, 0, 0]
        1,   // Row 1: [0, 1, 0, 0, 0]
        2,   // Row 2: [0, 0, 1, 0, 0]
        3,   // Row 3: [0, 0, 0, 1, 0]
        4,   // Row 4: [1, 0, 0, 0, 0]
        5,   // Row 5: [0, 1, 0, 0, 0]
        6,   // Row 6: [0, 0, 1, 0, 0]
        7,   // Row 7: [0, 0, 1, 0, 0]
        8,   // Row 8: [0, 0, 0, 1, 0]
        9,   // Row 9: [0, 0, 0, 0, 1]
        10,  // Row 10: [1, 0, 0, 0, 0]
        11,  // Row 11: [0, 0, 0, 1, 0]
        12,  // Row 12: [0, 0, 0, 0, 1]
        13   // End marker
    };
    std::vector<int64_t> A_colIndices = {
        0,  // Row 0
        1,  // Row 1
        2,  // Row 2
        3,  // Row 3
        0,  // Row 4
        1,  // Row 5
        2,  // Row 6
        2,  // Row 7
        3,  // Row 8
        4,  // Row 9
        0,  // Row 10
        3,  // Row 11
        4   // Row 12
    };
    std::vector<double> A_values(13, 1.0);  // All ones
    int64_t nnzA = 13;

    // b vector (RHS)
    std::vector<double> b_values = {
        1.0, 1.0,         // Zero cone
        2.0, 2.0,         // Nonnegative cone
        1.0, 1.0, 2.0,    // Exponential cone
        2.0, 2.0, 0.5,    // SOC
        1.0, 2.0, 0.5     // Power cone
    };

    // Cone configuration
    Cones cones;
    cones.numZeroCones = 2;
    cones.numNonnegCones = 2;
    cones.numExpCones = 1;
    cones.socConeDims = {3};  // 1 SOC cone of dimension 3
    cones.numPowerCones = 1;
    cones.powerAlphas = {0.6};

    // Settings
    Settings settings;
    settings.maxIter = 50;
    settings.verbose = true;
    settings.ipm.equilibrationSettings.enable = true;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_rowOffsets.data(), P_colIndices.data(), nnzP,
        A_rowOffsets.data(), A_colIndices.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory for problem data
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_values.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_values.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Print solution
    std::cout << "\n=== SOLUTION ===\n";
    std::vector<double> x_host(n * batchSize);
    std::vector<double> s_host(m * batchSize);
    std::vector<double> z_host(m * batchSize);

    cudaMemcpy(x_host.data(), solver.solution.x.data(),
               sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_host.data(), solver.solution.s.data(),
               sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_host.data(), solver.solution.z.data(),
               sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);

    std::cout << "x = [";
    for (int i = 0; i < n; i++) {
        std::cout << x_host[i];
        if (i < n-1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "s = [";
    for (int i = 0; i < m; i++) {
        std::cout << s_host[i];
        if (i < m-1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "z = [";
    for (int i = 0; i < m; i++) {
        std::cout << z_host[i];
        if (i < m-1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);

    return 0;
}
