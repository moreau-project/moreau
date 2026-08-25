#include "moreau/solver/solver.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

using namespace moreau;

int main(int argc, char** argv) {
    // Usage: large_qp_batched [batch_size] [n] [m] [density]

    // Default problem size
    int n = 5000;
    int m = 2500;
    int batchSize = 128;
    double density = 0.05;  // 5% density by default
    bool skip_solve = false;
    bool use_schur = false;

    // Parse arguments
    int pos_arg = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--skip-solve") {
            skip_solve = true;
        } else if (arg == "--schur") {
            use_schur = true;
        } else {
            // Positional arguments: batch_size, n, m, density
            if (pos_arg == 0) batchSize = std::atoi(argv[i]);
            else if (pos_arg == 1) n = std::atoi(argv[i]);
            else if (pos_arg == 2) m = std::atoi(argv[i]);
            else if (pos_arg == 3) density = std::atof(argv[i]);
            pos_arg++;
        }
    }

    std::cout << "Creating large batched QP problem:\n";
    std::cout << "  Variables: " << n << "\n";
    std::cout << "  Constraints: " << m << "\n";
    std::cout << "  Batch size: " << batchSize << "\n";
    std::cout << "  Target density: " << (density * 100) << "%\n";

    // Random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.1, 2.0);

    // P matrix: diagonal with random positive values (n x n)
    // Same structure for all batches
    std::vector<int64_t> P_rowOffsets(n + 1);
    std::vector<int64_t> P_colIndices(n);

    for (int i = 0; i < n; i++) {
        P_rowOffsets[i] = i;
        P_colIndices[i] = i;
    }
    P_rowOffsets[n] = n;
    int nnzP = n;

    // Generate different P values for each batch
    std::vector<double> P_values(nnzP * batchSize);
    for (int batch = 0; batch < batchSize; batch++) {
        for (int i = 0; i < n; i++) {
            P_values[batch * nnzP + i] = dist(rng);
        }
    }

    // q vector: different random linear cost for each batch
    std::vector<double> q_values(n * batchSize);
    std::uniform_real_distribution<double> q_dist(-1.0, 1.0);
    for (int batch = 0; batch < batchSize; batch++) {
        for (int i = 0; i < n; i++) {
            q_values[batch * n + i] = q_dist(rng);
        }
    }

    // A matrix: sparse random (m x n), with specified density
    // Same structure for all batches
    std::vector<int64_t> A_rowOffsets;
    std::vector<int64_t> A_colIndices;

    A_rowOffsets.push_back(0);
    int64_t nnz = 0;

    // Compute entries per row from density
    int entries_per_row = std::max(1, std::min(n, static_cast<int>(density * n + 0.5)));

    for (int i = 0; i < m; i++) {
        // Sample random columns without replacement
        std::vector<int> all_cols(n);
        std::iota(all_cols.begin(), all_cols.end(), 0);
        std::shuffle(all_cols.begin(), all_cols.end(), rng);

        // Take first entries_per_row columns
        std::vector<int> cols(all_cols.begin(), all_cols.begin() + entries_per_row);
        std::sort(cols.begin(), cols.end());

        for (int col : cols) {
            A_colIndices.push_back(col);
            nnz++;
        }

        A_rowOffsets.push_back(nnz);
    }

    int nnzA = nnz;
    std::cout << "A matrix sparsity: " << nnzA << " nonzeros ("
              << (100.0 * nnzA / (n * m)) << "% dense)\n";

    // Generate different A values for each batch
    std::vector<double> A_values(nnzA * batchSize);
    for (int batch = 0; batch < batchSize; batch++) {
        for (int i = 0; i < nnzA; i++) {
            A_values[batch * nnzA + i] = dist(rng);
        }
    }

    // b vector (RHS) - different for each batch
    std::vector<double> b_values(m * batchSize);
    std::uniform_real_distribution<double> b_dist(-2.0, 2.0);
    for (int batch = 0; batch < batchSize; batch++) {
        for (int i = 0; i < m; i++) {
            b_values[batch * m + i] = b_dist(rng);
        }
    }

    // Cone configuration: all nonnegative (simple box constraints)
    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = m;  // All constraints are x >= 0
    cones.numExpCones = 0;
    cones.numSocCones = 0;
    cones.numPowerCones = 0;

    // Settings
    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;  // Enable to see detailed timing breakdown

    std::cout << "Creating solver...\n";

    // Measure constructor time
    cudaEvent_t ctor_start, ctor_stop;
    cudaEventCreate(&ctor_start);
    cudaEventCreate(&ctor_stop);

    cudaEventRecord(ctor_start);
    CompiledSolver* solver_ptr;
    {
        // Create solver
        solver_ptr = new CompiledSolver(
            n, m, batchSize,
            P_rowOffsets.data(), P_colIndices.data(), nnzP,
            A_rowOffsets.data(), A_colIndices.data(), nnzA,
            cones,
            settings
        );
    }
    CompiledSolver& solver = *solver_ptr;
    cudaEventRecord(ctor_stop);
    cudaEventSynchronize(ctor_stop);

    float ctor_ms = 0;
    cudaEventElapsedTime(&ctor_ms, ctor_start, ctor_stop);
    std::cout << "Constructor time: " << ctor_ms << " ms\n";

    cudaEventDestroy(ctor_start);
    cudaEventDestroy(ctor_stop);

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

    std::cout << "\nSolving " << batchSize << " problems...\n";

    // Benchmark the solve
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure setup separately
    cudaEvent_t setup_start, setup_stop;
    cudaEventCreate(&setup_start);
    cudaEventCreate(&setup_stop);

    cudaEventRecord(setup_start);
    {
        solver.setup(d_P_values, d_A_values);
    }
    cudaEventRecord(setup_stop);
    cudaEventSynchronize(setup_stop);

    float setup_ms = 0;
    cudaEventElapsedTime(&setup_ms, setup_start, setup_stop);

    // Measure solve separately
    cudaEventRecord(start);
    if (!skip_solve) {
        solver.solve(d_q, d_b);
    } else {
        std::cout << "SKIPPED solve() call\n";
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float solve_ms = 0;
    cudaEventElapsedTime(&solve_ms, start, stop);

    float total_ms = setup_ms + solve_ms;

    std::cout << "\n=== BENCHMARK ===" << std::endl;
    std::cout << "Setup time: " << setup_ms << " ms" << std::endl;
    std::cout << "Solve time: " << solve_ms << " ms" << std::endl;
    std::cout << "Total time: " << total_ms << " ms" << std::endl;
    std::cout << "Time per problem (solve only): " << (solve_ms / batchSize) << " ms" << std::endl;
    std::cout << "Time per problem (total): " << (total_ms / batchSize) << " ms" << std::endl;
    std::cout << "Throughput (solve only): " << (1000.0 * batchSize / solve_ms) << " problems/second" << std::endl;

    cudaEventDestroy(setup_start);
    cudaEventDestroy(setup_stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print solution statistics
    std::vector<double> x_host(n * batchSize);
    cudaMemcpy(x_host.data(), solver.solution.x.data(),
               sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    (void)use_schur;  // unused for now

    std::cout << "\n=== SOLUTION (first problem, first 10 values) ===\n";
    std::cout << "x = [";
    for (int i = 0; i < std::min(10, n); i++) {
        std::cout << x_host[i];
        if (i < std::min(10, n)-1) std::cout << ", ";
    }
    std::cout << ", ...]\n";

    // Print final info for all batches
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    std::vector<int32_t> status_vals(batchSize);
    cudaMemcpy(status_vals.data(), solver.info.status_device,
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    std::cout << "\n=== BATCH RESULTS ===\n";
    std::cout << "Total iterations: " << solver.info.iterations << "\n";

    int num_solved = 0;
    int num_almost_solved = 0;
    int num_failed = 0;

    for (int batch = 0; batch < batchSize; batch++) {
        if (status_vals[batch] == 1) num_solved++;
        else if (status_vals[batch] == 6) num_almost_solved++;
        else num_failed++;
    }

    std::cout << "Solved: " << num_solved << " / " << batchSize << "\n";
    std::cout << "Almost solved: " << num_almost_solved << " / " << batchSize << "\n";
    std::cout << "Failed: " << num_failed << " / " << batchSize << "\n";

    if (batchSize <= 10) {
        std::cout << "\n=== PER-BATCH DETAILS ===\n";
        for (int batch = 0; batch < batchSize; batch++) {
            std::cout << "Batch " << batch << ": objective=" << cost_primal[batch]
                      << ", status=" << status_vals[batch] << "\n";
        }
    }

    // Clean up
    {
        cudaFree(d_P_values);
        cudaFree(d_A_values);
        cudaFree(d_q);
        cudaFree(d_b);
        delete solver_ptr;
    }

    return 0;
}
