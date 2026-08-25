/**
 * @brief Benchmark: YOLO mode vs normal solve on a large sparse random QP.
 *
 * Generates a random sparse QP with n variables, m constraints (zero + nonneg cones),
 * solves it normally to find the iteration count, then benchmarks both modes.
 */

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

using namespace moreau;

// Generate a random sparse symmetric positive-definite P (diagonal + some off-diag)
// Returns CSR arrays. P is stored as full symmetric.
void generate_random_P(int n, double density,
                       std::vector<int64_t>& ro, std::vector<int64_t>& ci,
                       std::vector<double>& vals, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Build as dense upper triangle, then symmetrize
    // For simplicity, use diagonal-dominant sparse P
    struct Entry { int64_t r, c; double v; };
    std::vector<std::vector<Entry>> rows(n);

    for (int i = 0; i < n; i++) {
        // Diagonal: always present, value in [1, 3]
        double diag = 1.0 + 2.0 * dist(rng);
        rows[i].push_back({i, i, diag});

        // Off-diagonal: sparse
        for (int j = i + 1; j < n; j++) {
            if (dist(rng) < density) {
                double v = 0.1 * (dist(rng) - 0.5);
                rows[i].push_back({i, j, v});
                rows[j].push_back({j, i, v});  // symmetric
            }
        }
    }

    // Sort each row by column
    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [](const Entry& a, const Entry& b) { return a.c < b.c; });
    }

    // Build CSR
    ro.resize(n + 1);
    ci.clear();
    vals.clear();
    ro[0] = 0;
    for (int i = 0; i < n; i++) {
        for (auto& e : rows[i]) {
            ci.push_back(e.c);
            vals.push_back(e.v);
        }
        ro[i + 1] = ci.size();
    }
}

// Generate random sparse A matrix in CSR
void generate_random_A(int m, int n, double density,
                       std::vector<int64_t>& ro, std::vector<int64_t>& ci,
                       std::vector<double>& vals, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    ro.resize(m + 1);
    ci.clear();
    vals.clear();
    ro[0] = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (coin(rng) < density) {
                ci.push_back(j);
                vals.push_back(dist(rng));
            }
        }
        // Ensure at least one entry per row
        if (ro[i] == static_cast<int64_t>(ci.size())) {
            int j = rng() % n;
            ci.push_back(j);
            vals.push_back(dist(rng));
        }
        ro[i + 1] = ci.size();
    }
}

int main(int argc, char** argv) {
    // Parse optional args: bench_yolo [n] [batch] [trials]
    int n = 500;
    int batchSize = 128;
    int num_trials = 5;
    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) batchSize = std::atoi(argv[2]);
    if (argc > 3) num_trials = std::atoi(argv[3]);

    const int num_zero = n / 5;
    const int num_nonneg = n - num_zero;
    const int m = num_zero + num_nonneg;
    const double P_density = std::min(0.1, 50.0 / n);  // ~50 off-diag per row max
    const double A_density = std::min(0.2, 100.0 / n);  // ~100 entries per row max

    std::mt19937 rng(42);

    // Generate problem structure
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_vals_host, A_vals_host;
    generate_random_P(n, P_density, P_ro, P_ci, P_vals_host, rng);
    generate_random_A(m, n, A_density, A_ro, A_ci, A_vals_host, rng);

    int64_t nnzP = P_vals_host.size();
    int64_t nnzA = A_vals_host.size();

    // Generate batched q and b
    std::uniform_real_distribution<double> qdist(-1.0, 1.0);
    std::vector<double> q_host(n * batchSize), b_host(m * batchSize);
    for (auto& v : q_host) v = qdist(rng);
    // b: make feasible — zero-cone rows get 0, nonneg get positive values
    for (int bi = 0; bi < batchSize; bi++) {
        for (int i = 0; i < num_zero; i++) b_host[bi * m + i] = 0.0;
        for (int i = num_zero; i < m; i++) b_host[bi * m + i] = 1.0 + std::abs(qdist(rng));
    }

    // Tile P and A values across batch
    std::vector<double> P_vals_batch(nnzP * batchSize), A_vals_batch(nnzA * batchSize);
    for (int bi = 0; bi < batchSize; bi++) {
        std::copy(P_vals_host.begin(), P_vals_host.end(), P_vals_batch.begin() + bi * nnzP);
        std::copy(A_vals_host.begin(), A_vals_host.end(), A_vals_batch.begin() + bi * nnzA);
    }

    Cones cones{};
    cones.numZeroCones = num_zero;
    cones.numNonnegCones = num_nonneg;

    std::cout << "Problem: n=" << n << " m=" << m << " batch=" << batchSize
              << " nnzP=" << nnzP << " nnzA=" << nnzA << "\n\n";

    // =========================================================================
    // Step 1: Normal solve to find iteration count
    // =========================================================================
    int normal_iters = 0;
    double normal_time = 0;
    {
        Settings settings;
        settings.verbose = false;
        settings.maxIter = 200;

        CompiledSolver solver(n, m, batchSize,
            P_ro.data(), P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
        cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);
        cudaMemcpy(d_P, P_vals_batch.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_vals_batch.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_host.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_host.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        // Warmup
        solver.solveAll(d_P, d_A, d_q, d_b);
        normal_iters = solver.info.iterations;

        // Check status
        std::vector<int32_t> status(batchSize);
        cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
        int solved = 0;
        for (auto s : status) if (s == 1 || s == 4) solved++;
        std::cout << "Normal solve: " << solved << "/" << batchSize << " solved in "
                  << normal_iters << " iterations\n";

        // Benchmark
        double total = 0;
        for (int t = 0; t < num_trials; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        normal_time = total / num_trials;
        std::cout << "Normal time:  " << std::fixed << std::setprecision(2)
                  << normal_time << " ms (avg of " << num_trials << " trials)\n\n";

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    // =========================================================================
    // Step 2: YOLO solve with same iteration count
    // =========================================================================
    {
        Settings settings;
        settings.verbose = false;
        settings.yolo = true;
        settings.yoloNumIters = normal_iters;

        CompiledSolver solver(n, m, batchSize,
            P_ro.data(), P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
        cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);
        cudaMemcpy(d_P, P_vals_batch.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_vals_batch.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_host.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_host.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        // Warmup
        solver.solveAll(d_P, d_A, d_q, d_b);

        // Check status and solution quality
        std::vector<int32_t> status(batchSize);
        cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
        int max_iters = 0;
        for (auto s : status) if (s == 7) max_iters++;
        std::cout << "YOLO solve (" << normal_iters << " iters): "
                  << max_iters << "/" << batchSize << " MaxIterations status\n";

        // Check for NaN per batch
        std::vector<double> x_sol(n * batchSize);
        cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
        int nan_count = 0;
        for (int bi = 0; bi < batchSize; bi++) {
            for (int j = 0; j < n; j++) {
                if (std::isnan(x_sol[bi * n + j])) { nan_count++; break; }
            }
        }
        std::cout << "YOLO NaN batches: " << nan_count << "\n";

        // Benchmark
        double total = 0;
        for (int t = 0; t < num_trials; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double yolo_time = total / num_trials;
        std::cout << "YOLO time:    " << std::fixed << std::setprecision(2)
                  << yolo_time << " ms (avg of " << num_trials << " trials)\n";
        std::cout << "Speedup:      " << std::setprecision(2)
                  << normal_time / yolo_time << "x\n\n";

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    return 0;
}
