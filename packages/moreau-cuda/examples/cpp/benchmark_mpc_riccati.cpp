/**
 * @file benchmark_mpc_riccati.cpp
 * @brief MPC QP benchmark for nsys profiling of Riccati solver path
 *
 * Constructs the same MPC problem as bench_riccati_nsys.py:
 *   nx=12, nu=6, T=50, B=512
 * and runs solve() in a tight loop so nsys can profile kernel-level breakdown.
 *
 * Usage:
 *   nsys profile --stats=true ./bench_mpc_riccati [nx] [nu] [T] [B] [n_solves]
 */
#include "moreau/solver/solver.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace moreau;

// Simple LCG-based RNG matching numpy's default_rng interface (good enough for benchmarks)
struct SimpleRNG {
    std::mt19937_64 gen;
    std::normal_distribution<double> normal{0.0, 1.0};

    SimpleRNG(uint64_t seed) : gen(seed) {}

    double randn() { return normal(gen); }
};

struct MpcQP {
    int n_vars, m_cons;
    int n_eq, n_ineq;
    int n_eq_dyn, n_eq_ic;

    // CSR P
    std::vector<int64_t> P_row_offsets;
    std::vector<int64_t> P_col_indices;
    std::vector<double>  P_values;  // (batch, nnzP)

    // CSR A
    std::vector<int64_t> A_row_offsets;
    std::vector<int64_t> A_col_indices;
    std::vector<double>  A_values;  // (batch, nnzA)

    // RHS
    std::vector<double> q_batch;  // (batch, n)
    std::vector<double> b_batch;  // (batch, m)

    Cones cones;
    int nnzP, nnzA;
};

MpcQP make_mpc_qp(int nx, int nu, int T, int batch_size) {
    MpcQP qp;
    SimpleRNG rng(42);

    // Dynamics: A_dyn = I + 0.01 * randn, B_dyn = 0.1 * randn
    std::vector<double> A_dyn(nx * nx), B_dyn(nx * nu);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < nx; j++)
            A_dyn[i * nx + j] = (i == j ? 1.0 : 0.0) + 0.01 * rng.randn();
    for (int i = 0; i < nx * nu; i++)
        B_dyn[i] = 0.1 * rng.randn();

    int n_vars = T * (nx + nu) + nx;
    qp.n_vars = n_vars;

    // P = diag(Q, R, Q, R, ..., Qf)  — diagonal
    std::vector<double> P_diag(n_vars, 0.0);
    for (int t = 0; t < T; t++) {
        int xs = t * (nx + nu);
        for (int i = 0; i < nx; i++) P_diag[xs + i] = 1.0;
        for (int i = 0; i < nu; i++) P_diag[xs + nx + i] = 0.1;
    }
    for (int i = 0; i < nx; i++) P_diag[T * (nx + nu) + i] = 10.0;

    // P is diagonal — full symmetric means both (i,j) and (j,i), but diagonal only has (i,i)
    qp.P_row_offsets.resize(n_vars + 1);
    qp.P_col_indices.resize(n_vars);
    for (int i = 0; i < n_vars; i++) {
        qp.P_row_offsets[i] = i;
        qp.P_col_indices[i] = i;
    }
    qp.P_row_offsets[n_vars] = n_vars;
    qp.nnzP = n_vars;

    // P values: same across batch
    qp.P_values.resize(batch_size * n_vars);
    for (int b = 0; b < batch_size; b++)
        for (int i = 0; i < n_vars; i++)
            qp.P_values[b * n_vars + i] = P_diag[i];

    // Constraint counts
    int n_eq_dyn = nx * T;
    int n_eq_ic = nx;
    int n_eq = n_eq_dyn + n_eq_ic;
    int n_ineq = 2 * nu * T;
    int m = n_eq + n_ineq;
    qp.m_cons = m;
    qp.n_eq = n_eq;
    qp.n_ineq = n_ineq;
    qp.n_eq_dyn = n_eq_dyn;
    qp.n_eq_ic = n_eq_ic;

    // Build A in COO, then convert to CSR
    struct Entry { int row, col; double val; };
    std::vector<Entry> entries;
    entries.reserve(n_eq_dyn * (nx + nu + nx) + n_eq_ic * 1 + n_ineq);

    int row_idx = 0;

    // Dynamics: A_dyn * x_t + B_dyn * u_t - x_{t+1} = 0
    for (int t = 0; t < T; t++) {
        int x_t = t * (nx + nu);
        int u_t = x_t + nx;
        int x_tp1 = (t < T - 1) ? (t + 1) * (nx + nu) : T * (nx + nu);

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < nx; j++) {
                if (std::abs(A_dyn[i * nx + j]) > 1e-10)
                    entries.push_back({row_idx + i, x_t + j, A_dyn[i * nx + j]});
            }
            for (int j = 0; j < nu; j++) {
                if (std::abs(B_dyn[i * nu + j]) > 1e-10)
                    entries.push_back({row_idx + i, u_t + j, B_dyn[i * nu + j]});
            }
            entries.push_back({row_idx + i, x_tp1 + i, -1.0});
        }
        row_idx += nx;
    }

    // Initial condition: x_0 = x0
    for (int i = 0; i < nx; i++)
        entries.push_back({row_idx + i, i, 1.0});
    row_idx += nx;

    // Control bounds: -u_max <= u <= u_max
    for (int t = 0; t < T; t++) {
        int u_t = t * (nx + nu) + nx;
        for (int j = 0; j < nu; j++) {
            entries.push_back({row_idx, u_t + j, -1.0});
            row_idx++;
        }
        for (int j = 0; j < nu; j++) {
            entries.push_back({row_idx, u_t + j, 1.0});
            row_idx++;
        }
    }

    // Sort by (row, col) for CSR
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.row < b.row || (a.row == b.row && a.col < b.col);
    });

    qp.nnzA = (int)entries.size();
    qp.A_row_offsets.resize(m + 1, 0);
    qp.A_col_indices.resize(qp.nnzA);
    std::vector<double> a_vals(qp.nnzA);

    for (int i = 0; i < qp.nnzA; i++) {
        qp.A_col_indices[i] = entries[i].col;
        a_vals[i] = entries[i].val;
        qp.A_row_offsets[entries[i].row + 1]++;
    }
    for (int i = 1; i <= m; i++)
        qp.A_row_offsets[i] += qp.A_row_offsets[i - 1];

    // A values: same across batch
    qp.A_values.resize(batch_size * qp.nnzA);
    for (int b = 0; b < batch_size; b++)
        for (int i = 0; i < qp.nnzA; i++)
            qp.A_values[b * qp.nnzA + i] = a_vals[i];

    // q, b
    double u_max = 1.0;
    SimpleRNG rng_batch(123);
    qp.q_batch.resize(batch_size * n_vars, 0.0);
    qp.b_batch.resize(batch_size * m, 0.0);
    for (int b = 0; b < batch_size; b++) {
        // Random initial state
        for (int i = 0; i < nx; i++) {
            double x0_i = rng_batch.randn() * 0.5;
            qp.b_batch[b * m + n_eq_dyn + i] = x0_i;
        }
        // u_max bounds
        for (int i = 0; i < n_ineq; i++)
            qp.b_batch[b * m + n_eq + i] = u_max;
    }

    qp.cones.numZeroCones = n_eq;
    qp.cones.numNonnegCones = n_ineq;
    qp.cones.numExpCones = 0;
    qp.cones.numSocCones = 0;
    qp.cones.numPowerCones = 0;

    return qp;
}

int main(int argc, char** argv) {
    int nx = 12, nu = 6, T = 50, batch_size = 512, n_solves = 11;

    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) nu = atoi(argv[2]);
    if (argc > 3) T = atoi(argv[3]);
    if (argc > 4) batch_size = atoi(argv[4]);
    if (argc > 5) n_solves = atoi(argv[5]);

    printf("MPC Riccati Benchmark (for nsys profiling)\n");
    printf("==========================================\n");
    printf("  nx=%d, nu=%d, T=%d, B=%d, solves=%d\n", nx, nu, T, batch_size, n_solves);

    auto qp = make_mpc_qp(nx, nu, T, batch_size);
    int n = qp.n_vars, m_cons = qp.m_cons;
    printf("  n=%d, m=%d, nnzP=%d, nnzA=%d\n", n, m_cons, qp.nnzP, qp.nnzA);

    // Create solver
    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;
    CompiledSolver solver(
        n, m_cons, batch_size,
        qp.P_row_offsets.data(), qp.P_col_indices.data(), qp.nnzP,
        qp.A_row_offsets.data(), qp.A_col_indices.data(), qp.nnzA,
        qp.cones, settings
    );

    // Upload data to GPU
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * batch_size * qp.nnzP);
    cudaMalloc(&d_A, sizeof(double) * batch_size * qp.nnzA);
    cudaMalloc(&d_q, sizeof(double) * batch_size * n);
    cudaMalloc(&d_b, sizeof(double) * batch_size * m_cons);

    cudaMemcpy(d_P, qp.P_values.data(), sizeof(double) * batch_size * qp.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, qp.A_values.data(), sizeof(double) * batch_size * qp.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, qp.q_batch.data(), sizeof(double) * batch_size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, qp.b_batch.data(), sizeof(double) * batch_size * m_cons, cudaMemcpyHostToDevice);

    solver.setup(d_P, d_A);

    // Warmup
    printf("Warmup...\n");
    solver.solve(d_q, d_b);
    cudaDeviceSynchronize();

    printf("  iters=%d\n", solver.info.iterations);

    // Timed solves
    printf("Running %d solves...\n", n_solves);
    std::vector<double> times(n_solves);

    for (int i = 0; i < n_solves; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        solver.solve(d_q, d_b);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Stats
    double sum = 0;
    double min_t = times[0];
    for (auto t : times) {
        sum += t;
        if (t < min_t) min_t = t;
    }
    double avg = sum / n_solves;
    int iters = solver.info.iterations;
    double per_iter_us = min_t * 1000.0 / iters;

    printf("\n=== RESULTS ===\n");
    printf("  %d solves: avg=%.2f ms, min=%.2f ms\n", n_solves, avg, min_t);
    printf("  IPM iters: %d\n", iters);
    printf("  Per iter: %.1f us\n", per_iter_us);
    printf("  Per iter per problem: %.2f us\n", per_iter_us / batch_size);

    // Print individual times
    printf("\n  Individual solve times (ms):");
    for (int i = 0; i < n_solves; i++)
        printf(" %.2f", times[i]);
    printf("\n");

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);

    return 0;
}
