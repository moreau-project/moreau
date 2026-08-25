// clarabel_test_helpers.hpp
// Helper functions for porting Clarabel.rs tests to Moreau
// Handles CSC (Clarabel) to CSR (Moreau) conversion and common test patterns

#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace clarabel_test {

using namespace moreau;

// =============================================================================
// CSC to CSR Conversion
// =============================================================================

/**
 * @brief Convert Clarabel's CSC format to Moreau's CSR format
 *
 * Clarabel stores matrices in CSC (Compressed Sparse Column):
 *   colptr[j] .. colptr[j+1] gives the range of entries in column j
 *   rowval[k] gives the row index of the k-th entry
 *   nzval[k] gives the value of the k-th entry
 *
 * Moreau stores matrices in CSR (Compressed Sparse Row):
 *   row_offsets[i] .. row_offsets[i+1] gives the range of entries in row i
 *   col_indices[k] gives the column index of the k-th entry
 *   values[k] gives the value of the k-th entry
 */
inline void cscToCSR(
    int64_t m,  // number of rows
    int64_t n,  // number of columns
    const std::vector<int64_t>& csc_colptr,
    const std::vector<int64_t>& csc_rowval,
    const std::vector<double>& csc_nzval,
    std::vector<int64_t>& csr_row_offsets,
    std::vector<int64_t>& csr_col_indices,
    std::vector<double>& csr_values
) {
    // Count entries per row
    std::vector<int64_t> row_counts(m, 0);
    for (auto row : csc_rowval) {
        row_counts[row]++;
    }

    // Compute row offsets
    csr_row_offsets.resize(m + 1);
    csr_row_offsets[0] = 0;
    for (int64_t i = 0; i < m; i++) {
        csr_row_offsets[i + 1] = csr_row_offsets[i] + row_counts[i];
    }

    // Allocate output arrays
    int64_t nnz = csc_nzval.size();
    csr_col_indices.resize(nnz);
    csr_values.resize(nnz);

    // Reset row counts to use as insertion pointers
    std::fill(row_counts.begin(), row_counts.end(), 0);

    // Fill CSR arrays
    for (int64_t col = 0; col < n; col++) {
        for (int64_t idx = csc_colptr[col]; idx < csc_colptr[col + 1]; idx++) {
            int64_t row = csc_rowval[idx];
            int64_t dest = csr_row_offsets[row] + row_counts[row];
            csr_col_indices[dest] = col;
            csr_values[dest] = csc_nzval[idx];
            row_counts[row]++;
        }
    }

    // Sort each row by column index (CSR should be sorted within rows)
    for (int64_t row = 0; row < m; row++) {
        int64_t start = csr_row_offsets[row];
        int64_t end = csr_row_offsets[row + 1];

        // Create index array for sorting
        std::vector<int64_t> indices(end - start);
        std::iota(indices.begin(), indices.end(), 0);

        // Sort by column index
        std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
            return csr_col_indices[start + a] < csr_col_indices[start + b];
        });

        // Apply permutation
        std::vector<int64_t> sorted_cols(end - start);
        std::vector<double> sorted_vals(end - start);
        for (int64_t i = 0; i < end - start; i++) {
            sorted_cols[i] = csr_col_indices[start + indices[i]];
            sorted_vals[i] = csr_values[start + indices[i]];
        }
        std::copy(sorted_cols.begin(), sorted_cols.end(), csr_col_indices.begin() + start);
        std::copy(sorted_vals.begin(), sorted_vals.end(), csr_values.begin() + start);
    }
}

// =============================================================================
// Matrix Construction Helpers (matching Clarabel's patterns)
// =============================================================================

/**
 * @brief Create a zero matrix in CSR format
 */
inline void createZeroMatrix(
    int64_t m, int64_t n,
    std::vector<int64_t>& row_offsets,
    std::vector<int64_t>& col_indices,
    std::vector<double>& values
) {
    row_offsets.resize(m + 1, 0);
    col_indices.clear();
    values.clear();
}

/**
 * @brief Create an identity matrix in CSR format (upper triangle only for symmetric)
 */
inline void createIdentityCSR(
    int64_t n,
    std::vector<int64_t>& row_offsets,
    std::vector<int64_t>& col_indices,
    std::vector<double>& values
) {
    row_offsets.resize(n + 1);
    col_indices.resize(n);
    values.resize(n);

    for (int64_t i = 0; i < n; i++) {
        row_offsets[i] = i;
        col_indices[i] = i;
        values[i] = 1.0;
    }
    row_offsets[n] = n;
}

/**
 * @brief Create a scaled identity matrix: alpha * I
 */
inline void createScaledIdentityCSR(
    int64_t n,
    double alpha,
    std::vector<int64_t>& row_offsets,
    std::vector<int64_t>& col_indices,
    std::vector<double>& values
) {
    createIdentityCSR(n, row_offsets, col_indices, values);
    for (auto& v : values) {
        v = alpha;
    }
}

/**
 * @brief Vertical concatenation of two CSR matrices
 * Result: [A; B]
 */
inline void vcatCSR(
    const std::vector<int64_t>& A_ro, const std::vector<int64_t>& A_ci, const std::vector<double>& A_val,
    const std::vector<int64_t>& B_ro, const std::vector<int64_t>& B_ci, const std::vector<double>& B_val,
    std::vector<int64_t>& C_ro, std::vector<int64_t>& C_ci, std::vector<double>& C_val
) {
    int64_t m_A = A_ro.size() - 1;
    int64_t m_B = B_ro.size() - 1;
    int64_t nnz_A = A_val.size();
    int64_t nnz_B = B_val.size();

    C_ro.resize(m_A + m_B + 1);
    C_ci.resize(nnz_A + nnz_B);
    C_val.resize(nnz_A + nnz_B);

    // Copy A
    std::copy(A_ro.begin(), A_ro.end() - 1, C_ro.begin());
    std::copy(A_ci.begin(), A_ci.end(), C_ci.begin());
    std::copy(A_val.begin(), A_val.end(), C_val.begin());

    // Copy B with offset
    for (int64_t i = 0; i <= m_B; i++) {
        C_ro[m_A + i] = nnz_A + B_ro[i];
    }
    std::copy(B_ci.begin(), B_ci.end(), C_ci.begin() + nnz_A);
    std::copy(B_val.begin(), B_val.end(), C_val.begin() + nnz_A);
}

/**
 * @brief Negate all values in a CSR matrix (in-place)
 */
inline void negateCSR(std::vector<double>& values) {
    for (auto& v : values) {
        v = -v;
    }
}

/**
 * @brief Scale all values in a CSR matrix (in-place)
 */
inline void scaleCSR(std::vector<double>& values, double alpha) {
    for (auto& v : values) {
        v *= alpha;
    }
}

// =============================================================================
// Test Helper Class
// =============================================================================

/**
 * @brief Helper class for running Moreau solver with Clarabel-style test data
 */
class ClarabelTestRunner {
public:
    // Problem dimensions
    int64_t n = 0;
    int64_t m = 0;
    int64_t batchSize = 1;

    // Problem data in CSR format
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_val;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_val;
    std::vector<double> q, b;

    // Cone specification
    Cones cones;

    // Solver settings
    Settings settings;

    // Solution storage
    std::vector<double> x_sol, s_sol, z_sol;
    std::vector<int32_t> status_sol;
    std::vector<double> obj_primal, obj_dual;

    ClarabelTestRunner() {
        settings.verbose = false;
        settings.maxIter = 200;
        settings.ipm.equilibrationSettings.enable = true;
    }

    /**
     * @brief Set problem data from Clarabel CSC format
     * Automatically converts to CSR format used by Moreau
     */
    void setProblemCSC(
        int64_t n_, int64_t m_,
        const std::vector<int64_t>& P_colptr,
        const std::vector<int64_t>& P_rowval,
        const std::vector<double>& P_nzval,
        const std::vector<int64_t>& A_colptr,
        const std::vector<int64_t>& A_rowval,
        const std::vector<double>& A_nzval,
        const std::vector<double>& q_,
        const std::vector<double>& b_
    ) {
        n = n_;
        m = m_;

        // Convert P from CSC to CSR (P is n x n)
        cscToCSR(n, n, P_colptr, P_rowval, P_nzval, P_ro, P_ci, P_val);

        // Convert A from CSC to CSR (A is m x n)
        cscToCSR(m, n, A_colptr, A_rowval, A_nzval, A_ro, A_ci, A_val);

        q = q_;
        b = b_;
    }

    /**
     * @brief Set problem data directly in CSR format
     */
    void setProblemCSR(
        int64_t n_, int64_t m_,
        std::vector<int64_t> P_ro_, std::vector<int64_t> P_ci_, std::vector<double> P_val_,
        std::vector<int64_t> A_ro_, std::vector<int64_t> A_ci_, std::vector<double> A_val_,
        std::vector<double> q_, std::vector<double> b_
    ) {
        n = n_;
        m = m_;
        P_ro = std::move(P_ro_);
        P_ci = std::move(P_ci_);
        P_val = std::move(P_val_);
        A_ro = std::move(A_ro_);
        A_ci = std::move(A_ci_);
        A_val = std::move(A_val_);
        q = std::move(q_);
        b = std::move(b_);
    }

    /**
     * @brief Run the solver
     */
    void solve() {
        int64_t nnzP = P_val.size();
        int64_t nnzA = A_val.size();

        CompiledSolver solver(
            n, m, batchSize,
            P_ro.data(), P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones,
            settings
        );

        // Allocate device memory
        double *d_P_val, *d_A_val, *d_q, *d_b;
        cudaMalloc(&d_P_val, sizeof(double) * std::max<int64_t>(1, nnzP * batchSize));
        cudaMalloc(&d_A_val, sizeof(double) * std::max<int64_t>(1, nnzA * batchSize));
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);

        // Copy to device
        if (nnzP > 0) {
            cudaMemcpy(d_P_val, P_val.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        }
        if (nnzA > 0) {
            cudaMemcpy(d_A_val, A_val.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        // Solve (convenience method - infinite bounds)
        solver.solveAll(d_P_val, d_A_val, d_q, d_b);

        // Copy solution to host
        x_sol.resize(n * batchSize);
        s_sol.resize(m * batchSize);
        z_sol.resize(m * batchSize);
        status_sol.resize(batchSize);
        obj_primal.resize(batchSize);
        obj_dual.resize(batchSize);

        cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

        solver.info.cost_primal.gpuToCpu(obj_primal.data());
        solver.info.cost_dual.gpuToCpu(obj_dual.data());

        cudaDeviceSynchronize();

        // Cleanup
        cudaFree(d_P_val);
        cudaFree(d_A_val);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    // Status check helpers
    bool isSolved(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::Solved) ||
               status_sol[batch] == static_cast<int32_t>(SolverStatus::AlmostSolved);
    }

    bool isPrimalInfeasible(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::PrimalInfeasible) ||
               status_sol[batch] == static_cast<int32_t>(SolverStatus::AlmostPrimalInfeasible);
    }

    bool isDualInfeasible(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::DualInfeasible) ||
               status_sol[batch] == static_cast<int32_t>(SolverStatus::AlmostDualInfeasible);
    }

    bool isMaxIterations(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::MaxIterations);
    }

    int32_t getStatus(int batch = 0) const {
        return status_sol[batch];
    }
};

// =============================================================================
// Vector Utilities
// =============================================================================

/**
 * @brief Compute L2 distance between two vectors
 */
inline double vectorDist(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Print a vector
 */
inline void printVector(const std::string& name, const std::vector<double>& v) {
    std::cout << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << std::setprecision(8) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

} // namespace clarabel_test
