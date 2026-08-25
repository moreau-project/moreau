/**
 * @file data.hpp
 * @brief Solver data structure for conic optimization problems
 *
 * This module defines SolverData which contains problem dimensions,
 * matrices, vectors, and cone structure for the optimization problem.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include "moreau/vector/vector.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/equilibration/equilibration.hpp"

namespace moreau {

/**
 * @brief Problem data and dimensions for conic optimization
 *
 * Contains the problem dimensions, matrices, vectors, and cone structure
 * for the optimization problem.
 */
struct SolverData {
    // Problem dimensions
    int64_t n;           // Number of primal variables
    int64_t m;           // Number of constraints
    int64_t batchSize;   // Number of problems to solve in parallel

    // Problem data (on device) - equilibrated values
    CSR P;               // Cost matrix (for quadratic objective: x'Px/2 + q'x)
    CSR A;               // Constraint matrix
    BatchedVector q;     // Linear cost vector (equilibrated)
    BatchedVector b;     // Constraint right-hand side (equilibrated)

    // Cone structure
    Cones cones;

    // Equilibration data
    EquilibrationData equilibration;

    // Cached norms for residual computation (computed once at setup)
    BatchedVector normb;  // ||b||_inf with equilibration scaling
    BatchedVector normq;  // ||q||_inf with equilibration scaling

    int64_t* matrixPRowOf;
    int64_t* matrixARowOf;

    // CSC representation of A (= A^T in CSR format) for atomic-free transpose SpMV
    int64_t* d_At_rowOffsets;  // [n+1] — CSC column offsets = A^T row offsets
    int64_t* d_At_colIndices;  // [nnzA] — CSC row indices = A^T column indices
    int64_t* d_At_val_perm;   // [nnzA] — maps A^T positions to A.values positions
    int64_t nnzA_;             // cached for cleanup

    /**
     * @brief Construct solver data for given problem structure
     */
    SolverData(
        int64_t n_, int64_t m_, int64_t batchSize_,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const Cones& cones_
    )
        : n(n_), m(m_), batchSize(batchSize_),
          P(n, n, nnzP, batchSize_),
          A(m, n, nnzA, batchSize_),
          q(n, batchSize_),
          b(m, batchSize_),
          cones(cones_),
          equilibration(n_, m_, batchSize_),
          normb(1, batchSize_),
          normq(1, batchSize_),
          d_At_rowOffsets(nullptr),
          d_At_colIndices(nullptr),
          d_At_val_perm(nullptr),
          nnzA_(nnzA)
    {
        // Initialize cone working data structures
        cones.initialize(batchSize_);

        cudaStream_t stream = 0;  // Use default stream
        std::vector<int64_t> h_matrixPRowOf = moreau::computeRowOfFromCSR(P_ro, n, nnzP);
        std::vector<int64_t> h_matrixARowOf = moreau::computeRowOfFromCSR(A_ro, m, nnzA);
        matrixPRowOf = moreau::allocateAndCopyRowOfToGPU(h_matrixPRowOf, stream);
        matrixARowOf = moreau::allocateAndCopyRowOfToGPU(h_matrixARowOf, stream);

        // Build CSC structure for A (= A^T in CSR) for atomic-free transpose SpMV
        if (nnzA > 0 && n > 0 && m > 0) {
            // Count entries per column of A
            std::vector<int64_t> col_count(n, 0);
            for (int64_t i = 0; i < nnzA; ++i) {
                col_count[A_ci[i]]++;
            }

            // Build At_rowOffsets (cumulative sum)
            std::vector<int64_t> h_At_rowOffsets(n + 1, 0);
            for (int64_t j = 0; j < n; ++j) {
                h_At_rowOffsets[j + 1] = h_At_rowOffsets[j] + col_count[j];
            }

            // Build At_colIndices and At_val_perm
            std::vector<int64_t> h_At_colIndices(nnzA);
            std::vector<int64_t> h_At_val_perm(nnzA);
            std::vector<int64_t> write_pos(n, 0);  // current write position per column

            for (int64_t row = 0; row < m; ++row) {
                for (int64_t idx = A_ro[row]; idx < A_ro[row + 1]; ++idx) {
                    int64_t col = A_ci[idx];
                    int64_t dest = h_At_rowOffsets[col] + write_pos[col];
                    h_At_colIndices[dest] = row;    // row of A = column of A^T
                    h_At_val_perm[dest] = idx;      // position in A.values
                    write_pos[col]++;
                }
            }

            // Upload to device
            cudaMalloc(&d_At_rowOffsets, sizeof(int64_t) * (n + 1));
            cudaMalloc(&d_At_colIndices, sizeof(int64_t) * nnzA);
            cudaMalloc(&d_At_val_perm, sizeof(int64_t) * nnzA);
            cudaMemcpyAsync(d_At_rowOffsets, h_At_rowOffsets.data(),
                            sizeof(int64_t) * (n + 1), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_At_colIndices, h_At_colIndices.data(),
                            sizeof(int64_t) * nnzA, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_At_val_perm, h_At_val_perm.data(),
                            sizeof(int64_t) * nnzA, cudaMemcpyHostToDevice, stream);
        }

        cudaStreamSynchronize(stream);

        // Copy matrix structure to device
        P.indicesCpuToGpu(P_ro, P_ci);
        A.indicesCpuToGpu(A_ro, A_ci);

        // Create cuSPARSE descriptors for fast SpMV operations
        // P is symmetric and stored as full matrix (both upper and lower triangle)
        P.createCusparseDescriptor();
        A.createCusparseDescriptor();
    }

    // No copying (large device buffers)
    SolverData(const SolverData&) = delete;
    SolverData& operator=(const SolverData&) = delete;

    SolverData(SolverData&& o) noexcept
        : n(o.n), m(o.m), batchSize(o.batchSize),
          P(std::move(o.P)), A(std::move(o.A)),
          q(std::move(o.q)), b(std::move(o.b)),
          cones(std::move(o.cones)),
          equilibration(std::move(o.equilibration)),
          normb(std::move(o.normb)), normq(std::move(o.normq)),
          matrixPRowOf(o.matrixPRowOf), matrixARowOf(o.matrixARowOf),
          d_At_rowOffsets(o.d_At_rowOffsets),
          d_At_colIndices(o.d_At_colIndices),
          d_At_val_perm(o.d_At_val_perm),
          nnzA_(o.nnzA_)
    {
        o.matrixPRowOf = nullptr;
        o.matrixARowOf = nullptr;
        o.d_At_rowOffsets = nullptr;
        o.d_At_colIndices = nullptr;
        o.d_At_val_perm = nullptr;
    }

    SolverData& operator=(SolverData&& o) noexcept {
        if (this != &o) {
            // Free our existing device allocations
            if (d_At_rowOffsets) cudaFree(d_At_rowOffsets);
            if (d_At_colIndices) cudaFree(d_At_colIndices);
            if (d_At_val_perm) cudaFree(d_At_val_perm);
            if (matrixPRowOf) cudaFree(matrixPRowOf);
            if (matrixARowOf) cudaFree(matrixARowOf);

            // Move scalars
            n = o.n; m = o.m; batchSize = o.batchSize;
            nnzA_ = o.nnzA_;

            // Move RAII members
            P = std::move(o.P);
            A = std::move(o.A);
            q = std::move(o.q);
            b = std::move(o.b);
            cones = std::move(o.cones);
            equilibration = std::move(o.equilibration);
            normb = std::move(o.normb);
            normq = std::move(o.normq);

            // Transfer raw pointer ownership
            matrixPRowOf = o.matrixPRowOf;
            matrixARowOf = o.matrixARowOf;
            d_At_rowOffsets = o.d_At_rowOffsets;
            d_At_colIndices = o.d_At_colIndices;
            d_At_val_perm = o.d_At_val_perm;

            o.matrixPRowOf = nullptr;
            o.matrixARowOf = nullptr;
            o.d_At_rowOffsets = nullptr;
            o.d_At_colIndices = nullptr;
            o.d_At_val_perm = nullptr;
        }
        return *this;
    }

    ~SolverData() {
        if (d_At_rowOffsets) cudaFree(d_At_rowOffsets);
        if (d_At_colIndices) cudaFree(d_At_colIndices);
        if (d_At_val_perm) cudaFree(d_At_val_perm);
        // matrixPRowOf and matrixARowOf are also device allocations
        if (matrixPRowOf) cudaFree(matrixPRowOf);
        if (matrixARowOf) cudaFree(matrixARowOf);
    }

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept {
        return P.memoryUsage() + A.memoryUsage() +
               q.memoryUsage() + b.memoryUsage() +
               equilibration.memoryUsage();
    }
};

} // namespace moreau
