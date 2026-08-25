// equilibration.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "moreau/cones/cones.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/vector/vector_kernels.cuh"

namespace moreau {

/**
 * @brief Settings for matrix equilibration
 */
struct EquilibrationSettings {
    bool enable = true;
    int max_iter = 10;           // Maximum number of equilibration iterations
    double tolerance = 1e-4;     // Convergence tolerance for equilibration
    double scale_min = 1e-4;     // Minimum scaling factor (prevents over-scaling)
    double scale_max = 1e4;      // Maximum scaling factor (prevents over-scaling)
};

// Holds all equilibration scalings and scratch, always owning.
struct EquilibrationData {
    // public for direct field access
    BatchedVector d;        // [batchSize][n]
    BatchedVector dinv;     // [batchSize][n]
    BatchedVector e;        // [batchSize][m]
    BatchedVector einv;     // [batchSize][m]
    BatchedVector c;        // [batchSize][1]
    BatchedVector dwork;    // [batchSize][n]
    BatchedVector ework;    // [batchSize][m]

    // Cached mean of equilibrated P row norms (for cost scaling computation)
    // Used when re-solving with new q/b but same P/A
    BatchedVector mean_P_row_norm;  // [batchSize][1]

    // Construct and allocate everything
    EquilibrationData(int n, int m, int batchSize)
        : d(n, batchSize),
          dinv(n, batchSize),
          e(m, batchSize),
          einv(m, batchSize),
          c(1, batchSize),
          dwork(n, batchSize),
          ework(m, batchSize),
          mean_P_row_norm(1, batchSize)
    {
        reset();  // identity scaling by default
    }

    // No copying (large device buffers)
    EquilibrationData(const EquilibrationData&) = delete;
    EquilibrationData& operator=(const EquilibrationData&) = delete;

    // Moves ok
    EquilibrationData(EquilibrationData&&) noexcept = default;
    EquilibrationData& operator=(EquilibrationData&&) noexcept = default;

    ~EquilibrationData() = default;

    // Reset scaling to identity (d=1, e=1, c=1; work = 1 to match previous behavior)
    // Uses batch kernel to set all vectors in one kernel launch
    void reset(cudaStream_t stream = 0) {
        std::vector<double*> ptrs = {
            d.data(), dinv.data(), e.data(), einv.data(),
            c.data(), dwork.data(), ework.data(), mean_P_row_norm.data()
        };
        std::vector<int64_t> sizes = {
            d.n() * d.batchSize(),
            dinv.n() * dinv.batchSize(),
            e.n() * e.batchSize(),
            einv.n() * einv.batchSize(),
            c.n() * c.batchSize(),
            dwork.n() * dwork.batchSize(),
            ework.n() * ework.batchSize(),
            mean_P_row_norm.n() * mean_P_row_norm.batchSize()
        };
        batchSetConstant(ptrs, sizes, 1.0, stream);
    }

    // Host → device for base vectors (any null pointer is skipped)
    void cpuToGpu(const double* h_d,
                  const double* h_dinv,
                  const double* h_e,
                  const double* h_einv,
                  const double* h_c,
                  const double* h_dwork,
                  const double* h_ework,
                  cudaStream_t stream = 0)
    {
        if (h_d)     d.cpuToGpu(h_d, stream);
        if (h_dinv)  dinv.cpuToGpu(h_dinv, stream);
        if (h_e)     e.cpuToGpu(h_e, stream);
        if (h_einv)  einv.cpuToGpu(h_einv, stream);
        if (h_c)     c.cpuToGpu(h_c, stream);
        if (h_dwork) dwork.cpuToGpu(h_dwork, stream);
        if (h_ework) ework.cpuToGpu(h_ework, stream);
    }

    // Device → host for base vectors (any null pointer is skipped)
    void gpuToCpu(double* h_d,
                  double* h_dinv,
                  double* h_e,
                  double* h_einv,
                  double* h_c,
                  double* h_dwork,
                  double* h_ework,
                  cudaStream_t stream = 0) const
    {
        if (h_d)     d.gpuToCpu(h_d, stream);
        if (h_dinv)  dinv.gpuToCpu(h_dinv, stream);
        if (h_e)     e.gpuToCpu(h_e, stream);
        if (h_einv)  einv.gpuToCpu(h_einv, stream);
        if (h_c)     c.gpuToCpu(h_c, stream);
        if (h_dwork) dwork.gpuToCpu(h_dwork, stream);
        if (h_ework) ework.gpuToCpu(h_ework, stream);
    }

    // Total device bytes
    size_t memoryUsage() const {
        return d.memoryUsage() + dinv.memoryUsage()
             + e.memoryUsage() + einv.memoryUsage()
             + c.memoryUsage() + dwork.memoryUsage() + ework.memoryUsage()
             + mean_P_row_norm.memoryUsage();
    }
};

// Full equilibration function - computes d, e, c and applies to P, A, q, b
// Also caches mean_P_row_norm for subsequent equilibrate_vectors_only calls
void equilibration(
    EquilibrationData& equil,
    CSR& P,
    int64_t* matrixPRowOf,
    CSR& A,
    int64_t* matrixARowOf,
    BatchedVector& q,
    BatchedVector& b,
    const EquilibrationSettings& settings,
    const Cones& cones,
    cudaStream_t stream = 0
);

// Equilibrate only q and b using stored factors (d, e) and cached mean_P_row_norm
// Call this when P and A are already equilibrated from a previous solve
// Computes new c based on cached mean_P_row_norm and inf_norm(d*q), adjusts P scaling if c changes
void equilibrate_vectors_only(
    EquilibrationData& equil,
    CSR& P,
    int64_t* matrixPRowOf,
    BatchedVector& q,
    BatchedVector& b,
    const EquilibrationSettings& settings,
    cudaStream_t stream = 0
);

// Helper function to compute row_of array from CSR rowOffsets on CPU
// For each nonzero index p, row_of[p] gives the row index i such that p is in [rowOffsets[i], rowOffsets[i+1])
std::vector<int64_t> computeRowOfFromCSR(const int64_t* rowOffsets, int64_t n, int64_t nnz);

// Helper function to allocate GPU memory and copy row_of array to device
int64_t* allocateAndCopyRowOfToGPU(const std::vector<int64_t>& h_row_of, cudaStream_t stream = 0);

} // namespace moreau