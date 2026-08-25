/**
 * @file test_psd_scaling_flag.cpp
 * @brief Regression for #177: PSD scaling-failure detection must be per-batch.
 *
 * update_psd_scaling reports a non-finite Hs by calling check_finite_kernel,
 * which writes the per-batch scaling-success array. The bug (mirroring SOC #118)
 * was that it wrote only element [0], so a failure in batch k != 0 was attributed
 * to batch 0 — or, with the interim fix, every batch was conservatively poisoned.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <vector>

#include "moreau/cones/psd_kernels.cuh"

using namespace moreau;

// Non-finite entries in batches 1 and 3 must flag exactly those batches.
TEST(PsdScalingFlag, FiniteCheckIsPerBatch) {
    constexpr int64_t batchSize = 4;
    constexpr int64_t stride = 6;  // Hs entries per batch element
    constexpr int64_t n = batchSize * stride;

    std::vector<double> host(n, 1.0);
    host[1 * stride + 2] = std::nan("");
    host[3 * stride + 5] = std::numeric_limits<double>::infinity();

    double* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(double) * n);
    cudaMemcpy(d_data, host.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    int32_t* d_flags = nullptr;
    cudaMalloc(&d_flags, sizeof(int32_t) * batchSize);
    std::vector<int32_t> flags(batchSize, 1);
    cudaMemcpy(d_flags, flags.data(), sizeof(int32_t) * batchSize, cudaMemcpyHostToDevice);

    check_finite_kernel(d_data, n, stride, d_flags);
    cudaDeviceSynchronize();
    cudaMemcpy(flags.data(), d_flags, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    EXPECT_EQ(flags[0], 1);
    EXPECT_EQ(flags[1], 0);
    EXPECT_EQ(flags[2], 1);
    EXPECT_EQ(flags[3], 0);

    cudaFree(d_data);
    cudaFree(d_flags);
}
