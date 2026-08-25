// moreau/vector/vector.hpp
#pragma once

#include "moreau/cuda/utils.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "moreau/vector/vector_kernels.cuh"

namespace moreau {

struct BatchedVector {
    // construct & allocate (always owning)
    BatchedVector(int n_, int batchSize_)
        : n_(n_), batchSize_(batchSize_), data_(nullptr), external_data_(nullptr)
    {
        if (n_ < 0 || batchSize_ <= 0)
            throw std::invalid_argument("BatchedVector: invalid dimensions");

        // Check for integer overflow before computing bytes
        // Max safe product is SIZE_MAX / sizeof(double)
        const size_t max_elements = SIZE_MAX / sizeof(double);
        const size_t n_sz = static_cast<size_t>(n_);
        const size_t batch_sz = static_cast<size_t>(batchSize_);

        // Check n * batchSize doesn't overflow
        if (n_sz > 0 && batch_sz > max_elements / n_sz) {
            throw std::overflow_error("BatchedVector: allocation size overflow (n * batchSize too large)");
        }

        const size_t bytes = sizeof(double) * n_sz * batch_sz;
        if (bytes > 0) {
            void* p = nullptr;
            auto e = cudaMalloc(&p, bytes);
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));

            // CRITICAL: Zero the memory immediately to prevent state corruption
            // cudaMalloc can return previously-used memory with stale data
            e = cudaMemset(p, 0, bytes);
            if (e != cudaSuccess) {
                // Clean up the allocated memory before throwing
                cudaFree(p);
                throw std::runtime_error(std::string("cudaMemset failed: ") + cudaGetErrorString(e));
            }

            data_.reset(static_cast<double*>(p));
        }
    }

    /**
     * @brief Non-owning view constructor - wraps external memory without taking ownership
     *
     * IMPORTANT: This is a non-owning view. The caller MUST ensure:
     * 1. The external memory outlives this BatchedVector
     * 2. The external memory is valid device memory
     * 3. The external memory has at least n * batchSize doubles allocated
     *
     * Use isView() to check if a BatchedVector is a view.
     * Misuse (use-after-free) will cause undefined behavior / GPU crashes.
     *
     * @param external_ptr Pointer to existing device memory (must not be null for non-empty vectors)
     * @param n_ Number of elements per batch
     * @param batchSize_ Number of batches
     * @throws std::invalid_argument if dimensions are invalid or external_ptr is null for non-empty vector
     */
    BatchedVector(double* external_ptr, int n_, int batchSize_)
        : n_(n_), batchSize_(batchSize_), data_(nullptr), external_data_(external_ptr)
    {
        if (n_ < 0 || batchSize_ <= 0)
            throw std::invalid_argument("BatchedVector: invalid dimensions");

        // Validate that external pointer is non-null for non-empty vectors
        if (n_ > 0 && external_ptr == nullptr) {
            throw std::invalid_argument("BatchedVector view: external_ptr cannot be null for non-empty vector");
        }
    }

    // no copy
    BatchedVector(const BatchedVector&) = delete;
    BatchedVector& operator=(const BatchedVector&) = delete;

    // move ok
    BatchedVector(BatchedVector&&) noexcept = default;
    BatchedVector& operator=(BatchedVector&&) noexcept = default;

    ~BatchedVector() = default;

    // raw device pointer (for kernels)
    // Returns external_data_ if this is a non-owning view, otherwise owned data_
    [[nodiscard]] double* data() noexcept {
        return external_data_ ? external_data_ : data_.get();
    }
    [[nodiscard]] const double* data() const noexcept {
        return external_data_ ? external_data_ : data_.get();
    }

    // Check if this is a non-owning view
    [[nodiscard]] bool isView() const noexcept { return external_data_ != nullptr; }

    // synchronous/async copies (throw on CUDA error)
    void cpuToGpu(const double* h_data, cudaStream_t stream = 0) {
        const size_t bytes = sizeof(double) * n_ * batchSize_;
        if (!bytes) return;
        CUDA_THROW(cudaMemcpyAsync(data(), h_data, bytes, cudaMemcpyHostToDevice, stream));
    }

    void gpuToCpu(double* h_data, cudaStream_t stream = 0) const {
        const size_t bytes = sizeof(double) * n_ * batchSize_;
        if (!bytes) return;
        CUDA_THROW(cudaMemcpyAsync(h_data, data(), bytes, cudaMemcpyDeviceToHost, stream));
    }

    // set all entries to a constant (batched, i.e., fills batchSize*n elements)
    void setToConstant(double constant, cudaStream_t stream = 0) {
        setConstantDevice(data(), constant, n_ * batchSize_, stream);
    }

    [[nodiscard]] size_t memoryUsage() const noexcept {
        return sizeof(double) * n_ * batchSize_;
    }

    [[nodiscard]] int64_t n() const noexcept { return n_; }
    [[nodiscard]] int64_t batchSize() const noexcept { return batchSize_; }

private:
    int64_t n_;
    int64_t batchSize_;
    device_unique_ptr<double> data_;       // Owned memory (null for views)
    double* external_data_ = nullptr;      // Non-owning view pointer (null for owned)
};

} // namespace moreau
