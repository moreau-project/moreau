// matrix/csr.hpp
#pragma once

#include "moreau/cuda/utils.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <vector>

namespace moreau {

struct CSR {
public:
    CSR(int n_, int m_, int nnz_, int batchSize_)
    : n_(n_), m_(m_), nnz_(nnz_), batchSize_(batchSize_) 
    {
        if (m_ < 0 || n_ < 0 || nnz_ < 0 || batchSize_ <= 0)
            throw std::invalid_argument("CSR: invalid dimensions");

        const size_t roBytes = sizeof(int64_t) * (n_ + 1);  // n_ is number of rows
        const size_t ciBytes = sizeof(int64_t) * nnz_;
        const size_t valBytes = sizeof(double)  * batchSize_ * nnz_;

        auto dmalloc = [](size_t bytes) -> void* {
            if (!bytes) return nullptr;
            void* p = nullptr;
            cudaError_t e = cudaMalloc(&p, bytes);
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            // CRITICAL: Zero the memory to prevent state corruption
            // cudaMalloc can return previously-used memory with stale data
            cudaMemset(p, 0, bytes);
            return p;
        };

        rowOffsets_.reset(static_cast<int64_t*>(dmalloc(roBytes)));
        colIndices_.reset(static_cast<int64_t*>(dmalloc(ciBytes)));
        values_.reset(static_cast<double*>(dmalloc(valBytes)));
    }

    CSR() = default;
    CSR(const CSR&) = delete;
    CSR& operator=(const CSR&) = delete;
    CSR(CSR&&) noexcept = default;
    CSR& operator=(CSR&&) noexcept = default;

    ~CSR() {
        if (cusparseDescr_) {
            cusparseDestroySpMat(cusparseDescr_);
        }
    }

    // Device pointer accessors
    [[nodiscard]] int64_t* rowOffsets() noexcept { return rowOffsets_.get(); }
    [[nodiscard]] int64_t* colIndices() noexcept { return colIndices_.get(); }
    [[nodiscard]] double*  values()     noexcept { return values_.get(); }

    [[nodiscard]] const int64_t* rowOffsets() const noexcept { return rowOffsets_.get(); }
    [[nodiscard]] const int64_t* colIndices() const noexcept { return colIndices_.get(); }
    [[nodiscard]] const double*  values()     const noexcept { return values_.get(); }

    // Getters for matrix dimensions and attributes
    [[nodiscard]] int64_t rows() const noexcept { return n_; }
    [[nodiscard]] int64_t cols() const noexcept { return m_; }
    [[nodiscard]] int64_t n() const noexcept { return n_; }  // alias for rows
    [[nodiscard]] int64_t m() const noexcept { return m_; }  // alias for cols
    [[nodiscard]] int64_t nnz() const noexcept { return nnz_; }
    [[nodiscard]] int64_t batchSize() const noexcept { return batchSize_; }

    void indicesCpuToGpu(const int64_t* h_ro, const int64_t* h_ci, cudaStream_t s = 0) {
        cudaMemcpyAsync(rowOffsets_.get(), h_ro, sizeof(int64_t)*(n_+1), cudaMemcpyHostToDevice, s);  // n_ is number of rows
        cudaMemcpyAsync(colIndices_.get(), h_ci, sizeof(int64_t)*nnz_, cudaMemcpyHostToDevice, s);
    }

    void indicesGpuToCpu(int64_t* h_ro, int64_t* h_ci, cudaStream_t s = 0) const {
        cudaMemcpyAsync(h_ro, rowOffsets_.get(), sizeof(int64_t)*(n_+1), cudaMemcpyDeviceToHost, s);  // n_ is number of rows
        cudaMemcpyAsync(h_ci, colIndices_.get(), sizeof(int64_t)*nnz_, cudaMemcpyDeviceToHost, s);
    }

    void valuesCpuToGpu(const double* h_val, cudaStream_t s = 0) {
        cudaMemcpyAsync(values_.get(), h_val, sizeof(double)*batchSize_*nnz_, cudaMemcpyHostToDevice, s);
    }

    void valuesGpuToCpu(double* h_val, cudaStream_t s = 0) const {
        cudaMemcpyAsync(h_val, values_.get(), sizeof(double)*batchSize_*nnz_, cudaMemcpyDeviceToHost, s);
    }

    [[nodiscard]] size_t memoryUsage() const noexcept {
        return sizeof(int64_t) * (n_ + 1) +     // rowOffsets (n_ is number of rows)
               sizeof(int64_t) * nnz_ +         // colIndices
               sizeof(double) * batchSize_ * nnz_;  // values
    }

    /**
     * @brief Create cuSPARSE descriptor for this matrix
     *
     * Should be called after indices are copied to GPU.
     * The descriptor uses the first batch's values pointer initially.
     *
     * @param is_symmetric If true, matrix is symmetric (only upper triangle stored)
     */
    void createCusparseDescriptor(bool is_symmetric = false) {
        if (n_ == 0 || m_ == 0) {
            return;  // Skip descriptor creation for empty matrices
        }
        if (cusparseDescr_) {
            cusparseDestroySpMat(cusparseDescr_);
        }

        cusparseCreateCsr(&cusparseDescr_, n_, m_, nnz_,
                         rowOffsets_.get(),
                         colIndices_.get(),
                         values_.get(),  // First batch
                         CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        // Note: is_symmetric parameter is currently unused because cuSPARSE SpMM
        // doesn't support FILL_MODE attributes. Kept for future use.
        (void)is_symmetric;
    }

    /**
     * @brief Get cuSPARSE descriptor
     */
    cusparseSpMatDescr_t cusparseDescriptor() const noexcept {
        return cusparseDescr_;
    }

    /**
     * @brief Update cuSPARSE descriptor to point to a specific batch's values
     */
    void setCusparseBatch(int64_t batch) {
        if (cusparseDescr_) {
            double* batch_values = values_.get() + batch * nnz_;
            cusparseCsrSetPointers(cusparseDescr_, rowOffsets_.get(),
                                  colIndices_.get(), batch_values);
        }
    }

private:
    int64_t n_;
    int64_t m_;
    int64_t nnz_;
    int64_t batchSize_;
    device_unique_ptr<int64_t> rowOffsets_{nullptr};
    device_unique_ptr<int64_t> colIndices_{nullptr};
    device_unique_ptr<double>  values_{nullptr};
    cusparseSpMatDescr_t cusparseDescr_{nullptr};
};

// Fills row_of[0..nnz-1] with the row index for each nonzero.
// Preconditions:
//   - rowOffsets has length m+1
//   - rowOffsets is non-decreasing
//   - rowOffsets[m] == nnz
inline void compute_row_indices_from_csr_host(
    int64_t m,
    const int64_t* rowOffsets,   // length m+1
    int64_t nnz,
    int64_t* row_of              // length nnz (output)
) {
    assert(m >= 0 && nnz >= 0);
    assert(rowOffsets != nullptr);
    assert(row_of != nullptr);

    for (int64_t i = 0; i < m; ++i) {
        const int64_t start = rowOffsets[i];
        const int64_t end   = rowOffsets[i + 1];
        // Optional sanity checks (debug only)
        assert(0 <= start && start <= end && end <= nnz);
        for (int64_t p = start; p < end; ++p) {
            row_of[p] = i;
        }
    }
    // Optional final check:
    assert(rowOffsets[m] == nnz);
}

// Convenience overload that returns a std::vector<int64_t>.
inline std::vector<int64_t> compute_row_indices_from_csr_host(
    int64_t m,
    const int64_t* rowOffsets,   // length m+1
    int64_t nnz
) {
    std::vector<int64_t> row_of(static_cast<size_t>(nnz));
    compute_row_indices_from_csr_host(m, rowOffsets, nnz, row_of.data());
    return row_of;
}

} // namespace moreau
