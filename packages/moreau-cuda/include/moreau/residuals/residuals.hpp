/**
 * @file residuals.hpp
 * @brief Residuals for standard-form conic optimization solver
 *
 * This module defines the residual vectors used in interior-point methods
 * for tracking convergence and detecting infeasibility.
 */

#pragma once

#include "moreau/vector/vector.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include "moreau/variables/variables.hpp"
#include "moreau/solver/data.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

namespace moreau {

/**
 * @brief Standard-form solver residuals structure
 *
 * Contains KKT residuals, partial residuals for infeasibility detection,
 * and various inner products used in convergence checks.
 */
struct Residuals {
    // Main KKT residuals
    BatchedVector rx;      // Primal residual: size n (same as x)
    BatchedVector rz;      // Dual residual: size m (same as z)
    BatchedVector rτ;      // Gap residual (size 1 per batch)

    // Partial residuals for infeasibility checks
    BatchedVector rx_inf;    // Infeasibility certificate residual (primal): size n
    BatchedVector rz_inf;    // Infeasibility certificate residual (dual): size m

    // Various inner products
    // NB: these are invariant w.r.t equilibration
    BatchedVector dot_qx;   // c'x (size 1 per batch)
    BatchedVector dot_bz;   // b'z (size 1 per batch)
    BatchedVector dot_sz;   // s'z (size 1 per batch)
    BatchedVector dot_xPx;  // x'Px (size 1 per batch)

    // The product Px by itself. Required for infeasibility checks
    BatchedVector Px;

    // Problem dimensions
    int64_t n_, m_, batchSize_;

    /**
     * @brief Construct residuals structure for given problem dimensions
     * @param n Number of primal variables
     * @param m Number of constraints
     * @param batchSize Number of problems to solve in parallel
     */
    Residuals(int64_t n, int64_t m, int64_t batchSize);

    // No copy
    Residuals(const Residuals&) = delete;
    Residuals& operator=(const Residuals&) = delete;

    // Move
    Residuals(Residuals&& other) noexcept
        : rx(std::move(other.rx)), rz(std::move(other.rz)),
          rτ(std::move(other.rτ)),
          rx_inf(std::move(other.rx_inf)), rz_inf(std::move(other.rz_inf)),
          dot_qx(std::move(other.dot_qx)), dot_bz(std::move(other.dot_bz)),
          dot_sz(std::move(other.dot_sz)), dot_xPx(std::move(other.dot_xPx)),
          Px(std::move(other.Px)),
          n_(other.n_), m_(other.m_), batchSize_(other.batchSize_)
    {
    }

    Residuals& operator=(Residuals&& other) noexcept {
        if (this != &other) {
            rx = std::move(other.rx); rz = std::move(other.rz);
            rτ = std::move(other.rτ);
            rx_inf = std::move(other.rx_inf); rz_inf = std::move(other.rz_inf);
            dot_qx = std::move(other.dot_qx); dot_bz = std::move(other.dot_bz);
            dot_sz = std::move(other.dot_sz); dot_xPx = std::move(other.dot_xPx);
            Px = std::move(other.Px);
            n_ = other.n_; m_ = other.m_; batchSize_ = other.batchSize_;
        }
        return *this;
    }

    ~Residuals() = default;

    void update(const Variables& variables, const SolverData& data,
                cusparseHandle_t cusparse_handle, cublasHandle_t cublas_handle,
                cudaStream_t stream = 0);

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept {
        return rx.memoryUsage() + rz.memoryUsage() +
               rτ.memoryUsage() +
               rx_inf.memoryUsage() + rz_inf.memoryUsage() +
               dot_qx.memoryUsage() + dot_bz.memoryUsage() +
               dot_sz.memoryUsage() + dot_xPx.memoryUsage() +
               Px.memoryUsage();
    }

};

} // namespace moreau
