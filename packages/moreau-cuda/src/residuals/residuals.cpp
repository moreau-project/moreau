/**
 * @file residuals.cu
 * @brief Implementation of residuals update function
 */

#include "moreau/residuals/residuals.hpp"
#include "moreau/residuals/residuals_kernels.cuh"
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <vector>

namespace moreau {

Residuals::Residuals(int64_t n, int64_t m, int64_t batchSize)
    : rx(n, batchSize),        // rx has size n (primal residual, same as x)
      rz(m, batchSize),        // rz has size m (dual residual, same as z)
      rτ(1, batchSize),
      rx_inf(n, batchSize),    // rx_inf has size n
      rz_inf(m, batchSize),    // rz_inf has size m
      dot_qx(1, batchSize),
      dot_bz(1, batchSize),
      dot_sz(1, batchSize),
      dot_xPx(1, batchSize),
      Px(n, batchSize),
      n_(n), m_(m), batchSize_(batchSize)
{
    // All computation now uses custom batched kernels, no cuSPARSE descriptors needed
}

void Residuals::update(const Variables& variables, const SolverData& data,
                       cusparseHandle_t /* cusparse_handle */, cublasHandle_t /* cublas_handle */,
                       cudaStream_t stream) {
    // NOTE: cusparse_handle and cublas_handle are unused - all computation uses
    // custom batched kernels. Keeping parameters for API compatibility.

    int64_t n = data.n;
    int64_t m = data.m;
    int64_t batchSize = data.batchSize;

    double alpha, beta;

    // 1. Compute Px = P * X (batched sparse-dense matrix multiply)
    // P is stored as full symmetric matrix
    // For LP (empty P), Px = 0
    if (data.P.nnz() > 0) {
        alpha = 1.0; beta = 0.0;
        csrSpMVBatched(
            n, n, data.P.nnz(), batchSize,
            data.P.rowOffsets(), data.P.colIndices(), data.P.values(),
            variables.x.data(), Px.data(),
            alpha, beta, stream
        );

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in csrSpMVBatched: ") + cudaGetErrorString(err));
        }
    } else {
        // P is empty (LP case), set Px = 0
        cudaMemsetAsync(Px.data(), 0, sizeof(double) * n * batchSize, stream);
    }

    // 2. Compute rx_inf = -A^T * Z (batched)
    // A is m×n, A^T is n×m
    // Using precomputed CSC structure (A^T in CSR format) for atomic-free transpose SpMV
    if (m == 0 || data.A.nnz() == 0) {
        cudaMemsetAsync(rx_inf.data(), 0, sizeof(double) * n * batchSize, stream);
    } else {
        alpha = -1.0; beta = 0.0;
        csrSpMVTransposeBatched_CSC(
            m, n, data.A.nnz(), batchSize,
            data.d_At_rowOffsets, data.d_At_colIndices,
            data.d_At_val_perm, data.A.values(),
            variables.z.data(), rx_inf.data(),
            alpha, beta, stream
        );
    }

    // 2b. Direct-x cones: fold +Σ_J E_J^T z_x into rx_inf so it carries the
    // full primal-infeasibility certificate residual `−(A^T z − Σ_J E_J^T z_x)`.
    // The downstream `rx = rx_inf − Px − qτ` then automatically picks up the
    // direct-x contribution; without this, the certificate test
    // `‖rx_inf‖ < tol·|b^T z|` never fires on direct-x problems.
    // No-op when totalXConeNumel == 0.
    const int64_t total_xcone_numel = variables.totalXConeNumel();
    if (total_xcone_numel > 0 && data.cones.d_xcone_indices != nullptr) {
        scatter_add_z_x_to_rx_inf(
            rx_inf.data(), variables.z_x.data(), data.cones.d_xcone_indices,
            batchSize, n, total_xcone_numel, stream);
    }

    // 3. Compute rz_inf = A * X + S (batched, fused: eliminates D2D memcpy)
    if (m > 0 && data.A.nnz() > 0) {
        alpha = 1.0;
        csrSpMVBatchedWithInit(
            m, n, data.A.nnz(), batchSize,
            data.A.rowOffsets(), data.A.colIndices(), data.A.values(),
            variables.x.data(), rz_inf.data(), variables.s.data(),
            alpha, stream
        );
    } else if (m > 0) {
        // No A entries: rz_inf = S
        cudaMemcpyAsync(rz_inf.data(), variables.s.data(),
                       sizeof(double) * m * batchSize, cudaMemcpyDeviceToDevice, stream);
    }

    // 4. Fused: compute 4 dot products + complete residuals in one kernel
    // dot_qx = q'x, dot_bz = b'z, dot_sz = s'z, dot_xPx = x'Px
    // rx = rx_inf - Px - q*τ   (rx_inf already includes direct-x scatter)
    // rz = rz_inf - b*τ
    // rτ = dot_qx + dot_bz + κ + dot_xPx/τ
    fusedCompleteResidualsAndDots(
        n, m, batchSize,
        data.q.data(), variables.x.data(),
        data.b.data(), variables.z.data(),
        variables.s.data(),
        Px.data(),
        rx_inf.data(), rx.data(),
        rz_inf.data(), rz.data(),
        variables.τ.data(), variables.κ.data(),
        rτ.data(),
        dot_qx.data(), dot_bz.data(),
        dot_sz.data(), dot_xPx.data(),
        stream
    );

    // Direct-x cones: extend dot_sz with x[J]'z_x so calc_mu picks up the
    // x-cone complementarity term. No-op when totalXConeNumel == 0.
    if (total_xcone_numel > 0 && data.cones.d_xcone_indices != nullptr) {
        accumulate_dot_xJ_zx(
            dot_sz.data(), variables.x.data(), variables.z_x.data(),
            data.cones.d_xcone_indices,
            batchSize, n, total_xcone_numel, stream);
    }

    // No sync needed - all residuals stay on GPU for subsequent operations
    // Buffer is kept allocated for reuse in next iteration
}

} // namespace moreau
