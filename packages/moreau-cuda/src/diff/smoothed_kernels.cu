/**
 * @file smoothed_kernels.cu
 * @brief Smoothed cone derivative kernels for smoothed differentiation
 *
 * Computes H = (I + μ·∇²φ*(z))⁻¹ for each cone type:
 * - Zero cones: H = I (handled by updateJ, not here)
 * - Nonneg cones: H[i] = z[i]² / (z[i]² + μ)
 * - SOC cones: Two-step Sherman-Morrison (stable when d0 = 1-μ/ζ ≈ 0)
 */

#include "moreau/diff/smoothed_kernels.cuh"
#include "moreau/cones/common.cuh"
#include "moreau/cones/cones.hpp"
#include "moreau/cuda/utils.cuh"

#include <algorithm>

namespace moreau {

// ============================================================================
// Smoothed Nonneg Derivative Kernel
// H[i] = z[i]² / (z[i]² + μ)
// ============================================================================

__global__ void smoothed_nonneg_derivative_kernel(
    double* __restrict__ H_diag,
    const double* __restrict__ z,
    const double* __restrict__ mu,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double mu_val = mu[batch];

    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t z_idx = batch * m + offset + i;
        int64_t h_idx = batch * numNonnegCones + i;
        double zi = z[z_idx];
        double zi_sq = zi * zi;
        H_diag[h_idx] = zi_sq / (zi_sq + mu_val);
    }
}

// ============================================================================
// Two-step Sherman-Morrison coefficients for SOC smoothed derivative.
//
// M = I + μ∇²φ*(z) = di·I + c·vvᵀ + (d0-di)·e₀e₀ᵀ
// M⁻¹ = (1/di)I - β·vvᵀ - γ·wwᵀ  where w = N⁻¹e₀
//
// Numerically stable even when d0 = 1-μ/ζ ≈ 0 (uses di·I as base).
// ============================================================================

struct SocSmoothedCoeffs {
    double inv_di;
    double beta;
    double beta_z0;
    double gamma;
    double n_inv_00;
};

__device__ __forceinline__ SocSmoothedCoeffs
soc_smoothed_coeffs(double z0, double norm_sq, double zeta, double mu_val) {
    double di = 1.0 + mu_val / zeta;
    double inv_di = 1.0 / di;
    double c = 2.0 * mu_val / (zeta * zeta);
    double v_norm_sq = z0 * z0 + norm_sq;

    double beta = c / (di * (di + c * v_norm_sq));
    double n_inv_00 = inv_di - beta * z0 * z0;
    double beta_z0 = beta * z0;
    double d0_minus_di = -2.0 * mu_val / zeta;
    double gamma = d0_minus_di / (1.0 + d0_minus_di * n_inv_00);

    return {inv_di, beta, beta_z0, gamma, n_inv_00};
}

// ============================================================================
// Smoothed SOC Derivative Kernel (dense storage, dim <= 4)
// ============================================================================

__global__ void smoothed_soc_dense_derivative_kernel(
    double* __restrict__ soc_H,                    // output: upper triangle entries
    const double* __restrict__ z,                  // [batchSize][m]
    const double* __restrict__ mu,                 // [batchSize]
    const int64_t* __restrict__ soc_dims,          // [numSocCones]
    const int64_t* __restrict__ soc_offsets,       // [numSocCones] — offset of each cone in z
    const int64_t* __restrict__ dense_Hs_offsets,  // [numSocCones+1] — offset into soc_H
    int64_t numSocCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double mu_val = mu[batch];

    for (int64_t cone = threadIdx.x; cone < numSocCones; cone += blockDim.x) {
        int64_t dim = soc_dims[cone];
        int64_t z_off = batch * m + soc_offsets[cone];
        int64_t h_off = batch * dense_Hs_offsets[numSocCones] + dense_Hs_offsets[cone];

        // Only handle dense cones (dim <= 4).
        // ConeDerivatives routes dim>4 to sparse storage; local arrays v[4]/w[4] rely on this.
        if (dim > 4) continue;

        double z0 = z[z_off];
        double norm_sq = 0.0;
        for (int64_t i = 1; i < dim; i++) {
            double zi = z[z_off + i];
            norm_sq += zi * zi;
        }
        double zeta = z0 * z0 - norm_sq;

        // If ζ ≤ 0 (not interior), H = I (matches CPU guard)
        if (zeta <= 0.0) {
            int64_t idx = 0;
            for (int64_t i = 0; i < dim; i++) {
                for (int64_t j = i; j < dim; j++) {
                    soc_H[h_off + idx] = (i == j) ? 1.0 : 0.0;
                    idx++;
                }
            }
            continue;
        }

        auto co = soc_smoothed_coeffs(z0, norm_sq, zeta, mu_val);

        // v = Jz, w = N⁻¹e₀
        double v[4], w[4];
        v[0] = z0;
        w[0] = co.n_inv_00;
        for (int64_t i = 1; i < dim; i++) {
            v[i] = -z[z_off + i];
            w[i] = co.beta_z0 * z[z_off + i];
        }

        // H = (1/di)I - β·vvᵀ - γ·wwᵀ, store upper triangle
        int64_t idx = 0;
        for (int64_t i = 0; i < dim; i++) {
            for (int64_t j = i; j < dim; j++) {
                double val = -co.beta * v[i] * v[j] - co.gamma * w[i] * w[j];
                if (i == j) val += co.inv_di;
                soc_H[h_off + idx] = val;
                idx++;
            }
        }
    }
}

// ============================================================================
// Smoothed SOC Derivative Kernel (sparse storage, dim > 4)
// ============================================================================

// Small-sparse-cone path. Loop is restricted to [0, numSmallSoc); large cones
// are handled by smoothed_soc_sparse_derivative_large_kernel.
__global__ void smoothed_soc_sparse_derivative_kernel(
    double* __restrict__ sparse_diag,    // [batchSize][totalSparseSocDim]
    double* __restrict__ sparse_v1,      // [batchSize][totalSparseSocDim]
    double* __restrict__ sparse_v2,      // [batchSize][totalSparseSocDim]
    double* __restrict__ sparse_c1,      // [batchSize][numSparseSoc]
    double* __restrict__ sparse_c2,      // [batchSize][numSparseSoc]
    const double* __restrict__ z,        // [batchSize][m]
    const double* __restrict__ mu,       // [batchSize]
    const int64_t* __restrict__ soc_dims,
    const int64_t* __restrict__ soc_offsets,
    const int64_t* __restrict__ sparse_indices,   // maps cone_idx -> sparse_idx (-1 if dense)
    const int64_t* __restrict__ sparse_offsets,   // prefix sum of sparse cone dims
    int64_t numSmallSoc,
    int64_t numSparseSoc,
    int64_t totalSparseSocDim,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double mu_val = mu[batch];

    for (int64_t cone = threadIdx.x; cone < numSmallSoc; cone += blockDim.x) {
        int64_t sparse_idx = sparse_indices[cone];
        if (sparse_idx < 0) continue;  // dense cone, skip

        int64_t dim = soc_dims[cone];
        int64_t z_off = batch * m + soc_offsets[cone];
        int64_t diag_off = batch * totalSparseSocDim + sparse_offsets[sparse_idx];
        int64_t c_off = batch * numSparseSoc + sparse_idx;

        double z0 = z[z_off];
        double norm_sq = 0.0;
        for (int64_t i = 1; i < dim; i++) {
            double zi = z[z_off + i];
            norm_sq += zi * zi;
        }
        double zeta = z0 * z0 - norm_sq;

        // If ζ ≤ 0 (not interior), H = I (matches CPU guard)
        if (zeta <= 0.0) {
            for (int64_t i = 0; i < dim; i++) {
                sparse_diag[diag_off + i] = 1.0;
                sparse_v1[diag_off + i] = 0.0;
                sparse_v2[diag_off + i] = 0.0;
            }
            sparse_c1[c_off] = 0.0;
            sparse_c2[c_off] = 0.0;
            continue;
        }

        auto co = soc_smoothed_coeffs(z0, norm_sq, zeta, mu_val);

        // Write uniform diagonal, v1 = Jz, v2 = w = N⁻¹e₀
        sparse_diag[diag_off] = co.inv_di;
        sparse_v1[diag_off] = z0;
        sparse_v2[diag_off] = co.n_inv_00;
        for (int64_t i = 1; i < dim; i++) {
            sparse_diag[diag_off + i] = co.inv_di;
            sparse_v1[diag_off + i] = -z[z_off + i];
            sparse_v2[diag_off + i] = co.beta_z0 * z[z_off + i];
        }

        sparse_c1[c_off] = -co.beta;
        sparse_c2[c_off] = -co.gamma;
    }
}

// Block-per-cone variant for large sparse SOCs (dim > SOC_PARALLEL_THRESHOLD).
// Mirrors smoothed_soc_sparse_derivative_kernel with intra-cone parallelism
// for the tail norm and per-entry writes.
__global__ void smoothed_soc_sparse_derivative_large_kernel(
    double* __restrict__ sparse_diag,
    double* __restrict__ sparse_v1,
    double* __restrict__ sparse_v2,
    double* __restrict__ sparse_c1,
    double* __restrict__ sparse_c2,
    const double* __restrict__ z,
    const double* __restrict__ mu,
    const int64_t* __restrict__ soc_dims,
    const int64_t* __restrict__ soc_offsets,
    const int64_t* __restrict__ sparse_indices,
    const int64_t* __restrict__ sparse_offsets,
    int64_t numSmallSoc,
    int64_t numLargeSoc,
    int64_t numSparseSoc,
    int64_t totalSparseSocDim,
    int64_t batchSize,
    int64_t m
) {
    int64_t large_idx = blockIdx.x;
    int64_t batch = blockIdx.y;
    if (large_idx >= numLargeSoc || batch >= batchSize) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    int64_t cone = numSmallSoc + large_idx;
    int64_t sparse_idx = sparse_indices[cone];
    if (sparse_idx < 0) return;  // large cones are always sparse; defensive

    int64_t dim = soc_dims[cone];
    int64_t z_off = batch * m + soc_offsets[cone];
    int64_t diag_off = batch * totalSparseSocDim + sparse_offsets[sparse_idx];
    int64_t c_off = batch * numSparseSoc + sparse_idx;

    double mu_val = mu[batch];
    double z0 = z[z_off];

    extern __shared__ double smem[];
    double my_sq = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double zi = z[z_off + i];
        my_sq += zi * zi;
    }
    double norm_sq = cones::block_sum_reduce(my_sq, smem, tid);
    double zeta = z0 * z0 - norm_sq;

    if (zeta <= 0.0) {
        // Not interior: H = I
        for (int64_t i = tid; i < dim; i += blockDimX) {
            sparse_diag[diag_off + i] = 1.0;
            sparse_v1[diag_off + i] = 0.0;
            sparse_v2[diag_off + i] = 0.0;
        }
        if (tid == 0) {
            sparse_c1[c_off] = 0.0;
            sparse_c2[c_off] = 0.0;
        }
        return;
    }

    auto co = soc_smoothed_coeffs(z0, norm_sq, zeta, mu_val);

    if (tid == 0) {
        sparse_diag[diag_off] = co.inv_di;
        sparse_v1[diag_off] = z0;
        sparse_v2[diag_off] = co.n_inv_00;
        sparse_c1[c_off] = -co.beta;
        sparse_c2[c_off] = -co.gamma;
    }
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double zi = z[z_off + i];
        sparse_diag[diag_off + i] = co.inv_di;
        sparse_v1[diag_off + i] = -zi;
        sparse_v2[diag_off + i] = co.beta_z0 * zi;
    }
}

// ============================================================================
// Host dispatch function
// ============================================================================

void compute_smoothed_cone_derivative(
    const BatchedVector& z,
    const BatchedVector& s,
    const BatchedVector& mu,
    ConeDerivatives& derivs,
    const Cones& cones,
    cudaStream_t stream
) {
    int64_t m_total = z.n();
    int64_t batchSize = z.batchSize();

    // Reject unsupported cones — smoothed diff only supports zero + nonneg + SOC
    if (cones.numExpCones > 0 || cones.numPowerCones > 0) {
        throw std::runtime_error(
            "Smoothed differentiation only supports zero + nonneg + SOC cones. "
            "Use diff_method='exact'."
        );
    }

    // Zero cones: H = I (no barrier), handled by updateJ.
    // Nonneg offset = numZeroCones (zero cones come first in the cone ordering).

    // Nonnegative cones
    if (cones.numNonnegCones > 0) {
        int threads = std::min((int)cones.numNonnegCones, 256);
        smoothed_nonneg_derivative_kernel<<<batchSize, threads, 0, stream>>>(
            derivs.nonneg_H.data(), z.data(), mu.data(),
            cones.numZeroCones, cones.numNonnegCones, batchSize, m_total
        );
    }

    // SOC cones
    if (cones.numSocCones > 0) {
        // Dense SOC cones (dim <= 4)
        if (derivs.totalDenseSocHsEntries > 0) {
            smoothed_soc_dense_derivative_kernel<<<batchSize, std::min((int)cones.numSocCones, 256), 0, stream>>>(
                derivs.soc_H.data(), z.data(), mu.data(),
                cones.d_soc_dims, cones.d_soc_offsets,
                derivs.d_dense_soc_Hs_offsets,
                cones.numSocCones, batchSize, m_total
            );
        }

        // Sparse SOC cones: small cones (4 < dim <= SOC_PARALLEL_THRESHOLD)
        // via thread-per-cone; large cones via block-per-cone.
        int64_t numSmallSoc = cones.numSocCones - cones.numLargeSoc;
        if (cones.numSparseSoc > 0 && numSmallSoc > 0) {
            smoothed_soc_sparse_derivative_kernel<<<batchSize, std::min((int)numSmallSoc, 256), 0, stream>>>(
                derivs.soc_sparse_diag.data(),
                derivs.soc_sparse_v1.data(),
                derivs.soc_sparse_v2.data(),
                derivs.soc_sparse_c1.data(),
                derivs.soc_sparse_c2.data(),
                z.data(), mu.data(),
                cones.d_soc_dims, cones.d_soc_offsets,
                cones.d_soc_sparse_indices, cones.d_soc_sparse_offsets,
                numSmallSoc, cones.numSparseSoc,
                derivs.totalSparseSocDim,
                batchSize, m_total
            );
        }
        if (cones.numLargeSoc > 0) {
            const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
            dim3 grid(static_cast<unsigned int>(cones.numLargeSoc),
                      static_cast<unsigned int>(batchSize));
            dim3 block(block_size);
            size_t smem_bytes = sizeof(double) * block_size;
            smoothed_soc_sparse_derivative_large_kernel<<<grid, block, smem_bytes, stream>>>(
                derivs.soc_sparse_diag.data(),
                derivs.soc_sparse_v1.data(),
                derivs.soc_sparse_v2.data(),
                derivs.soc_sparse_c1.data(),
                derivs.soc_sparse_c2.data(),
                z.data(), mu.data(),
                cones.d_soc_dims, cones.d_soc_offsets,
                cones.d_soc_sparse_indices, cones.d_soc_sparse_offsets,
                numSmallSoc, cones.numLargeSoc, cones.numSparseSoc,
                derivs.totalSparseSocDim,
                batchSize, m_total
            );
        }
    }
}

} // namespace moreau
