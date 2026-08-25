/**
 * @file cones.cu
 * @brief CUDA kernels for cone operations
 */

#include "moreau/cones/cones.hpp"
#include "moreau/cones/psd_kernels.cuh"
#include "moreau/cones/common.cuh"
#include "moreau/cuda/utils.cuh"
#include "moreau/variables/variables_kernels.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

namespace moreau {

// ============================================================================
// FUSED cone initialization kernel - combines all cone types (5→1)
// ============================================================================
__global__ void initAllConesKernel(
    double* __restrict__ s,
    double* __restrict__ z,
    // Cone offsets in constraint space
    int64_t offset_zero,
    int64_t offset_nonneg,
    int64_t offset_soc,
    int64_t offset_exp,
    int64_t offset_power,
    int64_t offset_genpow,
    // Cone counts
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t numGenPowerCones,
    // Power cone alphas
    const double* __restrict__ d_powerAlphas,
    // GenPowerCone params
    const double* __restrict__ d_genPowerAlphas,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    // Total constraints
    int64_t m,
    // Variable-dim SOC params
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch_idx = blockIdx.x;
    int64_t stride = blockDim.x;

    // ========== ZERO CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numZeroCones; cone_idx += stride) {
        int64_t idx = batch_idx * m + offset_zero + cone_idx;
        s[idx] = 0.0;
        z[idx] = 0.0;
    }
    __syncthreads();

    // ========== NONNEGATIVE CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numNonnegCones; cone_idx += stride) {
        int64_t idx = batch_idx * m + offset_nonneg + cone_idx;
        s[idx] = 1.0;
        z[idx] = 1.0;
    }
    __syncthreads();

    // ========== SOC CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numSocCones; cone_idx += stride) {
        int64_t dim = d_soc_dims[cone_idx];
        int64_t sz_off = d_soc_sz_offsets[cone_idx];
        int64_t base_idx = batch_idx * m + offset_soc + sz_off;
        s[base_idx] = 1.0;
        z[base_idx] = 1.0;
        for (int64_t i = 1; i < dim; i++) {
            s[base_idx + i] = 0.0;
            z[base_idx + i] = 0.0;
        }
    }
    __syncthreads();

    // ========== EXPONENTIAL CONES ==========
    // Constants from Clarabel for exponential cone unit initialization
    for (int64_t cone_idx = threadIdx.x; cone_idx < numExpCones; cone_idx += stride) {
        int64_t base_idx = batch_idx * m + offset_exp + cone_idx * 3;
        s[base_idx + 0] = -1.051383945322714;
        s[base_idx + 1] =  0.556409619469370;
        s[base_idx + 2] =  1.258967884768947;
        z[base_idx + 0] = -1.051383945322714;
        z[base_idx + 1] =  0.556409619469370;
        z[base_idx + 2] =  1.258967884768947;
    }
    __syncthreads();

    // ========== POWER CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numPowerCones; cone_idx += stride) {
        double alpha = d_powerAlphas[cone_idx];
        int64_t base_idx = batch_idx * m + offset_power + cone_idx * 3;

        double val0 = sqrt(1.0 + alpha);
        double val1 = sqrt(1.0 + (1.0 - alpha));

        s[base_idx + 0] = val0;
        s[base_idx + 1] = val1;
        s[base_idx + 2] = 0.0;
        z[base_idx + 0] = val0;
        z[base_idx + 1] = val1;
        z[base_idx + 2] = 0.0;
    }
    __syncthreads();

    // ========== GENERALIZED POWER CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += stride) {
        int64_t dim1 = d_genPowerDim1s[cone_idx];
        int64_t dim2 = d_genPowerDim2s[cone_idx];
        int64_t sz_off = d_genPowerSzOffsets[cone_idx];
        int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
        int64_t base_idx = batch_idx * m + offset_genpow + sz_off;

        for (int64_t i = 0; i < dim1; i++) {
            double val = sqrt(1.0 + d_genPowerAlphas[alpha_off + i]);
            s[base_idx + i] = val;
            z[base_idx + i] = val;
        }
        for (int64_t i = 0; i < dim2; i++) {
            s[base_idx + dim1 + i] = 0.0;
            z[base_idx + dim1 + i] = 0.0;
        }
    }
}

void Cones::unit_initialization(BatchedVector& s, BatchedVector& z, cudaStream_t stream) {
    int64_t m = totalConstraints();
    if (m == 0) return;

    // Compute cone offsets
    int64_t offset_zero = 0;
    int64_t offset_nonneg = offset_zero + numZeroCones;
    int64_t offset_soc = offset_nonneg + numNonnegCones;
    int64_t offset_exp = offset_soc + totalSocDim;
    int64_t offset_power = offset_exp + numExpCones * 3;
    int64_t offset_genpow = offset_power + numPowerCones * 3;

    // Determine thread count based on max cones
    int64_t maxCones = numZeroCones;
    maxCones = (numNonnegCones > maxCones) ? numNonnegCones : maxCones;
    maxCones = (numSocCones > maxCones) ? numSocCones : maxCones;
    maxCones = (numExpCones > maxCones) ? numExpCones : maxCones;
    maxCones = (numPowerCones > maxCones) ? numPowerCones : maxCones;
    maxCones = (numGenPowerCones > maxCones) ? numGenPowerCones : maxCones;

    if (maxCones == 0) return;

    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(initAllConesKernel, batchSize, threadsPerBlock, 0, stream,
        s.data(), z.data(),
        offset_zero, offset_nonneg, offset_soc, offset_exp, offset_power, offset_genpow,
        numZeroCones, numNonnegCones, numSocCones, numExpCones, numPowerCones, numGenPowerCones,
        d_powerAlphas,
        d_genPowerAlphas, d_genPowerDim1s, d_genPowerDim2s, d_genPowerOffsets, d_genPowerAlphaOffsets,
        d_genPowerSzOffsets,
        m,
        d_soc_dims, d_soc_offsets, d_soc_sz_offsets
    );

    // PSD cones: separate kernel (needs variable dim per cone)
    if (numPsdCones > 0) {
        int64_t offset_psd = offset_power + numPowerCones * 3;
        dim3 psd_grid(batchSize, numPsdCones);
        initPsdConesKernel<<<psd_grid, 256, 0, stream>>>(
            s.data(), z.data(), offset_psd,
            d_psd_dims, d_psd_sz_offsets, numPsdCones, m
        );
    }

    cudaStreamSynchronize(stream);
}

// ==================== Affine ds kernels ====================

// FUSED kernel for all cone types - eliminates 4 kernel launches per call
__global__ void affine_ds_all_cones_kernel(
    double* __restrict__ ds,
    const double* __restrict__ s,
    // Zero cone params
    int64_t offset_zero,
    int64_t numZeroCones,
    // Nonneg cone params
    const double* __restrict__ nonneg_lambda,
    int64_t offset_nonneg,
    int64_t numNonnegCones,
    // SOC cone params
    const double* __restrict__ soc_lambda,
    int64_t offset_soc,
    int64_t numSocCones,
    // Exp/Power cone params (both use ds = s copy)
    int64_t offset_exp,
    int64_t numExpCones,
    int64_t offset_power,
    int64_t numPowerCones,
    // GenPowerCone params (ds = s copy)
    int64_t offset_genpow,
    int64_t numGenPowerCones,
    int64_t totalGenPowerDim,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    // Common params
    int64_t m,
    int64_t batchSize,
    // Variable-dim SOC params
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    // Zero cones: ds = 0
    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        ds[batch_idx * m + offset_zero + i] = 0.0;
    }

    // Nonneg cones: ds = lambda²
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t ds_idx = batch_idx * m + offset_nonneg + i;
        int64_t lambda_idx = batch_idx * numNonnegCones + i;
        double lambda_val = nonneg_lambda[lambda_idx];
        ds[ds_idx] = lambda_val * lambda_val;
    }

    // SOC cones: ds = lambda ∘ lambda (circle product)
    for (int64_t cone_idx = threadIdx.x; cone_idx < numSocCones; cone_idx += blockDim.x) {
        int64_t dim = d_soc_dims[cone_idx];
        int64_t soc_off = d_soc_offsets[cone_idx];     // for internal arrays (lambda)
        int64_t sz_off = d_soc_sz_offsets[cone_idx];   // for ds output (s/z space)
        int64_t ds_base = batch_idx * m + offset_soc + sz_off;
        int64_t lambda_base = batch_idx * totalSocDim + soc_off;

        double lambda0 = soc_lambda[lambda_base];

        // Circle product: result[0] = ||lambda||², result[1:] = 2*lambda[0]*lambda[1:]
        // lambda0 can be negative during iterations (numerical error), but the
        // circle product formula is correct regardless of sign.
        double dot = lambda0 * lambda0;
        for (int64_t i = 1; i < dim; i++) {
            dot += soc_lambda[lambda_base + i] * soc_lambda[lambda_base + i];
        }
        ds[ds_base] = dot;
        for (int64_t i = 1; i < dim; i++) {
            ds[ds_base + i] = 2.0 * lambda0 * soc_lambda[lambda_base + i];
        }
    }

    // Exp cones: ds = s (copy)
    int64_t numExpElements = numExpCones * 3;
    for (int64_t i = threadIdx.x; i < numExpElements; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_exp + i;
        ds[idx] = s[idx];
    }

    // Power cones: ds = s (copy)
    int64_t numPowerElements = numPowerCones * 3;
    for (int64_t i = threadIdx.x; i < numPowerElements; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_power + i;
        ds[idx] = s[idx];
    }

    // GenPowerCone: ds = s (copy, variable dim)
    for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += blockDim.x) {
        int64_t sz_off = d_genPowerSzOffsets[cone_idx];
        int64_t dim = d_genPowerDim1s[cone_idx] + d_genPowerDim2s[cone_idx];
        int64_t base = batch_idx * m + offset_genpow + sz_off;
        for (int64_t i = 0; i < dim; i++) {
            ds[base + i] = s[base + i];
        }
    }
}


void Cones::affine_ds(BatchedVector& ds, const BatchedVector& s, cudaStream_t stream) {
    const int64_t m = ds.n();

    // Compute offsets
    int64_t offset_zero = 0;
    int64_t offset_nonneg = numZeroCones;
    int64_t offset_soc = numZeroCones + numNonnegCones;
    int64_t offset_exp = numZeroCones + numNonnegCones + totalSocDim;
    int64_t offset_power = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
    int64_t offset_genpow = offset_power + numPowerCones * 3;

    // Determine thread count based on largest cone type
    int64_t maxCones = max(max(max(numZeroCones, numNonnegCones),
                               max(numSocCones, numExpCones * 3)),
                          max(numPowerCones * 3, numGenPowerCones));
    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(affine_ds_all_cones_kernel, batchSize, threadsPerBlock, 0, stream,
        ds.data(), s.data(),
        offset_zero, numZeroCones,
        nonneg_lambda.data(), offset_nonneg, numNonnegCones,
        soc_lambda.data(), offset_soc, numSocCones,
        offset_exp, numExpCones,
        offset_power, numPowerCones,
        offset_genpow, numGenPowerCones, totalGenPowerDim,
        d_genPowerOffsets, d_genPowerDim1s, d_genPowerDim2s, d_genPowerSzOffsets,
        m, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets
    );

    // PSD cones: ds = svec(diag(λ²))
    if (numPsdCones > 0) {
        int64_t offset_psd = numZeroCones + numNonnegCones + totalSocDim
                           + numExpCones * 3 + numPowerCones * 3;
        dim3 psd_grid(batchSize, numPsdCones);
        psd_affine_ds_kernel<<<psd_grid, 256, 0, stream>>>(
            ds.data(), psd_lambda.data(), offset_psd,
            d_psd_dims, d_psd_sz_offsets, d_psd_mat_offsets,
            numPsdCones, totalPsdSvecDim, totalPsdMatDim, m
        );
    }
}

// ==================== Δs_from_Δz_offset kernels ====================

// FUSED kernel for all cone types - eliminates 4 kernel launches per call
__global__ void ds_from_dz_offset_all_cones_kernel(
    double* __restrict__ out,
    const double* __restrict__ ds,
    const double* __restrict__ z,
    // Zero cone params
    int64_t offset_zero,
    int64_t numZeroCones,
    // Nonneg cone params
    int64_t offset_nonneg,
    int64_t numNonnegCones,
    // SOC cone params
    const double* __restrict__ soc_lambda,
    const double* __restrict__ soc_w,
    const double* __restrict__ soc_eta,
    int64_t offset_soc,
    int64_t numSocCones,
    // Exp/Power cone params (both use out = ds copy)
    int64_t offset_exp,
    int64_t numExpCones,
    int64_t offset_power,
    int64_t numPowerCones,
    // GenPowerCone params (out = ds copy)
    int64_t offset_genpow,
    int64_t numGenPowerCones,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    // Common params
    int64_t m,
    int64_t batchSize,
    // Variable-dim SOC params
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    // Zero cones: out = 0
    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        out[batch_idx * m + offset_zero + i] = 0.0;
    }

    // Nonneg cones: out = ds / z
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_nonneg + i;
        out[idx] = ds[idx] / z[idx];
    }

    // SOC cones: Δs_from_Δz_offset
    for (int64_t cone_idx = threadIdx.x; cone_idx < numSocCones; cone_idx += blockDim.x) {
        int64_t dim = d_soc_dims[cone_idx];
        int64_t soc_off = d_soc_offsets[cone_idx];
        int64_t sz_off = d_soc_sz_offsets[cone_idx];
        int64_t base_idx = batch_idx * m + offset_soc + sz_off;
        int64_t lambda_base = batch_idx * totalSocDim + soc_off;
        int64_t w_base = lambda_base;
        int64_t eta_idx = batch_idx * numSocCones + cone_idx;

        double z0 = z[base_idx];
        double lambda0 = soc_lambda[lambda_base];
        double w0 = soc_w[w_base];
        double eta = soc_eta[eta_idx];
        double ds0 = ds[base_idx];

        // Compute dot products: lambda[1:] . ds[1:] and w[1:] . ds[1:]
        double lambda_tail_ds_tail = 0.0;
        double w_tail_ds_tail = 0.0;
        double z_tail_sq = 0.0;
        for (int64_t i = 1; i < dim; i++) {
            lambda_tail_ds_tail += soc_lambda[lambda_base + i] * ds[base_idx + i];
            w_tail_ds_tail += soc_w[w_base + i] * ds[base_idx + i];
            z_tail_sq += z[base_idx + i] * z[base_idx + i];
        }

        // SOC residual: z0² - ||z[1:]||²
        double z_tail_norm = sqrt(z_tail_sq);
        double resz = (z0 - z_tail_norm) * (z0 + z_tail_norm);

        // out = J(z) * c where c = (lambda0*ds0 - lambda_tail.ds_tail) / resz
        // J = diag(1, -1, -1, ...)
        double c = (lambda0 * ds0 - lambda_tail_ds_tail) / resz;
        double out0 = z0 * c;

        // out += eta * mul_W terms
        double factor = w_tail_ds_tail / (1.0 + w0);
        out0 += eta * w_tail_ds_tail;

        double inv_lambda0 = 1.0 / lambda0;
        out[base_idx] = out0 * inv_lambda0;
        for (int64_t i = 1; i < dim; i++) {
            double out_i = -z[base_idx + i] * c;
            out_i += eta * (ds[base_idx + i] + factor * soc_w[w_base + i]);
            out[base_idx + i] = out_i * inv_lambda0;
        }
    }

    // Exp cones: out = ds (copy)
    int64_t numExpElements = numExpCones * 3;
    for (int64_t i = threadIdx.x; i < numExpElements; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_exp + i;
        out[idx] = ds[idx];
    }

    // Power cones: out = ds (copy)
    int64_t numPowerElements = numPowerCones * 3;
    for (int64_t i = threadIdx.x; i < numPowerElements; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_power + i;
        out[idx] = ds[idx];
    }

    // GenPowerCone: out = ds (copy, variable dim)
    for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += blockDim.x) {
        int64_t sz_off = d_genPowerSzOffsets[cone_idx];
        int64_t dim = d_genPowerDim1s[cone_idx] + d_genPowerDim2s[cone_idx];
        int64_t base = batch_idx * m + offset_genpow + sz_off;
        for (int64_t i = 0; i < dim; i++) {
            out[base + i] = ds[base + i];
        }
    }
}


void Cones::Δs_from_Δz_offset(BatchedVector& out, const BatchedVector& ds, BatchedVector& work, const BatchedVector& z, cudaStream_t stream) {
    const int64_t m = out.n();

    // Compute offsets
    int64_t offset_zero = 0;
    int64_t offset_nonneg = numZeroCones;
    int64_t offset_soc = numZeroCones + numNonnegCones;
    int64_t offset_exp = numZeroCones + numNonnegCones + totalSocDim;
    int64_t offset_power = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
    int64_t offset_genpow = offset_power + numPowerCones * 3;

    // Determine thread count based on largest cone type
    int64_t maxCones = max(max(max(numZeroCones, numNonnegCones),
                               max(numSocCones, numExpCones * 3)),
                          max(numPowerCones * 3, numGenPowerCones));
    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(ds_from_dz_offset_all_cones_kernel, batchSize, threadsPerBlock, 0, stream,
        out.data(), ds.data(), z.data(),
        offset_zero, numZeroCones,
        offset_nonneg, numNonnegCones,
        soc_lambda.data(), soc_w.data(), soc_eta.data(), offset_soc, numSocCones,
        offset_exp, numExpCones,
        offset_power, numPowerCones,
        offset_genpow, numGenPowerCones,
        d_genPowerOffsets, d_genPowerDim1s, d_genPowerDim2s, d_genPowerSzOffsets,
        m, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets
    );

    // PSD cones: out = W^T(λ \ ds) (proper symmetric cone operation)
    if (numPsdCones > 0) {
        int64_t offset_psd = numZeroCones + numNonnegCones + totalSocDim
                           + numExpCones * 3 + numPowerCones * 3;
        psd_ds_from_dz_offset(*this, out.data(), ds.data(),
                              offset_psd, m, stream);
    }
}

// ==================== mul_Hs kernels ====================

// FUSED kernel for all cone types - eliminates 4 kernel launches per call
__global__ void mul_hs_all_cones_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    // Zero cone params
    int64_t offset_zero,
    int64_t numZeroCones,
    // Nonneg cone params
    const double* __restrict__ nonneg_w,
    int64_t offset_nonneg,
    int64_t numNonnegCones,
    // SOC cone params
    const double* __restrict__ soc_w,
    const double* __restrict__ soc_eta,
    int64_t offset_soc,
    int64_t numSocCones,
    // Exp cone params
    const double* __restrict__ exp_Hs,
    int64_t offset_exp,
    int64_t numExpCones,
    // Power cone params
    const double* __restrict__ power_Hs,
    int64_t offset_power,
    int64_t numPowerCones,
    // GenPowerCone params
    const double* __restrict__ genpow_p,
    const double* __restrict__ genpow_q,
    const double* __restrict__ genpow_r,
    const double* __restrict__ genpow_d1,
    const double* __restrict__ genpow_d2,
    const double* __restrict__ genpow_Hs,      // dense: packed upper-tri, sparse: diagonal only
    const double* __restrict__ genpow_mu,      // [batchSize] mu per batch (for sparse rank-3 multiply)
    int64_t offset_genpow,
    int64_t numGenPowerCones,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerHsOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerHsEntries,
    // Rank-6 sparse PD scaling (zero when inactive — Hs += Σ sign·coef·axis·axis').
    const double* __restrict__ genpow_pd_axes,    // (B, 6 * totalGenPowerDim)
    const double* __restrict__ genpow_pd_coefs,   // (B, 6 * numGenPowerCones)
    const double* __restrict__ genpow_pd_signs,   // (B, 6 * numGenPowerCones)
    const double* __restrict__ genpow_pd_active,  // (B, numGenPowerCones)
    // Common params
    int64_t m,
    int64_t batchSize,
    // Variable-dim SOC params
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    // Zero cones: y = 0
    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        y[batch_idx * m + offset_zero + i] = 0.0;
    }

    // Nonneg cones: y = w² * x
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t y_idx = batch_idx * m + offset_nonneg + i;
        int64_t w_idx = batch_idx * numNonnegCones + i;
        double w_val = nonneg_w[w_idx];
        y[y_idx] = w_val * w_val * x[y_idx];
    }

    // SOC cones: y = eta²(2ww^T - J)x where J = diag(1,-I)
    for (int64_t cone_idx = threadIdx.x; cone_idx < numSocCones; cone_idx += blockDim.x) {
        int64_t dim = d_soc_dims[cone_idx];
        int64_t soc_off = d_soc_offsets[cone_idx];
        int64_t sz_off = d_soc_sz_offsets[cone_idx];
        int64_t base_idx = batch_idx * m + offset_soc + sz_off;
        int64_t w_base = batch_idx * totalSocDim + soc_off;
        int64_t eta_idx = batch_idx * numSocCones + cone_idx;

        double eta = soc_eta[eta_idx];
        double eta2 = eta * eta;

        // Compute c = 2 * dot(w, x)
        double c = 0.0;
        for (int64_t i = 0; i < dim; i++) {
            c += soc_w[w_base + i] * x[base_idx + i];
        }
        c *= 2.0;

        y[base_idx] = eta2 * (c * soc_w[w_base] + (-x[base_idx]));
        for (int64_t i = 1; i < dim; i++) {
            y[base_idx + i] = eta2 * (c * soc_w[w_base + i] + x[base_idx + i]);
        }
    }

    // Exp cones: y = Hs * x (3x3 symmetric matrix multiply)
    for (int64_t cone_idx = threadIdx.x; cone_idx < numExpCones; cone_idx += blockDim.x) {
        int64_t base_idx = batch_idx * m + offset_exp + cone_idx * 3;
        int64_t Hs_base = batch_idx * numExpCones * 6 + cone_idx * 6;

        double H00 = exp_Hs[Hs_base + 0];
        double H01 = exp_Hs[Hs_base + 1];
        double H02 = exp_Hs[Hs_base + 2];
        double H11 = exp_Hs[Hs_base + 3];
        double H12 = exp_Hs[Hs_base + 4];
        double H22 = exp_Hs[Hs_base + 5];
        double x0 = x[base_idx + 0];
        double x1 = x[base_idx + 1];
        double x2 = x[base_idx + 2];

        y[base_idx + 0] = H00 * x0 + H01 * x1 + H02 * x2;
        y[base_idx + 1] = H01 * x0 + H11 * x1 + H12 * x2;
        y[base_idx + 2] = H02 * x0 + H12 * x1 + H22 * x2;
    }

    // Power cones: y = Hs * x (3x3 symmetric matrix multiply)
    for (int64_t cone_idx = threadIdx.x; cone_idx < numPowerCones; cone_idx += blockDim.x) {
        int64_t base_idx = batch_idx * m + offset_power + cone_idx * 3;
        int64_t Hs_base = batch_idx * numPowerCones * 6 + cone_idx * 6;

        double H00 = power_Hs[Hs_base + 0];
        double H01 = power_Hs[Hs_base + 1];
        double H02 = power_Hs[Hs_base + 2];
        double H11 = power_Hs[Hs_base + 3];
        double H12 = power_Hs[Hs_base + 4];
        double H22 = power_Hs[Hs_base + 5];
        double x0 = x[base_idx + 0];
        double x1 = x[base_idx + 1];
        double x2 = x[base_idx + 2];

        y[base_idx + 0] = H00 * x0 + H01 * x1 + H02 * x2;
        y[base_idx + 1] = H01 * x0 + H11 * x1 + H12 * x2;
        y[base_idx + 2] = H02 * x0 + H12 * x1 + H22 * x2;
    }

    // GenPowerCone: y = Hs * x
    // Dense (dim<=4): packed upper-triangle symmetric matvec
    // Sparse (dim>4): rank-3 multiply y = μ*(D*x + p*(p'x) - q*(q'x) - r*(r'x))
    for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += blockDim.x) {
        int64_t dim1 = d_genPowerDim1s[cone_idx];
        int64_t dim2 = d_genPowerDim2s[cone_idx];
        int64_t dim = dim1 + dim2;
        int64_t sz_off = d_genPowerSzOffsets[cone_idx];
        int64_t hs_off = d_genPowerHsOffsets[cone_idx];
        int64_t base_idx = batch_idx * m + offset_genpow + sz_off;
        int64_t Hs_base = batch_idx * totalGenPowerHsEntries + hs_off;

        if (dim > 4) {
            // Sparse: compute y = μ*(D*x + p*(p'x) - q*(q'x) - r*(r'x))
            int64_t gp_off = d_genPowerOffsets[cone_idx];
            int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
            int64_t r_off = gp_off - alpha_off;
            double mu_val = genpow_mu[batch_idx];

            int64_t p_base = batch_idx * totalGenPowerDim + gp_off;
            int64_t q_base = batch_idx * totalGenPowerAlphas + alpha_off;
            int64_t r_base = batch_idx * (totalGenPowerDim - totalGenPowerAlphas) + r_off;
            int64_t d1_base = batch_idx * totalGenPowerAlphas + alpha_off;

            // Compute dot products: p'x, q'x (dim1 part), r'x (dim2 part)
            double pdotx = 0.0, qdotx = 0.0, rdotx = 0.0;
            for (int64_t i = 0; i < dim1; i++) {
                pdotx += genpow_p[p_base + i] * x[base_idx + i];
                qdotx += genpow_q[q_base + i] * x[base_idx + i];
            }
            for (int64_t i = 0; i < dim2; i++) {
                pdotx += genpow_p[p_base + dim1 + i] * x[base_idx + dim1 + i];
                rdotx += genpow_r[r_base + i] * x[base_idx + dim1 + i];
            }

            // y_i = μ * (D_i*x_i + p_i*pdotx - q_i*qdotx)  for i < dim1
            double d2_val = genpow_d2[batch_idx * numGenPowerCones + cone_idx];
            for (int64_t i = 0; i < dim1; i++) {
                double d_val = genpow_d1[d1_base + i];
                y[base_idx + i] = mu_val * (d_val * x[base_idx + i]
                    + genpow_p[p_base + i] * pdotx
                    - genpow_q[q_base + i] * qdotx);
            }
            // y_i = μ * (d2*x_i + p_i*pdotx - r_{i-dim1}*rdotx)  for i >= dim1
            for (int64_t i = 0; i < dim2; i++) {
                y[base_idx + dim1 + i] = mu_val * (d2_val * x[base_idx + dim1 + i]
                    + genpow_p[p_base + dim1 + i] * pdotx
                    - genpow_r[r_base + i] * rdotx);
            }

            // Rank-6 PD scaling contribution: y += Σ_{k=0..5} sign_k · coef_k · ⟨axis_k, x⟩ · axis_k
            // When pd_active=0 the coefs are zero and this is a no-op.
            double pd_active_val = genpow_pd_active[batch_idx * numGenPowerCones + cone_idx];
            if (pd_active_val > 0.5) {
                int64_t pd_axes_base = batch_idx * 6 * totalGenPowerDim + 6 * gp_off;
                int64_t pd_coef_base = batch_idx * 6 * numGenPowerCones + 6 * cone_idx;
                for (int k = 0; k < 6; ++k) {
                    double coef = genpow_pd_coefs[pd_coef_base + k];
                    if (coef == 0.0) continue;
                    double sign = genpow_pd_signs[pd_coef_base + k];
                    double dot = 0.0;
                    for (int64_t i = 0; i < dim; ++i) {
                        dot += genpow_pd_axes[pd_axes_base + k * dim + i] * x[base_idx + i];
                    }
                    double scale = sign * coef * dot;
                    for (int64_t i = 0; i < dim; ++i) {
                        y[base_idx + i] += scale * genpow_pd_axes[pd_axes_base + k * dim + i];
                    }
                }
            }
        } else {
            // Dense: symmetric matvec with packed upper triangle
            for (int64_t i = 0; i < dim; i++) {
                double yi = 0.0;
                for (int64_t j = 0; j < dim; j++) {
                    int64_t r = (i <= j) ? i : j;
                    int64_t c = (i <= j) ? j : i;
                    int64_t idx = r * dim - r * (r + 1) / 2 + c;
                    yi += genpow_Hs[Hs_base + idx] * x[base_idx + j];
                }
                y[base_idx + i] = yi;
            }
        }
    }
}


void Cones::mul_Hs(BatchedVector& y, const BatchedVector& x, BatchedVector& work, cudaStream_t stream) {
    const int64_t m = y.n();

    // Compute offsets
    int64_t offset_zero = 0;
    int64_t offset_nonneg = numZeroCones;
    int64_t offset_soc = numZeroCones + numNonnegCones;
    int64_t offset_exp = numZeroCones + numNonnegCones + totalSocDim;
    int64_t offset_power = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
    int64_t offset_genpow = offset_power + numPowerCones * 3;

    // Determine thread count based on largest cone type
    int64_t maxCones = max(max(max(numZeroCones, numNonnegCones),
                               max(numSocCones, numExpCones)),
                          max(numPowerCones, numGenPowerCones));
    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(mul_hs_all_cones_kernel, batchSize, threadsPerBlock, 0, stream,
        y.data(), x.data(),
        offset_zero, numZeroCones,
        nonneg_w.data(), offset_nonneg, numNonnegCones,
        soc_w.data(), soc_eta.data(), offset_soc, numSocCones,
        exp_Hs.data(), offset_exp, numExpCones,
        power_Hs.data(), offset_power, numPowerCones,
        genpow_p.data(), genpow_q.data(), genpow_r.data(), genpow_d1.data(),
        genpow_d2.data(), genpow_Hs.data(), genpow_mu.data(),
        offset_genpow, numGenPowerCones,
        d_genPowerDim1s, d_genPowerDim2s, d_genPowerOffsets,
        d_genPowerHsOffsets, d_genPowerAlphaOffsets, d_genPowerSzOffsets,
        totalGenPowerDim, totalGenPowerAlphas, totalGenPowerHsEntries,
        genpow_pd_axes.data(), genpow_pd_coefs.data(),
        genpow_pd_signs.data(), genpow_pd_active.data(),
        m, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets
    );

    // PSD cones: y = Hs * x (dense symmetric matvec)
    if (numPsdCones > 0) {
        int64_t offset_psd = numZeroCones + numNonnegCones + totalSocDim
                           + numExpCones * 3 + numPowerCones * 3;
        dim3 psd_grid(batchSize, numPsdCones);
        psd_mul_Hs_kernel<<<psd_grid, 256, 0, stream>>>(
            y.data(), x.data(), psd_Hs.data(),
            offset_psd, d_psd_dims, d_psd_sz_offsets, d_psd_Hs_offsets,
            numPsdCones, totalPsdHsEntries, m
        );
    }
}

// ============================================================================
// Step length kernels
// ============================================================================

// Use the canonical logsafe from common.cuh (returns -INFINITY for x <= 0)
using cones::logsafe;

/**
 * @brief Device function to check if s is primal feasible for exponential cone
 *
 * Primal exponential cone: s₃ ≥ s₂*e^(s₁/s₂), s₃,s₂ > 0
 */
__device__ inline bool is_exp_primal_feasible(double s0, double s1, double s2) {
    if (s2 > 0.0 && s1 > 0.0) {
        double res = s1 * logsafe(s2 / s1) - s0;
        if (res > 0.0) return true;
    }
    return false;
}

/**
 * @brief Device function to check if z is dual feasible for exponential cone
 *
 * Dual exponential cone: z₃ ≥ -z₁*e^(z₂/z₁ - 1), z₃ > 0, z₁ < 0
 */
__device__ inline bool is_exp_dual_feasible(double z0, double z1, double z2) {
    if (z2 > 0.0 && z0 < 0.0) {
        double res = z1 - z0 - z0 * logsafe(-z2 / z0);
        if (res > 0.0) return true;
    }
    return false;
}

/**
 * @brief Device function to check if s is primal feasible for power cone
 *
 * Primal Power cone: s₁^α * s₂^(1-α) ≥ |s₃|, s₁,s₂ ≥ 0
 */
__device__ inline bool is_pow_primal_feasible(double s0, double s1, double s2, double alpha) {
    if (s0 > 0.0 && s1 > 0.0) {
        double res = exp(2.0 * alpha * logsafe(s0) + 2.0 * (1.0 - alpha) * logsafe(s1)) - s2 * s2;
        if (res > 0.0) return true;
    }
    return false;
}

/**
 * @brief Device function to check if z is dual feasible for power cone
 *
 * Dual Power cone: (z₁/α)^α * (z₂/(1-α))^(1-α) ≥ |z₃|, z₁,z₂ ≥ 0
 */
__device__ inline bool is_pow_dual_feasible(double z0, double z1, double z2, double alpha) {
    if (z0 > 0.0 && z1 > 0.0) {
        double res = exp(2.0 * alpha * logsafe(z0 / alpha) + 2.0 * (1.0 - alpha) * logsafe(z1 / (1.0 - alpha))) - z2 * z2;
        if (res > 0.0) return true;
    }
    return false;
}

/**
 * @brief Device function to check if s is primal feasible for generalized power cone
 *
 * GenPower primal: ∏ s_i^{2αi} ≥ ||w||², all s_i > 0
 */
__device__ inline bool is_genpow_primal_feasible(
    const double* s, int64_t dim1, int64_t dim2,
    const double* alphas
) {
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        if (s[i] <= 0.0) return false;
        log_prod += 2.0 * alphas[i] * logsafe(s[i]);
    }
    double norm2w = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        norm2w += s[dim1 + i] * s[dim1 + i];
    }
    return exp(log_prod) - norm2w > 0.0;
}

/**
 * @brief Device function to check if z is dual feasible for generalized power cone
 *
 * GenPower dual: ∏ (z_i/αi)^{2αi} ≥ ||w||², all z_i > 0
 */
__device__ inline bool is_genpow_dual_feasible(
    const double* z, int64_t dim1, int64_t dim2,
    const double* alphas
) {
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        if (z[i] <= 0.0) return false;
        log_prod += 2.0 * alphas[i] * logsafe(z[i] / alphas[i]);
    }
    double norm2w = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        norm2w += z[dim1 + i] * z[dim1 + i];
    }
    return exp(log_prod) - norm2w > 0.0;
}

/**
 * @brief Check primal feasibility at q + alpha * dq without materializing the sum
 */
__device__ inline bool is_genpow_primal_feasible_with_step(
    const double* q, const double* dq, double alpha,
    int64_t dim1, int64_t dim2, const double* alphas
) {
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        double val = q[i] + alpha * dq[i];
        if (val <= 0.0) return false;
        log_prod += 2.0 * alphas[i] * logsafe(val);
    }
    double norm2w = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        double val = q[dim1 + i] + alpha * dq[dim1 + i];
        norm2w += val * val;
    }
    return exp(log_prod) - norm2w > 0.0;
}

/**
 * @brief Check dual feasibility at q + alpha * dq without materializing the sum
 */
__device__ inline bool is_genpow_dual_feasible_with_step(
    const double* q, const double* dq, double alpha,
    int64_t dim1, int64_t dim2, const double* alphas
) {
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        double val = q[i] + alpha * dq[i];
        if (val <= 0.0) return false;
        log_prod += 2.0 * alphas[i] * logsafe(val / alphas[i]);
    }
    double norm2w = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        double val = q[dim1 + i] + alpha * dq[dim1 + i];
        norm2w += val * val;
    }
    return exp(log_prod) - norm2w > 0.0;
}

/**
 * @brief Variable-dimension backtracking line search for generalized power cone
 *
 * Zero stack allocation — feasibility is computed inline at q + alpha * dq.
 */
__device__ double backtrack_search_genpow(
    const double* q, const double* dq,
    int64_t dim1, int64_t dim2,
    const double* alphas,
    double alpha_init, double alpha_min, double step,
    bool is_primal
) {
    double alpha = alpha_init;
    constexpr int max_iters = 100;

    for (int iter = 0; iter < max_iters; iter++) {
        bool feasible = is_primal
            ? is_genpow_primal_feasible_with_step(q, dq, alpha, dim1, dim2, alphas)
            : is_genpow_dual_feasible_with_step(q, dq, alpha, dim1, dim2, alphas);

        if (feasible) break;

        alpha *= step;
        if (alpha < alpha_min) {
            alpha = 0.0;
            break;
        }
    }
    return alpha;
}

/**
 * @brief Device function to perform backtracking line search
 *
 * Find maximum α such that q + α*dq stays in the cone.
 * Uses backtracking from α_init with multiplicative step.
 */
template<typename FeasibilityCheck>
__device__ double backtrack_search(
    double q0, double q1, double q2,
    double dq0, double dq1, double dq2,
    double alpha_init,
    double alpha_min,
    double step,
    FeasibilityCheck is_feasible
) {
    double alpha = alpha_init;
    constexpr int max_iters = 100;  // Prevent infinite loops
    int iter = 0;

    while (iter < max_iters) {
        // work = q + α*dq
        double w0 = q0 + alpha * dq0;
        double w1 = q1 + alpha * dq1;
        double w2 = q2 + alpha * dq2;

        if (is_feasible(w0, w1, w2)) {
            break;
        }

        alpha *= step;
        if (alpha < alpha_min) {
            alpha = 0.0;
            break;
        }
        iter++;
    }

    return alpha;
}

/**
 * @brief Device function to compute SOC residual: x₀² - ||x₁||²
 * Uses (x₀ - ||x₁||)(x₀ + ||x₁||) to avoid catastrophic cancellation.
 * Variable-dim version reading from global memory.
 */
__device__ inline double soc_residual_varlen(const double* x, int64_t dim) {
    double x0 = x[0];
    double tail_sq = 0.0;
    for (int64_t i = 1; i < dim; i++) {
        tail_sq += x[i] * x[i];
    }
    double tail_norm = sqrt(tail_sq);
    return (x0 - tail_norm) * (x0 + tail_norm);
}


/**
 * @brief Device function to calculate step length for one SOC component (variable dim)
 *
 * Finds maximum α such that x + α*dx remains in second-order cone.
 * Uses quadratic formula on boundary equation: ||x₁+αy₁||² = (x₀ + αy₀)²
 */
__device__ double step_length_soc_component_varlen(
    const double* x, const double* dx,
    int64_t dim, double alpha_max
) {
    double x0 = x[0];
    double dx0 = dx[0];

    // Upper bound the step length by the maximum allowable
    // step length for the scalar part of the cone
    if (x0 >= 0.0 && dx0 < 0.0) {
        alpha_max = fmin(alpha_max, -x0 / dx0);
    }

    // Quadratic coefficients for ||x₁+αy₁||^2 = (x₀ + αy₀)^2
    const double two = 2.0;
    const double four = 4.0;

    // a = dx0^2 - ||dx[1:]||^2
    double a = dx0 * dx0;
    double b_inner = 0.0;
    double c_val = x0 * x0;
    for (int64_t i = 1; i < dim; i++) {
        a -= dx[i] * dx[i];
        b_inner += x[i] * dx[i];
        c_val -= x[i] * x[i];
    }
    double b = two * (x0 * dx0 - b_inner);
    double c = fmax(0.0, c_val);
    double d = b * b - four * a * c;

    if (c < 0.0) {
        // This should never be reachable since c ≥ 0 above
        // panic!("starting point of line search not in SOC");
        return 0.0;  // Conservative fallback
    }

    if ((a > 0.0 && b > 0.0) || d < 0.0) {
        // all negative roots / complex root pair
        // -> infinite step length
        return alpha_max;
    } else if (a == 0.0) {
        // Edge case with only one root.  This corresponds to
        // the case where the search direction is exactly on the
        // cone boundary.   The root should be -c/b, but b can't
        // be negative since both (x,y) are in the cone and it is
        // self dual, so <x,y> >= 0 necessarily.
        return alpha_max;
    } else if (c == 0.0) {
        // Edge case with one of the roots at 0.   This corresponds
        // to the case where the initial point is exactly on the
        // cone boundary.  The other root is -b/a.   If the search
        // direction is in the cone, then a >= 0 and b can't be
        // negative due to self-duality.  If a < 0, then the
        // direction is outside the cone and b can't be positive.
        // Either way, step length is determined by whether or not
        // the search direction is in the cone.
        return (a >= 0.0) ? alpha_max : 0.0;
    }

    // If we got this far then we need to calculate a pair
    // of real roots and choose the smallest positive one.
    // We need to be cautious about cancellations though.
    // See §1.4: Goldberg, ACM Computing Surveys, 1991
    // https://dl.acm.org/doi/pdf/10.1145/103162.103163

    double t;
    if (b >= 0.0) {
        t = -b - sqrt(d);
    } else {
        t = -b + sqrt(d);
    }

    double r1 = (two * c) / t;
    double r2 = t / (two * a);

    // Return the minimum positive root, up to αmax
    double r1_pos = (r1 < 0.0) ? INFINITY : r1;
    double r2_pos = (r2 < 0.0) ? INFINITY : r2;

    return fmin(alpha_max, fmin(r1_pos, r2_pos));
}

/**
 * @brief Simple kernel to clamp alpha values to a maximum ceiling
 */
__global__ void clamp_alpha_ceiling_kernel(
    double* __restrict__ alpha_z,
    double* __restrict__ alpha_s,
    double ceiling,
    int64_t batchSize
) {
    int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batchSize) {
        alpha_z[idx] = (alpha_z[idx] < ceiling) ? alpha_z[idx] : ceiling;
        alpha_s[idx] = (alpha_s[idx] < ceiling) ? alpha_s[idx] : ceiling;
    }
}

/**
 * @brief Fused kernel for step length calculation for symmetric cones (nonneg + SOC)
 */
__global__ void step_length_symmetric_kernel(
    double* __restrict__ alpha_z,
    double* __restrict__ alpha_s,
    const double* __restrict__ dz,
    const double* __restrict__ ds,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset_nonneg, int64_t numNonnegCones,
    int64_t offset_soc, int64_t numSocCones,
    int64_t m, int64_t batchSize,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    // Nonnegative cones: α ≤ -x/dx for negative dx
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t global_idx = batch_idx * m + offset_nonneg + i;

        double zi = z[global_idx];
        double dzi = dz[global_idx];
        double si = s[global_idx];
        double dsi = ds[global_idx];

        if (dzi < 0.0) {
            double candidate = -zi / dzi;
            atomicMinDouble(&alpha_z[batch_idx], candidate);
        }
        if (dsi < 0.0) {
            double candidate = -si / dsi;
            atomicMinDouble(&alpha_s[batch_idx], candidate);
        }
    }

    // SOC cones: quadratic formula for cone boundary (variable dim)
    for (int64_t i = threadIdx.x; i < numSocCones; i += blockDim.x) {
        int64_t dim = d_soc_dims[i];
        int64_t sz_off = d_soc_sz_offsets[i];
        int64_t base_idx = batch_idx * m + offset_soc + sz_off;

        double alpha_z_local = step_length_soc_component_varlen(
            &z[base_idx], &dz[base_idx], dim, alpha_z[batch_idx]);
        atomicMinDouble(&alpha_z[batch_idx], alpha_z_local);

        double alpha_s_local = step_length_soc_component_varlen(
            &s[base_idx], &ds[base_idx], dim, alpha_s[batch_idx]);
        atomicMinDouble(&alpha_s[batch_idx], alpha_s_local);
    }
}

/**
 * @brief Fused kernel for step length calculation for asymmetric cones (exp + power)
 *
 * This kernel also applies the ceiling clamp at the start before processing.
 */
__global__ void step_length_asymmetric_kernel(
    double* __restrict__ alpha_z,
    double* __restrict__ alpha_s,
    const double* __restrict__ dz,
    const double* __restrict__ ds,
    const double* __restrict__ z,
    const double* __restrict__ s,
    const double* __restrict__ power_cone_alpha,
    double ceiling,
    double alpha_min,
    double step,
    int64_t offset_exp, int64_t numExpCones,
    int64_t offset_power, int64_t numPowerCones,
    // GenPowerCone params
    int64_t offset_genpow, int64_t numGenPowerCones,
    const double* __restrict__ d_genPowerAlphas,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    int64_t m, int64_t batchSize
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    // Apply ceiling clamp (thread 0 only, before other threads proceed)
    if (threadIdx.x == 0) {
        double az = alpha_z[batch_idx];
        double as = alpha_s[batch_idx];
        alpha_z[batch_idx] = (az < ceiling) ? az : ceiling;
        alpha_s[batch_idx] = (as < ceiling) ? as : ceiling;
    }
    __syncthreads();

    // Exponential cones: backtracking line search
    for (int64_t i = threadIdx.x; i < numExpCones; i += blockDim.x) {
        int64_t base_idx = batch_idx * m + offset_exp + i * 3;

        double z0 = z[base_idx + 0];
        double z1 = z[base_idx + 1];
        double z2 = z[base_idx + 2];
        double dz0 = dz[base_idx + 0];
        double dz1 = dz[base_idx + 1];
        double dz2 = dz[base_idx + 2];

        auto is_dual_feasible = [](double v0, double v1, double v2) {
            return is_exp_dual_feasible(v0, v1, v2);
        };
        double alpha_z_local = backtrack_search(
            z0, z1, z2, dz0, dz1, dz2,
            alpha_z[batch_idx], alpha_min, step, is_dual_feasible
        );
        atomicMinDouble(&alpha_z[batch_idx], alpha_z_local);

        double s0 = s[base_idx + 0];
        double s1 = s[base_idx + 1];
        double s2 = s[base_idx + 2];
        double ds0 = ds[base_idx + 0];
        double ds1 = ds[base_idx + 1];
        double ds2 = ds[base_idx + 2];

        auto is_primal_feasible = [](double v0, double v1, double v2) {
            return is_exp_primal_feasible(v0, v1, v2);
        };
        double alpha_s_local = backtrack_search(
            s0, s1, s2, ds0, ds1, ds2,
            alpha_s[batch_idx], alpha_min, step, is_primal_feasible
        );
        atomicMinDouble(&alpha_s[batch_idx], alpha_s_local);
    }

    // Power cones: backtracking line search
    for (int64_t i = threadIdx.x; i < numPowerCones; i += blockDim.x) {
        int64_t base_idx = batch_idx * m + offset_power + i * 3;
        double alpha_param = power_cone_alpha[i];

        double z0 = z[base_idx + 0];
        double z1 = z[base_idx + 1];
        double z2 = z[base_idx + 2];
        double dz0 = dz[base_idx + 0];
        double dz1 = dz[base_idx + 1];
        double dz2 = dz[base_idx + 2];

        auto is_dual_feasible_power = [alpha_param](double v0, double v1, double v2) {
            return is_pow_dual_feasible(v0, v1, v2, alpha_param);
        };
        double alpha_z_local = backtrack_search(
            z0, z1, z2, dz0, dz1, dz2,
            alpha_z[batch_idx], alpha_min, step, is_dual_feasible_power
        );
        atomicMinDouble(&alpha_z[batch_idx], alpha_z_local);

        double s0 = s[base_idx + 0];
        double s1 = s[base_idx + 1];
        double s2 = s[base_idx + 2];
        double ds0 = ds[base_idx + 0];
        double ds1 = ds[base_idx + 1];
        double ds2 = ds[base_idx + 2];

        auto is_primal_feasible_power = [alpha_param](double v0, double v1, double v2) {
            return is_pow_primal_feasible(v0, v1, v2, alpha_param);
        };
        double alpha_s_local = backtrack_search(
            s0, s1, s2, ds0, ds1, ds2,
            alpha_s[batch_idx], alpha_min, step, is_primal_feasible_power
        );
        atomicMinDouble(&alpha_s[batch_idx], alpha_s_local);
    }

    // GenPowerCone: variable-dim backtracking line search
    for (int64_t i = threadIdx.x; i < numGenPowerCones; i += blockDim.x) {
        int64_t dim1 = d_genPowerDim1s[i];
        int64_t dim2 = d_genPowerDim2s[i];
        int64_t sz_off = d_genPowerSzOffsets[i];
        int64_t alpha_off = d_genPowerAlphaOffsets[i];
        int64_t base_idx = batch_idx * m + offset_genpow + sz_off;
        const double* cone_alphas = &d_genPowerAlphas[alpha_off];

        double alpha_z_local = backtrack_search_genpow(
            &z[base_idx], &dz[base_idx], dim1, dim2, cone_alphas,
            alpha_z[batch_idx], alpha_min, step, false
        );
        atomicMinDouble(&alpha_z[batch_idx], alpha_z_local);

        double alpha_s_local = backtrack_search_genpow(
            &s[base_idx], &ds[base_idx], dim1, dim2, cone_alphas,
            alpha_s[batch_idx], alpha_min, step, true
        );
        atomicMinDouble(&alpha_s[batch_idx], alpha_s_local);
    }
}


void Cones::step_length(
    const BatchedVector& dz,
    const BatchedVector& ds,
    const BatchedVector& z,
    const BatchedVector& s,
    const BatchedVector& alpha_max,
    BatchedVector& alpha_z,
    BatchedVector& alpha_s,
    double backtrack_step,
    double min_step_length,
    cudaStream_t stream
) {
    // alpha_z and alpha_s are pre-initialized by the caller
    // (calc_step_length_tau_kappa_init_kernel writes them along with alpha)

    int64_t m = totalConstraints();

    // Calculate offsets for each cone type
    int64_t offset_nonneg = numZeroCones;
    int64_t offset_soc = numZeroCones + numNonnegCones;
    int64_t offset_exp = numZeroCones + numNonnegCones + totalSocDim;
    int64_t offset_power = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;

    // Fused symmetric cones (nonneg + SOC)
    if (numNonnegCones > 0 || numSocCones > 0) {
        int64_t maxCones = max(numNonnegCones, numSocCones);
        int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
        if (threadsPerBlock < 32) threadsPerBlock = 32;

        MOREAU_KERNEL_LAUNCH(step_length_symmetric_kernel, batchSize, threadsPerBlock, 0, stream,
            alpha_z.data(), alpha_s.data(), dz.data(), ds.data(), z.data(), s.data(),
            offset_nonneg, numNonnegCones,
            offset_soc, numSocCones,
            m, batchSize,
            d_soc_dims, d_soc_sz_offsets
        );
    }

    // Fused asymmetric cones (exp + power + genpow) with inlined ceiling clamp
    int64_t offset_genpow = offset_power + numPowerCones * 3;
    if (numExpCones > 0 || numPowerCones > 0 || numGenPowerCones > 0) {
        double alpha_min = min_step_length;
        double step = backtrack_step;
        double ceiling = 1.0 - std::sqrt(std::numeric_limits<double>::epsilon());

        int64_t maxCones = max(max(numExpCones, numPowerCones), numGenPowerCones);
        int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
        if (threadsPerBlock < 32) threadsPerBlock = 32;

        MOREAU_KERNEL_LAUNCH(step_length_asymmetric_kernel, batchSize, threadsPerBlock, 0, stream,
            alpha_z.data(), alpha_s.data(), dz.data(), ds.data(), z.data(), s.data(),
            d_powerAlphas, ceiling, alpha_min, step,
            offset_exp, numExpCones,
            offset_power, numPowerCones,
            offset_genpow, numGenPowerCones,
            d_genPowerAlphas, d_genPowerDim1s, d_genPowerDim2s,
            d_genPowerOffsets, d_genPowerAlphaOffsets, d_genPowerSzOffsets,
            m, batchSize
        );
    }

    // PSD cones: eigendecomp-based step length
    if (numPsdCones > 0) {
        int64_t psd_offset = numZeroCones + numNonnegCones + totalSocDim
                           + numExpCones * 3 + numPowerCones * 3;
        psd_step_length(*this, dz.data(), ds.data(), z.data(), s.data(),
                       alpha_z.data(), alpha_s.data(),
                       psd_offset, m, stream);
    }

    // Note: Clarabel's compositecone.rs step_length() returns (α, α) where α = min(αz, αs)
    // across all cones. However, we compute alpha_z and alpha_s separately with atomicMinDouble,
    // and the caller (Variables::calc_step_length) takes min(alpha_z, alpha_s) via elementwise_min_kernel.
    // So we do NOT overwrite alpha_s here - let the caller take the proper minimum.
}

/**
 * @brief Fused CUDA kernel for combined DS shift for all cone types
 *
 * Computes the combined DS shift for all cone types in a single kernel launch.
 * This replaces 4-5 separate kernel launches with one fused kernel.
 */
__global__ void combined_ds_shift_all_cones_kernel(
    double* __restrict__ shift,
    double* __restrict__ step_z,
    double* __restrict__ step_s,
    const double* __restrict__ sigma_mu,
    // Zero cone parameters
    int64_t offset_zero, int64_t numZeroCones,
    // Nonneg cone parameters
    int64_t offset_nonneg, int64_t numNonnegCones,
    // SOC cone parameters
    const double* __restrict__ soc_w, const double* __restrict__ soc_eta,
    int64_t offset_soc, int64_t numSocCones,
    // Exp cone parameters
    const double* __restrict__ exp_grad, const double* __restrict__ exp_H_dual, const double* __restrict__ exp_z,
    int64_t offset_exp, int64_t numExpCones,
    // Power cone parameters
    const double* __restrict__ power_grad, const double* __restrict__ power_H_dual, const double* __restrict__ power_z, const double* __restrict__ power_alphas,
    int64_t offset_power, int64_t numPowerCones,
    // GenPowerCone parameters
    const double* __restrict__ genpow_grad,
    int64_t offset_genpow, int64_t numGenPowerCones,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerGradOffsets,  // same as d_genPowerOffsets for grad indexing
    const int64_t* __restrict__ d_genPowerSzOffsets,
    int64_t totalGenPowerDim,
    // GenPowerCone η-correction parameters (closed-form 3rd-order Mehrotra
    // term + ∞-norm magnitude cap; see CPU `higher_correction` for the math).
    const double* __restrict__ genpow_z,           // (B, totalGenPowerDim) z at scaling point
    const double* __restrict__ genpow_p,           // (B, totalGenPowerDim) rank-3 axis p
    const double* __restrict__ genpow_q,           // (B, totalGenPowerAlphas) rank-3 axis q (i<dim1)
    const double* __restrict__ genpow_r_axis,      // (B, totalGenPowerDim2) axis r (i≥dim1)
    const double* __restrict__ genpow_d1,          // (B, totalGenPowerAlphas) diagonal d1
    const double* __restrict__ genpow_d2,          // (B, numGenPowerCones) scalar d2 per cone
    const double* __restrict__ d_genPowerAlphas,   // (totalGenPowerAlphas) flattened alphas
    const int64_t* __restrict__ d_genPowerAlphaOffsets, // (numGenPowerCones+1) prefix-sum of dim1
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerDim2,
    double* __restrict__ genpow_correction_u,      // (B, totalGenPowerDim) workspace
    double* __restrict__ genpow_correction_gpsi,   // (B, totalGenPowerDim) workspace
    // Common parameters
    int64_t m, int64_t batchSize,
    // Variable-dim SOC params
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    double sigma_mu_val = sigma_mu[batch_idx];

    // Zero cones: shift = 0
    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_zero + i;
        shift[idx] = 0.0;
    }

    // Nonneg cones: shift = step_s * step_z - σμ
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t idx = batch_idx * m + offset_nonneg + i;
        shift[idx] = step_s[idx] * step_z[idx] - sigma_mu_val;
    }

    // SOC cones: W^-1(step_s) ∘ W(step_z) - σμe
    for (int64_t i = threadIdx.x; i < numSocCones; i += blockDim.x) {
        int64_t dim = d_soc_dims[i];
        int64_t soc_off = d_soc_offsets[i];
        int64_t sz_off = d_soc_sz_offsets[i];
        int64_t base_idx = batch_idx * m + offset_soc + sz_off;
        int64_t soc_base = batch_idx * totalSocDim + soc_off;

        double w0 = soc_w[soc_base];
        double eta = soc_eta[batch_idx * numSocCones + i];
        double inv_eta = 1.0 / eta;

        // Compute dot products: w[1:] . dz[1:] and w[1:] . ds[1:]
        double zeta_z = 0.0;
        double zeta_s = 0.0;
        for (int64_t j = 1; j < dim; j++) {
            zeta_z += soc_w[soc_base + j] * step_z[base_idx + j];
            zeta_s += soc_w[soc_base + j] * step_s[base_idx + j];
        }

        // Compute W(dz)
        double dz0 = step_z[base_idx];
        double c_z = dz0 + zeta_z / (1.0 + w0);
        double Wdz0 = eta * (w0 * dz0 + zeta_z);

        // Compute W^-1(ds)
        // Note: (w0 * ds0 - zeta_s) can suffer catastrophic cancellation when both
        // terms are large and nearly equal. This follows Clarabel's formulation;
        // reformulating requires reworking the NT scaling math.
        double ds0 = step_s[base_idx];
        double c_s = -ds0 + zeta_s / (1.0 + w0);
        double Winv_ds0 = inv_eta * (w0 * ds0 - zeta_s);

        // Compute circle product dot and write results
        double dot = Winv_ds0 * Wdz0;
        for (int64_t j = 1; j < dim; j++) {
            double Wdz_j = eta * (soc_w[soc_base + j] * c_z + step_z[base_idx + j]);
            double Winv_ds_j = inv_eta * (soc_w[soc_base + j] * c_s + step_s[base_idx + j]);
            dot += Winv_ds_j * Wdz_j;
            // Circle product tail: circ[j] = Winv_ds[0]*Wdz[j] + Wdz[0]*Winv_ds[j]
            shift[base_idx + j] = Winv_ds0 * Wdz_j + Wdz0 * Winv_ds_j;
            // Write transformed step_z and step_s in place
            step_z[base_idx + j] = Wdz_j;
            step_s[base_idx + j] = Winv_ds_j;
        }

        // Circle product head: circ[0] = dot - σμ
        shift[base_idx] = dot - sigma_mu_val;

        // Write transformed step_z[0] and step_s[0]
        step_z[base_idx] = Wdz0;
        step_s[base_idx] = Winv_ds0;
    }

    // Exp cones: shift = grad * σμ - eta (with 3rd order correction)
    for (int64_t i = threadIdx.x; i < numExpCones; i += blockDim.x) {
        int64_t grad_base = batch_idx * (numExpCones * 3) + i * 3;
        int64_t H_base = batch_idx * (numExpCones * 6) + i * 6;
        int64_t z_base = batch_idx * (numExpCones * 3) + i * 3;
        int64_t step_base = batch_idx * m + offset_exp + i * 3;

        // Load data
        double grad[3], H[6], z[3], ds[3], dz[3];
        grad[0] = exp_grad[grad_base];
        grad[1] = exp_grad[grad_base + 1];
        grad[2] = exp_grad[grad_base + 2];

        H[0] = exp_H_dual[H_base];
        H[1] = exp_H_dual[H_base + 1];
        H[2] = exp_H_dual[H_base + 2];
        H[3] = exp_H_dual[H_base + 3];
        H[4] = exp_H_dual[H_base + 4];
        H[5] = exp_H_dual[H_base + 5];

        z[0] = exp_z[z_base];
        z[1] = exp_z[z_base + 1];
        z[2] = exp_z[z_base + 2];

        ds[0] = step_s[step_base];
        ds[1] = step_s[step_base + 1];
        ds[2] = step_s[step_base + 2];

        dz[0] = step_z[step_base];
        dz[1] = step_z[step_base + 1];
        dz[2] = step_z[step_base + 2];

        // Compute eta via higher_correction
        double eta[3] = {0.0, 0.0, 0.0};
        double u[3] = {0.0, 0.0, 0.0};

        // Cholesky factorization and solve H*u = ds
        double L00 = sqrt(H[0]);
        if (L00 > 1e-12) {
            double L10 = H[1] / L00;
            double L20 = H[2] / L00;
            double L11 = sqrt(H[3] - L10*L10);
            if (L11 > 1e-12) {
                double L21 = (H[4] - L20*L10) / L11;
                double L22 = sqrt(H[5] - L20*L20 - L21*L21);
                if (L22 > 1e-12) {
                    // Forward substitution
                    double y0 = ds[0] / L00;
                    double y1 = (ds[1] - L10*y0) / L11;
                    double y2 = (ds[2] - L20*y0 - L21*y1) / L22;

                    // Backward substitution
                    u[2] = y2 / L22;
                    u[1] = (y1 - L21*u[2]) / L11;
                    u[0] = (y0 - L10*u[1] - L20*u[2]) / L00;

                    // Compute eta for exp cone
                    double psi_grad[3];
                    psi_grad[1] = 1.0;
                    psi_grad[2] = -z[0] / z[2];
                    psi_grad[0] = log(fabs(psi_grad[2]) + 1e-16);

                    double psi = z[0] * psi_grad[0] - z[0] + z[1];
                    double dotPsiU = u[0]*psi_grad[0] + u[1]*psi_grad[1] + u[2]*psi_grad[2];
                    double dotPsiV = dz[0]*psi_grad[0] + dz[1]*psi_grad[1] + dz[2]*psi_grad[2];

                    double coef = ((u[0]*(dz[0]/z[0] - dz[2]/z[2]) + u[2]*(z[0]*dz[2]/z[2] - dz[0])/z[2])*psi
                        - 2.0*dotPsiU*dotPsiV) / (psi*psi*psi);

                    eta[0] = psi_grad[0] * coef;
                    eta[1] = psi_grad[1] * coef;
                    eta[2] = psi_grad[2] * coef;

                    double inv_psi2 = 1.0 / (psi*psi);
                    eta[0] += (1.0/psi - 2.0/z[0])*u[0]*dz[0]/(z[0]*z[0])
                        - u[2]*dz[2]/(z[2]*z[2])/psi
                        + dotPsiU*inv_psi2*(dz[0]/z[0] - dz[2]/z[2])
                        + dotPsiV*inv_psi2*(u[0]/z[0] - u[2]/z[2]);
                    eta[2] += 2.0*(z[0]/psi - 1.0)*u[2]*dz[2]/(z[2]*z[2]*z[2])
                        - (u[2]*dz[0] + u[0]*dz[2])/(z[2]*z[2])/psi
                        + dotPsiU*inv_psi2*(z[0]*dz[2]/(z[2]*z[2]) - dz[0]/z[2])
                        + dotPsiV*inv_psi2*(z[0]*u[2]/(z[2]*z[2]) - u[0]/z[2]);

                    eta[0] *= 0.5;
                    eta[1] *= 0.5;
                    eta[2] *= 0.5;
                }
            }
        }

        // shift = grad * σμ - eta
        shift[step_base] = grad[0] * sigma_mu_val - eta[0];
        shift[step_base + 1] = grad[1] * sigma_mu_val - eta[1];
        shift[step_base + 2] = grad[2] * sigma_mu_val - eta[2];
    }

    // Power cones: shift = grad * σμ - eta (with 3rd order correction)
    for (int64_t i = threadIdx.x; i < numPowerCones; i += blockDim.x) {
        int64_t grad_base = batch_idx * (numPowerCones * 3) + i * 3;
        int64_t H_base = batch_idx * (numPowerCones * 6) + i * 6;
        int64_t z_base = batch_idx * (numPowerCones * 3) + i * 3;
        int64_t step_base = batch_idx * m + offset_power + i * 3;

        double alpha = power_alphas[i];

        // Load data
        double grad[3], H[6], z[3], ds[3], dz[3];
        grad[0] = power_grad[grad_base];
        grad[1] = power_grad[grad_base + 1];
        grad[2] = power_grad[grad_base + 2];

        H[0] = power_H_dual[H_base];
        H[1] = power_H_dual[H_base + 1];
        H[2] = power_H_dual[H_base + 2];
        H[3] = power_H_dual[H_base + 3];
        H[4] = power_H_dual[H_base + 4];
        H[5] = power_H_dual[H_base + 5];

        z[0] = power_z[z_base];
        z[1] = power_z[z_base + 1];
        z[2] = power_z[z_base + 2];

        ds[0] = step_s[step_base];
        ds[1] = step_s[step_base + 1];
        ds[2] = step_s[step_base + 2];

        dz[0] = step_z[step_base];
        dz[1] = step_z[step_base + 1];
        dz[2] = step_z[step_base + 2];

        // Compute eta via higher_correction (power cone version)
        double eta[3] = {0.0, 0.0, 0.0};
        double u[3] = {0.0, 0.0, 0.0};

        // Cholesky factorization and solve H*u = ds
        double L00 = sqrt(H[0]);
        if (L00 > 1e-12) {
            double L10 = H[1] / L00;
            double L20 = H[2] / L00;
            double L11 = sqrt(H[3] - L10*L10);
            if (L11 > 1e-12) {
                double L21 = (H[4] - L20*L10) / L11;
                double L22 = sqrt(H[5] - L20*L20 - L21*L21);
                if (L22 > 1e-12) {
                    // Forward substitution
                    double y0 = ds[0] / L00;
                    double y1 = (ds[1] - L10*y0) / L11;
                    double y2 = (ds[2] - L20*y0 - L21*y1) / L22;

                    // Backward substitution
                    u[2] = y2 / L22;
                    u[1] = (y1 - L21*u[2]) / L11;
                    u[0] = (y0 - L10*u[1] - L20*u[2]) / L00;

                    // Compute eta for power cone
                    double phi = pow(z[0]/alpha, 2.0*alpha) * pow(z[1]/(1.0-alpha), 2.0-2.0*alpha);
                    // Avoid catastrophic cancellation: phi - z[2]² = (√φ - |z₂|)(√φ + |z₂|)
                    double sqrt_phi = sqrt(phi);
                    double abs_z2 = fabs(z[2]);
                    double psi = (sqrt_phi - abs_z2) * (sqrt_phi + abs_z2);

                    // Gradient of psi
                    double psi_grad[3];
                    psi_grad[0] = 2.0*alpha*phi/z[0];
                    psi_grad[1] = 2.0*(1.0-alpha)*phi/z[1];
                    psi_grad[2] = -2.0*z[2];

                    double dotPsiU = u[0]*psi_grad[0] + u[1]*psi_grad[1] + u[2]*psi_grad[2];
                    double dotPsiV = dz[0]*psi_grad[0] + dz[1]*psi_grad[1] + dz[2]*psi_grad[2];

                    // Hessian of psi
                    double Hpsi[6];
                    Hpsi[0] = 2.0*alpha*(2.0*alpha-1.0)*phi/(z[0]*z[0]);
                    Hpsi[1] = 4.0*alpha*(1.0-alpha)*phi/(z[0]*z[1]);
                    Hpsi[2] = 0.0;
                    Hpsi[3] = 2.0*(1.0-alpha)*(1.0-2.0*alpha)*phi/(z[1]*z[1]);
                    Hpsi[4] = 0.0;
                    Hpsi[5] = -2.0;

                    // Hpsi * dz
                    double HpsiV[3];
                    HpsiV[0] = Hpsi[0]*dz[0] + Hpsi[1]*dz[1] + Hpsi[2]*dz[2];
                    HpsiV[1] = Hpsi[1]*dz[0] + Hpsi[3]*dz[1] + Hpsi[4]*dz[2];
                    HpsiV[2] = Hpsi[2]*dz[0] + Hpsi[4]*dz[1] + Hpsi[5]*dz[2];

                    double coef = (u[0]*HpsiV[0] + u[1]*HpsiV[1] + u[2]*HpsiV[2])*psi - 2.0*dotPsiU*dotPsiV;
                    coef /= (psi*psi*psi);

                    double coef2 = 4.0 * alpha * (2.0*alpha - 1.0) * (1.0 - alpha) * phi
                        * (u[0]/z[0] - u[1]/z[1]) * (dz[0]/z[0] - dz[1]/z[1]) / psi;

                    double inv_psi2 = 1.0/(psi*psi);

                    eta[0] = coef * psi_grad[0]
                        - 2.0 * (1.0 - alpha) * u[0] * dz[0] / (z[0] * z[0] * z[0])
                        + coef2 / z[0]
                        + HpsiV[0] * dotPsiU * inv_psi2;

                    eta[1] = coef * psi_grad[1]
                        - 2.0 * alpha * u[1] * dz[1] / (z[1] * z[1] * z[1])
                        - coef2 / z[1]
                        + HpsiV[1] * dotPsiU * inv_psi2;

                    eta[2] = coef * psi_grad[2] + HpsiV[2] * dotPsiU * inv_psi2;

                    // Hpsi * u
                    double HpsiU[3];
                    HpsiU[0] = Hpsi[0]*u[0] + Hpsi[1]*u[1] + Hpsi[2]*u[2];
                    HpsiU[1] = Hpsi[1]*u[0] + Hpsi[3]*u[1] + Hpsi[4]*u[2];
                    HpsiU[2] = Hpsi[2]*u[0] + Hpsi[4]*u[1] + Hpsi[5]*u[2];

                    eta[0] = (eta[0] + HpsiU[0] * dotPsiV * inv_psi2) * 0.5;
                    eta[1] = (eta[1] + HpsiU[1] * dotPsiV * inv_psi2) * 0.5;
                    eta[2] = (eta[2] + HpsiU[2] * dotPsiV * inv_psi2) * 0.5;
                }
            }
        }

        // shift = grad * σμ - eta
        shift[step_base] = grad[0] * sigma_mu_val - eta[0];
        shift[step_base + 1] = grad[1] * sigma_mu_val - eta[1];
        shift[step_base + 2] = grad[2] * sigma_mu_val - eta[2];
    }

    // GenPowerCone: shift = grad · σμ − η, where η is the closed-form
    // 3rd-order Mehrotra correction. See `apply_eta_cap` and CPU
    // `higher_correction` (genpowcone.rs) for the full derivation. One
    // thread per cone; rank-3 SMW solve avoids the O(dim²) Hessian.
    for (int64_t i = threadIdx.x; i < numGenPowerCones; i += blockDim.x) {
        int64_t gp_off       = d_genPowerOffsets[i];
        int64_t alpha_off    = d_genPowerAlphaOffsets[i];
        int64_t r_off        = gp_off - alpha_off;     // dim2 prefix sum
        int64_t sz_off       = d_genPowerSzOffsets[i];
        int64_t dim1         = d_genPowerDim1s[i];
        int64_t dim2         = d_genPowerDim2s[i];
        int64_t dim          = dim1 + dim2;
        int64_t grad_base    = batch_idx * totalGenPowerDim   + gp_off;
        int64_t z_base       = batch_idx * totalGenPowerDim   + gp_off;
        int64_t p_base       = batch_idx * totalGenPowerDim   + gp_off;
        int64_t q_base       = batch_idx * totalGenPowerAlphas + alpha_off;
        int64_t r_base       = batch_idx * totalGenPowerDim2  + r_off;
        int64_t d1_base      = batch_idx * totalGenPowerAlphas + alpha_off;
        double d2_val        = genpow_d2[batch_idx * numGenPowerCones + i];
        int64_t step_base    = batch_idx * m + offset_genpow + sz_off;
        int64_t u_base       = batch_idx * totalGenPowerDim   + gp_off;
        int64_t gpsi_base    = batch_idx * totalGenPowerDim   + gp_off;
        const double* alphas = &d_genPowerAlphas[alpha_off];

        // Step 1: solve H_dual · u = ds via rank-3 Sherman-Morrison-Woodbury.
        //   H_dual = D + U Σ U',  U = [p, q_full, r_full],  Σ = diag(1, −1, −1)
        //   D = diag(d1, d2 I_{n2})
        //   u = D⁻¹ ds − D⁻¹ U · M⁻¹ · U' D⁻¹ ds,  M = Σ⁻¹ + U' D⁻¹ U  (3×3)
        //
        // Off-diagonal `m12` of M is zero by disjoint q/r supports.
        double* u_arr = &genpow_correction_u[u_base];
        double b0 = 0.0, b1 = 0.0, b2 = 0.0;
        for (int64_t j = 0; j < dim1; ++j) {
            double dsj = step_s[step_base + j];
            double tmp = dsj / genpow_d1[d1_base + j];   // (D⁻¹ ds)[j]
            u_arr[j] = tmp;
            b0 += genpow_p[p_base + j] * tmp;
            b1 += genpow_q[q_base + j] * tmp;
        }
        double inv_d2 = 1.0 / d2_val;
        for (int64_t j = 0; j < dim2; ++j) {
            double dsj = step_s[step_base + dim1 + j];
            double tmp = dsj * inv_d2;
            u_arr[dim1 + j] = tmp;
            b0 += genpow_p[p_base + dim1 + j] * tmp;
            b2 += genpow_r_axis[r_base + j] * tmp;
        }

        double m00 = 1.0, m11 = -1.0, m22 = -1.0, m01 = 0.0, m02 = 0.0;
        for (int64_t j = 0; j < dim1; ++j) {
            double inv = 1.0 / genpow_d1[d1_base + j];
            double pj  = genpow_p[p_base + j];
            double qj  = genpow_q[q_base + j];
            m00 += pj * pj * inv;
            m11 += qj * qj * inv;
            m01 += pj * qj * inv;
        }
        for (int64_t j = 0; j < dim2; ++j) {
            double pj = genpow_p[p_base + dim1 + j];
            double rj = genpow_r_axis[r_base + j];
            m00 += pj * pj * inv_d2;
            m22 += rj * rj * inv_d2;
            m02 += pj * rj * inv_d2;
        }

        double det = m00 * m11 * m22 - m01 * m01 * m22 - m02 * m02 * m11;
        if (fabs(det) < 1e-300) {
            // M near-singular: fall back to η = 0 (shift = grad·σμ).
            for (int64_t j = 0; j < dim; ++j) {
                shift[step_base + j] = genpow_grad[grad_base + j] * sigma_mu_val;
            }
            continue;
        }
        double inv_det = 1.0 / det;
        double y0 = ((m11 * m22) * b0 + (-m01 * m22) * b1 + (-m02 * m11) * b2) * inv_det;
        double y1 = ((-m01 * m22) * b0 + (m00 * m22 - m02 * m02) * b1 + (m01 * m02) * b2) * inv_det;
        double y2 = ((-m02 * m11) * b0 + (m01 * m02) * b1 + (m00 * m11 - m01 * m01) * b2) * inv_det;

        // u = (D⁻¹ ds) − D⁻¹ (p·y0 + q_full·y1 + r_full·y2)
        for (int64_t j = 0; j < dim1; ++j) {
            u_arr[j] -= (genpow_p[p_base + j] * y0 + genpow_q[q_base + j] * y1)
                       / genpow_d1[d1_base + j];
        }
        for (int64_t j = 0; j < dim2; ++j) {
            u_arr[dim1 + j] -= (genpow_p[p_base + dim1 + j] * y0
                              + genpow_r_axis[r_base + j]   * y2) * inv_d2;
        }

        // Step 2: φ, ψ, g_ψ, σ_u, σ_v, ρ_uv. ds plays the rôle of `ds`,
        // step_z (passed as `v` in the CPU code) plays the rôle of v.
        double log_phi = 0.0;
        for (int64_t j = 0; j < dim1; ++j) {
            double zj = genpow_z[z_base + j];
            log_phi += 2.0 * alphas[j] * log(zj / alphas[j]);
        }
        double phi = exp(log_phi);
        double norm2_w = 0.0;
        for (int64_t j = 0; j < dim2; ++j) {
            double wj = genpow_z[z_base + dim1 + j];
            norm2_w += wj * wj;
        }
        double psi = phi - norm2_w;
        const double sqrt_eps = 1.49011611938476562e-8;  // sqrt(DBL_EPSILON)

        // Near-boundary guard: ψ → 0 makes 1/ψ³ blow up. Drop η to zero.
        if (psi <= 0.0 || psi < sqrt_eps * phi) {
            for (int64_t j = 0; j < dim; ++j) {
                shift[step_base + j] = genpow_grad[grad_base + j] * sigma_mu_val;
            }
            continue;
        }

        double* gpsi = &genpow_correction_gpsi[gpsi_base];
        for (int64_t j = 0; j < dim1; ++j) {
            gpsi[j] = 2.0 * alphas[j] * phi / genpow_z[z_base + j];
        }
        for (int64_t j = 0; j < dim2; ++j) {
            gpsi[dim1 + j] = -2.0 * genpow_z[z_base + dim1 + j];
        }

        double sigma_u = 0.0, sigma_v = 0.0, rho_uv = 0.0;
        for (int64_t j = 0; j < dim1; ++j) {
            double aj = 2.0 * alphas[j];
            double zj = genpow_z[z_base + j];
            double uj = u_arr[j];
            double vj = step_z[step_base + j];
            sigma_u += aj * uj / zj;
            sigma_v += aj * vj / zj;
            rho_uv  += aj * uj * vj / (zj * zj);
        }

        double dot_u = 0.0, dot_v = 0.0;
        for (int64_t j = 0; j < dim; ++j) {
            dot_u += gpsi[j] * u_arr[j];
            dot_v += gpsi[j] * step_z[step_base + j];
        }
        double uw_dot_vw = 0.0;
        for (int64_t j = 0; j < dim2; ++j) {
            uw_dot_vw += u_arr[dim1 + j] * step_z[step_base + dim1 + j];
        }
        double u_hpsi_v = phi * (sigma_u * sigma_v - rho_uv) - 2.0 * uw_dot_vw;
        double coef = (u_hpsi_v * psi - 2.0 * dot_u * dot_v) / (psi * psi * psi);
        double inv_psi2 = 1.0 / (psi * psi);

        // Step 3: assemble η[j] = 0.5·{ −T_ψ[u,v,j]/ψ + coef·g_ψ[j] + ... }.
        // Two passes: compute η in `shift[step_base..]` (re-using the
        // output buffer as scratch), apply ∞-norm cap, then convert to
        // shift = grad·σμ − η.
        double max_eta = 0.0, max_ds = 0.0;
        for (int64_t j = 0; j < dim; ++j) {
            double dsj = step_s[step_base + j];
            double abs_ds = fabs(dsj);
            if (abs_ds > max_ds) max_ds = abs_ds;

            double hpsi_u_j, hpsi_v_j, t_psi_j;
            double zj = genpow_z[z_base + j];
            double uj = u_arr[j];
            double vj = step_z[step_base + j];
            if (j < dim1) {
                hpsi_u_j = gpsi[j] * (sigma_u - uj / zj);
                hpsi_v_j = gpsi[j] * (sigma_v - vj / zj);
                double bracket = sigma_u * sigma_v
                               - rho_uv
                               - sigma_v * uj / zj
                               - sigma_u * vj / zj
                               + 2.0 * uj * vj / (zj * zj);
                t_psi_j = gpsi[j] * bracket;
            } else {
                hpsi_u_j = -2.0 * uj;
                hpsi_v_j = -2.0 * vj;
                t_psi_j  = 0.0;
            }
            double eta_j = -t_psi_j / psi
                         + coef * gpsi[j]
                         + (hpsi_u_j * dot_v + hpsi_v_j * dot_u) * inv_psi2;
            if (j < dim1) {
                double beta = 1.0 - alphas[j];
                eta_j -= 2.0 * beta * uj * vj / (zj * zj * zj);
            }
            eta_j *= 0.5;

            double abs_eta = fabs(eta_j);
            if (abs_eta > max_eta) max_eta = abs_eta;
            shift[step_base + j] = eta_j;   // stash η in the output slot
        }

        // ∞-norm cap: ‖η‖_∞ ≤ 7·‖ds‖_∞. Same K=7 as CPU
        // `apply_eta_cap`. Strict aggregate improvement over no cap on
        // the parity bench.
        double scale = 1.0;
        if (max_eta > 7.0 * max_ds && max_ds > 0.0) {
            scale = (7.0 * max_ds) / max_eta;
        }

        for (int64_t j = 0; j < dim; ++j) {
            double eta_j = shift[step_base + j] * scale;
            shift[step_base + j] = genpow_grad[grad_base + j] * sigma_mu_val - eta_j;
        }
    }
}


void Cones::combined_ds_shift(
    BatchedVector& shift,
    BatchedVector& step_z,
    BatchedVector& step_s,
    const BatchedVector& sigma_mu,
    cudaStream_t stream
) {
    const int64_t m = totalConstraints();

    // Calculate offsets for each cone type
    int64_t offset_zero = 0;
    int64_t offset_nonneg = numZeroCones;
    int64_t offset_soc = numZeroCones + numNonnegCones;
    int64_t offset_exp = numZeroCones + numNonnegCones + totalSocDim;
    int64_t offset_power = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
    int64_t offset_genpow = offset_power + numPowerCones * 3;

    // Determine thread count based on max cone count
    int64_t maxCones = max(max(max(numZeroCones, numNonnegCones),
                               max(numSocCones, numExpCones)),
                          max(numPowerCones, numGenPowerCones));
    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(combined_ds_shift_all_cones_kernel, batchSize, threadsPerBlock, 0, stream,
        shift.data(), step_z.data(), step_s.data(), sigma_mu.data(),
        offset_zero, numZeroCones,
        offset_nonneg, numNonnegCones,
        soc_w.data(), soc_eta.data(), offset_soc, numSocCones,
        exp_grad.data(), exp_H_dual.data(), exp_z.data(), offset_exp, numExpCones,
        power_grad.data(), power_H_dual.data(), power_z.data(), d_powerAlphas, offset_power, numPowerCones,
        genpow_grad.data(), offset_genpow, numGenPowerCones,
        d_genPowerDim1s, d_genPowerDim2s, d_genPowerOffsets, d_genPowerOffsets, d_genPowerSzOffsets,
        totalGenPowerDim,
        // GenPow η-correction extra args:
        genpow_z.data(), genpow_p.data(), genpow_q.data(), genpow_r.data(),
        genpow_d1.data(), genpow_d2.data(),
        d_genPowerAlphas, d_genPowerAlphaOffsets,
        totalGenPowerAlphas, totalGenPowerDim2,
        genpow_correction_u.data(), genpow_correction_gpsi.data(),
        m, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets);

    // PSD cones: shift = W^{-T}(step_s) ∘ W(step_z) - σμ·e (proper symmetric cone operation)
    if (numPsdCones > 0) {
        int64_t offset_psd = numZeroCones + numNonnegCones + totalSocDim
                           + numExpCones * 3 + numPowerCones * 3;
        psd_combined_ds_shift(*this, shift.data(), step_z.data(), step_s.data(),
                              sigma_mu.data(), offset_psd, m, stream);
    }
}

} // namespace moreau
