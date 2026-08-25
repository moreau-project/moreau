/**
 * @file soc_kernels.cu
 * @brief CUDA kernels for second-order cone (SOC) operations
 *
 * Implements scaling operations for variable-dimension second-order cones (Lorentz cones).
 *
 * The per-kind small-SOC scaling and dense-Hs population kernels were
 * retired in favour of `update_scaling_all_cones_kernel` in
 * composite_kernels.cu, which folds nonneg/SOC/exp/power scaling into a
 * single launch. This file now only carries `update_large_soc_scaling_kernel`
 * for the dim > SOC_PARALLEL_THRESHOLD suffix, where a block-per-cone
 * parallel reduction beats the composite kernel's single-thread path.
 */

#include "moreau/cones/cone_kernels.cuh"
#include "moreau/cones/cones.hpp"
#include "moreau/cones/common.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace moreau {

// ============================================================================
// Block-per-cone SOC scaling for high-dimensional cones
// ============================================================================
//
// For SOC cones with dim > SOC_PARALLEL_THRESHOLD, the single-threaded
// serial reduction over the tail (||x||₂ = sqrt(Σ xᵢ²)) becomes a bottleneck.
// This kernel assigns one thread block per (batch, large_cone) and uses a
// shared-memory reduction so all block threads cooperate on the norm
// computations. All subsequent per-element writes (w, u, v, λ, Hs diagonal)
// are also parallelized across the block.
//
// Because socConeDims is sorted ascending in Cones::initialize(), the large
// cones occupy the contiguous suffix [numSocCones - numLargeSoc, numSocCones).
// Grid: (numLargeSoc, batchSize). Block: SOC_PARALLEL_BLOCK_SIZE threads.
//
// Degenerate-state handling matches the composite kernel exactly: on s/z/w
// non-interior, we write identity-like values and set scaling_success[batch]=0.
// Multiple blocks may concurrently write 0 to the same batch slot; the write
// is idempotent.

__global__ void update_large_soc_scaling_kernel(
    double* __restrict__ soc_u,
    double* __restrict__ soc_v,
    double* __restrict__ soc_d,
    double* __restrict__ soc_w,
    double* __restrict__ soc_eta,
    double* __restrict__ soc_lambda,
    double* __restrict__ soc_Hs,
    const double* __restrict__ s,
    const double* __restrict__ z,
    int64_t offset_soc,
    int64_t m_total,
    int64_t numSocCones,
    int64_t numSmallSoc,
    int64_t numLargeSoc,
    int64_t batchSize,
    int32_t* __restrict__ scaling_success,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_Hs_offsets,
    const int64_t* __restrict__ d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries
) {
    int64_t large_idx = blockIdx.x;
    int64_t batch = blockIdx.y;
    if (large_idx >= numLargeSoc || batch >= batchSize) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    int64_t cone_idx = numSmallSoc + large_idx;
    int64_t dim = d_soc_dims[cone_idx];
    int64_t soc_off = d_soc_offsets[cone_idx];
    int64_t sz_off = d_soc_sz_offsets[cone_idx];
    int64_t hs_off = d_soc_Hs_offsets[cone_idx];

    int64_t s_base = batch * m_total + offset_soc + sz_off;
    int64_t z_base = batch * m_total + offset_soc + sz_off;
    int64_t out_base_uvw = batch * totalSocDim + soc_off;
    int64_t out_base_d = batch * numSocCones + cone_idx;
    int64_t Hs_base = batch * totalSocHsEntries + hs_off;

    // Shared scratch: two reductions (s_tail_sq, z_tail_sq) share the same
    // buffer sequentially, then w_tail_sq uses it again.
    extern __shared__ double smem[];
    double* s_red = smem;

    // ---- Step 1: parallel reduction for s_tail_sq and z_tail_sq ----
    double my_s_sq = 0.0, my_z_sq = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double si = s[s_base + i];
        double zi = z[z_base + i];
        my_s_sq += si * si;
        my_z_sq += zi * zi;
    }
    double s_tail_sq = cones::block_sum_reduce(my_s_sq, s_red, tid);
    double z_tail_sq = cones::block_sum_reduce(my_z_sq, s_red, tid);

    double s0 = s[s_base];
    double z0 = z[z_base];
    double s_tail_norm = sqrt(s_tail_sq);
    double z_tail_norm = sqrt(z_tail_sq);
    double s_res = (s0 - s_tail_norm) * (s0 + s_tail_norm);
    double z_res = (z0 - z_tail_norm) * (z0 + z_tail_norm);
    double sscale = (s_res > 0.0) ? sqrt(s_res) : 0.0;
    double zscale = (z_res > 0.0) ? sqrt(z_res) : 0.0;

    // ---- Degenerate s/z: identity-equivalent scaling ----
    if (zscale == 0.0 || sscale == 0.0) {
        if (tid == 0) {
            scaling_success[batch] = 0;
            soc_w[out_base_uvw] = 1.0;
            soc_eta[out_base_d] = 1.0;
            soc_lambda[out_base_uvw] = 1.0;
            // dim > SOC_PARALLEL_THRESHOLD > 4, always sparse
            soc_u[out_base_uvw] = 0.7071067811865476;  // 1/√2
            soc_v[out_base_uvw] = 0.0;
            soc_d[out_base_d] = 0.5;
            soc_Hs[Hs_base] = 0.5;  // η²·d = 1·0.5
        }
        for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
            soc_w[out_base_uvw + i] = 0.0;
            soc_lambda[out_base_uvw + i] = 0.0;
            soc_u[out_base_uvw + i] = 0.0;
            soc_v[out_base_uvw + i] = 0.0;
            soc_Hs[Hs_base + i] = 1.0;  // η²·1
        }
        return;
    }

    double eta = sqrt(sscale / zscale);

    // ---- Step 2: compute w tail = s/sscale - z/zscale and reduce w_tail_sq ----
    double my_w_sq = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double wi = s[s_base + i] / sscale - z[z_base + i] / zscale;
        soc_w[out_base_uvw + i] = wi;  // temp store (normalized later)
        my_w_sq += wi * wi;
    }
    double w_tail_sq = cones::block_sum_reduce(my_w_sq, s_red, tid);

    double w0 = s0 / sscale + z0 / zscale;
    double w_tail_norm = sqrt(w_tail_sq);
    double w_res = (w0 - w_tail_norm) * (w0 + w_tail_norm);
    double wscale = (w_res > 0.0) ? sqrt(w_res) : 0.0;

    double eta_sq = eta * eta;

    // ---- Degenerate w: identity-scaled with preserved eta ----
    if (wscale == 0.0) {
        if (tid == 0) {
            scaling_success[batch] = 0;
            soc_w[out_base_uvw] = 1.0;
            soc_eta[out_base_d] = eta;
            soc_lambda[out_base_uvw] = sqrt(sscale * zscale);
            soc_u[out_base_uvw] = 0.7071067811865476;
            soc_v[out_base_uvw] = 0.0;
            soc_d[out_base_d] = 0.5;
            soc_Hs[Hs_base] = eta_sq * 0.5;
        }
        for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
            soc_w[out_base_uvw + i] = 0.0;
            soc_lambda[out_base_uvw + i] = 0.0;
            soc_u[out_base_uvw + i] = 0.0;
            soc_v[out_base_uvw + i] = 0.0;
            soc_Hs[Hs_base + i] = eta_sq;
        }
        return;
    }

    // ---- Step 3: normalize w tail in parallel; derive scalar quantities ----
    double inv_wscale = 1.0 / wscale;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        soc_w[out_base_uvw + i] *= inv_wscale;
    }

    // Every thread independently reconstructs the scalar coefficients.
    // w1sq is derived from w_tail_sq (already reduced), not from shared buffers.
    double w1sq = w_tail_sq * (inv_wscale * inv_wscale);
    w0 = sqrt(1.0 + w1sq);

    // Sparse expansion constants (dim > 4 always holds here)
    double wsq = w0 * w0 + w1sq;
    double wsqinv = 1.0 / wsq;
    double d_val = 0.5 * wsqinv;
    double u0 = sqrt(wsq - d_val);
    double alpha_uv = 2.0 * w0;
    double u1_scale = alpha_uv / u0;
    double v1_scale = sqrt(2.0 * (2.0 + wsqinv) / (2.0 * wsq - wsqinv));

    // lambda coefficients
    double gamma = wscale * 0.5;
    double a = (gamma + z0 / zscale) / sscale;
    double b = (gamma + s0 / sscale) / zscale;
    double denom = s0 / sscale + z0 / zscale + 2.0 * gamma;
    double lambda_scale = sqrt(sscale * zscale);

    // ---- Step 4: scalar writes (thread 0) ----
    if (tid == 0) {
        soc_w[out_base_uvw] = w0;
        soc_eta[out_base_d] = eta;
        soc_lambda[out_base_uvw] = gamma * lambda_scale;
        soc_u[out_base_uvw] = u0;
        soc_v[out_base_uvw] = 0.0;
        soc_d[out_base_d] = d_val;
        soc_Hs[Hs_base] = eta_sq * d_val;
    }

    // Ensure normalized soc_w tail is visible before we read it for u, v writes.
    __syncthreads();

    // ---- Step 5: parallel writes for tail entries (lambda, u, v, Hs) ----
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double li = (a * s[s_base + i] + b * z[z_base + i]) / denom;
        soc_lambda[out_base_uvw + i] = li * lambda_scale;
        double wi = soc_w[out_base_uvw + i];
        soc_u[out_base_uvw + i] = u1_scale * wi;
        soc_v[out_base_uvw + i] = v1_scale * wi;
        soc_Hs[Hs_base + i] = eta_sq;
    }
}

void update_large_soc_scaling(
    double* soc_u,
    double* soc_v,
    double* soc_d,
    double* soc_w,
    double* soc_eta,
    double* soc_lambda,
    double* soc_Hs,
    const double* s,
    const double* z,
    int64_t offset_soc,
    int64_t m_total,
    int64_t numSocCones,
    int64_t numSmallSoc,
    int64_t numLargeSoc,
    int64_t batchSize,
    int32_t* scaling_success,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_offsets,
    const int64_t* d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries,
    cudaStream_t stream
) {
    if (numLargeSoc == 0 || batchSize == 0) return;

    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    dim3 grid(static_cast<unsigned int>(numLargeSoc),
              static_cast<unsigned int>(batchSize));
    dim3 block(block_size);
    size_t smem_bytes = sizeof(double) * block_size;

    MOREAU_KERNEL_LAUNCH(update_large_soc_scaling_kernel, grid, block, smem_bytes, stream,
        soc_u, soc_v, soc_d, soc_w, soc_eta, soc_lambda, soc_Hs,
        s, z, offset_soc, m_total,
        numSocCones, numSmallSoc, numLargeSoc, batchSize,
        scaling_success,
        d_soc_dims, d_soc_offsets, d_soc_Hs_offsets, d_soc_sz_offsets,
        totalSocDim, totalSocHsEntries
    );
}

} // namespace moreau
