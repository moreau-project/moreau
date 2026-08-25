/**
 * @file kkt_kernels.cu
 * @brief CUDA kernel implementations for KKT system operations
 *
 * Implements GPU kernels for parallel construction and manipulation
 * of KKT system matrices used in interior-point methods.
 */

#include "moreau/kkt/kkt_kernels.cuh"
#include "moreau/cones/common.cuh"
#include "moreau/cones/cones.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>

namespace moreau {

// Direct-x SOC kernels use one block per (batch, cone) with
// `SOC_PARALLEL_BLOCK_SIZE` threads cooperating via
// `cones::block_sum_reduce` for tail-norm reductions. Per-element work
// streams in parallel passes — no stack arrays of dim size, so dim is
// unbounded on CUDA, matching the CPU side.

__global__ void populate_values_via_map_kernel(
    const double* __restrict__ src_values,
    double* __restrict__ dst_values,
    const int64_t* __restrict__ idx_map,
    int64_t src_nnz,
    int64_t dst_nnz,
    int64_t batch_size
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;

    if (i >= src_nnz || b >= batch_size) return;

    int64_t dst_idx = idx_map[i];
    if (dst_idx >= 0) {
        dst_values[b * dst_nnz + dst_idx] = src_values[b * src_nnz + i];
    }
}

void populate_values_via_map(
    CSR& src,
    CSR& dst,
    int64_t* idx_map,
    cudaStream_t stream
) {
    const int64_t src_nnz = src.nnz();
    const int64_t dst_nnz = dst.nnz();
    const int64_t batch_size = src.batchSize();

    if (src_nnz == 0 || batch_size == 0) return;

    dim3 block(256, 1, 1);
    dim3 grid((src_nnz + block.x - 1) / block.x, batch_size, 1);

    MOREAU_KERNEL_LAUNCH(populate_values_via_map_kernel, grid, block, 0, stream,
        src.values(),
        dst.values(),
        idx_map,
        src_nnz,
        dst_nnz,
        batch_size
    );
}

// Pack RHS for uniform batched KKT solve: const_rhs = [x_part; b_part; zeros(p)]
__global__ void pack_const_rhs_kernel(
    const double* __restrict__ x_part,
    const double* __restrict__ b_part,
    double* __restrict__ rhs,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    const int64_t N = n + m + p;
    if (batch >= batch_size || idx >= N) return;

    int64_t base = batch * N;
    if (idx < n) {
        rhs[base + idx] = x_part[batch * n + idx];
    } else if (idx < n + m) {
        rhs[base + idx] = b_part[batch * m + (idx - n)];
    } else {
        rhs[base + idx] = 0.0;  // expansion entries
    }
}

void pack_const_rhs(
    const double* x_part,
    const double* b_part,
    double* rhs,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream)
{
    const int64_t N = n + m + p;
    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, batch_size, 1);
    MOREAU_KERNEL_LAUNCH(pack_const_rhs_kernel, grid, block, 0, stream, x_part, b_part, rhs, n, m, p, batch_size);
}

// Fused: negate q and pack with b into const RHS (eliminates separate axpby)
// Layout: [-q; b; 0..0] per batch where 0..0 is p expansion entries
__global__ void negate_and_pack_const_rhs_impl(
    const double* __restrict__ q,
    const double* __restrict__ b,
    double* __restrict__ rhs,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    const int64_t N = n + m + p;
    if (batch >= batch_size || idx >= N) return;

    int64_t base = batch * N;
    if (idx < n) {
        rhs[base + idx] = -q[batch * n + idx];
    } else if (idx < n + m) {
        rhs[base + idx] = b[batch * m + (idx - n)];
    } else {
        rhs[base + idx] = 0.0;
    }
}

void negate_and_pack_const_rhs_kernel(
    const double* q,
    const double* b,
    double* rhs,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream)
{
    const int64_t N = n + m + p;
    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, batch_size, 1);
    MOREAU_KERNEL_LAUNCH(negate_and_pack_const_rhs_impl, grid, block, 0, stream, q, b, rhs, n, m, p, batch_size);
}

// Unpack solution from uniform batch layout into x2/z2 (expansion entries discarded)
__global__ void unpack_const_sol_kernel(
    const double* __restrict__ sol,
    double* __restrict__ x2,
    double* __restrict__ z2,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    const int64_t N = n + m + p;
    if (batch >= batch_size || idx >= N) return;

    int64_t base = batch * N;
    if (idx < n) {
        x2[batch * n + idx] = sol[base + idx];
    } else if (idx < n + m) {
        z2[batch * m + (idx - n)] = sol[base + idx];
    }
    // else: expansion entry, discard
}

void unpack_const_sol(
    const double* sol,
    double* x2,
    double* z2,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream)
{
    const int64_t N = n + m + p;
    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, batch_size, 1);
    MOREAU_KERNEL_LAUNCH(unpack_const_sol_kernel, grid, block, 0, stream, sol, x2, z2, n, m, p, batch_size);
}

// Pack two RHS into 2-column format for batched multi-RHS solve
// Layout: [all_col0s, all_col1s] where each is [(n+m+p) * batch_size]
// = [batch0_col0, batch1_col0, ..., batch0_col1, batch1_col1, ...]
// This matches what solve2() expects: two consecutive solve() calls
__global__ void pack_rhs2_kernel(
    const double* __restrict__ x_part0,
    const double* __restrict__ z_part0,
    const double* __restrict__ x_part1,
    const double* __restrict__ z_part1,
    double* __restrict__ rhs2,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    const int64_t N = n + m + p;
    if (batch >= batch_size || idx >= N) return;

    // col0 starts at 0, col1 starts at N * batch_size
    int64_t col0_base = batch * N;
    int64_t col1_base = N * batch_size + batch * N;

    if (idx < n) {
        // x part
        rhs2[col0_base + idx] = x_part0[batch * n + idx];      // col0
        rhs2[col1_base + idx] = x_part1[batch * n + idx];      // col1
    } else if (idx < n + m) {
        // z part
        int64_t z_idx = idx - n;
        rhs2[col0_base + idx] = z_part0[batch * m + z_idx];    // col0
        rhs2[col1_base + idx] = z_part1[batch * m + z_idx];    // col1
    } else {
        // expansion entries: zero-fill
        rhs2[col0_base + idx] = 0.0;
        rhs2[col1_base + idx] = 0.0;
    }
}

void pack_rhs2(
    const double* x_part0,
    const double* z_part0,
    const double* x_part1,
    const double* z_part1,
    double* rhs2,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream)
{
    const int64_t N = n + m + p;
    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, batch_size, 1);
    MOREAU_KERNEL_LAUNCH(pack_rhs2_kernel, grid, block, 0, stream,
        x_part0, z_part0, x_part1, z_part1, rhs2, n, m, p, batch_size);
}

// Unpack 2-column solution into separate x/z buffers
// Layout matches pack_rhs2: [all_col0s, all_col1s] where each is [(n+m+p) * batch_size]
__global__ void unpack_sol2_kernel(
    const double* __restrict__ sol2,
    double* __restrict__ x0,
    double* __restrict__ z0,
    double* __restrict__ x1,
    double* __restrict__ z1,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    const int64_t N = n + m + p;
    if (batch >= batch_size || idx >= N) return;

    // col0 starts at 0, col1 starts at N * batch_size
    int64_t col0_base = batch * N;
    int64_t col1_base = N * batch_size + batch * N;

    if (idx < n) {
        // x part
        x0[batch * n + idx] = sol2[col0_base + idx];      // col0
        x1[batch * n + idx] = sol2[col1_base + idx];      // col1
    } else if (idx < n + m) {
        // z part
        int64_t z_idx = idx - n;
        z0[batch * m + z_idx] = sol2[col0_base + idx];    // col0
        z1[batch * m + z_idx] = sol2[col1_base + idx];    // col1
    }
    // else: expansion entries, discard
}

void unpack_sol2(
    const double* sol2,
    double* x0,
    double* z0,
    double* x1,
    double* z1,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream)
{
    const int64_t N = n + m + p;
    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, batch_size, 1);
    MOREAU_KERNEL_LAUNCH(unpack_sol2_kernel, grid, block, 0, stream,
        sol2, x0, z0, x1, z1, n, m, p, batch_size);
}

__global__ void update_kkt_H_block_kernel(KKTHBlockArgs a) {
    // Alias struct fields to __restrict__ locals so the existing kernel
    // body reads field-by-name and the compiler keeps the no-alias hints.
    double* __restrict__ kkt_values = a.kkt_values;
    const int64_t* __restrict__ H_diagIdx = a.H_diagIdx;
    const double* __restrict__ nonneg_w = a.nonneg_w;
    const int64_t* __restrict__ H_exp_idx = a.H_exp_idx;
    const double* __restrict__ exp_Hs = a.exp_Hs;
    const int64_t* __restrict__ H_soc_idx = a.H_soc_idx;
    const double* __restrict__ soc_Hs = a.soc_Hs;
    const int64_t* __restrict__ H_power_idx = a.H_power_idx;
    const double* __restrict__ power_Hs = a.power_Hs;
    const int64_t numZeroCones = a.numZeroCones;
    const int64_t numNonnegCones = a.numNonnegCones;
    const int64_t numExpCones = a.numExpCones;
    const int64_t numSocCones = a.numSocCones;
    const int64_t numPowerCones = a.numPowerCones;
    const int64_t nnzKKT = a.nnzKKT;
    const int64_t totalSocHsEntries = a.totalSocHsEntries;
    const int64_t* __restrict__ H_soc_u_idx = a.H_soc_u_idx;
    const int64_t* __restrict__ H_soc_v_idx = a.H_soc_v_idx;
    const int64_t* __restrict__ H_soc_exp_diag_idx = a.H_soc_exp_diag_idx;
    const double* __restrict__ soc_u = a.soc_u;
    const double* __restrict__ soc_v = a.soc_v;
    const double* __restrict__ soc_eta = a.soc_eta;
    const int64_t* __restrict__ d_soc_dims = a.d_soc_dims;
    const int64_t* __restrict__ d_soc_offsets = a.d_soc_offsets;
    const int64_t totalSocDim = a.totalSocDim;
    const int64_t numSparseSoc = a.numSparseSoc;
    const int64_t* __restrict__ d_soc_sparse_offsets = a.d_soc_sparse_offsets;
    const int64_t* __restrict__ d_soc_sparse_indices = a.d_soc_sparse_indices;
    const int64_t* __restrict__ H_genpow_idx = a.H_genpow_idx;
    const int64_t* __restrict__ H_genpow_q_idx = a.H_genpow_q_idx;
    const int64_t* __restrict__ H_genpow_r_idx = a.H_genpow_r_idx;
    const int64_t* __restrict__ H_genpow_p_idx = a.H_genpow_p_idx;
    const int64_t* __restrict__ H_genpow_exp_diag_idx = a.H_genpow_exp_diag_idx;
    const double* __restrict__ genpow_Hs = a.genpow_Hs;
    const double* __restrict__ genpow_q = a.genpow_q;
    const double* __restrict__ genpow_r = a.genpow_r;
    const double* __restrict__ genpow_p = a.genpow_p;
    const double* __restrict__ genpow_mu = a.genpow_mu;
    const int64_t numGenPowerCones = a.numGenPowerCones;
    const int64_t numSparseGenPow = a.numSparseGenPow;
    const int64_t totalGenPowerHsEntries = a.totalGenPowerHsEntries;
    const int64_t* __restrict__ d_genPowerDim1s = a.d_genPowerDim1s;
    const int64_t* __restrict__ d_genPowerDim2s = a.d_genPowerDim2s;
    const int64_t* __restrict__ d_genPowerOffsets = a.d_genPowerOffsets;
    const int64_t* __restrict__ d_genPowerAlphaOffsets = a.d_genPowerAlphaOffsets;
    const int64_t* __restrict__ d_genPowerSparseOffsets = a.d_genPowerSparseOffsets;
    const int64_t* __restrict__ d_genPowerSparseIndices = a.d_genPowerSparseIndices;
    const int64_t totalGenPowerDim = a.totalGenPowerDim;
    const int64_t totalGenPowerAlphas = a.totalGenPowerAlphas;
    const double* __restrict__ genpow_pd_axes = a.genpow_pd_axes;
    const double* __restrict__ genpow_pd_coefs = a.genpow_pd_coefs;
    const double* __restrict__ genpow_pd_signs = a.genpow_pd_signs;
    const double* __restrict__ genpow_pd_active = a.genpow_pd_active;

    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;
    const int64_t* pd_axis_idx_arr[6] = {
        a.H_genpow_pd_axis_idx[0], a.H_genpow_pd_axis_idx[1], a.H_genpow_pd_axis_idx[2],
        a.H_genpow_pd_axis_idx[3], a.H_genpow_pd_axis_idx[4], a.H_genpow_pd_axis_idx[5]
    };

    // Grid-stride loop to handle > 1024 threads
    for (int64_t tid = threadIdx.x; tid < numZeroCones + numNonnegCones; tid += stride) {
        // Zero cones: H = 0 (diagonal), starting at H_diagIdx[0]
        if (tid < numZeroCones) {
            int64_t idx = H_diagIdx[tid];
            kkt_values[batch * nnzKKT + idx] = 0.0;
        }
        // Nonnegative cones: H = w*w (diagonal), starting at H_diagIdx[numZeroCones]
        // KKT has -H, so negate
        else if (tid < numZeroCones + numNonnegCones) {
            int64_t cone_idx = tid - numZeroCones;
            int64_t idx = H_diagIdx[tid];
            double w = nonneg_w[batch * numNonnegCones + cone_idx];
            kkt_values[batch * nnzKKT + idx] = -(w * w);
        }
    }

    // Exp cone 3x3 blocks: 6 elements per cone (grid-stride loop)
    // KKT has -H, so negate
    for (int64_t tid = threadIdx.x; tid < numExpCones * 6; tid += stride) {
        int64_t kkt_idx = H_exp_idx[tid];
        double hs_val = exp_Hs[batch * (numExpCones * 6) + tid];
        kkt_values[batch * nnzKKT + kkt_idx] = -hs_val;
    }

    // SOC blocks: diagonal entries (both dense upper-tri and sparse diagonal)
    // H_soc_idx is a flat array of KKT value indices for Hs entries
    // soc_Hs contains: dense upper-tri for dim<=4, diagonal for dim>4
    // KKT has -H, so negate
    for (int64_t tid = threadIdx.x; tid < totalSocHsEntries; tid += stride) {
        int64_t kkt_idx = H_soc_idx[tid];
        double hs_val = soc_Hs[batch * totalSocHsEntries + tid];
        kkt_values[batch * nnzKKT + kkt_idx] = -hs_val;
    }

    // Sparse SOC expansion: u column, v column, and expansion diagonal
    // Per-cone approach: one thread per SOC cone, skip dense cones
    // Uses precomputed prefix-sum arrays for O(1) offset lookup
    if (numSparseSoc > 0) {
        for (int64_t tid = threadIdx.x; tid < numSocCones; tid += stride) {
            int64_t sparse_cone_idx = d_soc_sparse_indices[tid];
            if (sparse_cone_idx < 0) continue;  // Skip dense cones

            int64_t dim = d_soc_dims[tid];
            int64_t soc_off = d_soc_offsets[tid];  // offset within totalSocDim
            double eta = soc_eta[batch * numSocCones + tid];
            double eta_sq = eta * eta;

            // Precomputed offset within H_soc_u_idx / H_soc_v_idx (sparse cones only)
            int64_t sparse_off = d_soc_sparse_offsets[tid];

            // Write u column: -η² * u[i] for i in 0..dim
            for (int64_t i = 0; i < dim; ++i) {
                int64_t kkt_idx = H_soc_u_idx[sparse_off + i];
                double u_val = soc_u[batch * totalSocDim + soc_off + i];
                kkt_values[batch * nnzKKT + kkt_idx] = -eta_sq * u_val;
            }

            // Write v column: -η² * v[i] for i in 0..dim
            // Both u and v columns are scaled by -η²; the sign difference
            // between uu^T and vv^T is handled by the expansion diagonal [-η², +η²]
            for (int64_t i = 0; i < dim; ++i) {
                int64_t kkt_idx = H_soc_v_idx[sparse_off + i];
                double v_val = soc_v[batch * totalSocDim + soc_off + i];
                kkt_values[batch * nnzKKT + kkt_idx] = -eta_sq * v_val;
            }

            // Expansion diagonal: [-η², +η²]
            kkt_values[batch * nnzKKT + H_soc_exp_diag_idx[sparse_cone_idx * 2]] = -eta_sq;
            kkt_values[batch * nnzKKT + H_soc_exp_diag_idx[sparse_cone_idx * 2 + 1]] = eta_sq;
        }
    }

    // Power cone 3x3 blocks: 6 elements per cone (grid-stride loop)
    // KKT has -H, so negate
    for (int64_t tid = threadIdx.x; tid < numPowerCones * 6; tid += stride) {
        int64_t kkt_idx = H_power_idx[tid];
        double hs_val = power_Hs[batch * (numPowerCones * 6) + tid];
        kkt_values[batch * nnzKKT + kkt_idx] = -hs_val;
    }

    // GenPowerCone: dense (upper-tri Hs) + sparse (diagonal) entries
    // H_genpow_idx has mixed content: dense cones → dim*(dim+1)/2, sparse → dim entries
    // KKT has -H, so negate
    for (int64_t tid = threadIdx.x; tid < totalGenPowerHsEntries; tid += stride) {
        int64_t kkt_idx = H_genpow_idx[tid];
        double hs_val = genpow_Hs[batch * totalGenPowerHsEntries + tid];
        kkt_values[batch * nnzKKT + kkt_idx] = -hs_val;
    }

    // Sparse GenPowerCone expansion: q/r/p columns per sparse cone (dim > 4)
    if (numSparseGenPow > 0) {
        // Cones are sorted ascending by dim, so all dense cones (dim<=4) are at the start.
        // Sparse-only q/r/p index arrays need sparse-only alpha/dim2 offsets.
        // Derive from full offset arrays by subtracting the dense cone totals.
        int64_t numDenseGenPow = numGenPowerCones - numSparseGenPow;
        int64_t dense_alpha_total = d_genPowerAlphaOffsets[numDenseGenPow];
        int64_t dense_gp_total = d_genPowerOffsets[numDenseGenPow];
        int64_t dense_dim2_total = dense_gp_total - dense_alpha_total;

        for (int64_t tid = threadIdx.x; tid < numGenPowerCones; tid += stride) {
            int64_t sparse_cone_idx = d_genPowerSparseIndices[tid];
            if (sparse_cone_idx < 0) continue;  // Skip dense cones

            int64_t dim1 = d_genPowerDim1s[tid];
            int64_t dim2 = d_genPowerDim2s[tid];
            int64_t dim = dim1 + dim2;
            int64_t gp_off = d_genPowerOffsets[tid];
            int64_t alpha_off = d_genPowerAlphaOffsets[tid];
            int64_t r_off = gp_off - alpha_off;
            int64_t sparse_off = d_genPowerSparseOffsets[tid];

            // Sparse-only offsets for q and r index arrays
            int64_t sparse_q_off = alpha_off - dense_alpha_total;
            int64_t sparse_r_off = r_off - dense_dim2_total;

            double mu_val = genpow_mu[batch];
            double sqrtmu = sqrt(mu_val);

            // Adaptive per-axis equilibration (mirrors CPU
            // `csc_update_sparsecone` in moreau-cpu/datamaps.rs). For each
            // axis with weight w = μ·||v||², if w > 1e12 we swap the
            // off-diagonal column from `±√μ·v` to `±v/||v||` (unit norm)
            // and the sentinel from `±1` to `±1/w`. Schur contribution
            // `-c²/d = λ` is preserved exactly; only entry magnitudes
            // change. Catches the catastrophic 1e22-magnitude regime
            // without disturbing well-behaved iterates.
            const double EQ_THRESHOLD = 1.0e12;
            double q_norm_sq = 0.0;
            for (int64_t i = 0; i < dim1; ++i) {
                double q_val = genpow_q[batch * totalGenPowerAlphas + alpha_off + i];
                q_norm_sq += q_val * q_val;
            }
            double r_norm_sq = 0.0;
            for (int64_t i = 0; i < dim2; ++i) {
                double r_val = genpow_r[batch * (totalGenPowerDim - totalGenPowerAlphas) + r_off + i];
                r_norm_sq += r_val * r_val;
            }
            double p_norm_sq = 0.0;
            for (int64_t i = 0; i < dim; ++i) {
                double p_val = genpow_p[batch * totalGenPowerDim + gp_off + i];
                p_norm_sq += p_val * p_val;
            }
            double q_w = mu_val * q_norm_sq;
            double r_w = mu_val * r_norm_sq;
            double p_w = mu_val * p_norm_sq;
            double q_off_scale = (q_w > EQ_THRESHOLD && q_norm_sq > 0.0)
                ? -1.0 / sqrt(q_norm_sq) : -sqrtmu;
            double r_off_scale = (r_w > EQ_THRESHOLD && r_norm_sq > 0.0)
                ? -1.0 / sqrt(r_norm_sq) : -sqrtmu;
            double p_off_scale = (p_w > EQ_THRESHOLD && p_norm_sq > 0.0)
                ? -1.0 / sqrt(p_norm_sq) : -sqrtmu;
            double q_sent = (q_w > EQ_THRESHOLD) ? -1.0 / q_w : -1.0;
            double r_sent = (r_w > EQ_THRESHOLD) ? -1.0 / r_w : -1.0;
            double p_sent = (p_w > EQ_THRESHOLD) ?  1.0 / p_w :  1.0;

            // Write q column
            for (int64_t i = 0; i < dim1; ++i) {
                double q_val = genpow_q[batch * totalGenPowerAlphas + alpha_off + i];
                kkt_values[batch * nnzKKT + H_genpow_q_idx[sparse_q_off + i]] = q_off_scale * q_val;
            }

            // Write r column
            for (int64_t i = 0; i < dim2; ++i) {
                double r_val = genpow_r[batch * (totalGenPowerDim - totalGenPowerAlphas) + r_off + i];
                kkt_values[batch * nnzKKT + H_genpow_r_idx[sparse_r_off + i]] = r_off_scale * r_val;
            }

            // Write p column
            for (int64_t i = 0; i < dim; ++i) {
                double p_val = genpow_p[batch * totalGenPowerDim + gp_off + i];
                kkt_values[batch * nnzKKT + H_genpow_p_idx[sparse_off + i]] = p_off_scale * p_val;
            }

            // Expansion diagonals: q, r, p sentinels (signs preserved).
            kkt_values[batch * nnzKKT + H_genpow_exp_diag_idx[sparse_cone_idx * 9 + 0]] = q_sent;
            kkt_values[batch * nnzKKT + H_genpow_exp_diag_idx[sparse_cone_idx * 9 + 1]] = r_sent;
            kkt_values[batch * nnzKKT + H_genpow_exp_diag_idx[sparse_cone_idx * 9 + 2]] = p_sent;

            // PD-axis off-diagonal columns + sentinels. The CPU mirror is
            // in moreau-cpu `csc_update_sparsecone`. When pd_active=0 we
            // still write structurally-non-zero values: a tiny ε in the
            // off-diagonal column (k-th canonical basis to make each
            // column distinct) plus a sign·1 sentinel. This avoids cuDSS
            // EXECUTION_FAILED that triggers when the symbolic structure
            // has all-zero columns. When active we mirror the CPU
            // adaptive-equilibration logic with threshold 1e12.
            const double INACTIVE_EPS = 1.0e-8;  // tiny but structurally non-zero
            // PD storage layout (mirrors `compute_genpow_pd_axes`):
            //   axes_out  = &pd_axes[batch*6*totalGenPowerDim + 6*gp_off]
            //   coefs/signs = &pd_state[batch*6*numGenPowerCones + 6*tid]
            //   pd_active = &pd_active[batch*numGenPowerCones + tid]
            // axes for a cone are stored as 6 axes × dim, contiguous:
            // axes_out[k*dim + i] is element i of axis k.
            int64_t pd_axes_cone_base = batch * 6 * totalGenPowerDim + 6 * gp_off;
            int64_t pd_state_cone_base = batch * 6 * numGenPowerCones + 6 * tid;
            int64_t pd_active_idx = batch * numGenPowerCones + tid;
            double active_flag = (genpow_pd_active != nullptr)
                ? genpow_pd_active[pd_active_idx] : 0.0;

            for (int axk = 0; axk < 6; ++axk) {
                double sign = (genpow_pd_signs != nullptr)
                    ? genpow_pd_signs[pd_state_cone_base + axk] : 1.0;
                double coef = (genpow_pd_coefs != nullptr)
                    ? genpow_pd_coefs[pd_state_cone_base + axk] : 0.0;

                if (active_flag > 0.5 && coef > 0.0) {
                    double n_sq = 0.0;
                    for (int64_t i = 0; i < dim; ++i) {
                        double a_val = genpow_pd_axes[pd_axes_cone_base + axk * dim + i];
                        n_sq += a_val * a_val;
                    }
                    double w = coef * n_sq;
                    double off_scale, sent;
                    if (w > 1.0e12 && n_sq > 0.0) {
                        off_scale = -1.0 / sqrt(n_sq);
                        sent = sign / w;
                    } else {
                        off_scale = -sqrt(coef);
                        sent = sign;
                    }
                    for (int64_t i = 0; i < dim; ++i) {
                        double a_val = genpow_pd_axes[pd_axes_cone_base + axk * dim + i];
                        kkt_values[batch * nnzKKT
                            + pd_axis_idx_arr[axk][sparse_off + i]]
                            = off_scale * a_val;
                    }
                    kkt_values[batch * nnzKKT
                        + H_genpow_exp_diag_idx[sparse_cone_idx * 9 + 3 + axk]]
                        = sent;
                } else {
                    // Inactive: tiny ε on a single basis vector position and
                    // sign·1 sentinel. Schur contribution is ε² which is
                    // negligible numerically (~1e-300) but keeps cuDSS happy.
                    for (int64_t i = 0; i < dim; ++i) {
                        double v = (i == (int64_t)axk % dim) ? INACTIVE_EPS : 0.0;
                        kkt_values[batch * nnzKKT
                            + pd_axis_idx_arr[axk][sparse_off + i]] = v;
                    }
                    double sent_sign = (sign >= 0.0) ? 1.0 : -1.0;
                    kkt_values[batch * nnzKKT
                        + H_genpow_exp_diag_idx[sparse_cone_idx * 9 + 3 + axk]]
                        = sent_sign;
                }
            }
        }
    }
}

// =====================================================================
// Direct-x cone kernels — primal/dual convention
// =====================================================================
//
// These kernels implement the direct-x cone pipeline on CUDA, mirroring
// the CPU `direct_x_*` trait methods in
// `packages/moreau-cpu/src/solver/core/cones/mod.rs` (see the block
// comment above `direct_x_update_scaling` there for the authoritative
// primal/dual convention and the primal↔dual-swap justification).
//
// # Direct-x naming
//
// In direct-x a sub-vector `x[J]` is constrained to the cone directly
// with `z_J` (a.k.a. `z_x`) as its dual. So:
//   direct-x PRIMAL = `x[J]`   (gathered via `d_xcone_indices`)
//   direct-x DUAL   = `z_x`    (stored flat at `d_xcone_numel_offsets`)
//
// This is the OPPOSITE of slack naming, where `s` is primal and `z` is
// dual. Every direct-x kernel that delegates to a slack-style NT formula
// must apply the primal↔dual swap at the slack boundary — i.e. pass the
// direct-x dual into slack's s-slot and the direct-x primal into slack's
// z-slot. For symmetric cones (nonneg, dense SOC) the swap is legal
// because the cone is self-dual and the barrier is self-conjugate: the
// swap replaces `W` with `W⁻ᵀ` and therefore `Hs` with `Hs_inv`, which
// is exactly what the direct-x (1,1) block needs additively.
//
// # Where the swap lives
//
// The swap is applied at the GATHER boundary of each kernel: we read
// the direct-x dual into a local `slack_s[·]` buffer and the direct-x
// primal into a local `slack_z[·]` buffer, then run the standard slack
// NT math (or a pure `__device__` helper that embodies it) on those
// locals. After that, the stored `w`, `λ`, `η` carry the dual-side
// interpretation and downstream step-math kernels (affine_ds,
// combined_ds_shift, Δs_from_Δz, mul_Hs, step_length) use them
// consistently without needing their own per-kernel swap.
//
// Getting the swap wrong at the gather boundary silently corrupts the
// stored NT triple in a way that's invisible when the iterates are
// symmetric (e.g. x==z_x at unit init) but diverges the moment they
// break symmetry. A fix landed in ba9320a6; the refactor below makes
// the swap one documented line instead of inlined throughout.
//
// Asymmetric direct-x cones (exp, power, genpow) and PSD direct-x are
// handled by their own dedicated kernels below — they do NOT use the
// symmetric slack-delegation pattern because their primal and dual
// barriers differ. The symmetric Nonneg + SOC kernels in this block
// use the swap; the asymmetric kernels further down implement the
// direct-x primal-barrier formulas explicitly. See
// update_xcones_{exp,pow,genpow}_scaling_kernel and the
// xcone_psd_* family in packages/moreau-cuda/src/cones/psd_kernels.cu.
// =====================================================================

// Direct-x SOC scaling. Mirrors CPU `direct_x_update_scaling` for SOC:
// gather the direct-x (primal, dual) pair, apply the primal↔dual swap
// at the slack boundary (see file-level block comment and CPU mod.rs
// trait), run the standard SOC NT scaling math inline, persist w/λ/η,
// and pack Hs in the form the KKT assembly expects.
//
// Hs storage splits on `dim ≤ 4` (dense) vs `dim > 4` (rank-2 sparse):
//   - dim ≤ 4: column-major upper-tri of η²·(2ww' − J), dim·(dim+1)/2
//              entries, packed into xcone_Hs at hs_off.
//   - dim > 4: DIAGONAL η²·[d, 1, 1, ..., 1] (dim entries) at hs_off.
//              Off-diagonal coupling is absorbed by the rank-2 columns
//              (u, v, d) written to xcone_u/v/d, which refresh_xcone_hs
//              scatters into the KKT expansion columns.
//
// One block per (batch, cone) with blockDim threads cooperating across
// the cone's entries: three parallel reductions (s_tail_sq, z_tail_sq,
// w_tail_sq_raw) plus per-element parallel writes for w/λ/u/v/Hs. No
// per-thread stack buffers of dim size, so dim has no upper bound.
// Shared memory: sizeof(double) * blockDim for the block-sum scratch.
__global__ void update_xcones_soc_scaling_kernel(
    const double* __restrict__ x,              // direct-x primal (gathered via indices)
    const double* __restrict__ z_x,            // direct-x dual (flat, offset via numel_offsets)
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_sparse_indices,   // [numXCones], ≥0 iff SOC+dim>4
    const int64_t* __restrict__ d_xcone_sparse_offsets,   // [numXCones+1] prefix of dim over sparse
    const int64_t* __restrict__ d_xcone_cone_pos_for_sorted, // [totalXConeNumel], for sparse expansion scatter
    double* __restrict__ xcone_w,
    double* __restrict__ xcone_lambda,
    double* __restrict__ xcone_eta,
    double* __restrict__ xcone_Hs,
    double* __restrict__ xcone_u,                         // [batchSize * totalSparseXSocDim]
    double* __restrict__ xcone_v,                         // [batchSize * totalSparseXSocDim]
    double* __restrict__ xcone_d,                         // [batchSize * numSparseXSoc]
    // Fused KKT scatter targets (null → classic two-pass path, see
    // refresh_xcone_hs / refresh_xcone_sparse_expansion).
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_px_baseline,
    const int64_t* __restrict__ H_xcone_hs_idx,
    const int64_t* __restrict__ H_xcone_u_idx,
    const int64_t* __restrict__ H_xcone_v_idx,
    const int64_t* __restrict__ H_xcone_exp_diag_idx,
    int64_t nnzKKT,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalSparseXSocDim,
    int64_t numSparseXSoc)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    // Skip nonneg cones (handled by update_xcones_nonneg_scaling).
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t hs_off = d_xcone_hs_offsets[c];
    const int64_t x_off_b  = b * n;
    const int64_t zx_off_b = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;
    const int64_t w_off_b  = b * totalXConeNumel + num_off;

    // Fused scatter: when `kkt_values` is non-null the scaling kernel writes
    // H_s / expansion entries straight into the shared KKT (1,1) block, so
    // the per-iter `refresh_xcone_hs` + `refresh_xcone_sparse_expansion`
    // passes become no-ops. Saves the extra global read-write round trip.
    const bool fused = (kkt_values != nullptr) && (xcone_px_baseline != nullptr)
                       && (H_xcone_hs_idx != nullptr);
    const int64_t kkt_off = fused ? b * nnzKKT : 0;

    // Per-cone accessors. `s` follows slack semantics (primal) after the
    // direct-x primal↔dual swap: direct-x dual `z_x` → slack `s`;
    // direct-x primal `x[J]` → slack `z`. Lambdas read straight from
    // global memory; threads independently coalesce within each pass.
    auto s_at = [&](int64_t i) -> double {
        return z_x[zx_off_b + num_off + i];
    };
    auto z_at = [&](int64_t i) -> double {
        return x[x_off_b + d_xcone_indices[num_off + i]];
    };

    // ---------- Identity-fallback defaults (w=e_0, λ=e_0, η=eta_val) ----------
    auto write_identity = [&](double eta_val, double lam0_val) {
        if (tid == 0) {
            xcone_w[w_off_b + 0]      = 1.0;
            xcone_lambda[w_off_b + 0] = lam0_val;
            xcone_eta[b * numXCones + c] = eta_val;
        }
        for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
            xcone_w[w_off_b + i]      = 0.0;
            xcone_lambda[w_off_b + i] = 0.0;
        }
    };

    // ---------- Pass 1: parallel reduction for s/z tail norms ----------
    const double s0 = s_at(0), z0 = z_at(0);
    double my_s_sq = 0.0, my_z_sq = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double si = s_at(i);
        const double zi = z_at(i);
        my_s_sq += si * si;
        my_z_sq += zi * zi;
    }
    // Fused 2-value reduction.
    double vals2_sz[2] = { my_s_sq, my_z_sq };
    cones::block_sum_reduce_N<2>(vals2_sz, smem, tid);
    const double s_tail_sq = vals2_sz[0];
    const double z_tail_sq = vals2_sz[1];
    const double s_tail = sqrt(s_tail_sq);
    const double z_tail = sqrt(z_tail_sq);
    const double s_res = (s0 - s_tail) * (s0 + s_tail);
    const double z_res = (z0 - z_tail) * (z0 + z_tail);
    const double sscale = (s_res > 0.0) ? sqrt(s_res) : 0.0;
    const double zscale = (z_res > 0.0) ? sqrt(z_res) : 0.0;
    if (sscale <= 0.0 || zscale <= 0.0) {
        write_identity(/*eta=*/1.0, /*lam0=*/1.0);
        // Leave Hs slots as-is; caller will reseed identity on next scaling.
        return;
    }
    const double eta = sqrt(sscale / zscale);

    // ---------- Pass 2: parallel reduction for w raw tail norm ----------
    // w_raw_0 = s0/sscale + z0/zscale.  w_raw_i = s_i/sscale - z_i/zscale for i≥1.
    // wscale = sqrt((w_raw_0 − ||w_raw_tail||)(w_raw_0 + ||w_raw_tail||)).
    const double w_raw_0 = s0 / sscale + z0 / zscale;
    double my_w_sq_raw = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wri = s_at(i) / sscale - z_at(i) / zscale;
        my_w_sq_raw += wri * wri;
    }
    const double w_tail_sq_raw = cones::block_sum_reduce(my_w_sq_raw, smem, tid);
    const double w_tail_raw = sqrt(w_tail_sq_raw);
    const double w_res = (w_raw_0 - w_tail_raw) * (w_raw_0 + w_tail_raw);
    const double wscale = (w_res > 0.0) ? sqrt(w_res) : 0.0;
    if (wscale <= 0.0) {
        // Degenerate w — keep the identity w but recover the lam scale.
        write_identity(/*eta=*/eta, /*lam0=*/sqrt(sscale * zscale));
        return;
    }

    // Every thread reconstructs the scalar coefficients independently —
    // cheaper than broadcasting via shared memory.
    const double w1sq_normalized = w_tail_sq_raw / (wscale * wscale);
    const double w0 = sqrt(1.0 + w1sq_normalized);
    const double gamma = 0.5 * wscale;
    const double a_coef  = (gamma + z0 / zscale) / sscale;
    const double b_coef  = (gamma + s0 / sscale) / zscale;
    const double denom   = s0 / sscale + z0 / zscale + 2.0 * gamma;
    const double lam_scale = sqrt(sscale * zscale);

    // ---------- Pass 3: parallel writes for w[i], λ[i] ----------
    if (tid == 0) {
        xcone_w[w_off_b + 0]      = w0;
        xcone_lambda[w_off_b + 0] = gamma * lam_scale;
        xcone_eta[b * numXCones + c] = eta;
    }
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double si = s_at(i);
        const double zi = z_at(i);
        const double wi = (si / sscale - zi / zscale) / wscale;
        xcone_w[w_off_b + i] = wi;
        const double li = (a_coef * si + b_coef * zi) / denom;
        xcone_lambda[w_off_b + i] = li * lam_scale;
    }

    const double eta_sq = eta * eta;

    if (dim <= 4) {
        // ---- Dense Hs = η²·(2·w·w' − J), column-major upper-tri.
        // Up to 10 entries total — thread 0 emits them serially after
        // a sync so the w writes above are visible.
        __syncthreads();
        if (tid == 0) {
            int64_t pos = 0;
            for (int64_t cc = 0; cc < dim; ++cc) {
                const double wcc = xcone_w[w_off_b + cc];
                for (int64_t rr = 0; rr <= cc; ++rr) {
                    const double wrr = xcone_w[w_off_b + rr];
                    double val = 2.0 * eta_sq * wrr * wcc;
                    if (rr == cc) {
                        val += (rr == 0) ? -eta_sq : eta_sq;  // J diag
                    }
                    const int64_t hs_global = hs_off_b + hs_off + pos;
                    xcone_Hs[hs_global] = val;
                    if (fused) {
                        kkt_values[kkt_off + H_xcone_hs_idx[hs_off + pos]] =
                            xcone_px_baseline[hs_global] + val;
                    }
                    ++pos;
                }
            }
        }
        return;
    }

    // ---- Rank-2 sparse form (mirror of socone.rs::update_scaling). All
    // threads independently recover the scalar coefficients; the tail
    // writes (u, v, Hs diag) parallelise across the block. ----
    const double w1sq = w1sq_normalized;
    const double wsq = w0 * w0 + w1sq;
    const double wsqinv = 1.0 / wsq;
    const double d_val = 0.5 * wsqinv;
    const double alpha = 2.0 * w0;
    const double u0_val = sqrt(wsq - d_val);
    const double u1_coef = alpha / u0_val;
    const double v1_coef =
        sqrt(2.0 * (2.0 + wsqinv) / (2.0 * wsq - wsqinv));

    const int64_t sparse_idx = d_xcone_sparse_indices[c];
    const int64_t sp_off = d_xcone_sparse_offsets[c];
    const int64_t u_off_b = b * totalSparseXSocDim + sp_off;

    if (tid == 0) {
        xcone_u[u_off_b + 0] = u0_val;
        xcone_v[u_off_b + 0] = 0.0;
        xcone_d[b * numSparseXSoc + sparse_idx] = d_val;
        xcone_Hs[hs_off_b + hs_off + 0] = eta_sq * d_val;
    }

    // Need normalized w tail (written above) visible before reading.
    __syncthreads();
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wi = xcone_w[w_off_b + i];
        xcone_u[u_off_b + i] = u1_coef * wi;
        xcone_v[u_off_b + i] = v1_coef * wi;
        xcone_Hs[hs_off_b + hs_off + i] = eta_sq;
    }

    if (fused) {
        // Sparse diagonal (Hs): kkt[slot] = baseline + η² · d for entry 0,
        // baseline + η² for entries 1..dim-1.
        const int64_t hs_base = hs_off_b + hs_off;
        if (tid == 0) {
            kkt_values[kkt_off + H_xcone_hs_idx[hs_off + 0]] =
                xcone_px_baseline[hs_base + 0] + eta_sq * d_val;
        }
        for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
            kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                xcone_px_baseline[hs_base + i] + eta_sq;
        }

        // Rank-2 expansion: scatter +η²·v, +η²·u into v/u columns using
        // the sorted ordering (mirrors refresh_xcone_sparse_expansion).
        // cone_pos_for_sorted[num_off + p] gives the physical position
        // inside (xcone_u, xcone_v) for the p-th column-sorted entry.
        if (d_xcone_cone_pos_for_sorted != nullptr &&
            H_xcone_u_idx != nullptr && H_xcone_v_idx != nullptr) {
            // u/v writes above must be visible before we read them here.
            __syncthreads();
            for (int64_t p = tid; p < dim; p += blockDimX) {
                const int64_t cone_pos =
                    d_xcone_cone_pos_for_sorted[num_off + p];
                const int64_t flat = sp_off + p;
                const double u_val = xcone_u[u_off_b + cone_pos];
                const double v_val = xcone_v[u_off_b + cone_pos];
                kkt_values[kkt_off + H_xcone_v_idx[flat]] = eta_sq * v_val;
                kkt_values[kkt_off + H_xcone_u_idx[flat]] = eta_sq * u_val;
            }
        }

        // Expansion diag: [v-diag = +η², u-diag = -η²] for direct-x.
        if (tid == 0 && H_xcone_exp_diag_idx != nullptr) {
            kkt_values[kkt_off + H_xcone_exp_diag_idx[2 * sparse_idx]]     =  eta_sq;
            kkt_values[kkt_off + H_xcone_exp_diag_idx[2 * sparse_idx + 1]] = -eta_sq;
        }
    }
}

void update_xcones_soc_scaling(
    const double* x,
    const double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_sparse_indices,
    const int64_t* d_xcone_sparse_offsets,
    const int64_t* d_xcone_cone_pos_for_sorted,
    double* xcone_w,
    double* xcone_lambda,
    double* xcone_eta,
    double* xcone_Hs,
    double* xcone_u,
    double* xcone_v,
    double* xcone_d,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    const int64_t* H_xcone_u_idx,
    const int64_t* H_xcone_v_idx,
    const int64_t* H_xcone_exp_diag_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalSparseXSocDim,
    int64_t numSparseXSoc,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    update_xcones_soc_scaling_kernel<<<grid, block_size, smem_bytes, stream>>>(
        x, z_x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        d_xcone_sparse_indices, d_xcone_sparse_offsets,
        d_xcone_cone_pos_for_sorted,
        xcone_w, xcone_lambda, xcone_eta, xcone_Hs,
        xcone_u, xcone_v, xcone_d,
        kkt_values, xcone_px_baseline,
        H_xcone_hs_idx, H_xcone_u_idx, H_xcone_v_idx, H_xcone_exp_diag_idx,
        nnzKKT,
        n, numXCones, totalXConeNumel, totalXConeHsEntries,
        totalSparseXSocDim, numSparseXSoc);
}

// Direct-x nonneg NT scaling. CPU `direct_x_update_scaling` for nonneg
// applies the primal↔dual swap to slack `update_scaling(s, z)`:
//   slack nonneg stores w² = s/z. Swap gives stored w² = z_direct/x_direct,
//   which is the Hs_inv of the primal log-barrier at x — exactly what
//   the direct-x (1,1) block needs added to P. See the block comment
//   above `_nt_scaling_soc_slack` and CPU mod.rs trait for the
//   full primal↔dual convention.
__global__ void update_xcones_nonneg_scaling_kernel(
    const double* __restrict__ x,                      // [batchSize * n]        — direct-x primal
    const double* __restrict__ z_x,                    // [batchSize * totalXConeNumel] — direct-x dual
    const int64_t* __restrict__ d_xcone_kinds,         // [numXCones]
    const int64_t* __restrict__ d_xcone_dims,          // [numXCones]
    const int64_t* __restrict__ d_xcone_numel_offsets, // [numXCones + 1]
    const int64_t* __restrict__ d_xcone_hs_offsets,    // [numXCones + 1]
    const int64_t* __restrict__ d_xcone_indices,       // [totalXConeNumel]
    double* __restrict__ xcone_w,                      // [batchSize * totalXConeNumel]
    double* __restrict__ xcone_lambda,                 // [batchSize * totalXConeNumel]
    double* __restrict__ xcone_Hs,                     // [batchSize * totalXConeHsEntries]
    // Fused KKT scatter targets (nullable, see SOC kernel comment).
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_px_baseline,
    const int64_t* __restrict__ H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t batch = blockIdx.x;
    const int64_t x_off     = batch * n;
    const int64_t numel_off = batch * totalXConeNumel;
    const int64_t hs_off_b  = batch * totalXConeHsEntries;
    const bool fused = (kkt_values != nullptr) && (xcone_px_baseline != nullptr)
                       && (H_xcone_hs_idx != nullptr);
    const int64_t kkt_off = fused ? batch * nnzKKT : 0;

    // One block per batch; threads stride over (cone, cone-local entry) pairs.
    for (int64_t cone = 0; cone < numXCones; ++cone) {
        // SOC cones handled by update_xcones_soc_scaling.
        if (static_cast<XConeKind>(d_xcone_kinds[cone]) != XConeKind::Nonneg) continue;

        const int64_t dim     = d_xcone_dims[cone];
        const int64_t num_off = d_xcone_numel_offsets[cone];
        const int64_t hs_off  = d_xcone_hs_offsets[cone];

        for (int64_t p = threadIdx.x; p < dim; p += blockDim.x) {
            const int64_t idx = d_xcone_indices[num_off + p];
            // Primal↔dual swap at the slack boundary: slack sees
            // (s_slack=z_direct, z_slack=x_direct). Nonneg is scalar
            // per entry so the swap is just an arg-order rename.
            const double slack_s = z_x[numel_off + num_off + p];  // direct-x dual
            const double slack_z = x[x_off + idx];                // direct-x primal
            // Slack nonneg NT: w = sqrt(s/z), λ = sqrt(s·z), Hs = w².
            // After swap: w = sqrt(z_direct/x_direct), Hs = z_direct/x_direct
            // — the direct-x Hs_inv contribution.
            const double w_i   = sqrt(slack_s / slack_z);
            const double lam_i = sqrt(slack_s * slack_z);
            const double hs_i  = slack_s / slack_z;
            xcone_w     [numel_off + num_off + p] = w_i;
            xcone_lambda[numel_off + num_off + p] = lam_i;
            const int64_t hs_global = hs_off_b + hs_off + p;
            xcone_Hs[hs_global] = hs_i;
            if (fused) {
                kkt_values[kkt_off + H_xcone_hs_idx[hs_off + p]] =
                    xcone_px_baseline[hs_global] + hs_i;
            }
        }
    }
}

void update_xcones_nonneg_scaling(
    const double* x,
    const double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    double* xcone_w,
    double* xcone_lambda,
    double* xcone_Hs,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0 || totalXConeNumel <= 0) return;
    int threads = static_cast<int>(
        totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    update_xcones_nonneg_scaling_kernel<<<grid, threads, 0, stream>>>(
        x, z_x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        xcone_w, xcone_lambda, xcone_Hs,
        kkt_values, xcone_px_baseline, H_xcone_hs_idx, nnzKKT,
        n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// Direct-x nonneg step-math kernels. One block per batch, threads stride
// over direct-x entries. Indices disjoint across x-cones → no atomics.
//
// Each of these kernels mirrors a CPU `direct_x_*` trait method; the
// primal↔dual swap (direct-x dual ↔ slack primal) has already been
// applied at the SCALING step above, so the stored w/λ/η/Hs already
// carry the dual interpretation. These step-math kernels consume those
// stored quantities directly and use direct-x-named arguments — the swap
// is handled once at scaling and not re-applied here. See the top-of-
// section block comment above `_nt_scaling_soc_slack` for details.

// Direct-x nonneg `affine_ds`. CPU trait: `direct_x_affine_ds(out, z)`
// delegates to slack `affine_ds(out, z)`. For nonneg, slack affine_ds
// is `out[i] = λ[i]² = s·z` elementwise. The stored `λ` already
// reflects the primal↔dual swap from scaling (λ = sqrt(z_direct · x_direct)
// for nonneg), so `λ² = x[J] · z_direct` — which we compute directly
// here as `x[idx] * var_z_x[k]` instead of reading λ²  back.
__global__ void fill_step_rhs_zx_nonneg_affine_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ x,
    const double* __restrict__ var_z_x,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_kind_per_entry,
    int64_t n, int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        // SOC entries overwritten by fill_step_rhs_zx_soc_affine.
        if (d_xcone_kind_per_entry[k] != 0) continue;
        const int64_t idx = d_xcone_indices[k];
        step_rhs_z_x[zx_off + k] = x[x_off + idx] * var_z_x[zx_off + k];
    }
}

void fill_step_rhs_zx_nonneg_affine(
    double* step_rhs_z_x,
    const double* x,
    const double* var_z_x,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    fill_step_rhs_zx_nonneg_affine_kernel<<<grid, threads, 0, stream>>>(
        step_rhs_z_x, x, var_z_x, d_xcone_indices,
        d_xcone_kind_per_entry, n, totalXConeNumel);
}

// Direct-x SOC `affine_ds`. CPU trait: `direct_x_affine_ds(out, z)` —
// the symmetric default calls slack `affine_ds(out, z)` unchanged, which
// for SOC is the Jordan square `λ∘λ`. No explicit swap needed here
// because the result depends only on the stored `λ` (which already
// reflects the primal↔dual swap from scaling).
//
// Writes step_rhs.z_x[off..off+dim] = λ∘λ:
//   (λ∘λ)[0]  = ||λ||²
//   (λ∘λ)[i]  = 2·λ[0]·λ[i]  for i ≥ 1
// One (batch, cone) block with blockDim threads cooperating: parallel
// tail-norm reduction for ||λ_tail||², then parallel writes for tail
// entries. Dim-generic (dense + rank-2 both use the same λ).
__global__ void fill_step_rhs_zx_soc_affine_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ xcone_lambda,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t w_off_b = b * totalXConeNumel + num_off;
    const int64_t out_off = b * totalXConeNumel + num_off;

    const double lam0 = xcone_lambda[w_off_b + 0];
    double my_tail_sq = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double li = xcone_lambda[w_off_b + i];
        my_tail_sq += li * li;
    }
    const double tail_sq = cones::block_sum_reduce(my_tail_sq, smem, tid);

    if (tid == 0) {
        step_rhs_z_x[out_off + 0] = lam0 * lam0 + tail_sq;
    }
    const double two_lam0 = 2.0 * lam0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        step_rhs_z_x[out_off + i] = two_lam0 * xcone_lambda[w_off_b + i];
    }
}

void fill_step_rhs_zx_soc_affine(
    double* step_rhs_z_x,
    const double* xcone_lambda,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    fill_step_rhs_zx_soc_affine_kernel<<<grid, block_size, smem_bytes, stream>>>(
        step_rhs_z_x, xcone_lambda,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        totalXConeNumel);
}

// Direct-x nonneg `combined_ds_shift`. CPU trait:
// `direct_x_combined_ds_shift(shift, step_x, step_z, σμ)` delegates to
// slack `combined_ds_shift(shift, step_z=step_x_direct, step_s=step_z_direct, σμ)`.
// For nonneg, slack `_combined_ds_shift_symmetric` computes
//   shift = (W·step_z) ∘ (W⁻¹·step_s) − σμ
// which for scalar nonneg becomes (w·step_z)·(step_s/w) = step_z·step_s,
// so `shift = step_z · step_s − σμ`. After the swap:
//   shift = step_x_direct · (m · step_z_direct) − σμ
// where the Mehrotra factor `m` is folded in locally (the CPU scales
// step.z_x by m in-place just before the call; we apply m at the
// multiply here instead).
__global__ void add_combined_ds_shift_nonneg_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ step_aff_x,
    const double* __restrict__ step_aff_z_x,
    const double* __restrict__ sigma_mu,
    const double* __restrict__ mehrotra_m,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_kind_per_entry,
    int64_t n, int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const double sm = sigma_mu[b];
    const double mb = mehrotra_m[b];
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        if (d_xcone_kind_per_entry[k] != 0) continue;
        const int64_t idx = d_xcone_indices[k];
        step_rhs_z_x[zx_off + k] +=
            step_aff_x[x_off + idx] * (mb * step_aff_z_x[zx_off + k]) - sm;
    }
}

void add_combined_ds_shift_nonneg(
    double* step_rhs_z_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    add_combined_ds_shift_nonneg_kernel<<<grid, threads, 0, stream>>>(
        step_rhs_z_x, step_aff_x, step_aff_z_x, sigma_mu, mehrotra_m,
        d_xcone_indices, d_xcone_kind_per_entry, n, totalXConeNumel);
}

// Direct-x SOC `combined_ds_shift`. CPU trait: `direct_x_combined_ds_shift(
// shift, step_x, step_z, σμ)` calls slack `combined_ds_shift(shift, step_z,
// step_s, σμ)` — primal↔dual swap at the argument level: direct-x primal
// step `step_x` fills slack's `step_z` slot, direct-x dual step `step_z`
// (scaled by Mehrotra m) fills slack's `step_s` slot. The slack math is
//   shift = (W·step_z) ∘ (W⁻¹·step_s) − σμ·e
// which with the swap becomes
//   shift = (W·step_x_direct) ∘ (W⁻¹·m·step_z_direct) − σμ·e.
// step_rhs.z_x[off..off+dim] += shift. One block per (batch, cone) with
// blockDim threads cooperating: three parallel reductions (ζ_a, ζ_d,
// tail_dot) plus per-element parallel writes for the tail shift.
__global__ void add_combined_ds_shift_soc_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ step_aff_x,
    const double* __restrict__ step_aff_z_x,
    const double* __restrict__ sigma_mu,
    const double* __restrict__ mehrotra_m,
    const double* __restrict__ xcone_w,
    const double* __restrict__ xcone_eta,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t w_off_b = b * totalXConeNumel + num_off;

    const double sm  = sigma_mu[b];
    const double mb  = mehrotra_m[b];
    const double eta = xcone_eta[b * numXCones + c];
    const double inv_eta = 1.0 / eta;

    //   y_a[0] = η·(w[0]·a[0] + ζ_a),   ζ_a = Σ_{i≥1} w[i]·a[i]
    //   y_a[i] = η·(ca·w[i] + a[i]),    ca = a[0] + ζ_a/(1+w[0])      (i ≥ 1)
    //   y_d[0] = (1/η)·(w[0]·d[0] − ζ_d),  ζ_d = Σ_{i≥1} w[i]·d[i]
    //   y_d[i] = (1/η)·(cd·w[i] + d[i]),   cd = −d[0] + ζ_d/(1+w[0])   (i ≥ 1)
    //   shift[0]  = y_a·y_d − σμ
    //   shift[i]  = y_a[0]·y_d[i] + y_d[0]·y_a[i]                      (i ≥ 1)

    auto a_at = [&](int64_t i) -> double {
        return step_aff_x[x_off + d_xcone_indices[num_off + i]];
    };
    auto d_at = [&](int64_t i) -> double {
        return mb * step_aff_z_x[zx_off + num_off + i];
    };
    auto w_at = [&](int64_t i) -> double {
        return xcone_w[w_off_b + i];
    };

    // Pass 1: parallel reductions for ζ_a and ζ_d over the tail.
    const double w0 = w_at(0);
    const double a0 = a_at(0);
    const double d0 = d_at(0);
    double my_zeta_a = 0.0, my_zeta_d = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wi = w_at(i);
        my_zeta_a += wi * a_at(i);
        my_zeta_d += wi * d_at(i);
    }
    // Fused 2-value reduction.
    double vals2_zz[2] = { my_zeta_a, my_zeta_d };
    cones::block_sum_reduce_N<2>(vals2_zz, smem, tid);
    const double zeta_a = vals2_zz[0];
    const double zeta_d = vals2_zz[1];

    const double ca = a0 + zeta_a / (1.0 + w0);
    const double cd = -d0 + zeta_d / (1.0 + w0);
    const double y_a_0 = eta * (w0 * a0 + zeta_a);
    const double y_d_0 = inv_eta * (w0 * d0 - zeta_d);

    // Pass 2: parallel reduction for Σ_{i≥1} y_a[i]·y_d[i].
    double my_tail_dot = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wi = w_at(i);
        const double ai = a_at(i);
        const double di = d_at(i);
        const double y_a_i = eta    * (ca * wi + ai);
        const double y_d_i = inv_eta * (cd * wi + di);
        my_tail_dot += y_a_i * y_d_i;
    }
    const double tail_dot = cones::block_sum_reduce(my_tail_dot, smem, tid);

    if (tid == 0) {
        step_rhs_z_x[zx_off + num_off + 0] += y_a_0 * y_d_0 + tail_dot - sm;
    }

    // Pass 3: parallel writes for tail shift entries.
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wi = w_at(i);
        const double ai = a_at(i);
        const double di = d_at(i);
        const double y_a_i = eta    * (ca * wi + ai);
        const double y_d_i = inv_eta * (cd * wi + di);
        step_rhs_z_x[zx_off + num_off + i] += y_a_0 * y_d_i + y_d_0 * y_a_i;
    }
}

void add_combined_ds_shift_soc(
    double* step_rhs_z_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const double* xcone_w,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    add_combined_ds_shift_soc_kernel<<<grid, block_size, smem_bytes, stream>>>(
        step_rhs_z_x, step_aff_x, step_aff_z_x, sigma_mu, mehrotra_m,
        xcone_w, xcone_eta,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

// Direct-x affine KKT-RHS correction, kind-agnostic. CPU trait:
// `direct_x_affine_offset(out, z) → out.copy_from(z)` for any symmetric
// cone — the cone's *dual* slot (z_x) plays the role of slack's primal
// slack in the affine step's c_J. This kernel subtracts c_J = z_x into
// workx at the cone's indices, mirroring the slack path's
// `workx.waxpby(1, Δs_const_term=s, -1, rhs.z)` reduction. Works for
// both nonneg and SOC direct-x because the symmetric default is
// identical across symmetric cones.
__global__ void subtract_xcone_affine_from_workx_kernel(
    double* __restrict__ workx,
    const double* __restrict__ var_z_x,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        const int64_t idx = d_xcone_indices[k];
        workx[x_off + idx] -= var_z_x[zx_off + k];
    }
}

void subtract_xcone_affine_from_workx(
    double* workx,
    const double* var_z_x,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    subtract_xcone_affine_from_workx_kernel<<<grid, threads, 0, stream>>>(
        workx, var_z_x, d_xcone_indices, n, totalXConeNumel);
}

// Direct-x nonneg combined-step KKT-RHS correction. CPU trait:
// `direct_x_combined_offset(out, ds, work, x)` delegates to slack
// `Δs_from_Δz_offset(out, ds, work, z=x_direct)` — primal↔dual swap
// at the final arg. For nonneg, slack Δs_from_Δz_offset is
// `out[i] = ds[i] / z[i]`; after the swap (z := x_direct) this is
// `c_J[i] = rhs.z_x[i] / x[J][i]`, and we subtract c_J from workx at
// the direct-x indices.
__global__ void subtract_xcone_combined_from_workx_kernel(
    double* __restrict__ workx,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ var_x,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_kind_per_entry,
    int64_t n, int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        if (d_xcone_kind_per_entry[k] != 0) continue;
        const int64_t idx = d_xcone_indices[k];
        workx[x_off + idx] -= rhs_z_x[zx_off + k] / var_x[x_off + idx];
    }
}

void subtract_xcone_combined_from_workx(
    double* workx,
    const double* rhs_z_x,
    const double* var_x,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    subtract_xcone_combined_from_workx_kernel<<<grid, threads, 0, stream>>>(
        workx, rhs_z_x, var_x, d_xcone_indices,
        d_xcone_kind_per_entry, n, totalXConeNumel);
}

// Direct-x SOC combined-step offset. CPU trait: `direct_x_combined_offset`
// delegates to slack `Δs_from_Δz_offset(out, ds, work, z)` with the
// primal↔dual swap: direct-x primal `x[J]` is passed into slack's `z`
// slot (slack's "z" is dual; direct-x's "x" is primal, and by self-
// duality the swapped invocation computes the dual-barrier analog).
// The SOC Δs_from_Δz formula (socone.rs:280-301) — using the NT scaling
// (w, λ, η) which was itself computed with the swap at scaling time —
// is:
//
//   resz = (z[0]-||z[1..]||)·(z[0]+||z[1..]||)
//   out  = [z[0], -z[1..]]                             (J-reflection of z)
//   out *= (λ[0]·ds[0] − λ[1..]·ds[1..]) / resz
//   out[0]  += η · (w[1..]·ds[1..])
//   out[i]  += η · (ds[i] + w[1..]·ds[1..] / (1+w[0]) · w[i])    for i ≥ 1
//   out    /= λ[0]
//
// Here we substitute `z := x[J]` (the swap) and `ds := rhs.z_x`.
// Then workx[J] -= out. One block per (batch, cone) with
// `SOC_PARALLEL_BLOCK_SIZE` threads cooperating; dim unbounded.
__global__ void subtract_xcone_combined_from_workx_soc_kernel(
    double* __restrict__ workx,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ var_x,
    const double* __restrict__ xcone_w,
    const double* __restrict__ xcone_lambda,
    const double* __restrict__ xcone_eta,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t w_off_b = b * totalXConeNumel + num_off;
    const double eta = xcone_eta[b * numXCones + c];

    auto z_at  = [&](int64_t i) -> double {
        return var_x[x_off + d_xcone_indices[num_off + i]];
    };
    auto ds_at = [&](int64_t i) -> double {
        return rhs_z_x[zx_off + num_off + i];
    };
    auto w_at  = [&](int64_t i) -> double {
        return xcone_w[w_off_b + i];
    };
    auto lam_at = [&](int64_t i) -> double {
        return xcone_lambda[w_off_b + i];
    };

    // Pass 1: scalars at i=0 and parallel tail-dot reductions.
    const double z0   = z_at(0);
    const double ds0  = ds_at(0);
    const double w0   = w_at(0);
    const double lam0 = lam_at(0);
    double my_z1_sq = 0.0, my_lam1_ds1 = 0.0, my_w1_ds1 = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double zi = z_at(i);
        const double dsi = ds_at(i);
        my_z1_sq    += zi * zi;
        my_lam1_ds1 += lam_at(i) * dsi;
        my_w1_ds1   += w_at(i) * dsi;
    }
    // Fused 3-value reduction.
    double vals3[3] = { my_z1_sq, my_lam1_ds1, my_w1_ds1 };
    cones::block_sum_reduce_N<3>(vals3, smem, tid);
    const double z1_sq    = vals3[0];
    const double lam1_ds1 = vals3[1];
    const double w1_ds1   = vals3[2];

    const double z1_norm = sqrt(z1_sq);
    const double resz = (z0 - z1_norm) * (z0 + z1_norm);
    const double cfac = (lam0 * ds0 - lam1_ds1) / resz;
    const double inv_lam0 = 1.0 / lam0;
    const double w0p1 = 1.0 + w0;

    // Pass 2: write outputs in parallel.
    if (tid == 0) {
        const double out0 = inv_lam0 * (cfac * z0 + eta * w1_ds1);
        workx[x_off + d_xcone_indices[num_off + 0]] -= out0;
    }
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const int64_t idx = d_xcone_indices[num_off + i];
        const double zi = z_at(i);
        const double dsi = ds_at(i);
        const double wi = w_at(i);
        // out[i] = -z[i]·cfac + η·(ds[i] + (w1_ds1/(1+w0))·w[i]), then /λ0
        const double out_i = inv_lam0 *
            (cfac * (-zi) + eta * (dsi + (w1_ds1 / w0p1) * wi));
        workx[x_off + idx] -= out_i;
    }
}

void subtract_xcone_combined_from_workx_soc(
    double* workx,
    const double* rhs_z_x,
    const double* var_x,
    const double* xcone_w,
    const double* xcone_lambda,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    subtract_xcone_combined_from_workx_soc_kernel<<<grid, block_size, smem_bytes, stream>>>(
        workx, rhs_z_x, var_x, xcone_w, xcone_lambda, xcone_eta,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

// Direct-x nonneg Δz_J recovery (affine step). CPU `recover_direct_x_dual`:
//   Δz_J = -direct_x_mul_Hs(Δx[J]) - c_J_affine
// where c_J_affine = variables.z_x[J] (from direct_x_affine_offset).
// For nonneg, `direct_x_mul_Hs` delegates to slack `mul_Hs` which is
// scalar elementwise: `(Hs·x)[i] = Hs[i] · x[i]`. The stored `Hs` is
// `z_direct / x_direct` (Hs_inv of the primal barrier — see scaling
// kernel comment), so this gives the correct (1,1)-block recovery.
//
// Layout note: for nonneg the Hs slot index equals the cone-local numel
// index (1:1). SOC entries are packed differently (upper-tri, dim×(dim+1)/2)
// and are handled by recover_dz_x_affine_soc.
__global__ void recover_dz_x_affine_nonneg_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_kind_per_entry,
    int64_t n, int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;
    const int64_t hs_off = b * totalXConeHsEntries;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        if (d_xcone_kind_per_entry[k] != 0) continue;
        const int64_t idx = d_xcone_indices[k];
        const double Hs_k = xcone_Hs[hs_off + k];
        dz_x[zx_off + k] = -Hs_k * lhs_x[x_off + idx] - var_z_x[zx_off + k];
    }
}

void recover_dz_x_affine_nonneg(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_affine_nonneg_kernel<<<grid, threads, 0, stream>>>(
        dz_x, lhs_x, var_z_x, xcone_Hs, d_xcone_indices,
        d_xcone_kind_per_entry, n, totalXConeNumel, totalXConeHsEntries);
}

// Direct-x nonneg Δz_J recovery (combined/centering step). Same
// recipe as the affine version but with c_J = rhs.z_x / x[J] (the
// nonneg specialization of `direct_x_combined_offset` — slack
// Δs_from_Δz_offset `ds/z` with the primal↔dual-swapped z := x).
__global__ void recover_dz_x_combined_nonneg_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ var_x,
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_kind_per_entry,
    int64_t n, int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;
    const int64_t hs_off = b * totalXConeHsEntries;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        if (d_xcone_kind_per_entry[k] != 0) continue;
        const int64_t idx = d_xcone_indices[k];
        const double Hs_k = xcone_Hs[hs_off + k];
        const double c_k = rhs_z_x[zx_off + k] / var_x[x_off + idx];
        dz_x[zx_off + k] = -Hs_k * lhs_x[x_off + idx] - c_k;
    }
}

void recover_dz_x_combined_nonneg(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* var_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_combined_nonneg_kernel<<<grid, threads, 0, stream>>>(
        dz_x, lhs_x, rhs_z_x, var_x, xcone_Hs, d_xcone_indices,
        d_xcone_kind_per_entry, n, totalXConeNumel, totalXConeHsEntries);
}

// Direct-x SOC Δz_J recovery. CPU trait: `recover_direct_x_dual`
// computes Δz_J = -direct_x_mul_Hs(Δx[J]) - c_J, where
//   Affine:    c_J = variables.z_x[J]                   (direct_x_affine_offset)
//   Combined:  c_J = Δs_from_Δz_offset(rhs.z_x, x[J])   (direct_x_combined_offset)
//
// `direct_x_mul_Hs` is the symmetric-default delegation to slack
// `mul_Hs`, using the stored NT quantities — which were computed with
// the primal↔dual swap at scaling time, so `Hs` stored here IS the
// direct-x Hessian (= slack Hs_inv of the primal barrier). Applying it
// to Δx[J] therefore gives the correct map Δx → Δz for the (1,1)
// recovery. See the section block comment for the convention.
//
// Uses the analytic SOC mul_Hs form (socone.rs:262-270):
//   y = η²·(2·(w·Δx)·w − J·Δx),  J = diag(1, -I)
// Reads only w and η — works for both dense (dim ≤ 4) and rank-2
// (dim > 4) storage since neither is touched here.
//
// One block per (batch, cone) with blockDim threads cooperating:
// parallel reduction for w·Δx then per-element parallel writes.
__global__ void recover_dz_x_affine_soc_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ xcone_w,
    const double* __restrict__ xcone_eta,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t w_off_b = b * totalXConeNumel + num_off;

    const double eta = xcone_eta[b * numXCones + c];
    const double eta_sq = eta * eta;

    auto dx_at = [&](int64_t i) -> double {
        return lhs_x[x_off + d_xcone_indices[num_off + i]];
    };
    auto w_at  = [&](int64_t i) -> double {
        return xcone_w[w_off_b + i];
    };

    // Pass 1: wdx = Σ w[i]·Δx[i] via parallel reduction over all i.
    double my_wdx = 0.0;
    for (int64_t i = tid; i < dim; i += blockDimX) {
        my_wdx += w_at(i) * dx_at(i);
    }
    const double wdx = cones::block_sum_reduce(my_wdx, smem, tid);
    const double two_wdx = 2.0 * wdx;

    // Pass 2: parallel writes. y[0] = η²·(c·w[0] - dx[0]); y[i ≥ 1] = η²·(c·w[i] + dx[i])
    if (tid == 0) {
        const double w0  = w_at(0);
        const double dx0 = dx_at(0);
        const double y0  = eta_sq * (two_wdx * w0 - dx0);
        dz_x[zx_off + num_off + 0] = -y0 - var_z_x[zx_off + num_off + 0];
    }
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double yi = eta_sq * (two_wdx * w_at(i) + dx_at(i));
        dz_x[zx_off + num_off + i] = -yi - var_z_x[zx_off + num_off + i];
    }
}

void recover_dz_x_affine_soc(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_w,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    recover_dz_x_affine_soc_kernel<<<grid, block_size, smem_bytes, stream>>>(
        dz_x, lhs_x, var_z_x, xcone_w, xcone_eta,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

__global__ void recover_dz_x_combined_soc_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ var_x,
    const double* __restrict__ xcone_w,
    const double* __restrict__ xcone_lambda,
    const double* __restrict__ xcone_eta,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t w_off_b = b * totalXConeNumel + num_off;
    const double eta = xcone_eta[b * numXCones + c];
    const double eta_sq = eta * eta;

    auto dx_at = [&](int64_t i) -> double {
        return lhs_x[x_off + d_xcone_indices[num_off + i]];
    };
    auto ds_at = [&](int64_t i) -> double {
        return rhs_z_x[zx_off + num_off + i];
    };
    auto z_at  = [&](int64_t i) -> double {
        return var_x[x_off + d_xcone_indices[num_off + i]];
    };
    auto w_at  = [&](int64_t i) -> double {
        return xcone_w[w_off_b + i];
    };
    auto lam_at = [&](int64_t i) -> double {
        return xcone_lambda[w_off_b + i];
    };

    // Pass 1: parallel reductions for all four scalar accumulators.
    //   wdx      = Σ w[i]·Δx[i]              (for Hs·Δx via analytic mul_Hs)
    //   z1_sq    = Σ_{i≥1} z[i]²             (for resz)
    //   lam1_ds1 = Σ_{i≥1} λ[i]·ds[i]        (for cfac)
    //   w1_ds1   = Σ_{i≥1} w[i]·ds[i]        (for c_J tail)
    const double z0 = z_at(0);
    const double ds0 = ds_at(0);
    const double w0 = w_at(0);
    const double lam0 = lam_at(0);
    // wdx covers the i=0 term too — split out so the tail loop only carries
    // the (i ≥ 1) accumulators z1_sq/lam1_ds1/w1_ds1.
    double my_wdx_tail = 0.0, my_z1_sq = 0.0, my_lam1_ds1 = 0.0, my_w1_ds1 = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wi = w_at(i);
        const double zi = z_at(i);
        const double dsi = ds_at(i);
        my_wdx_tail += wi * dx_at(i);
        my_z1_sq    += zi * zi;
        my_lam1_ds1 += lam_at(i) * dsi;
        my_w1_ds1   += wi * dsi;
    }
    // Fused 4-value reduction.
    double vals4[4] = { my_wdx_tail, my_z1_sq, my_lam1_ds1, my_w1_ds1 };
    cones::block_sum_reduce_N<4>(vals4, smem, tid);
    const double wdx_tail = vals4[0];
    const double z1_sq    = vals4[1];
    const double lam1_ds1 = vals4[2];
    const double w1_ds1   = vals4[3];
    const double wdx = w0 * dx_at(0) + wdx_tail;

    const double two_wdx = 2.0 * wdx;
    const double z1_norm = sqrt(z1_sq);
    const double resz = (z0 - z1_norm) * (z0 + z1_norm);
    const double cfac = (lam0 * ds0 - lam1_ds1) / resz;
    const double inv_lam0 = 1.0 / lam0;
    const double w0p1 = 1.0 + w0;

    // Pass 2: parallel writes.
    //   y[i]  = η²·(2(w·Δx)·w[i] − J_ii·Δx[i])
    //   cj[0] = (+z[0])·cfac + η·w1_ds1                  /λ0
    //   cj[i] = (−z[i])·cfac + η·(ds[i] + (w1_ds1/(1+w0))·w[i])  /λ0
    //   dz_x[i] = -y[i] - cj[i]
    if (tid == 0) {
        const double dx0 = dx_at(0);
        const double y0v = eta_sq * (two_wdx * w0 - dx0);
        const double cj0 = inv_lam0 * (cfac * z0 + eta * w1_ds1);
        dz_x[zx_off + num_off + 0] = -y0v - cj0;
    }
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const double wi = w_at(i);
        const double zi = z_at(i);
        const double dsi = ds_at(i);
        const double dxi = dx_at(i);
        const double yi = eta_sq * (two_wdx * wi + dxi);
        const double cji = inv_lam0 *
            (cfac * (-zi) + eta * (dsi + (w1_ds1 / w0p1) * wi));
        dz_x[zx_off + num_off + i] = -yi - cji;
    }
}

void recover_dz_x_combined_soc(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* var_x,
    const double* xcone_w,
    const double* xcone_lambda,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    recover_dz_x_combined_soc_kernel<<<grid, block_size, smem_bytes, stream>>>(
        dz_x, lhs_x, rhs_z_x, var_x,
        xcone_w, xcone_lambda, xcone_eta,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

// Margin-based cone-interior shift for direct-x nonneg. Mirror of CPU
// `_shift_single_cone_to_interior` (variables.rs) applied to `x[J]`
// with `PrimalOrDualCone::PrimalCone` — this is the primal side of
// direct-x (the cone's `x`), using `margins()` and `scaled_unit_shift()`
// specialized to nonneg. One block per (batch, cone) with blockDim
// threads cooperating via `cones::block_min_reduce` for the min margin
// and `cones::block_sum_reduce` for the positive-sum.
__global__ void shift_x_into_nonneg_interior_kernel(
    double* __restrict__ x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    // Only nonneg cones (kind == 0) are handled here.
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Nonneg) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off = b * n;

    double my_mn = 1e300;
    double my_ps = 0.0;
    for (int64_t p = tid; p < dim; p += blockDimX) {
        const int64_t idx = d_xcone_indices[num_off + p];
        const double v = x[x_off + idx];
        if (v < my_mn) my_mn = v;
        if (v > 0.0) my_ps += v;
    }
    const double mn = cones::block_min_reduce(my_mn, smem, tid);
    const double ps = cones::block_sum_reduce(my_ps, smem, tid);

    const double dim_d = static_cast<double>(dim);
    double target = 0.1 * ps / dim_d;
    if (target < 1.0) target = 1.0;

    // Mirror CPU _shift_single_cone_to_interior:
    //   if min <= 0: shift = -min + target
    //   elif min < target: shift = target - min
    //   else: shift = 0
    double shift = 0.0;
    if (mn <= 0.0) shift = -mn + target;
    else if (mn < target) shift = target - mn;

    if (shift != 0.0) {
        for (int64_t p = tid; p < dim; p += blockDimX) {
            const int64_t idx = d_xcone_indices[num_off + p];
            x[x_off + idx] += shift;
        }
    }
}

void shift_x_into_nonneg_interior(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    shift_x_into_nonneg_interior_kernel<<<grid, block_size, smem_bytes, stream>>>(
        x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_indices, n);
}

// Margin-based cone-interior shift for direct-x SOC.
// Mirror of CPU `_shift_single_cone_to_interior` applied to `x[J]` on
// the primal side, specialized to SOC:
//   SOC margin (primal):  min_margin = x[0] − ||x[1..]||
//   target (degree=1):    target = max(1, 0.1 · max(0, min_margin))
//   scaled_unit_shift(α): x[0] += α  (SOC-specific: only touches the
//                         scalar part). We collapse CPU's two-stage
//                         "shift by -min then by target" into one
//                         `x[0] += target − min_margin` — equivalent
//                         exactly because scaled_unit_shift is additive
//                         on `x[0]`.
__global__ void shift_x_into_soc_interior_kernel(
    double* __restrict__ x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off = b * n;

    const double x0 = x[x_off + d_xcone_indices[num_off + 0]];
    double my_tail_sq = 0.0;
    for (int64_t p = 1 + tid; p < dim; p += blockDimX) {
        const double v = x[x_off + d_xcone_indices[num_off + p]];
        my_tail_sq += v * v;
    }
    const double tail_sq = cones::block_sum_reduce(my_tail_sq, smem, tid);

    if (tid == 0) {
        const double tail = sqrt(tail_sq);
        const double margin = x0 - tail;
        const double pos_margin = (margin > 0.0) ? margin : 0.0;
        // degree = 1 for SOC
        double target = 0.1 * pos_margin;
        if (target < 1.0) target = 1.0;

        double shift = 0.0;
        if (margin <= 0.0) shift = -margin + target;
        else if (margin < target) shift = target - margin;
        if (shift != 0.0) {
            x[x_off + d_xcone_indices[num_off + 0]] = x0 + shift;
        }
    }
}

void shift_x_into_soc_interior(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    shift_x_into_soc_interior_kernel<<<grid, block_size, smem_bytes, stream>>>(
        x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_indices, n);
}

// Initialize x[J] for direct-x Exp cones to the unit-init point. The
// standard ExpCone self-conjugate point is used — same point for both
// primal and dual. Mirrors CPU direct_x_unit_initialization. One block
// per batch, threads stride over cones, filter by kind == 3.
__global__ void init_xcone_x_exp_kernel(
    double* __restrict__ x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;

    static constexpr double pt[3] = {
        -1.051383945322714,
         0.556409619469370,
         1.258967884768947
    };
    for (int i = 0; i < 3; ++i) {
        x[x_off + d_xcone_indices[num_off + i]] = pt[i];
    }
    } // end cone loop
}

void init_xcone_x_exp(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    init_xcone_x_exp_kernel<<<grid, 256, 0, stream>>>(
        x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_indices, n, numXCones);
}

// Initialize z_x (the direct-x DUAL; see section block comment) to a
// cone-interior default per kind. Mirror of CPU
// `direct_x_unit_initialization`, which delegates to slack
// `unit_initialization(z, s)` with the primal↔dual swap — direct-x's
// dual fills slack's primal slot, and for symmetric cones the unit
// initialization is the same on both sides, giving:
//   Nonneg: z_x[k] = 1.0     (positive orthant interior)
//   SOC:    z_x[off + 0] = 1, z_x[off + i>=1] = 0   (= e_0, SOC interior)
__global__ void init_xcone_z_x_kernel(
    double* __restrict__ z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t zx_off = b * totalXConeNumel + num_off;
    const int64_t kind = d_xcone_kinds[c];

    if (kind == 0 /* Nonneg */) {
        for (int64_t p = tid; p < dim; p += blockDimX) z_x[zx_off + p] = 1.0;
    } else if (kind == 1 /* SOC */) {
        if (tid == 0) z_x[zx_off + 0] = 1.0;
        for (int64_t p = 1 + tid; p < dim; p += blockDimX) z_x[zx_off + p] = 0.0;
    } else if (kind == 2 /* PSD */) {
        // PSD direct-x: dim is svec_dim = k(k+1)/2. Identity matrix in svec
        // form has 1.0 at the k diagonal positions (col*(col+1)/2 + col)
        // and 0 elsewhere. Zero everything in parallel, then overwrite
        // the diagonal slots from thread 0 (k is small relative to dim).
        for (int64_t p = tid; p < dim; p += blockDimX) z_x[zx_off + p] = 0.0;
        __syncthreads();
        if (tid == 0) {
            int64_t col = 0;
            int64_t pos = 0;
            while (pos < dim) {
                int64_t diag_pos = (col + 1) * (col + 2) / 2 - 1;
                if (diag_pos >= dim) break;
                z_x[zx_off + diag_pos] = 1.0;
                pos = diag_pos + 1;
                ++col;
            }
        }
    } else if (kind == 3 /* Exp */) {
        // Asymmetric direct-x Exp: initialize z_x to the standard
        // self-dual unit point. Matches CPU `direct_x_unit_initialization`
        // for ExpCone. Power (kind=4) does NOT share this point — the
        // caller invokes `init_xcone_z_x_pow` after this kernel to write
        // the correct Power point (sqrt(1+α), sqrt(2-α), 0). GenPower
        // (kind=5) is handled by `init_xcone_z_x_genpow`.
        if (tid == 0) {
            z_x[zx_off + 0] = -1.051383945322714;
            z_x[zx_off + 1] =  0.556409619469370;
            z_x[zx_off + 2] =  1.258967884768947;
        }
    }
}

void init_xcone_z_x(
    double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0 || totalXConeNumel <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    init_xcone_z_x_kernel<<<grid, block_size, 0, stream>>>(
        z_x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        totalXConeNumel);
}

__global__ void xcone_step_length_nonneg_reduce_kernel(
    double* __restrict__ alpha_s,      // [batchSize] — min-reduced in-place
    double* __restrict__ alpha_z,      // [batchSize] — min-reduced in-place
    const double* __restrict__ var_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ step_x,
    const double* __restrict__ step_z_x,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_kind_per_entry,
    int64_t n, int64_t totalXConeNumel,
    double max_step)
{
    const int64_t b = blockIdx.x;
    const int64_t x_off  = b * n;
    const int64_t zx_off = b * totalXConeNumel;

    double local_s = max_step;
    double local_z = max_step;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        if (d_xcone_kind_per_entry[k] != 0) continue;
        const int64_t idx = d_xcone_indices[k];
        const double xi   = var_x[x_off + idx];
        const double dxi  = step_x[x_off + idx];
        const double zi   = var_z_x[zx_off + k];
        const double dzi  = step_z_x[zx_off + k];
        if (dxi < 0.0) {
            const double r = -xi / dxi;
            if (r < local_s) local_s = r;
        }
        if (dzi < 0.0) {
            const double r = -zi / dzi;
            if (r < local_z) local_z = r;
        }
    }
    // Warp reduction for min.
    for (int off = 16; off > 0; off >>= 1) {
        const double other_s = __shfl_down_sync(0xffffffff, local_s, off);
        const double other_z = __shfl_down_sync(0xffffffff, local_z, off);
        if (other_s < local_s) local_s = other_s;
        if (other_z < local_z) local_z = other_z;
    }
    __shared__ double warp_s[32];
    __shared__ double warp_z[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_s[warp] = local_s; warp_z[warp] = local_z; }
    __syncthreads();
    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        double s = (threadIdx.x < num_warps) ? warp_s[threadIdx.x] : max_step;
        double z = (threadIdx.x < num_warps) ? warp_z[threadIdx.x] : max_step;
        for (int off = 16; off > 0; off >>= 1) {
            const double o_s = __shfl_down_sync(0xffffffff, s, off);
            const double o_z = __shfl_down_sync(0xffffffff, z, off);
            if (o_s < s) s = o_s;
            if (o_z < z) z = o_z;
        }
        if (threadIdx.x == 0) {
            if (s < alpha_s[b]) alpha_s[b] = s;
            if (z < alpha_z[b]) alpha_z[b] = z;
        }
    }
}

void xcone_step_length_nonneg_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    double max_step,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    // IMPORTANT: always launch at least one full warp's worth of
    // threads (32). The kernel's warp shuffle intrinsics require all
    // 32 lanes to participate — launching fewer leaves lanes with
    // undefined register values, which __shfl_down_sync can propagate
    // as garbage into the min-reduction. Inactive lanes see k ≥
    // totalXConeNumel so their local_s/local_z stay at max_step.
    const int threads = 256;
    dim3 grid(static_cast<unsigned int>(batchSize));
    xcone_step_length_nonneg_reduce_kernel<<<grid, threads, 0, stream>>>(
        alpha_s, alpha_z, var_x, var_z_x, step_x, step_z_x,
        d_xcone_indices, d_xcone_kind_per_entry,
        n, totalXConeNumel, max_step);
}

// Direct-x SOC step length. CPU trait: `direct_x_step_length(dx, dz,
// x, z, αmax)` delegates to slack `step_length(dz, ds, z, s, αmax)`
// with the primal↔dual swap: direct-x primal pair `(x, dx)` fills
// slack's `(z, dz)` slots and direct-x dual pair `(z, dz)` fills
// slack's `(s, ds)` slots. The slack SOC step_length runs
// `_step_length_soc_component` twice — once per side — so after the
// swap we compute one quadratic min-ratio on (x[J], Δx[J]) and another
// on (z_x, Δz_x). Each solves
//   ||x_{1..} + α·y_{1..}||² = (x_0 + α·y_0)²
// and returns the minimum positive root up to αmax. One block per
// (batch, cone) with `SOC_PARALLEL_BLOCK_SIZE` threads cooperating
// (six tail reductions: x1_sq, y1_sq, xy1 per side); dim unbounded.
// Inline quadratic step-root solver for SOC: given scalars at position 0
// and tail accumulators (x1_sq, y1_sq, xy1), return the minimum positive
// α ≤ amax such that `x + α·y` stays in the SOC. Shared by both primal
// and dual step-length passes; caller gathers the scalars and tail
// accumulations appropriately.
__device__ __forceinline__ double _soc_step_root_from_accums(
    double x0, double y0,
    double x1_sq, double y1_sq, double xy1,
    double amax)
{
    double am = amax;
    if (x0 >= 0.0 && y0 < 0.0) {
        const double r = -x0 / y0;
        if (r < am) am = r;
    }
    // From socone.rs: a = resid(y), b = 2·(x[0]·y[0] − x_tail·y_tail),
    // c = max(0, resid(x)).
    const double a = (y0 - sqrt(y1_sq)) * (y0 + sqrt(y1_sq));
    const double b = 2.0 * (x0 * y0 - xy1);
    double c = (x0 - sqrt(x1_sq)) * (x0 + sqrt(x1_sq));
    if (c < 0.0) c = 0.0;
    const double d = b * b - 4.0 * a * c;

    if ((a > 0.0 && b > 0.0) || d < 0.0) return am;   // no positive root
    if (a == 0.0) return am;                          // linear; b ≥ 0 by duality
    if (c == 0.0) return (a >= 0.0) ? am : 0.0;       // one root at 0

    const double sqrt_d = sqrt(d);
    const double t = (b >= 0.0) ? (-b - sqrt_d) : (-b + sqrt_d);
    double r1 = (2.0 * c) / t;
    double r2 = t / (2.0 * a);
    if (r1 < 0.0) r1 = 1e300;
    if (r2 < 0.0) r2 = 1e300;
    double r = (r1 < r2) ? r1 : r2;
    return (am < r) ? am : r;
}

__global__ void xcone_step_length_soc_reduce_kernel(
    double* __restrict__ alpha_s,
    double* __restrict__ alpha_z,
    const double* __restrict__ var_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ step_x,
    const double* __restrict__ step_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::SOC) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t dim = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    // Index-0 scalars (all threads read; cheap broadcast through cache).
    const int64_t idx0 = d_xcone_indices[num_off + 0];
    const double xp0 = var_x[x_off + idx0];
    const double yp0 = step_x[x_off + idx0];
    const double xd0 = var_z_x[zx_off + num_off + 0];
    const double yd0 = step_z_x[zx_off + num_off + 0];

    // Single tail loop accumulates both primal and dual sums.
    double my_xp_sq = 0.0, my_yp_sq = 0.0, my_xyp = 0.0;
    double my_xd_sq = 0.0, my_yd_sq = 0.0, my_xyd = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        const int64_t idx = d_xcone_indices[num_off + i];
        const double xpi = var_x[x_off + idx];
        const double ypi = step_x[x_off + idx];
        my_xp_sq += xpi * xpi;
        my_yp_sq += ypi * ypi;
        my_xyp   += xpi * ypi;
        const double xdi = var_z_x[zx_off + num_off + i];
        const double ydi = step_z_x[zx_off + num_off + i];
        my_xd_sq += xdi * xdi;
        my_yd_sq += ydi * ydi;
        my_xyd   += xdi * ydi;
    }
    // Fused 6-value reduction: one shared-mem round-trip instead of six.
    double vals6[6] = { my_xp_sq, my_yp_sq, my_xyp, my_xd_sq, my_yd_sq, my_xyd };
    cones::block_sum_reduce_N<6>(vals6, smem, tid);
    const double xp_sq = vals6[0];
    const double yp_sq = vals6[1];
    const double xyp   = vals6[2];
    const double xd_sq = vals6[3];
    const double yd_sq = vals6[4];
    const double xyd   = vals6[5];

    if (tid != 0) return;

    // α_z from primal boundary (x[J], Δx[J]); α_s from dual (z_x, Δz_x).
    // Matches CPU mapping (dx, dz, x, z) → (dz, ds, z, s) → αz on primal.
    const double a_primal = _soc_step_root_from_accums(
        xp0, yp0, xp_sq, yp_sq, xyp, max_step);
    const double a_dual   = _soc_step_root_from_accums(
        xd0, yd0, xd_sq, yd_sq, xyd, max_step);    cones::atomic_min_pos_double(&alpha_z[b], a_primal);
    cones::atomic_min_pos_double(&alpha_s[b], a_dual);
}

void xcone_step_length_soc_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    xcone_step_length_soc_reduce_kernel<<<grid, block_size, smem_bytes, stream>>>(
        alpha_s, alpha_z, var_x, var_z_x, step_x, step_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel, max_step);
}

__global__ void snapshot_kkt_at_xcone_slots_kernel(
    const double* __restrict__ kkt_values,
    double* __restrict__ xcone_px_baseline,
    const int64_t* __restrict__ H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t totalXConeHsEntries)
{
    int64_t batch = blockIdx.x;
    const int64_t kkt_off = batch * nnzKKT;
    const int64_t buf_off = batch * totalXConeHsEntries;
    for (int64_t k = threadIdx.x; k < totalXConeHsEntries; k += blockDim.x) {
        int64_t slot = H_xcone_hs_idx[k];
        xcone_px_baseline[buf_off + k] = kkt_values[kkt_off + slot];
    }
}

void snapshot_kkt_at_xcone_slots(
    const double* kkt_values,
    double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (totalXConeHsEntries <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(
        totalXConeHsEntries < 256 ? totalXConeHsEntries : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    snapshot_kkt_at_xcone_slots_kernel<<<grid, threads, 0, stream>>>(
        kkt_values, xcone_px_baseline, H_xcone_hs_idx,
        nnzKKT, totalXConeHsEntries);
}

// Seed xcone_Hs with the identity-scaling Hs per cone. Must be called
// once in default_start before the first refresh_xcone_hs, since the
// generic setToConstant(1.0) only works for diagonal-packed storage
// (nonneg any dim; sparse SOC dim > 4). For dense SOC (dim ≤ 4) the
// correct identity Hs is the 3×3 or 4×4 identity matrix packed as
// column-major upper-tri, which is `[1, 0, 1, 0, 0, 1, ...]` — NOT
// all-ones. Getting this wrong makes the initial KKT factor strongly
// rank-deficient (all-ones is rank-1), blowing up μ ~200× at iter 0
// and costing ~10 extra IPM iterations to recover.
__global__ void seed_xcone_Hs_identity_kernel(
    double* __restrict__ xcone_Hs,                     // [batchSize * totalXConeHsEntries]
    const int64_t* __restrict__ d_xcone_kinds,         // [numXCones]
    const int64_t* __restrict__ d_xcone_dims,          // [numXCones]
    const int64_t* __restrict__ d_xcone_hs_offsets,    // [numXCones + 1]
    int64_t numXCones,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    const int64_t kind = d_xcone_kinds[c];
    const int64_t dim  = d_xcone_dims[c];
    const int64_t hs_off_b = b * totalXConeHsEntries + d_xcone_hs_offsets[c];

    // Packed column-major upper-tri identity: entry at (col cc, row rr ≤ cc)
    // lives at offset cc*(cc+1)/2 + rr; identity has value 1 iff rr == cc.
    // Recover cc from pos via the triangular-root formula
    // cc = floor((-1 + sqrt(1 + 8·pos)) / 2); a correction step handles
    // floating-point rounding for large pos (matters for PSD svec_dim ≫ 1).
    auto write_packed_uppertri_identity = [&](int64_t d) {
        const int64_t total = d * (d + 1) / 2;
        for (int64_t pos = tid; pos < total; pos += blockDimX) {
            int64_t cc = (int64_t)((sqrt((double)(8 * pos + 1)) - 1.0) * 0.5);
            int64_t base = cc * (cc + 1) / 2;
            if (base > pos) {
                --cc;
                base = cc * (cc + 1) / 2;
            } else if (base + cc + 1 <= pos) {
                ++cc;
                base = cc * (cc + 1) / 2;
            }
            const int64_t rr = pos - base;
            xcone_Hs[hs_off_b + pos] = (rr == cc) ? 1.0 : 0.0;
        }
    };

    if (kind == 0 /* Nonneg */) {
        // Nonneg: diagonal packed, identity = all ones.
        for (int64_t i = tid; i < dim; i += blockDimX) xcone_Hs[hs_off_b + i] = 1.0;
    } else if (kind == 1 /* SOC */) {
        if (dim <= 4) {
            write_packed_uppertri_identity(dim);
        } else {
            // Rank-2 sparse SOC: diagonal-only storage. Identity Hs at
            // identity NT scaling is η²·[d, 1, 1, ..., 1] with η=1, d=0.5
            // (CPU set_identity_scaling sets d=0.5). Leading entry is 0.5.
            if (tid == 0) xcone_Hs[hs_off_b + 0] = 0.5;
            for (int64_t i = 1 + tid; i < dim; i += blockDimX) xcone_Hs[hs_off_b + i] = 1.0;
        }
    } else if (kind == 2 /* PSD */) {
        // PSD direct-x: dim is the svec_dim. Identity Hs at the identity
        // NT scaling is the svec_dim × svec_dim identity (skron(I) = I
        // in svec space). Packed column-major upper-tri: 1 on diagonal,
        // 0 off-diagonal — same layout as SOC dense.
        write_packed_uppertri_identity(dim);
    } else if (kind == 3 /* Exp */ || kind == 4 /* Power */) {
        // Asymmetric direct-x Exp/Power: 3×3 dense Hs (packed upper-tri =
        // 6 entries). Iter-0 scaling kernel overwrites these from the
        // closed-form primal-barrier Hessian; identity here is a safe
        // placeholder pre-scaling.
        write_packed_uppertri_identity(dim);
    } else if (kind == 5 /* GenPower */) {
        // GenPowerCone: sparse (dim>4) → diagonal-only Hs with dim entries;
        // dense (dim<=4) → packed upper-tri Hs. Identity = 1 on diagonal.
        // Expansion column vectors (q/r/p) are zero-initialized via cudaMemset
        // and are written by refresh_xcone_genpow_expansion on the first call.
        if (dim > 4) {
            for (int64_t i = tid; i < dim; i += blockDimX) xcone_Hs[hs_off_b + i] = 1.0;
        } else {
            write_packed_uppertri_identity(dim);
        }
    }
}

void seed_xcone_Hs_identity(
    double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_hs_offsets,
    int64_t batchSize,
    int64_t numXCones,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0 || totalXConeHsEntries <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numXCones));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    seed_xcone_Hs_identity_kernel<<<grid, block_size, 0, stream>>>(
        xcone_Hs, d_xcone_kinds, d_xcone_dims, d_xcone_hs_offsets,
        numXCones, totalXConeHsEntries);
}

__global__ void refresh_xcone_hs_kernel(
    double* __restrict__ kkt_values,                       // [batchSize * nnzKKT]
    const double* __restrict__ xcone_Hs,                   // [batchSize * totalXConeHsEntries]
    const double* __restrict__ xcone_px_baseline,          // [batchSize * totalXConeHsEntries]
    const int64_t* __restrict__ H_xcone_hs_idx,            // [totalXConeHsEntries]
    int64_t nnzKKT,
    int64_t totalXConeHsEntries)
{
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;
    const int64_t kkt_off = batch * nnzKKT;
    const int64_t buf_off = batch * totalXConeHsEntries;

    for (int64_t k = threadIdx.x; k < totalXConeHsEntries; k += stride) {
        int64_t slot = H_xcone_hs_idx[k];
        kkt_values[kkt_off + slot] =
            xcone_px_baseline[buf_off + k] + xcone_Hs[buf_off + k];
    }
}

void refresh_xcone_hs(
    double* kkt_values,
    const double* xcone_Hs,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (totalXConeHsEntries <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(
        totalXConeHsEntries < 256 ? totalXConeHsEntries : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    refresh_xcone_hs_kernel<<<grid, threads, 0, stream>>>(
        kkt_values, xcone_Hs, xcone_px_baseline, H_xcone_hs_idx,
        nnzKKT, totalXConeHsEntries);
}

// Refresh the rank-2 expansion columns and diagonal for direct-x sparse
// SOC cones. Mirror of CPU `refresh_hx_blocks` sparse branch
// (directldlkktsolver.rs:211-243):
//   KKT[H_xcone_v_idx[flat]]         = +η² · v[cone_pos_for_sorted[flat]]
//   KKT[H_xcone_u_idx[flat]]         = +η² · u[cone_pos_for_sorted[flat]]
//   KKT[H_xcone_exp_diag_idx[2*c]]   = +η²    (v-col diag, direct-x sign)
//   KKT[H_xcone_exp_diag_idx[2*c+1]] = -η²    (u-col diag, direct-x sign)
//
// Signs are OPPOSITE of slack's `-η²·u, -η²·v, [-η², +η²]` because
// direct-x contributes `+Hs` (not `-Hs`) to the (1,1) block; the two
// sign flips (columns and diag) compose to give `+Hs` after Schur
// elimination of the 2 expansion cols.
__global__ void refresh_xcone_sparse_expansion_kernel(
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_u,
    const double* __restrict__ xcone_v,
    const double* __restrict__ xcone_eta,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_sparse_indices,    // [numXCones]
    const int64_t* __restrict__ d_xcone_sparse_offsets,    // [numXCones+1]
    const int64_t* __restrict__ d_xcone_cone_pos_for_sorted, // [totalXConeNumel]
    const int64_t* __restrict__ d_xcone_numel_offsets,     // [numXCones+1]
    const int64_t* __restrict__ H_xcone_u_idx,             // [totalSparseXSocDim]
    const int64_t* __restrict__ H_xcone_v_idx,             // [totalSparseXSocDim]
    const int64_t* __restrict__ H_xcone_exp_diag_idx,      // [2 * numSparseXSoc]
    int64_t nnzKKT,
    int64_t numXCones,
    int64_t totalSparseXSocDim)
{
    const int64_t b = blockIdx.x;
    const int64_t kkt_off = b * nnzKKT;

    // One thread per cone (strided); each thread serializes its cone's entries.
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
        const int64_t sparse_idx = d_xcone_sparse_indices[c];
        if (sparse_idx < 0) continue;   // dense / nonneg — skip
        const int64_t dim = d_xcone_dims[c];
        const int64_t sp_off = d_xcone_sparse_offsets[c];
        const int64_t num_off = d_xcone_numel_offsets[c];
        const double eta = xcone_eta[b * numXCones + c];
        const double eta_sq = eta * eta;
        const int64_t u_off_b = b * totalSparseXSocDim + sp_off;

        for (int64_t p = 0; p < dim; ++p) {
            const int64_t cone_pos = d_xcone_cone_pos_for_sorted[num_off + p];
            const int64_t flat = sp_off + p;
            const double u_val = xcone_u[u_off_b + cone_pos];
            const double v_val = xcone_v[u_off_b + cone_pos];
            kkt_values[kkt_off + H_xcone_v_idx[flat]] = eta_sq * v_val;
            kkt_values[kkt_off + H_xcone_u_idx[flat]] = eta_sq * u_val;
        }

        // Expansion diag: [v-diag = +η², u-diag = -η²] for direct-x.
        kkt_values[kkt_off + H_xcone_exp_diag_idx[2 * sparse_idx]]     =  eta_sq;
        kkt_values[kkt_off + H_xcone_exp_diag_idx[2 * sparse_idx + 1]] = -eta_sq;
    }
}

void refresh_xcone_sparse_expansion(
    double* kkt_values,
    const double* xcone_u,
    const double* xcone_v,
    const double* xcone_eta,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_sparse_indices,
    const int64_t* d_xcone_sparse_offsets,
    const int64_t* d_xcone_cone_pos_for_sorted,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* H_xcone_u_idx,
    const int64_t* H_xcone_v_idx,
    const int64_t* H_xcone_exp_diag_idx,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t numXCones,
    int64_t totalSparseXSocDim,
    int64_t numSparseXSoc,
    cudaStream_t stream)
{
    if (numSparseXSoc <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(numXCones < 256 ? numXCones : 256);
    if (threads <= 0) threads = 1;
    dim3 grid(static_cast<unsigned int>(batchSize));
    refresh_xcone_sparse_expansion_kernel<<<grid, threads, 0, stream>>>(
        kkt_values, xcone_u, xcone_v, xcone_eta,
        d_xcone_dims,
        d_xcone_sparse_indices, d_xcone_sparse_offsets,
        d_xcone_cone_pos_for_sorted, d_xcone_numel_offsets,
        H_xcone_u_idx, H_xcone_v_idx, H_xcone_exp_diag_idx,
        nnzKKT, numXCones, totalSparseXSocDim);
}

void update_kkt_H_block(const KKTHBlockArgs& args, cudaStream_t stream) {
    // Calculate max threads needed across all cone types
    int64_t max_threads = 0;
    max_threads = (args.numZeroCones + args.numNonnegCones > max_threads)
        ? args.numZeroCones + args.numNonnegCones : max_threads;
    max_threads = (args.numExpCones * 6 > max_threads) ? args.numExpCones * 6 : max_threads;
    max_threads = (args.totalSocHsEntries > max_threads) ? args.totalSocHsEntries : max_threads;
    max_threads = (args.numSocCones > max_threads) ? args.numSocCones : max_threads;  // sparse SOC per-cone
    max_threads = (args.numPowerCones * 6 > max_threads) ? args.numPowerCones * 6 : max_threads;
    max_threads = (args.totalGenPowerHsEntries > max_threads) ? args.totalGenPowerHsEntries : max_threads;
    max_threads = (args.numGenPowerCones > max_threads) ? args.numGenPowerCones : max_threads;

    if (max_threads == 0) return; // No cones to update

    // Launch kernel: one block per batch, 512 threads max.
    //
    // The 6-axis PD-scaling block added enough per-thread
    // locals (the `pd_axis_idx_arr[6]` pointer table, the 6-iter axis loop's
    // per-iter `sign`, `coef`, `axes_ptr`, `target_norm_sq`, `Mw`, etc.) that
    // the kernel's per-thread register count under 1024 threads/block
    // exceeds the SM register file (≤ 65536 regs on sm_75/sm_80) — driver
    // returns CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, surfacing later as
    // "too many resources requested for launch" via the next launch's
    // cudaGetLastError(). Halving to 512 stays well under the budget.
    // Grid-stride loops in the body (already present via `for tid`) handle
    // the larger work counts.
    int threadsPerBlock = (max_threads < 512) ? ((max_threads + 31) / 32 * 32) : 512;

    MOREAU_KERNEL_LAUNCH(update_kkt_H_block_kernel, args.batchSize, threadsPerBlock, 0, stream,
                         args);
}

// ============================================================================
// PSD H block scatter kernel
// ============================================================================

__global__ void update_kkt_psd_H_block_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ H_psd_idx,
    const double* __restrict__ psd_Hs,
    int64_t totalPsdHsEntries,
    int64_t nnzKKT
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    // Scatter psd_Hs values into KKT at H_psd_idx positions
    for (int64_t i = threadIdx.x; i < totalPsdHsEntries; i += stride) {
        int64_t kkt_idx = H_psd_idx[i];
        kkt_values[batch * nnzKKT + kkt_idx] = -psd_Hs[batch * totalPsdHsEntries + i];
    }
}

void update_kkt_psd_H_block(
    double* kkt_values,
    const int64_t* H_psd_idx,
    const double* psd_Hs,
    const int64_t* d_psd_Hs_offsets,
    int64_t totalPsdHsEntries,
    int64_t numPsdCones,
    int64_t batchSize,
    int64_t nnzKKT,
    cudaStream_t stream
) {
    if (numPsdCones == 0 || totalPsdHsEntries == 0) return;

    const int threadsPerBlock = 256;
    update_kkt_psd_H_block_kernel<<<batchSize, threadsPerBlock, 0, stream>>>(
        kkt_values, H_psd_idx, psd_Hs, totalPsdHsEntries, nnzKKT
    );
}

/**
 * @brief Fused kernel for backup + regularize diagonal in one pass
 *
 * Combines backup_diagonal_kernel and regularize_diagonal_kernel into a single kernel.
 * This saves one kernel launch per iteration.
 */
__global__ void backup_and_regularize_diagonal_kernel(
    double* __restrict__ kkt_values,
    double* __restrict__ work_diag,
    const int64_t* __restrict__ diag_full,
    const int8_t* dsigns,
    const double* __restrict__ eps,
    int64_t nnzKKT,
    int64_t N
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;
    double eps_batch = eps[batch];

    // Grid-stride loop to handle > 1024 elements
    for (int64_t i = threadIdx.x; i < N; i += stride) {
        int64_t diag_idx = diag_full[i];
        int64_t kkt_idx = batch * nnzKKT + diag_idx;
        int8_t sign = dsigns[i];

        // Read current value and backup
        double val = kkt_values[kkt_idx];
        work_diag[batch * N + i] = val;

        // Apply regularization in place
        if (sign == 1) {
            kkt_values[kkt_idx] = val + eps_batch;
        } else {
            kkt_values[kkt_idx] = val - eps_batch;
        }
    }
}

__global__ void backup_diagonal_kernel(
    const double* __restrict__ kkt_values,
    double* __restrict__ work_diag,
    const int64_t* __restrict__ diag_full,
    int64_t nnzKKT,
    int64_t N
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    // Grid-stride loop to handle > 1024 elements
    for (int64_t i = threadIdx.x; i < N; i += stride) {
        int64_t diag_idx = diag_full[i];
        work_diag[batch * N + i] = kkt_values[batch * nnzKKT + diag_idx];
    }
}

void backup_diagonal(
    const double* kkt_values,
    double* work_diag,
    const int64_t* diag_full,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    int threadsPerBlock = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;

    MOREAU_KERNEL_LAUNCH(backup_diagonal_kernel, batchSize, threadsPerBlock, 0, stream,
        kkt_values, work_diag, diag_full, nnzKKT, N
    );
}

void backup_and_regularize_diagonal(
    double* kkt_values,
    double* work_diag,
    const int64_t* diag_full,
    const int8_t* dsigns,
    const double* eps,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    int threadsPerBlock = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;

    MOREAU_KERNEL_LAUNCH(backup_and_regularize_diagonal_kernel, batchSize, threadsPerBlock, 0, stream,
        kkt_values, work_diag, diag_full, dsigns, eps, nnzKKT, N
    );
}

__global__ void regularize_diagonal_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ diag_full,
    const int8_t* dsigns,
    const double* __restrict__ eps,
    int64_t nnzKKT,
    int64_t N
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    // Grid-stride loop to handle > 1024 elements
    for (int64_t i = threadIdx.x; i < N; i += stride) {
        int64_t diag_idx = diag_full[i];
        int8_t sign = dsigns[i];
        double eps_batch = eps[batch];

        // Add signed regularization: +eps for positive (P), -eps for negative (H)
        if (sign == 1) {
            kkt_values[batch * nnzKKT + diag_idx] += eps_batch;
        } else {
            kkt_values[batch * nnzKKT + diag_idx] -= eps_batch;
        }
    }
}

void regularize_diagonal(
    double* kkt_values,
    const int64_t* diag_full,
    const int8_t* dsigns,
    const double* eps,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    int threadsPerBlock = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;

    MOREAU_KERNEL_LAUNCH(regularize_diagonal_kernel, batchSize, threadsPerBlock, 0, stream,
        kkt_values, diag_full, dsigns, eps, nnzKKT, N
    );
}

__global__ void restore_diagonal_kernel(
    double* __restrict__ kkt_values,
    const double* __restrict__ work_diag,
    const int64_t* __restrict__ diag_full,
    int64_t nnzKKT,
    int64_t N
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    // Grid-stride loop to handle > 1024 elements
    for (int64_t i = threadIdx.x; i < N; i += stride) {
        int64_t diag_idx = diag_full[i];
        kkt_values[batch * nnzKKT + diag_idx] = work_diag[batch * N + i];
    }
}

void restore_diagonal(
    double* kkt_values,
    const double* work_diag,
    const int64_t* diag_full,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    int threadsPerBlock = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;

    MOREAU_KERNEL_LAUNCH(restore_diagonal_kernel, batchSize, threadsPerBlock, 0, stream,
        kkt_values, work_diag, diag_full, nnzKKT, N
    );
}

// Helper for atomicMax on double
__device__ void atomicMax_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

__global__ void diagonal_inf_norm_kernel(
    const double* __restrict__ kkt_values,
    const int64_t* __restrict__ diag_full,
    double* __restrict__ result,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N
) {
    // Each block processes one batch
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t i = threadIdx.x;

    __shared__ double sdata[256];

    // Compute local max for this thread
    double local_max = 0.0;
    for (int64_t idx = i; idx < N; idx += blockDim.x) {
        int64_t diag_idx = diag_full[idx] + batch * nnzKKT;
        local_max = fmax(local_max, fabs(kkt_values[diag_idx]));
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write result for this batch
    if (threadIdx.x == 0) {
        result[batch] = sdata[0];
    }
}

void diagonal_inf_norm(
    const double* kkt_values,
    const int64_t* diag_full,
    double* result,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    // Initialize result to 0
    cudaMemsetAsync(result, 0, sizeof(double) * batchSize, stream);

    // One block per batch, 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = batchSize;

    MOREAU_KERNEL_LAUNCH(diagonal_inf_norm_kernel, blocksPerGrid, threadsPerBlock, 0, stream,
        kkt_values, diag_full, result, batchSize, nnzKKT, N
    );
}

__global__ void compute_regularizer_per_batch_kernel(
    double* __restrict__ eps_values,
    double eps_c,
    double eps_p,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    // eps[i] = eps_c + eps_p * max_diag[i]
    eps_values[batch] = eps_c + eps_p * eps_values[batch];
}

void compute_regularizer_per_batch(
    double* eps_values,
    double eps_c,
    double eps_p,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(compute_regularizer_per_batch_kernel, blocksPerGrid, threadsPerBlock, 0, stream,
        eps_values, eps_c, eps_p, batchSize
    );
}

__global__ void set_kkt_diagonal_to_ones_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ diag_full,
    int64_t nnzKKT,
    int64_t N
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    // Grid-stride loop to handle > 1024 elements
    for (int64_t i = threadIdx.x; i < N; i += stride) {
        int64_t diag_idx = diag_full[i];
        kkt_values[batch * nnzKKT + diag_idx] = 1.0;
    }
}

void set_kkt_diagonal_to_ones(
    double* kkt_values,
    const int64_t* diag_full,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    int threadsPerBlock = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;

    MOREAU_KERNEL_LAUNCH(set_kkt_diagonal_to_ones_kernel, batchSize, threadsPerBlock, 0, stream,
        kkt_values, diag_full, nnzKKT, N
    );
}

// ============================================================================
// Fused backup + inf-norm + regularize (4→1 kernel)
// ============================================================================
__global__ void fused_backup_infnorm_regularize_kernel(
    double* __restrict__ kkt_values,
    double* __restrict__ work_diag,
    const int64_t* __restrict__ diag_full,
    const int8_t* __restrict__ dsigns,
    double* __restrict__ eps_out,        // output: regularization eps per batch
    double eps_c,
    double eps_p,
    int64_t nnzKKT,
    int64_t N
) {
    int64_t batch = blockIdx.x;

    __shared__ double sdata[256];

    // Phase 1: backup diagonal + compute local max |diag| for inf-norm
    double local_max = 0.0;
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        int64_t diag_idx = diag_full[i];
        int64_t kkt_idx = batch * nnzKKT + diag_idx;
        double val = kkt_values[kkt_idx];
        work_diag[batch * N + i] = val;
        local_max = fmax(local_max, fabs(val));
    }

    // Reduce to find max across block
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(s)) {
            sdata[threadIdx.x] = fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Phase 2: thread 0 computes eps and broadcasts via shared memory
    if (threadIdx.x == 0) {
        double eps = eps_c + eps_p * sdata[0];
        eps_out[batch] = eps;
        sdata[0] = eps; // broadcast
    }
    __syncthreads();
    double eps_batch = sdata[0];

    // Phase 3: apply regularization
    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
        int64_t diag_idx = diag_full[i];
        int64_t kkt_idx = batch * nnzKKT + diag_idx;
        int8_t sign = dsigns[i];
        if (sign == 1) {
            kkt_values[kkt_idx] += eps_batch;
        } else {
            kkt_values[kkt_idx] -= eps_batch;
        }
    }
}

void fused_backup_infnorm_regularize(
    double* kkt_values,
    double* work_diag,
    const int64_t* diag_full,
    const int8_t* dsigns,
    double* eps_out,
    double eps_c,
    double eps_p,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream
) {
    // One block per batch, 256 threads
    MOREAU_KERNEL_LAUNCH(fused_backup_infnorm_regularize_kernel, batchSize, 256, 0, stream,
        kkt_values, work_diag, diag_full, dsigns, eps_out,
        eps_c, eps_p, nnzKKT, N);
}

// ========== Full matrix expansion for Streams strategy ==========

__global__ void expand_upper_to_full_kernel(
    const double* __restrict__ upper_values,
    double* __restrict__ full_values,
    const int64_t* __restrict__ upper_to_full_map,
    const int64_t* __restrict__ upper_to_transpose_map,
    int64_t nnz_upper,
    int64_t nnz_full
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;

    if (i >= nnz_upper) return;

    double val = upper_values[b * nnz_upper + i];
    int64_t full_idx = upper_to_full_map[i];
    int64_t transpose_idx = upper_to_transpose_map[i];

    // Copy to main position
    if (full_idx >= 0) {
        full_values[b * nnz_full + full_idx] = val;
    }

    // Copy to transpose position (for off-diagonal entries)
    if (transpose_idx >= 0) {
        full_values[b * nnz_full + transpose_idx] = val;
    }
}

void expand_upper_to_full(
    const double* upper_values,
    double* full_values,
    const int64_t* upper_to_full_map,
    const int64_t* upper_to_transpose_map,
    int64_t nnz_upper,
    int64_t nnz_full,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnz_upper == 0 || batchSize == 0) return;

    dim3 block(256);
    dim3 grid((nnz_upper + block.x - 1) / block.x, batchSize);

    MOREAU_KERNEL_LAUNCH(expand_upper_to_full_kernel, grid, block, 0, stream,
        upper_values, full_values, upper_to_full_map, upper_to_transpose_map,
        nnz_upper, nnz_full);
}

// ============================================================================
// Direct-x ExpCone kernels (Phase 2: asymmetric asymmetric cones)
// ============================================================================
//
// These four kernels mirror the CPU `direct_x_*` trait methods for ExpCone.
// The primal↔dual swap used for symmetric cones does NOT apply here; the
// asymmetric cones use the primal barrier directly.
//
// References:
//   CPU: packages/moreau-cpu/src/solver/core/cones/expcone.rs (direct_x_* impls)
//   Feasibility: is_primal_feasible / is_dual_feasible in NonsymmetricCone impl
//
// Storage layout (shared with other xcone kinds):
//   xcone_Hs         [batchSize * totalXConeHsEntries] — packed upper-tri 3x3 (6 entries)
//   xcone_grad_primal [batchSize * totalXConeNumel]    — ∇F_primal(x), 3 entries per cone
//
// Identity fallback: if x or z is infeasible, write identity Hs and skip
// grad_primal (left at zero/previous value). This matches the SOC identity
// fallback pattern and lets the outer solver continue.

// Helper: is_primal_feasible for the exponential cone.
// K_exp = {(x0, x1, x2) : x1*log(x2/x1) - x0 >= 0, x1 > 0, x2 > 0}
__device__ __forceinline__ bool exp_is_primal_feasible(const double* x) {
    if (x[1] <= 0.0 || x[2] <= 0.0) return false;
    double u = x[1] * cones::logsafe(x[2] / x[1]) - x[0];
    return u > 0.0;
}

// Helper: is_dual_feasible for the exponential cone.
// K_exp* = {(z0, z1, z2) : z1 - z0 - z0*log(-z2/z0) >= 0, z0 < 0, z2 > 0}
__device__ __forceinline__ bool exp_is_dual_feasible(const double* z) {
    if (z[0] >= 0.0 || z[2] <= 0.0) return false;
    double res = z[1] - z[0] - z[0] * cones::logsafe(-z[2] / z[0]);
    return res > 0.0;
}

// --------------------------------------------------------------------------
// 1. update_xcones_exp_scaling
//
// Computes Hs = mu * H_primal(x) and grad_primal = ∇F_primal(x) for each
// Exp x-cone. Filters by `static_cast<XConeKind>(d_xcone_kinds[c]) == XConeKind::Exp`.
// One block per batch; threads stride over cones (256 threads/block).
//
// H_primal packed upper-tri layout (column-major): [H00, H01, H02, H11, H12, H22]
//   H00 = inv_u²
//   H01 = -v * inv_u²
//   H02 = -w * inv_u²
//   H11 = v²*inv_u² + inv_u/x1 + 1/x1²
//   H12 = v*w*inv_u² - inv_u/x2
//   H22 = w²*inv_u² + x1*inv_u/x2² + 1/x2²
// where u = x1*log(x2/x1) - x0, v = log(x2/x1) - 1, w = x1/x2.
// --------------------------------------------------------------------------
__global__ void update_xcones_exp_scaling_kernel(
    const double* __restrict__ x,
    const double* __restrict__ z_x,
    const double* __restrict__ mu,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    double* __restrict__ xcone_Hs,
    double* __restrict__ xcone_grad_primal,
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_px_baseline,
    const int64_t* __restrict__ H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off  = d_xcone_numel_offsets[c];
    const int64_t hs_off   = d_xcone_hs_offsets[c];
    const int64_t x_off_b  = b * n;
    const int64_t zx_off   = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;
    const int64_t w_off_b  = b * totalXConeNumel + num_off;

    const bool fused = (kkt_values != nullptr) && (xcone_px_baseline != nullptr)
                       && (H_xcone_hs_idx != nullptr);
    const int64_t kkt_off = fused ? b * nnzKKT : 0;

    // Gather x[J] and z_x for this cone.
    double xj[3], zj[3];
    for (int i = 0; i < 3; ++i) {
        xj[i] = x[x_off_b + d_xcone_indices[num_off + i]];
        zj[i] = z_x[zx_off + num_off + i];
    }

    // Identity fallback if either side is infeasible.
    if (!exp_is_primal_feasible(xj) || !exp_is_dual_feasible(zj)) {
        // Write identity Hs: 6-entry packed upper-tri, 1 on diagonal positions.
        // Diagonal positions: pos 0 (0,0), pos 3 (1,1), pos 5 (2,2).
        xcone_Hs[hs_off_b + hs_off + 0] = 1.0;
        xcone_Hs[hs_off_b + hs_off + 1] = 0.0;
        xcone_Hs[hs_off_b + hs_off + 2] = 0.0;
        xcone_Hs[hs_off_b + hs_off + 3] = 1.0;
        xcone_Hs[hs_off_b + hs_off + 4] = 0.0;
        xcone_Hs[hs_off_b + hs_off + 5] = 1.0;
        if (fused) {
            double identity_vals[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
            for (int i = 0; i < 6; ++i) {
                kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                    xcone_px_baseline[hs_off_b + hs_off + i] + identity_vals[i];
            }
        }
        continue;
    }

    // Compute closed-form primal-barrier gradient and Hessian at xj.
    // F(s) = -log(u) - log(s1) - log(s2),  u = s1*log(s2/s1) - s0.
    double log_ratio = cones::logsafe(xj[2] / xj[1]);
    double u     = xj[1] * log_ratio - xj[0];
    double v     = log_ratio - 1.0;
    double w_    = xj[1] / xj[2];
    double inv_u  = 1.0 / u;
    double inv_u2 = inv_u * inv_u;

    // Gradient ∇F_primal = -(1/u)∇u - {0, 1/x1, 1/x2}, ∇u = (-1, v, w_).
    double grad[3];
    grad[0] = inv_u;
    grad[1] = -v * inv_u - 1.0 / xj[1];
    grad[2] = -w_ * inv_u - 1.0 / xj[2];

    // Write grad to xcone_grad_primal.
    for (int i = 0; i < 3; ++i) {
        xcone_grad_primal[w_off_b + i] = grad[i];
    }

    // Hessian ∇²F_primal — column-major upper-tri (matches KKT slot order
    // built in kkt.hpp::Exp/Power branch and seed_xcone_Hs_identity_kernel).
    // Layout: [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)].
    double H[6];
    H[0] = inv_u2;                                                                 // (0,0)
    H[1] = -v * inv_u2;                                                            // (0,1)
    H[2] = v * v * inv_u2 + inv_u / xj[1] + 1.0 / (xj[1] * xj[1]);                  // (1,1)
    H[3] = -w_ * inv_u2;                                                           // (0,2)
    H[4] = v * w_ * inv_u2 - inv_u / xj[2];                                        // (1,2)
    H[5] = w_ * w_ * inv_u2 + (xj[1] * inv_u) / (xj[2] * xj[2]) + 1.0 / (xj[2] * xj[2]); // (2,2)

    // Hs = mu * H_primal.
    double mu_b = mu[b];
    for (int i = 0; i < 6; ++i) {
        double val = mu_b * H[i];
        xcone_Hs[hs_off_b + hs_off + i] = val;
        if (fused) {
            kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                xcone_px_baseline[hs_off_b + hs_off + i] + val;
        }
    }
    } // end cone loop
}

void update_xcones_exp_scaling(
    const double* x,
    const double* z_x,
    const double* mu,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    double* xcone_Hs,
    double* xcone_grad_primal,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    update_xcones_exp_scaling_kernel<<<grid, 256, 0, stream>>>(
        x, z_x, mu,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        xcone_Hs, xcone_grad_primal,
        kkt_values, xcone_px_baseline, H_xcone_hs_idx,
        nnzKKT, batchSize, n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// --------------------------------------------------------------------------
// 2. fill_step_rhs_zx_exp_affine
//
// CPU: direct_x_affine_offset(out, z) { out = z }.
// For Exp, the affine offset is just a copy of the dual z_x to step_rhs.z_x.
// One block per batch, threads stride over cones, filter by kind == 3.
// --------------------------------------------------------------------------
__global__ void fill_step_rhs_zx_exp_affine_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ var_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    int64_t totalXConeNumel, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t zx_off  = b * totalXConeNumel;

    for (int i = 0; i < 3; ++i) {
        step_rhs_z_x[zx_off + num_off + i] = var_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void fill_step_rhs_zx_exp_affine(
    double* step_rhs_z_x,
    const double* var_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    fill_step_rhs_zx_exp_affine_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, var_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        totalXConeNumel, numXCones);
}

// --------------------------------------------------------------------------
// 3. add_combined_ds_shift_exp
//
// CPU: direct_x_combined_ds_shift(shift, step_x, step_z, σμ):
//   shift[i] = grad_primal[i] * σμ - η[i]
// where η is the Mehrotra 3rd-order correction
//   η = (1/2) D³F(x_pt)[H_primal⁻¹·step_x, m·step_z]
// (CPU expcone.rs::direct_x_higher_correction). Falls back to η=0 if the
// 3×3 Cholesky fails or u(x)=x[1]·log(x[2]/x[1])-x[0] is near zero.
// --------------------------------------------------------------------------
__global__ void add_combined_ds_shift_exp_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ sigma_mu,
    const double* __restrict__ mehrotra_m,
    const double* __restrict__ mu,
    const double* __restrict__ var_x,
    const double* __restrict__ step_aff_x,
    const double* __restrict__ step_aff_z_x,
    const double* __restrict__ xcone_Hs,
    const double* __restrict__ xcone_grad_primal,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    const double sm = sigma_mu[b];
    const double mb = mehrotra_m[b];
    const double mu_b = mu[b];

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t hs_off  = d_xcone_hs_offsets[c];
    const int64_t zx_b    = b * totalXConeNumel + num_off;
    const int64_t hs_b    = b * totalXConeHsEntries + hs_off;
    const int64_t x_b     = b * n;

    // Gather x_pt and step_aff_x at the cone's primal indices.
    double x[3], sx[3];
    for (int i = 0; i < 3; ++i) {
        const int64_t idx = d_xcone_indices[num_off + i];
        x[i]  = var_x[x_b + idx];
        sx[i] = step_aff_x[x_b + idx];
    }
    (void)mb;        // step_z unused in primal-direct formula
    (void)hs_off; (void)hs_b; (void)xcone_Hs; (void)totalXConeHsEntries;
    (void)step_aff_z_x;

    // Primal-direct η = ½·D³F(x)[step_x, step_x]. Both contraction slots
    // are step_x — see CPU `direct_x_higher_correction` for the
    // decomposition. F = -log(u) - log(x[1]) - log(x[2]),
    // u = x[1]·log(x[2]/x[1]) - x[0].
    double eta0 = 0.0, eta1 = 0.0, eta2 = 0.0;
    // Mirror CPU `T::epsilon().sqrt()` ≈ 1.49e-8. Looser than 1e-15 so we
    // zero η near the cone boundary instead of blowing it up.
    const double eps = 1.4901161193847656e-8;
    if (fabs(x[1]) > eps && fabs(x[2]) > eps) {
        const double l = log(x[2] / x[1]);
        const double ucone = x[1] * l - x[0];
        if (fabs(ucone) > eps) {
            const double a0 = sx[0], a1 = sx[1], a2 = sx[2];
            const double b0 = sx[0], b1 = sx[1], b2 = sx[2];
            // ∇u(x) = (-1, l - 1, x[1]/x[2]).
            const double gu0 = -1.0;
            const double gu1 = l - 1.0;
            const double gu2 = x[1] / x[2];
            const double dot_a = a0 * gu0 + a1 * gu1 + a2 * gu2;
            const double dot_b = b0 * gu0 + b1 * gu1 + b2 * gu2;
            // a^T H_u b. H_u nonzeros: (1,1)=-1/x[1], (1,2)=(2,1)=1/x[2],
            // (2,2)=-x[1]/x[2]².
            const double q_u = -a1 * b1 / x[1]
                + (a1 * b2 + a2 * b1) / x[2]
                - x[1] * a2 * b2 / (x[2] * x[2]);
            const double coef = (q_u * ucone - 2.0 * dot_a * dot_b)
                / (ucone * ucone * ucone);
            eta0 = coef * gu0;
            eta1 = coef * gu1;
            eta2 = coef * gu2;
            const double inv_u = 1.0 / ucone;
            const double inv_u2 = inv_u * inv_u;
            // η[1] additions:
            const double Hua1 = -a1 / x[1] + a2 / x[2];
            const double Hub1 = -b1 / x[1] + b2 / x[2];
            const double d3u1 = a1 * b1 / (x[1] * x[1])
                - a2 * b2 / (x[2] * x[2]);
            eta1 += dot_b * Hua1 * inv_u2 + dot_a * Hub1 * inv_u2
                - d3u1 * inv_u
                - 2.0 * a1 * b1 / (x[1] * x[1] * x[1]);
            // η[2] additions:
            const double Hua2 = a1 / x[2] - a2 * x[1] / (x[2] * x[2]);
            const double Hub2 = b1 / x[2] - b2 * x[1] / (x[2] * x[2]);
            const double d3u2 = -(a1 * b2 + a2 * b1) / (x[2] * x[2])
                + 2.0 * x[1] * a2 * b2 / (x[2] * x[2] * x[2]);
            eta2 += dot_b * Hua2 * inv_u2 + dot_a * Hub2 * inv_u2
                - d3u2 * inv_u
                - 2.0 * a2 * b2 / (x[2] * x[2] * x[2]);
            eta0 *= 0.5;
            eta1 *= 0.5;
            eta2 *= 0.5;
        }
    }

    // μ⁴ scaling + K=0.5 ∞-norm cap: mirrors CPU
    // `expcone.rs::{DIRECT_X_ETA_MU_EXP, DIRECT_X_ETA_CAP_K}` (sweep-tuned
    // defaults).
    const double mu_sq = mu_b * mu_b;
    const double mu_quad = mu_sq * mu_sq;
    eta0 *= mu_quad;
    eta1 *= mu_quad;
    eta2 *= mu_quad;

    const double K = 0.5;
    double max_ds = fmax(fmax(fabs(sx[0]), fabs(sx[1])), fabs(sx[2]));
    double max_eta = fmax(fmax(fabs(eta0), fabs(eta1)), fabs(eta2));
    double cap = K * max_ds;
    if (max_eta > cap && max_ds > 0.0) {
        double scale = cap / max_eta;
        eta0 *= scale; eta1 *= scale; eta2 *= scale;
    }

    // Apply: shift = σμ·∇F - η, accumulated into step_rhs_z_x.
    step_rhs_z_x[zx_b + 0] += sm * xcone_grad_primal[zx_b + 0] - eta0;
    step_rhs_z_x[zx_b + 1] += sm * xcone_grad_primal[zx_b + 1] - eta1;
    step_rhs_z_x[zx_b + 2] += sm * xcone_grad_primal[zx_b + 2] - eta2;
    } // end cone loop
}

void add_combined_ds_shift_exp(
    double* step_rhs_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const double* mu,
    const double* var_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* xcone_Hs,
    const double* xcone_grad_primal,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    add_combined_ds_shift_exp_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, sigma_mu, mehrotra_m, mu,
        var_x, step_aff_x, step_aff_z_x,
        xcone_Hs, xcone_grad_primal,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        n, totalXConeNumel, totalXConeHsEntries, numXCones);
}

// --------------------------------------------------------------------------
// 4. xcone_step_length_exp_reduce
//
// CPU: direct_x_step_length — backtrack on is_primal_feasible(x + α·dx) for
// primal (α_z) and is_dual_feasible(z + α·dz) for dual (α_s). Uses 0.8
// backtrack step and 1e-7 min step length, matching IPMSettings defaults.
//
// One block per batch; threads stride over cones, filter by kind == Exp.
// Atomic-min into alpha_z (primal x constraint) and alpha_s (dual z_x
// constraint) using CAS on reinterpreted double bits.
// --------------------------------------------------------------------------
__global__ void xcone_step_length_exp_reduce_kernel(
    double* __restrict__ alpha_s,
    double* __restrict__ alpha_z,
    const double* __restrict__ var_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ step_x,
    const double* __restrict__ step_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack)
{
    const int64_t b = blockIdx.x;

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    // Gather x[J] and dx[J] for the primal backtrack.
    double xj[3], dxj[3];
    for (int i = 0; i < 3; ++i) {
        const int64_t idx = d_xcone_indices[num_off + i];
        xj[i]  = var_x[x_off + idx];
        dxj[i] = step_x[x_off + idx];
    }
    // Gather z_x and dz_x for the dual backtrack (flat indexed).
    double zj[3], dzj[3];
    for (int i = 0; i < 3; ++i) {
        zj[i]  = var_z_x[zx_off + num_off + i];
        dzj[i] = step_z_x[zx_off + num_off + i];
    }

    // Primal backtrack: find largest α such that x + α·dx is feasible.
    // CPU `backtrack_search` returns 0 when no feasible α ≥ α_min is found
    // (nonsymmetric_common.rs:1174-1176). Match that semantic so the IPM
    // sees the same infeasibility signal on CUDA.
    //
    // Start from `min(max_step, alpha_{z,s}[b])` (the already-applied τ/κ +
    // slack-cone bound). Starting only from `max_step=1.0` causes a
    // divergence: when α=1.0 is infeasible but the τ/κ-bounded αmax
    // (e.g. 0.893) is feasible, CUDA backtracks once to 0.8 while CPU
    // returns 0.893 — propagates through σ=(1−α)³ to a 6× σμ mismatch in
    // the combined-step shift. Same fix as the GenPow kernel below.
    double alpha_primal = fmin(max_step, alpha_z[b]);
    {
        double trial[3];
        bool found = false;
        while (alpha_primal > min_step) {
            for (int i = 0; i < 3; ++i) trial[i] = xj[i] + alpha_primal * dxj[i];
            if (exp_is_primal_feasible(trial)) { found = true; break; }
            alpha_primal *= backtrack;
        }
        if (!found) alpha_primal = 0.0;
    }

    // Dual backtrack: same αmax convention as primal.
    double alpha_dual = fmin(max_step, alpha_s[b]);
    {
        double trial[3];
        bool found = false;
        while (alpha_dual > min_step) {
            for (int i = 0; i < 3; ++i) trial[i] = zj[i] + alpha_dual * dzj[i];
            if (exp_is_dual_feasible(trial)) { found = true; break; }
            alpha_dual *= backtrack;
        }
        if (!found) alpha_dual = 0.0;
    }

    // Fold into the per-batch alpha arrays.
    // α_z receives the primal constraint (x → z slot convention: direct-x primal is alpha_z).
    // α_s receives the dual constraint (z_x → s slot convention).
    cones::atomic_min_pos_double(&alpha_z[b], alpha_primal);
    cones::atomic_min_pos_double(&alpha_s[b], alpha_dual);
    } // end cone loop
}

void xcone_step_length_exp_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    xcone_step_length_exp_reduce_kernel<<<grid, 256, 0, stream>>>(
        alpha_s, alpha_z, var_x, var_z_x, step_x, step_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel, max_step, min_step, backtrack);
}

// --------------------------------------------------------------------------
// 5. subtract_xcone_combined_from_workx_exp
//
// CPU: direct_x_combined_offset(out, ds, work, x) { out = ds } (identity).
// So c_J = rhs_z_x (no division by x), and workx[J] -= rhs_z_x.
// One block per batch, threads stride over cones, filter by kind == 3.
// --------------------------------------------------------------------------
__global__ void subtract_xcone_combined_from_workx_exp_kernel(
    double* __restrict__ workx,
    const double* __restrict__ rhs_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    for (int i = 0; i < 3; ++i) {
        const int64_t idx = d_xcone_indices[num_off + i];
        workx[x_off + idx] -= rhs_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void subtract_xcone_combined_from_workx_exp(
    double* workx,
    const double* rhs_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    subtract_xcone_combined_from_workx_exp_kernel<<<grid, 256, 0, stream>>>(
        workx, rhs_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

// --------------------------------------------------------------------------
// 6. recover_dz_x_affine_exp
//
// CPU: recover_direct_x_dual (affine):
//   Δz_J = -direct_x_mul_Hs(Δx[J]) - variables.z_x[J]
// For Exp, Hs is the dense 3×3 (packed upper-tri). mul_Hs does a
// symmetric matvec. One block per (batch, cone), single-threaded.
// --------------------------------------------------------------------------
__global__ void recover_dz_x_affine_exp_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off  = d_xcone_numel_offsets[c];
    const int64_t hs_off   = d_xcone_hs_offsets[c];
    const int64_t x_off    = b * n;
    const int64_t zx_off   = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;

    // Gather Δx[J].
    double dx[3];
    for (int i = 0; i < 3; ++i) {
        dx[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];
    }

    // Read packed column-major upper-tri Hs:
    // [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)].
    const double* H = xcone_Hs + hs_off_b + hs_off;

    // Symmetric matvec: y = H * dx
    double y[3];
    y[0] = H[0]*dx[0] + H[1]*dx[1] + H[3]*dx[2];
    y[1] = H[1]*dx[0] + H[2]*dx[1] + H[4]*dx[2];
    y[2] = H[3]*dx[0] + H[4]*dx[1] + H[5]*dx[2];

    for (int i = 0; i < 3; ++i) {
        dz_x[zx_off + num_off + i] = -y[i] - var_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void recover_dz_x_affine_exp(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_affine_exp_kernel<<<grid, 256, 0, stream>>>(
        dz_x, lhs_x, var_z_x, xcone_Hs,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// --------------------------------------------------------------------------
// 7. recover_dz_x_combined_exp
//
// CPU: recover_direct_x_dual (combined):
//   Δz_J = -direct_x_mul_Hs(Δx[J]) - c_J
// For Exp, direct_x_combined_offset gives c_J = rhs_z_x (identity copy).
// So: dz_x = -Hs·dx[J] - rhs_z_x.
// --------------------------------------------------------------------------
__global__ void recover_dz_x_combined_exp_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Exp) continue;

    const int64_t num_off  = d_xcone_numel_offsets[c];
    const int64_t hs_off   = d_xcone_hs_offsets[c];
    const int64_t x_off    = b * n;
    const int64_t zx_off   = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;

    // Gather Δx[J].
    double dx[3];
    for (int i = 0; i < 3; ++i) {
        dx[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];
    }

    // Read packed column-major upper-tri Hs:
    // [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)].
    const double* H = xcone_Hs + hs_off_b + hs_off;

    double y[3];
    y[0] = H[0]*dx[0] + H[1]*dx[1] + H[3]*dx[2];
    y[1] = H[1]*dx[0] + H[2]*dx[1] + H[4]*dx[2];
    y[2] = H[3]*dx[0] + H[4]*dx[1] + H[5]*dx[2];

    for (int i = 0; i < 3; ++i) {
        dz_x[zx_off + num_off + i] = -y[i] - rhs_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void recover_dz_x_combined_exp(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_combined_exp_kernel<<<grid, 256, 0, stream>>>(
        dz_x, lhs_x, rhs_z_x, xcone_Hs,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// ============================================================================
// Direct-x PowerCone kernels (Phase 3: asymmetric cones)
// ============================================================================
//
// These kernels mirror the CPU `direct_x_*` trait methods for PowerCone.
// The closed-form primal barrier is
//   F(s) = -log(phi - s[2]²) - (1-α)·log(s[0]) - α·log(s[1])
// where phi = s[0]^(2α) · s[1]^(2(1-α)).
//
// References:
//   CPU: packages/moreau-cpu/src/solver/core/cones/powcone.rs
//   Feasibility: is_primal_feasible / is_dual_feasible in NonsymmetricCone impl
//
// Storage layout (shared with Exp and other xcone kinds):
//   xcone_Hs          [batchSize * totalXConeHsEntries] — packed upper-tri 3x3 (6 entries)
//   xcone_grad_primal [batchSize * totalXConeNumel]     — ∇F_primal(x), 3 entries per cone
//
// Identity fallback: if x or z is infeasible, write identity Hs and return.
// Unit init point for PowerCone(α): (sqrt(1+α), sqrt(2-α), 0) — lies in both
// primal and dual power cones for all α ∈ (0,1). Depends on α per cone.

// Helper: is_primal_feasible for the power cone.
// K_pow(α) = {(x0,x1,x2) : x0^α · x1^(1-α) >= |x2|, x0>=0, x1>=0}
// Equivalent: exp(2α·log(x0) + 2(1-α)·log(x1)) - x2² > 0 with x0,x1 > 0.
__device__ __forceinline__ bool pow_is_primal_feasible(const double* x, double alpha) {
    if (x[0] <= 0.0 || x[1] <= 0.0) return false;
    double res = exp(2.0 * alpha * log(x[0]) + 2.0 * (1.0 - alpha) * log(x[1]))
                 - x[2] * x[2];
    return res > 0.0;
}

// Helper: is_dual_feasible for the power cone.
// K_pow(α)* = {(z0,z1,z2) : (z0/α)^α · (z1/(1-α))^(1-α) >= |z2|, z0>=0, z1>=0}
// Equivalent: exp(2α·log(z0/α) + 2(1-α)·log(z1/(1-α))) - z2² > 0 with z0,z1 > 0.
__device__ __forceinline__ bool pow_is_dual_feasible(const double* z, double alpha) {
    if (z[0] <= 0.0 || z[1] <= 0.0) return false;
    double res = exp(2.0 * alpha * log(z[0] / alpha)
                     + 2.0 * (1.0 - alpha) * log(z[1] / (1.0 - alpha)))
                 - z[2] * z[2];
    return res > 0.0;
}

// --------------------------------------------------------------------------
// init_xcone_x_pow
//
// Initialize x[J] for Power x-cones to the unit-init point
//   (sqrt(1+α), sqrt(2-α), 0)
// which lies in the Power cone interior for all α ∈ (0,1).
// One block per batch, threads stride over cones, filter by kind == 4.
// --------------------------------------------------------------------------
__global__ void init_xcone_x_pow_kernel(
    double* __restrict__ x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_pow_idx,
    const double* __restrict__ d_xcone_pow_alpha,
    int64_t n, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;

    const double alpha = d_xcone_pow_alpha[d_xcone_pow_idx[c]];
    const double pt[3] = {
        sqrt(1.0 + alpha),
        sqrt(2.0 - alpha),
        0.0
    };
    for (int i = 0; i < 3; ++i) {
        x[x_off + d_xcone_indices[num_off + i]] = pt[i];
    }
    } // end cone loop
}

void init_xcone_x_pow(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    init_xcone_x_pow_kernel<<<grid, 256, 0, stream>>>(
        x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_indices, d_xcone_pow_idx, d_xcone_pow_alpha, n, numXCones);
}

// --------------------------------------------------------------------------
// init_xcone_z_x_pow
//
// Overwrite z_x for Power x-cones with the correct unit-init point
//   (sqrt(1+α), sqrt(2-α), 0).
// This corrects the wrong Exp unit point that init_xcone_z_x_kernel wrote
// for kind==4 entries (the generic init uses the Exp point for both Exp
// and Power, which is wrong for Power). Called immediately after
// init_xcone_z_x when numXPowerCones > 0.
// One block per batch, threads stride over cones, filter by kind == 4.
// --------------------------------------------------------------------------
__global__ void init_xcone_z_x_pow_kernel(
    double* __restrict__ z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_pow_idx,
    const double* __restrict__ d_xcone_pow_alpha,
    int64_t totalXConeNumel, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t zx_off  = b * totalXConeNumel;

    const double alpha = d_xcone_pow_alpha[d_xcone_pow_idx[c]];
    z_x[zx_off + num_off + 0] = sqrt(1.0 + alpha);
    z_x[zx_off + num_off + 1] = sqrt(2.0 - alpha);
    z_x[zx_off + num_off + 2] = 0.0;
    } // end cone loop
}

void init_xcone_z_x_pow(
    double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t numXCones, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    init_xcone_z_x_pow_kernel<<<grid, 256, 0, stream>>>(
        z_x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_pow_idx, d_xcone_pow_alpha, totalXConeNumel, numXCones);
}

// --------------------------------------------------------------------------
// 1. update_xcones_pow_scaling
//
// Computes Hs = mu * H_primal(x) and grad_primal = ∇F_primal(x) for each
// Power x-cone. Filters by static_cast<XConeKind>(d_xcone_kinds[c]) == XConeKind::Power (Power). One block per
// (batch, cone), single-threaded.
//
// Uses the standard primal barrier:
//   F(s) = -log(phi - s[2]²) - (1-α)·log(s[0]) - α·log(s[1])
// where phi = s[0]^(2α) · s[1]^(2(1-α)).
//
// H_primal packed upper-tri layout (column-major): [H00, H01, H02, H11, H12, H22]
// --------------------------------------------------------------------------
__global__ void update_xcones_pow_scaling_kernel(
    const double* __restrict__ x,
    const double* __restrict__ z_x,
    const double* __restrict__ mu,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_pow_idx,
    const double* __restrict__ d_xcone_pow_alpha,
    double* __restrict__ xcone_Hs,
    double* __restrict__ xcone_grad_primal,
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_px_baseline,
    const int64_t* __restrict__ H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off  = d_xcone_numel_offsets[c];
    const int64_t hs_off   = d_xcone_hs_offsets[c];
    const int64_t x_off_b  = b * n;
    const int64_t zx_off   = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;
    const int64_t w_off_b  = b * totalXConeNumel + num_off;

    const bool fused = (kkt_values != nullptr) && (xcone_px_baseline != nullptr)
                       && (H_xcone_hs_idx != nullptr);
    const int64_t kkt_off = fused ? b * nnzKKT : 0;

    const double alpha = d_xcone_pow_alpha[d_xcone_pow_idx[c]];

    // Gather x[J] and z_x for this cone.
    double xj[3], zj[3];
    for (int i = 0; i < 3; ++i) {
        xj[i] = x[x_off_b + d_xcone_indices[num_off + i]];
        zj[i] = z_x[zx_off + num_off + i];
    }

    // Identity fallback if either side is infeasible.
    if (!pow_is_primal_feasible(xj, alpha) || !pow_is_dual_feasible(zj, alpha)) {
        xcone_Hs[hs_off_b + hs_off + 0] = 1.0;
        xcone_Hs[hs_off_b + hs_off + 1] = 0.0;
        xcone_Hs[hs_off_b + hs_off + 2] = 0.0;
        xcone_Hs[hs_off_b + hs_off + 3] = 1.0;
        xcone_Hs[hs_off_b + hs_off + 4] = 0.0;
        xcone_Hs[hs_off_b + hs_off + 5] = 1.0;
        if (fused) {
            double identity_vals[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
            for (int i = 0; i < 6; ++i) {
                kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                    xcone_px_baseline[hs_off_b + hs_off + i] + identity_vals[i];
            }
        }
        continue;
    }

    // Compute closed-form primal-barrier gradient and Hessian at xj.
    // F(s) = -log(phi - s[2]²) - (1-α)·log(s[0]) - α·log(s[1])
    // where phi = s[0]^(2α) · s[1]^(2(1-α)).
    const double two = 2.0;
    const double phi = pow(xj[0], two * alpha) * pow(xj[1], two * (1.0 - alpha));
    const double psi = phi - xj[2] * xj[2];

    // gpsi = (1/psi) * grad(psi)
    const double gpsi_0 = two * alpha * phi / (xj[0] * psi);
    const double gpsi_1 = two * (1.0 - alpha) * phi / (xj[1] * psi);
    const double gpsi_2 = -two * xj[2] / psi;

    // Gradient ∇F_primal = -gpsi - {(1-α)/x[0], α/x[1], 0}
    double grad[3];
    grad[0] = -gpsi_0 - (1.0 - alpha) / xj[0];
    grad[1] = -gpsi_1 - alpha / xj[1];
    grad[2] = -gpsi_2;  // = +2 * x[2] / psi

    // Write grad to xcone_grad_primal.
    for (int i = 0; i < 3; ++i) {
        xcone_grad_primal[w_off_b + i] = grad[i];
    }

    // Hessian ∇²F_primal — column-major upper-tri (matches KKT slot order
    // built in kkt.hpp::Exp/Power branch and seed_xcone_Hs_identity_kernel).
    // Layout: [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)].
    double H[6];
    H[0] = gpsi_0 * gpsi_0                                                              // (0,0)
           - two * alpha * (two * alpha - 1.0) * phi / (xj[0] * xj[0] * psi)
           + (1.0 - alpha) / (xj[0] * xj[0]);
    H[1] = gpsi_0 * gpsi_1                                                              // (0,1)
           - (two * two) * alpha * (1.0 - alpha) * phi / (xj[0] * xj[1] * psi);
    H[2] = gpsi_1 * gpsi_1                                                              // (1,1)
           - two * (1.0 - alpha) * (1.0 - two * alpha) * phi / (xj[1] * xj[1] * psi)
           + alpha / (xj[1] * xj[1]);
    H[3] = gpsi_0 * gpsi_2;                                                             // (0,2)
    H[4] = gpsi_1 * gpsi_2;                                                             // (1,2)
    H[5] = gpsi_2 * gpsi_2 + two / psi;                                                 // (2,2)

    // Hs = mu * H_primal.
    double mu_b = mu[b];
    for (int i = 0; i < 6; ++i) {
        double val = mu_b * H[i];
        xcone_Hs[hs_off_b + hs_off + i] = val;
        if (fused) {
            kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                xcone_px_baseline[hs_off_b + hs_off + i] + val;
        }
    }
    } // end cone loop
}

void update_xcones_pow_scaling(
    const double* x,
    const double* z_x,
    const double* mu,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    double* xcone_Hs,
    double* xcone_grad_primal,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    update_xcones_pow_scaling_kernel<<<grid, 256, 0, stream>>>(
        x, z_x, mu,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        d_xcone_pow_idx, d_xcone_pow_alpha,
        xcone_Hs, xcone_grad_primal,
        kkt_values, xcone_px_baseline, H_xcone_hs_idx,
        nnzKKT, batchSize, n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// --------------------------------------------------------------------------
// 2. fill_step_rhs_zx_pow_affine
//
// CPU: direct_x_affine_offset(out, z) { out = z }.
// For Power, the affine offset is just a copy of the dual z_x to step_rhs.z_x.
// One block per batch, threads stride over cones, filter by kind == 4.
// --------------------------------------------------------------------------
__global__ void fill_step_rhs_zx_pow_affine_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ var_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    int64_t totalXConeNumel, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t zx_off  = b * totalXConeNumel;

    for (int i = 0; i < 3; ++i) {
        step_rhs_z_x[zx_off + num_off + i] = var_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void fill_step_rhs_zx_pow_affine(
    double* step_rhs_z_x,
    const double* var_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    fill_step_rhs_zx_pow_affine_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, var_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        totalXConeNumel, numXCones);
}

// --------------------------------------------------------------------------
// 3. add_combined_ds_shift_pow
//
// CPU: direct_x_combined_ds_shift(shift, step_x, step_z, σμ):
//   shift[i] = grad_primal[i] * σμ - η[i]
// where η is the Mehrotra 3rd-order correction
//   η = (1/2) D³F(s)[H_primal⁻¹·step_x, m·step_z]
// using the closed-form primal barrier
//   F(s) = -log(ψ) - (1-α)·log(s[0]) - α·log(s[1]),
//   ψ(s) = s[0]^(2α)·s[1]^(2(1-α)) - s[2]².
// (CPU powcone.rs::direct_x_higher_correction.) Falls back to η=0 if 3×3
// Cholesky fails or ψ ≈ 0.
// --------------------------------------------------------------------------
__global__ void add_combined_ds_shift_pow_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ sigma_mu,
    const double* __restrict__ mehrotra_m,
    const double* __restrict__ mu,
    const double* __restrict__ var_x,
    const double* __restrict__ step_aff_x,
    const double* __restrict__ step_aff_z_x,
    const double* __restrict__ xcone_Hs,
    const double* __restrict__ xcone_grad_primal,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_pow_idx,
    const double* __restrict__ d_xcone_pow_alpha,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    const double sm = sigma_mu[b];
    const double mb = mehrotra_m[b];
    const double mu_b = mu[b];

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t hs_off  = d_xcone_hs_offsets[c];
    const int64_t zx_b    = b * totalXConeNumel + num_off;
    const int64_t hs_b    = b * totalXConeHsEntries + hs_off;
    const int64_t x_b     = b * n;
    const double alpha    = d_xcone_pow_alpha[d_xcone_pow_idx[c]];

    double s[3], sx[3];
    for (int i = 0; i < 3; ++i) {
        const int64_t idx = d_xcone_indices[num_off + i];
        s[i]  = var_x[x_b + idx];
        sx[i] = step_aff_x[x_b + idx];
    }
    (void)mb; (void)mu_b;
    (void)hs_off; (void)hs_b; (void)xcone_Hs; (void)totalXConeHsEntries;
    (void)step_aff_z_x;

    // Primal-direct η = ½·D³F(s)[step_x, step_x] for PowerCone.
    // F(s) = -log(ψ) - (1-α)·log(s[0]) - α·log(s[1]),
    // ψ(s) = s[0]^(2α)·s[1]^(2(1-α)) - s[2]².
    double eta0 = 0.0, eta1 = 0.0, eta2 = 0.0;
    // Mirror CPU `T::epsilon().sqrt()` ≈ 1.49e-8. Looser than 1e-15 so we
    // zero η near the cone boundary instead of blowing it up.
    const double eps = 1.4901161193847656e-8;
    if (fabs(s[0]) > eps && fabs(s[1]) > eps) {
        const double phi = pow(s[0], 2.0 * alpha) * pow(s[1], 2.0 * (1.0 - alpha));
        const double psi = phi - s[2] * s[2];
        if (fabs(psi) > eps) {
            const double a0 = sx[0], a1 = sx[1], a2 = sx[2];
            const double b0 = sx[0], b1 = sx[1], b2 = sx[2];
            // ∇ψ = (2α·φ/s[0], 2(1-α)·φ/s[1], -2 s[2]).
            const double gp0 = 2.0 * alpha * phi / s[0];
            const double gp1 = 2.0 * (1.0 - alpha) * phi / s[1];
            const double gp2 = -2.0 * s[2];
            const double dot_a = a0 * gp0 + a1 * gp1 + a2 * gp2;
            const double dot_b = b0 * gp0 + b1 * gp1 + b2 * gp2;

            const double hp00 = 2.0 * alpha * (2.0 * alpha - 1.0) * phi / (s[0] * s[0]);
            const double hp01 = 4.0 * alpha * (1.0 - alpha) * phi / (s[0] * s[1]);
            const double hp11 = 2.0 * (1.0 - alpha) * (1.0 - 2.0 * alpha) * phi / (s[1] * s[1]);
            const double hp22 = -2.0;

            const double q_psi = a0 * (hp00 * b0 + hp01 * b1)
                + a1 * (hp01 * b0 + hp11 * b1)
                + a2 * hp22 * b2;
            const double coef = (q_psi * psi - 2.0 * dot_a * dot_b)
                / (psi * psi * psi);
            eta0 = coef * gp0;
            eta1 = coef * gp1;
            eta2 = coef * gp2;

            const double inv_p = 1.0 / psi;
            const double inv_p2 = inv_p * inv_p;
            const double Hpa0 = hp00 * a0 + hp01 * a1;
            const double Hpa1 = hp01 * a0 + hp11 * a1;
            const double Hpa2 = hp22 * a2;
            const double Hpb0 = hp00 * b0 + hp01 * b1;
            const double Hpb1 = hp01 * b0 + hp11 * b1;
            const double Hpb2 = hp22 * b2;

            const double psi_000 = 2.0 * alpha * (2.0 * alpha - 1.0) * (2.0 * alpha - 2.0)
                * phi / (s[0] * s[0] * s[0]);
            const double psi_001 = 4.0 * alpha * (2.0 * alpha - 1.0) * (1.0 - alpha)
                * phi / (s[0] * s[0] * s[1]);
            const double psi_011 = 4.0 * alpha * (1.0 - alpha) * (1.0 - 2.0 * alpha)
                * phi / (s[0] * s[1] * s[1]);
            const double psi_111 = -4.0 * alpha * (1.0 - alpha) * (1.0 - 2.0 * alpha)
                * phi / (s[1] * s[1] * s[1]);

            const double d3p0 = a0 * b0 * psi_000
                + (a0 * b1 + a1 * b0) * psi_001
                + a1 * b1 * psi_011;
            const double d3p1 = a0 * b0 * psi_001
                + (a0 * b1 + a1 * b0) * psi_011
                + a1 * b1 * psi_111;

            eta0 += dot_b * Hpa0 * inv_p2 + dot_a * Hpb0 * inv_p2
                - d3p0 * inv_p
                - 2.0 * (1.0 - alpha) * a0 * b0 / (s[0] * s[0] * s[0]);
            eta1 += dot_b * Hpa1 * inv_p2 + dot_a * Hpb1 * inv_p2
                - d3p1 * inv_p
                - 2.0 * alpha * a1 * b1 / (s[1] * s[1] * s[1]);
            eta2 += dot_b * Hpa2 * inv_p2 + dot_a * Hpb2 * inv_p2;

            eta0 *= 0.5;
            eta1 *= 0.5;
            eta2 *= 0.5;
        }
    }

    // μ⁴ scaling + K=0.5 ∞-norm cap: mirrors CPU
    // `powcone.rs` which imports `DIRECT_X_ETA_{MU_EXP,CAP_K}` from
    // `expcone.rs` (sweep-tuned defaults).
    const double mu_sq = mu_b * mu_b;
    const double mu_quad = mu_sq * mu_sq;
    eta0 *= mu_quad;
    eta1 *= mu_quad;
    eta2 *= mu_quad;

    const double K = 0.5;
    double max_ds = fmax(fmax(fabs(sx[0]), fabs(sx[1])), fabs(sx[2]));
    double max_eta = fmax(fmax(fabs(eta0), fabs(eta1)), fabs(eta2));
    double cap = K * max_ds;
    if (max_eta > cap && max_ds > 0.0) {
        double scale = cap / max_eta;
        eta0 *= scale; eta1 *= scale; eta2 *= scale;
    }

    step_rhs_z_x[zx_b + 0] += sm * xcone_grad_primal[zx_b + 0] - eta0;
    step_rhs_z_x[zx_b + 1] += sm * xcone_grad_primal[zx_b + 1] - eta1;
    step_rhs_z_x[zx_b + 2] += sm * xcone_grad_primal[zx_b + 2] - eta2;
    } // end cone loop
}

void add_combined_ds_shift_pow(
    double* step_rhs_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const double* mu,
    const double* var_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* xcone_Hs,
    const double* xcone_grad_primal,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    add_combined_ds_shift_pow_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, sigma_mu, mehrotra_m, mu,
        var_x, step_aff_x, step_aff_z_x,
        xcone_Hs, xcone_grad_primal,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        d_xcone_pow_idx, d_xcone_pow_alpha,
        n, totalXConeNumel, totalXConeHsEntries, numXCones);
}

// --------------------------------------------------------------------------
// 4. xcone_step_length_pow_reduce
//
// CPU: direct_x_step_length — backtrack on is_primal_feasible(x + α·dx) for
// primal (alpha_z) and is_dual_feasible(z + α·dz) for dual (alpha_s).
// Per (batch, cone) block, single-threaded, filter by kind == 4.
// Atomic-min into alpha_z (primal x constraint) and alpha_s (dual z_x).
// --------------------------------------------------------------------------
__global__ void xcone_step_length_pow_reduce_kernel(
    double* __restrict__ alpha_s,
    double* __restrict__ alpha_z,
    const double* __restrict__ var_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ step_x,
    const double* __restrict__ step_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_pow_idx,
    const double* __restrict__ d_xcone_pow_alpha,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack)
{
    const int64_t b = blockIdx.x;

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    const double alpha = d_xcone_pow_alpha[d_xcone_pow_idx[c]];

    // Gather x[J] and dx[J] for the primal backtrack.
    double xj[3], dxj[3];
    for (int i = 0; i < 3; ++i) {
        const int64_t idx = d_xcone_indices[num_off + i];
        xj[i]  = var_x[x_off + idx];
        dxj[i] = step_x[x_off + idx];
    }
    // Gather z_x and dz_x for the dual backtrack (flat indexed).
    double zj[3], dzj[3];
    for (int i = 0; i < 3; ++i) {
        zj[i]  = var_z_x[zx_off + num_off + i];
        dzj[i] = step_z_x[zx_off + num_off + i];
    }

    // CPU `backtrack_search` returns 0 when no feasible α ≥ α_min is found
    // (nonsymmetric_common.rs:1174-1176). Match that semantic.
    //
    // Start from `min(max_step, alpha_{z,s}[b])` (the already-applied τ/κ +
    // slack-cone bound) — see the Exp kernel above for the rationale.
    double alpha_primal = fmin(max_step, alpha_z[b]);
    {
        double trial[3];
        bool found = false;
        while (alpha_primal > min_step) {
            for (int i = 0; i < 3; ++i) trial[i] = xj[i] + alpha_primal * dxj[i];
            if (pow_is_primal_feasible(trial, alpha)) { found = true; break; }
            alpha_primal *= backtrack;
        }
        if (!found) alpha_primal = 0.0;
    }

    double alpha_dual = fmin(max_step, alpha_s[b]);
    {
        double trial[3];
        bool found = false;
        while (alpha_dual > min_step) {
            for (int i = 0; i < 3; ++i) trial[i] = zj[i] + alpha_dual * dzj[i];
            if (pow_is_dual_feasible(trial, alpha)) { found = true; break; }
            alpha_dual *= backtrack;
        }
        if (!found) alpha_dual = 0.0;
    }

    // Fold into the per-batch alpha arrays.
    cones::atomic_min_pos_double(&alpha_z[b], alpha_primal);
    cones::atomic_min_pos_double(&alpha_s[b], alpha_dual);
    } // end cone loop
}

void xcone_step_length_pow_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    xcone_step_length_pow_reduce_kernel<<<grid, 256, 0, stream>>>(
        alpha_s, alpha_z, var_x, var_z_x, step_x, step_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        d_xcone_pow_idx, d_xcone_pow_alpha,
        n, numXCones, totalXConeNumel, max_step, min_step, backtrack);
}

// --------------------------------------------------------------------------
// 5. subtract_xcone_combined_from_workx_pow
//
// CPU: direct_x_combined_offset(out, ds, work, x) { out = ds } (identity).
// So c_J = rhs_z_x (no division by x), and workx[J] -= rhs_z_x.
// One block per batch, threads stride over cones, filter by kind == 4.
// --------------------------------------------------------------------------
__global__ void subtract_xcone_combined_from_workx_pow_kernel(
    double* __restrict__ workx,
    const double* __restrict__ rhs_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    for (int i = 0; i < 3; ++i) {
        const int64_t idx = d_xcone_indices[num_off + i];
        workx[x_off + idx] -= rhs_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void subtract_xcone_combined_from_workx_pow(
    double* workx,
    const double* rhs_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    subtract_xcone_combined_from_workx_pow_kernel<<<grid, 256, 0, stream>>>(
        workx, rhs_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

// --------------------------------------------------------------------------
// 6. recover_dz_x_affine_pow
//
// CPU: recover_direct_x_dual (affine):
//   Δz_J = -direct_x_mul_Hs(Δx[J]) - variables.z_x[J]
// For Power, Hs is the dense 3×3 (packed upper-tri). mul_Hs does a
// symmetric matvec. One block per batch, threads stride over cones.
// --------------------------------------------------------------------------
__global__ void recover_dz_x_affine_pow_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off  = d_xcone_numel_offsets[c];
    const int64_t hs_off   = d_xcone_hs_offsets[c];
    const int64_t x_off    = b * n;
    const int64_t zx_off   = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;

    // Gather Δx[J].
    double dx[3];
    for (int i = 0; i < 3; ++i) {
        dx[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];
    }

    // Read packed column-major upper-tri Hs:
    // [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)].
    const double* H = xcone_Hs + hs_off_b + hs_off;

    double y[3];
    y[0] = H[0]*dx[0] + H[1]*dx[1] + H[3]*dx[2];
    y[1] = H[1]*dx[0] + H[2]*dx[1] + H[4]*dx[2];
    y[2] = H[3]*dx[0] + H[4]*dx[1] + H[5]*dx[2];

    for (int i = 0; i < 3; ++i) {
        dz_x[zx_off + num_off + i] = -y[i] - var_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void recover_dz_x_affine_pow(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_affine_pow_kernel<<<grid, 256, 0, stream>>>(
        dz_x, lhs_x, var_z_x, xcone_Hs,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// --------------------------------------------------------------------------
// 7. recover_dz_x_combined_pow
//
// CPU: recover_direct_x_dual (combined):
//   Δz_J = -direct_x_mul_Hs(Δx[J]) - c_J
// For Power, direct_x_combined_offset gives c_J = rhs_z_x (identity copy).
// So: dz_x = -Hs·dx[J] - rhs_z_x.
// --------------------------------------------------------------------------
__global__ void recover_dz_x_combined_pow_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::Power) continue;

    const int64_t num_off  = d_xcone_numel_offsets[c];
    const int64_t hs_off   = d_xcone_hs_offsets[c];
    const int64_t x_off    = b * n;
    const int64_t zx_off   = b * totalXConeNumel;
    const int64_t hs_off_b = b * totalXConeHsEntries;

    // Gather Δx[J].
    double dx[3];
    for (int i = 0; i < 3; ++i) {
        dx[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];
    }

    // Read packed column-major upper-tri Hs:
    // [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)].
    const double* H = xcone_Hs + hs_off_b + hs_off;

    double y[3];
    y[0] = H[0]*dx[0] + H[1]*dx[1] + H[3]*dx[2];
    y[1] = H[1]*dx[0] + H[2]*dx[1] + H[4]*dx[2];
    y[2] = H[3]*dx[0] + H[4]*dx[1] + H[5]*dx[2];

    for (int i = 0; i < 3; ++i) {
        dz_x[zx_off + num_off + i] = -y[i] - rhs_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void recover_dz_x_combined_pow(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_combined_pow_kernel<<<grid, 256, 0, stream>>>(
        dz_x, lhs_x, rhs_z_x, xcone_Hs,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel, totalXConeHsEntries);
}

// ============================================================================
// Direct-x GenPowerCone kernels (Phase 4)
// ============================================================================
//
// Variable-dim cone K = {(p,w): ∏ p_i^αi ≥ ‖w‖₂, p_i≥0}
// dim1 = len(alphas), dim2 = len(w), dim = dim1 + dim2.
// Primal barrier: F(x) = -log(ζ) - Σ(1-αᵢ)log(xᵢ)  (renormalized)
//   where ζ = phi - norm2w, phi = exp(Σ 2αᵢ log xᵢ)
//
// Hessian Hs = μ·(D + p·p' - q·q' - r·r') (rank-3 sparse for dim>4)
// Feasibility helpers:

__device__ __forceinline__ bool genpow_is_primal_feasible(
    const double* xj, int64_t dim1, int64_t dim2, const double* alphas)
{
    for (int i = 0; i < dim1; ++i)
        if (xj[i] <= 0.0) return false;
    double log_phi = 0.0;
    for (int i = 0; i < dim1; ++i)
        log_phi += 2.0 * alphas[i] * log(xj[i]);
    double phi = exp(log_phi);
    double norm2w = 0.0;
    for (int j = 0; j < dim2; ++j)
        norm2w += xj[dim1 + j] * xj[dim1 + j];
    return phi > norm2w;
}

__device__ __forceinline__ bool genpow_is_dual_feasible(
    const double* zj, int64_t dim1, int64_t dim2, const double* alphas)
{
    for (int i = 0; i < dim1; ++i)
        if (zj[i] <= 0.0) return false;
    double log_phi = 0.0;
    for (int i = 0; i < dim1; ++i) {
        double ai = alphas[i];
        log_phi += 2.0 * ai * log(zj[i] / ai);
    }
    double phi = exp(log_phi);
    double norm2w = 0.0;
    for (int j = 0; j < dim2; ++j)
        norm2w += zj[dim1 + j] * zj[dim1 + j];
    return phi > norm2w;
}

// --------------------------------------------------------------------------
// init_xcone_x_genpow
// x[i] = sqrt(1 + alphas[i]) for i < dim1, x[dim1+j] = 0 for j < dim2.
// One block per batch, threads stride over cones, filter by kind == 5.
// --------------------------------------------------------------------------
__global__ void init_xcone_x_genpow_kernel(
    double* __restrict__ x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const double* __restrict__ d_xcone_genpow_alphas,
    int64_t n, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t gidx    = d_xcone_genpow_idx[c];
    const int64_t dim1    = d_xcone_genpow_dim1s[gidx];
    const int64_t dim     = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];
    const int64_t x_off   = b * n;

    for (int64_t i = 0; i < dim1; ++i) {
        double ai = d_xcone_genpow_alphas[a_off + i];
        x[x_off + d_xcone_indices[num_off + i]] = sqrt(1.0 + ai);
    }
    for (int64_t j = 0; j < dim - dim1; ++j) {
        x[x_off + d_xcone_indices[num_off + dim1 + j]] = 0.0;
    }
    } // end cone loop
}

void init_xcone_x_genpow(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    init_xcone_x_genpow_kernel<<<grid, 256, 0, stream>>>(
        x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_indices, d_xcone_genpow_idx, d_xcone_genpow_dim1s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_alphas, n, numXCones);
}

// --------------------------------------------------------------------------
// init_xcone_z_x_genpow
// Same as init_xcone_x_genpow but writes to z_x (flat numel layout).
// --------------------------------------------------------------------------
__global__ void init_xcone_z_x_genpow_kernel(
    double* __restrict__ z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const double* __restrict__ d_xcone_genpow_alphas,
    int64_t totalXConeNumel, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t gidx    = d_xcone_genpow_idx[c];
    const int64_t dim1    = d_xcone_genpow_dim1s[gidx];
    const int64_t dim     = d_xcone_dims[c];
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];
    const int64_t zx_off  = b * totalXConeNumel;

    for (int64_t i = 0; i < dim1; ++i) {
        double ai = d_xcone_genpow_alphas[a_off + i];
        z_x[zx_off + num_off + i] = sqrt(1.0 + ai);
    }
    for (int64_t j = 0; j < dim - dim1; ++j) {
        z_x[zx_off + num_off + dim1 + j] = 0.0;
    }
    } // end cone loop
}

void init_xcone_z_x_genpow(
    double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t batchSize, int64_t numXCones, int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    init_xcone_z_x_genpow_kernel<<<grid, 256, 0, stream>>>(
        z_x, d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_alphas,
        totalXConeNumel, numXCones);
}

// --------------------------------------------------------------------------
// update_xcones_genpow_scaling
//
// Computes Hs = μ*(D + p·p' - q·q' - r·r') from primal-barrier Hessian.
// Dense (dim<=4): writes full upper-tri to xcone_Hs.
// Sparse (dim>4): writes diagonal to xcone_Hs; p/q/r vectors to working
// storage; optionally scatters into KKT (fused path).
// --------------------------------------------------------------------------
__global__ void update_xcones_genpow_scaling_kernel(
    const double* __restrict__ x,
    const double* __restrict__ z_x,
    const double* __restrict__ mu,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_dim2s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_idx,
    const int64_t* __restrict__ d_xcone_genpow_sparse_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_q_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_r_offsets,
    const double* __restrict__ d_xcone_genpow_alphas,
    double* __restrict__ xcone_Hs,
    double* __restrict__ xcone_grad_primal,
    double* __restrict__ xcone_genpow_p,
    double* __restrict__ xcone_genpow_q,
    double* __restrict__ xcone_genpow_r,
    double* __restrict__ xcone_genpow_d1,
    double* __restrict__ xcone_genpow_d2,
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_px_baseline,
    const int64_t* __restrict__ H_xcone_hs_idx,
    const int64_t* __restrict__ H_xcone_genpow_q_idx,
    const int64_t* __restrict__ H_xcone_genpow_r_idx,
    const int64_t* __restrict__ H_xcone_genpow_p_idx,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_0,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_1,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_2,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_3,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_4,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_5,
    const int64_t* __restrict__ H_xcone_genpow_exp_diag_idx,
    const double* __restrict__ xgenpow_pd_axes,
    const double* __restrict__ xgenpow_pd_coefs,
    const double* __restrict__ xgenpow_pd_signs,
    const double* __restrict__ xgenpow_pd_active,
    int64_t nnzKKT,
    int64_t n,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    int64_t totalSparseXGenPowDim)
{
    const int64_t b = blockIdx.x;
    const int64_t* xpd_axis_idx_arr[6] = {
        H_xcone_genpow_pd_axis_idx_0, H_xcone_genpow_pd_axis_idx_1,
        H_xcone_genpow_pd_axis_idx_2, H_xcone_genpow_pd_axis_idx_3,
        H_xcone_genpow_pd_axis_idx_4, H_xcone_genpow_pd_axis_idx_5
    };
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t gidx    = d_xcone_genpow_idx[c];
    const int64_t dim1    = d_xcone_genpow_dim1s[gidx];
    const int64_t dim2    = d_xcone_genpow_dim2s[gidx];
    const int64_t dim     = dim1 + dim2;
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t hs_off  = d_xcone_hs_offsets[c];
    const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];
    const int64_t d_off   = d_xcone_genpow_dim_offsets[gidx];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t hs_b    = b * totalXConeHsEntries;
    const int64_t p_b     = b * totalXGenPowerDim;
    const int64_t q_b     = b * totalXGenPowerAlphas;
    const int64_t r_b     = b * totalXGenPowerDim2;
    const double  mu_b    = mu[b];

    const bool fused = (kkt_values != nullptr) && (xcone_px_baseline != nullptr)
                       && (H_xcone_hs_idx != nullptr);
    const int64_t kkt_off = fused ? b * nnzKKT : 0;

    // Gather x[J] (using d_xcone_indices for problem space, note: xj[i] for i<dim1 are p-part, w-part for i>=dim1)
    // We use streaming loops since dim may be large.

    // Compute log_phi = Σ 2αᵢ log(xᵢ) and check feasibility.
    double log_phi = 0.0;
    bool feasible = true;
    for (int64_t i = 0; i < dim1; ++i) {
        double xi = x[x_off + d_xcone_indices[num_off + i]];
        if (xi <= 0.0) { feasible = false; break; }
        log_phi += 2.0 * d_xcone_genpow_alphas[a_off + i] * log(xi);
    }
    // Check w part just for feasibility
    if (feasible) {
        double phi = exp(log_phi);
        double norm2w = 0.0;
        for (int64_t j = 0; j < dim2; ++j) {
            double wj = x[x_off + d_xcone_indices[num_off + dim1 + j]];
            norm2w += wj * wj;
        }
        if (phi <= norm2w) feasible = false;
    }

    if (!feasible) {
        // Identity fallback: Hs = I (diagonal 1, no off-diagonal)
        if (dim > 4) {
            for (int64_t i = 0; i < dim; ++i) {
                xcone_Hs[hs_b + hs_off + i] = 1.0;
                // Write p/q/r = 0 for identity
                xcone_genpow_p[p_b + d_off + i] = 0.0;
                if (i < dim1) xcone_genpow_q[q_b + a_off + i] = 0.0;
                if (i >= dim1) xcone_genpow_r[r_b + (d_off - a_off) + (i - dim1)] = 0.0;
            }
            if (fused) {
                for (int64_t i = 0; i < dim; ++i) {
                    kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                        xcone_px_baseline[hs_b + hs_off + i] + 1.0;
                }
                // Scatter expansion cols: all zero (identity has no rank-3 component)
                const int64_t sidx   = d_xcone_genpow_sparse_idx[gidx];
                const int64_t sp_off = d_xcone_genpow_sparse_offsets[sidx];  // sparse-only dim prefix
                const int64_t sq_off = d_xcone_genpow_sparse_q_offsets[sidx]; // sparse-only dim1 prefix
                const int64_t sr_off = d_xcone_genpow_sparse_r_offsets[sidx]; // sparse-only dim2 prefix
                for (int64_t i = 0; i < dim; ++i) {
                    kkt_values[kkt_off + H_xcone_genpow_p_idx[sp_off + i]] = 0.0;
                    if (i < dim1) kkt_values[kkt_off + H_xcone_genpow_q_idx[sq_off + i]] = 0.0;
                    if (i >= dim1) kkt_values[kkt_off + H_xcone_genpow_r_idx[sr_off + (i - dim1)]] = 0.0;
                }
                // Expansion diag: q,r,p = +1, +1, -1; PD axes inactive sentinels.
                // Schur: K_eff = P + Hs_diag - μ·q·q'/d_q - μ·r·r'/d_r - μ·p·p'/d_p
                //              + Σ_k (-1/d_pd_k) · (off_pd_k)·(off_pd_k)'.
                // q,r diag = +1 → contribution -μ·qq'/-μ·rr'; p diag = -1 → +μ·pp'.
                kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 0]] =  1.0; // q-diag
                kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 1]] =  1.0; // r-diag
                kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 2]] = -1.0; // p-diag
                // 6 PD-axis slots: inactive (ε on a single basis position, sign=+1 sentinel).
                const double XPD_INACTIVE_EPS = 1.0e-8;
                for (int axk = 0; axk < 6; ++axk) {
                    for (int64_t i = 0; i < dim; ++i) {
                        double v = (i == (int64_t)axk % dim) ? XPD_INACTIVE_EPS : 0.0;
                        kkt_values[kkt_off + xpd_axis_idx_arr[axk][sp_off + i]] = v;
                    }
                    kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 3 + axk]] = 1.0;
                }
            }
        } else {
            // Dense: write identity upper-tri
            int64_t k = 0;
            for (int64_t cc = 0; cc < dim; ++cc) {
                for (int64_t rr = 0; rr <= cc; ++rr) {
                    double val = (rr == cc) ? 1.0 : 0.0;
                    xcone_Hs[hs_b + hs_off + k] = val;
                    if (fused)
                        kkt_values[kkt_off + H_xcone_hs_idx[hs_off + k]] =
                            xcone_px_baseline[hs_b + hs_off + k] + val;
                    ++k;
                }
            }
        }
        // Write identity gradient
        for (int64_t i = 0; i < dim; ++i) {
            xcone_grad_primal[b * totalXConeNumel + num_off + i] = 0.0;
        }
        if (dim > 4) {
            xcone_genpow_d2[b * numXGenPowerCones + gidx] = 1.0;
        }
        continue;
    }

    // Compute phi, norm2w, ζ
    double phi = exp(log_phi);
    double norm2w = 0.0;
    for (int64_t j = 0; j < dim2; ++j) {
        double wj = x[x_off + d_xcone_indices[num_off + dim1 + j]];
        norm2w += wj * wj;
    }
    const double zeta = phi - norm2w;

    // Compute τᵢ = 2αᵢ/xᵢ for i < dim1
    // Gradient and diagonal d1, d2
    for (int64_t i = 0; i < dim1; ++i) {
        double xi  = x[x_off + d_xcone_indices[num_off + i]];
        double ai  = d_xcone_genpow_alphas[a_off + i];
        double tau = 2.0 * ai / xi;
        double g   = -tau * phi / zeta - (1.0 - ai) / xi;
        xcone_grad_primal[b * totalXConeNumel + num_off + i] = g;
    }
    for (int64_t j = 0; j < dim2; ++j) {
        double wj = x[x_off + d_xcone_indices[num_off + dim1 + j]];
        double g  = 2.0 * wj / zeta;
        xcone_grad_primal[b * totalXConeNumel + num_off + dim1 + j] = g;
    }

    // Compute rank-3 expansion scalars
    // p0 = sqrt(phi*(phi+norm2w)/2), p1 = -2*phi/p0
    // q0 = sqrt(ζ*phi/2), r1 = 2*sqrt(ζ/(phi+norm2w))
    // Match CPU `update_primal_grad_H` (genpowcone.rs:912-919): guard the
    // ratios with > ε so a near-boundary iterate doesn't produce NaN/Inf
    // in the Hs decomposition.
    const double phi_plus_norm2w = phi + norm2w;
    const double eps_dbl = 2.220446049250313e-16;
    const double p0 = sqrt(phi * phi_plus_norm2w / 2.0);
    const double p1 = (p0 > eps_dbl) ? (-2.0 * phi / p0) : 0.0;
    const double q0 = sqrt(zeta * phi / 2.0);
    const double r1 = (phi_plus_norm2w > eps_dbl)
        ? (2.0 * sqrt(zeta / phi_plus_norm2w))
        : 0.0;

    if (dim > 4) {
        // Sparse path: write diagonal Hs and p/q/r vectors
        const int64_t sidx    = d_xcone_genpow_sparse_idx[gidx];
        const int64_t sp_off  = d_xcone_genpow_sparse_offsets[sidx];   // sparse-only dim prefix
        const int64_t sq_off  = d_xcone_genpow_sparse_q_offsets[sidx]; // sparse-only dim1 prefix
        const int64_t sr_off  = d_xcone_genpow_sparse_r_offsets[sidx]; // sparse-only dim2 prefix
        // dim2 offset in global q/r data storage
        const int64_t r_cone_off = d_off - a_off;  // = dim_offset - alpha_offset

        for (int64_t i = 0; i < dim1; ++i) {
            double xi  = x[x_off + d_xcone_indices[num_off + i]];
            double ai  = d_xcone_genpow_alphas[a_off + i];
            double tau = 2.0 * ai / xi;
            double d1i = tau * phi / (zeta * xi) + (1.0 - ai) / (xi * xi);
            double pi  = (p0 / zeta) * tau;
            double qi  = (q0 / zeta) * tau;
            xcone_Hs[hs_b + hs_off + i]        = mu_b * d1i;
            // d1 stored μ-scaled to match the p,q,r,d2 storage convention
            // (sqrt(μ)·raw for vectors, μ·raw for diagonals). qr6's helper
            // is then called with mu=1.0 for direct-x to avoid double-scaling
            // (compute_xgenpow_pd_axes path). Without this write, qr6's
            // mul_μH·rank3 is missing the diagonal contribution and produces
            // wrong eigenvectors / coefs (~100× off vs CPU); apparently
            // benign for many problem shapes because the resulting
            // rank-9 contribution becomes near-zero, but mathematically
            // wrong and reveals as InsufficientProgress on tight cones.
            // q/r/p stored with the same negated `sqrt(μ)` scale CPU
            // applies in `directldlkktsolver.rs::refresh_hx_blocks`
            // (q_scale = -sqrtμ). Schur contributes -c·c'/d which is
            // sign-invariant in c, but matching the matrix entry-for-
            // entry simplifies CPU/CUDA dump diffs.
            xcone_genpow_d1[q_b + a_off + i]   = mu_b * d1i;
            xcone_genpow_p[p_b + d_off + i]    = -sqrt(mu_b) * pi;
            xcone_genpow_q[q_b + a_off + i]    = -sqrt(mu_b) * qi;
            if (fused) {
                kkt_values[kkt_off + H_xcone_hs_idx[hs_off + i]] =
                    xcone_px_baseline[hs_b + hs_off + i] + mu_b * d1i;
                kkt_values[kkt_off + H_xcone_genpow_q_idx[sq_off + i]] =
                    -sqrt(mu_b) * qi;
                kkt_values[kkt_off + H_xcone_genpow_p_idx[sp_off + i]] =
                    -sqrt(mu_b) * pi;
            }
        }
        for (int64_t j = 0; j < dim2; ++j) {
            double wj  = x[x_off + d_xcone_indices[num_off + dim1 + j]];
            double d2  = 2.0 / zeta;
            double pj  = (p1 / zeta) * wj;
            double rj  = (r1 / zeta) * wj;
            xcone_Hs[hs_b + hs_off + dim1 + j]                    = mu_b * d2;
            xcone_genpow_p[p_b + d_off + dim1 + j]               = -sqrt(mu_b) * pj;
            xcone_genpow_r[r_b + r_cone_off + j]                  = -sqrt(mu_b) * rj;
            if (fused) {
                kkt_values[kkt_off + H_xcone_hs_idx[hs_off + dim1 + j]] =
                    xcone_px_baseline[hs_b + hs_off + dim1 + j] + mu_b * d2;
                kkt_values[kkt_off + H_xcone_genpow_r_idx[sr_off + j]] =
                    -sqrt(mu_b) * rj;
                kkt_values[kkt_off + H_xcone_genpow_p_idx[sp_off + dim1 + j]] =
                    -sqrt(mu_b) * pj;
            }
        }
        // Store d2 (shared across w-part) for feasibility check
        xcone_genpow_d2[b * numXGenPowerCones + gidx] = mu_b * (2.0 / zeta);
        if (fused) {
            // Expansion diagonals: q,r,p (rank-3) and 6 PD-axis (rank-6 PD).
            // Schur: K_eff = P + Hs_diag - μ·qq'/d_q - μ·rr'/d_r - μ·pp'/d_p
            //              + Σ_k (-1/d_pd_k) · (off_pd_k)·(off_pd_k)'.
            // q,r diag = +1 → contribution -μ·qq'/-μ·rr'; p diag = -1 → +μ·pp'.
            kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 0]] =  1.0; // q-col diag
            kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 1]] =  1.0; // r-col diag
            kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 2]] = -1.0; // p-col diag

            // PD-axis writes. When pd_active=0 (or pointers null) we still must
            // emit structurally non-zero values to keep cuDSS factorisation
            // happy — tiny ε on a single basis position + sign·1 sentinel.
            // When active we mirror the slack-side equilibration logic with
            // threshold 1e12.
            const double XPD_INACTIVE_EPS = 1.0e-8;
            int64_t pd_active_idx = b * numXGenPowerCones + gidx;
            double active_flag = (xgenpow_pd_active != nullptr)
                ? xgenpow_pd_active[pd_active_idx] : 0.0;
            int64_t pd_axes_cone_base = b * 6 * totalXGenPowerDim + 6 * d_off;
            int64_t pd_state_cone_base = b * 6 * numXGenPowerCones + 6 * gidx;

            for (int axk = 0; axk < 6; ++axk) {
                double sign = (xgenpow_pd_signs != nullptr)
                    ? xgenpow_pd_signs[pd_state_cone_base + axk] : 1.0;
                double coef = (xgenpow_pd_coefs != nullptr)
                    ? xgenpow_pd_coefs[pd_state_cone_base + axk] : 0.0;
                if (active_flag > 0.5 && coef > 0.0) {
                    double n_sq = 0.0;
                    for (int64_t i = 0; i < dim; ++i) {
                        double a_val = xgenpow_pd_axes[pd_axes_cone_base + axk * dim + i];
                        n_sq += a_val * a_val;
                    }
                    double w = coef * n_sq;
                    double off_scale, sent;
                    // Schur contribution wants -1/d · c·c' = +sign·coef·a·a'.
                    // With c = ±sqrt(coef)·a, that's c·c'/d = +coef·a·a' when
                    // d = +1 (sign=-1) or -coef·a·a' when d = -1 (sign=+1).
                    // I.e., dsign = -sign (matches slack convention).
                    if (w > 1.0e12 && n_sq > 0.0) {
                        off_scale = -1.0 / sqrt(n_sq);
                        sent = sign / w;
                    } else {
                        off_scale = -sqrt(coef);
                        sent = sign;
                    }
                    for (int64_t i = 0; i < dim; ++i) {
                        double a_val = xgenpow_pd_axes[pd_axes_cone_base + axk * dim + i];
                        kkt_values[kkt_off + xpd_axis_idx_arr[axk][sp_off + i]]
                            = off_scale * a_val;
                    }
                    kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 3 + axk]]
                        = sent;
                } else {
                    for (int64_t i = 0; i < dim; ++i) {
                        double v = (i == (int64_t)axk % dim) ? XPD_INACTIVE_EPS : 0.0;
                        kkt_values[kkt_off + xpd_axis_idx_arr[axk][sp_off + i]] = v;
                    }
                    double sent_sign = (sign >= 0.0) ? 1.0 : -1.0;
                    kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 3 + axk]]
                        = sent_sign;
                }
            }
        }
    } else {
        // Dense path (dim <= 4): compute the full Mosek-Tunçel direct-x Hs
        // matching CPU `pd_scaling_nd_dense` (called from
        // `GenPowerCone::try_build_dense_pd_hs_primal`). The matrix is
        //   Hs = P_⊥·(μ·H_primal)·P_⊥ + (1/⟨z_x,x⟩)·z_x·z_xᵀ
        //                              + (1/⟨δz_x,δx⟩)·δz_x·δz_xᵀ,
        // where μ·H_primal is the rank-3 form (D + p·pᵀ − q·qᵀ − r·rᵀ),
        // δz_x = z_x + μ_local·g_x, δx = x + μ_local·g_zx,
        // μ_local = ⟨z_x, x⟩ / dim1, and P_⊥ projects orthogonal to
        // span{z_x, δz_x}. Without the secant rank-2 terms (and the
        // matching P_⊥ cleanup) the Newton coefficient property
        // `Hs·x = z_x` fails — CPU and CUDA then disagree on the iter-1
        // KKT direction on dim≤4 problems.
        double pv[4], qv[4], rv[4], dv[4];
        for (int64_t i = 0; i < dim1; ++i) {
            double xi  = x[x_off + d_xcone_indices[num_off + i]];
            double ai  = d_xcone_genpow_alphas[a_off + i];
            double tau = 2.0 * ai / xi;
            double d1i = tau * phi / (zeta * xi) + (1.0 - ai) / (xi * xi);
            pv[i]  = (p0 / zeta) * tau;
            qv[i]  = (q0 / zeta) * tau;
            dv[i]  = d1i;
        }
        double d2 = 2.0 / zeta;
        for (int64_t j = 0; j < dim2; ++j) {
            double wj = x[x_off + d_xcone_indices[num_off + dim1 + j]];
            pv[dim1 + j] = (p1 / zeta) * wj;
            rv[j]        = (r1 / zeta) * wj;
            dv[dim1 + j] = d2;
        }

        // Build the full symmetric μ·H in a fixed-size buffer (dim ≤ 4 → 16
        // entries). Stored row-major; later we'll extract the upper-tri
        // back into xcone_Hs.
        double H[16];
        for (int64_t cc = 0; cc < dim; ++cc) {
            for (int64_t rr = 0; rr < dim; ++rr) {
                double val = (rr == cc ? dv[rr] : 0.0)
                           + pv[rr] * pv[cc]
                           - (rr < dim1 && cc < dim1 ? qv[rr] * qv[cc] : 0.0)
                           - (rr >= dim1 && cc >= dim1
                                ? rv[rr - dim1] * rv[cc - dim1] : 0.0);
                H[rr * 4 + cc] = val * mu_b;
            }
        }
        // Snapshot the rank-3 μ·H_primal before applying the projector
        // and secant rank-2 updates below. CPU `try_build_dense_pd_hs_primal`
        // verifies the resulting H against the secants `H·x = z_x` and
        // `H·δx = δz_x` (genpowcone.rs:466-505) and falls back to this
        // rank-3 form if either residual exceeds tolerance. Without that
        // verification, CUDA may emit an Hs that satisfies neither secant
        // when the projector basis is ill-conditioned but escaped the
        // norm-based bails above.
        double H_rank3[16];
        for (int i = 0; i < 16; ++i) H_rank3[i] = H[i];

        // Gather z_x and compute the dual-barrier gradient g_zx at z_x.
        // CPU `try_build_dense_pd_hs_primal` builds g_zx from z_x using
        //   ψ_z = Π_{i<dim1} (z_i/α_i)^(2α_i),
        //   ζ_z = ψ_z − ‖z_w‖²,
        //   g_zx[i<dim1]  = -(2α_i/z_i)·ψ_z/ζ_z − (1−α_i)/z_i,
        //   g_zx[i≥dim1]  = (2/ζ_z)·z_i.
        double zxv[4];
        for (int64_t i = 0; i < dim; ++i) {
            zxv[i] = z_x[zx_off + num_off + i];
        }
        double log_phi_z = 0.0;
        bool z_feasible = true;
        for (int64_t i = 0; i < dim1; ++i) {
            double zi = zxv[i];
            double ai = d_xcone_genpow_alphas[a_off + i];
            if (zi <= 0.0 || ai <= 0.0) { z_feasible = false; break; }
            log_phi_z += 2.0 * ai * log(zi / ai);
        }
        double phi_z = z_feasible ? exp(log_phi_z) : 0.0;
        double norm2w_z = 0.0;
        for (int64_t j = 0; j < dim2; ++j) {
            double wj = zxv[dim1 + j];
            norm2w_z += wj * wj;
        }
        double zeta_z = phi_z - norm2w_z;
        const double eps_sqrt = 1.4901161193847656e-8;
        bool secant_ok = z_feasible && (zeta_z > 0.0);

        // ⟨z_x, x⟩ and μ_local = ⟨z_x, x⟩ / dim1 (cone degree).
        double xz = 0.0;
        if (secant_ok) {
            double xv[4];
            for (int64_t i = 0; i < dim; ++i) {
                xv[i] = x[x_off + d_xcone_indices[num_off + i]];
            }
            for (int64_t i = 0; i < dim; ++i) xz += xv[i] * zxv[i];
            if (!(xz > eps_sqrt)) secant_ok = false;

            if (secant_ok) {
                // CPU `GenPowerCone::degree() = dim1 + 1` (not dim1) —
                // see `genpowcone.rs::Cone::degree`. μ_local = ⟨z_x,x⟩/ν.
                const double mu_local = xz / (double)(dim1 + 1);

                // g_zx = ∇F*(z_x).
                double g_zx[4];
                for (int64_t i = 0; i < dim1; ++i) {
                    double tau_z = 2.0 * d_xcone_genpow_alphas[a_off + i] / zxv[i];
                    g_zx[i] = -tau_z * phi_z / zeta_z
                              - (1.0 - d_xcone_genpow_alphas[a_off + i]) / zxv[i];
                }
                for (int64_t j = 0; j < dim2; ++j) {
                    g_zx[dim1 + j] = (2.0 / zeta_z) * zxv[dim1 + j];
                }

                // g_x = ∇F(x): identical to the gradient we just wrote to
                // xcone_grad_primal[zx_off + num_off + i] above (line 5204/5209).
                double g_x[4];
                for (int64_t i = 0; i < dim; ++i) {
                    g_x[i] = xcone_grad_primal[zx_off + num_off + i];
                }

                // δz_x = z_x + μ_local·g_x, δx = x + μ_local·g_zx.
                double dz[4], dxv[4];
                for (int64_t i = 0; i < dim; ++i) {
                    dz[i]  = zxv[i] + mu_local * g_x[i];
                    dxv[i] = xv[i]  + mu_local * g_zx[i];
                }
                double dd = 0.0;
                for (int64_t i = 0; i < dim; ++i) dd += dz[i] * dxv[i];
                if (!(dd > eps_sqrt)) { secant_ok = false; }

                if (secant_ok) {
                    // Gram-Schmidt on span{x, δx} (the `z` argument to
                    // `pd_scaling_nd_dense` is x after the direct-x swap).
                    // e1 = x/‖x‖, e2 = (δx − ⟨δx,e1⟩·e1)/‖·‖ with one Kahan
                    // re-orthogonalisation pass. The projector P_⊥ ⊥ {e1,e2}
                    // is the one that's "cleaned out" of μ·H; the secant
                    // rank-2 update lives in span{z_x, δz_x}, a different
                    // (in general non-orthogonal) subspace.
                    double x_norm = 0.0;
                    for (int64_t i = 0; i < dim; ++i) x_norm += xv[i] * xv[i];
                    x_norm = sqrt(x_norm);
                    if (!(x_norm > eps_sqrt)) {
                        secant_ok = false;
                    } else {
                        double e1[4];
                        for (int64_t i = 0; i < dim; ++i) e1[i] = xv[i] / x_norm;
                        double dot_dx_e1 = 0.0;
                        for (int64_t i = 0; i < dim; ++i) dot_dx_e1 += dxv[i] * e1[i];
                        double e2[4];
                        for (int64_t i = 0; i < dim; ++i)
                            e2[i] = dxv[i] - dot_dx_e1 * e1[i];
                        double e2_e1_resid = 0.0;
                        for (int64_t i = 0; i < dim; ++i) e2_e1_resid += e2[i] * e1[i];
                        for (int64_t i = 0; i < dim; ++i)
                            e2[i] -= e2_e1_resid * e1[i];
                        double e2_norm = 0.0;
                        for (int64_t i = 0; i < dim; ++i) e2_norm += e2[i] * e2[i];
                        e2_norm = sqrt(e2_norm);
                        if (!(e2_norm > eps_sqrt)) {
                            secant_ok = false;
                        } else {
                            for (int64_t i = 0; i < dim; ++i) e2[i] /= e2_norm;

                            // h1 = μH·e1, h2 = μH·e2; q11/q22/q12 quadratic forms.
                            double h1[4] = {0,0,0,0};
                            double h2[4] = {0,0,0,0};
                            for (int64_t rr = 0; rr < dim; ++rr) {
                                for (int64_t cc = 0; cc < dim; ++cc) {
                                    h1[rr] += H[rr * 4 + cc] * e1[cc];
                                    h2[rr] += H[rr * 4 + cc] * e2[cc];
                                }
                            }
                            double q11 = 0.0, q22 = 0.0, q12 = 0.0;
                            for (int64_t i = 0; i < dim; ++i) {
                                q11 += e1[i] * h1[i];
                                q22 += e2[i] * h2[i];
                                q12 += e1[i] * h2[i];
                            }

                            // P_⊥·H·P_⊥ = H − h1·e1ᵀ − e1·h1ᵀ − h2·e2ᵀ − e2·h2ᵀ
                            //             + q11·e1·e1ᵀ + q22·e2·e2ᵀ
                            //             + q12·(e1·e2ᵀ + e2·e1ᵀ).
                            for (int64_t rr = 0; rr < dim; ++rr) {
                                for (int64_t cc = 0; cc < dim; ++cc) {
                                    double v = H[rr * 4 + cc];
                                    v -= h1[rr] * e1[cc] + e1[rr] * h1[cc];
                                    v -= h2[rr] * e2[cc] + e2[rr] * h2[cc];
                                    v += q11 * e1[rr] * e1[cc]
                                       + q22 * e2[rr] * e2[cc]
                                       + q12 * (e1[rr] * e2[cc] + e2[rr] * e1[cc]);
                                    // Secant rank-2 updates.
                                    v += (zxv[rr] * zxv[cc]) / xz
                                       + (dz[rr]  * dz[cc])  / dd;
                                    H[rr * 4 + cc] = v;
                                }
                            }
                            // Verify the secants `H·x ≈ z_x` and
                            // `H·δx ≈ δz_x` — mirrors CPU
                            // `try_build_dense_pd_hs_primal` lines
                            // 447-505. The projector cleanup uses
                            // `μ·H`'s spectrum, and the rank-2 secant
                            // update lives in `span{z_x, δz_x}`; when
                            // those subspaces are nearly co-linear
                            // the resulting H satisfies neither
                            // secant. Fall back to dual-only
                            // `Hs = μ·H_primal` (the saved
                            // `H_rank3`) in that case.
                            double y_x[4]  = {0.0, 0.0, 0.0, 0.0};
                            double y_dx[4] = {0.0, 0.0, 0.0, 0.0};
                            for (int64_t rr = 0; rr < dim; ++rr) {
                                for (int64_t cc = 0; cc < dim; ++cc) {
                                    y_x[rr]  += H[rr * 4 + cc] * xv[cc];
                                    y_dx[rr] += H[rr * 4 + cc] * dxv[cc];
                                }
                            }
                            double z_norm_sq = 0.0, dz_norm_sq = 0.0;
                            double err_x_sq = 0.0, err_dx_sq = 0.0;
                            for (int64_t i = 0; i < dim; ++i) {
                                z_norm_sq  += zxv[i] * zxv[i];
                                dz_norm_sq += dz[i]  * dz[i];
                                double dx_err  = y_x[i]  - zxv[i];
                                double ddx_err = y_dx[i] - dz[i];
                                err_x_sq  += dx_err  * dx_err;
                                err_dx_sq += ddx_err * ddx_err;
                            }
                            double z_norm  = sqrt(z_norm_sq);
                            double dz_norm = sqrt(dz_norm_sq);
                            // CPU uses tol = max(1e-7, sqrt(n)·ulp·max|coef|);
                            // the dense path doesn't keep per-axis coefs
                            // separately, so use the constant floor.
                            const double tol_secant = 1e-7;
                            double r_x  = sqrt(err_x_sq)  / fmax(z_norm,  1.0);
                            double r_dx = sqrt(err_dx_sq) / fmax(dz_norm, 1.0);
                            if (r_x > tol_secant || r_dx > tol_secant) {
                                for (int i = 0; i < 16; ++i) {
                                    H[i] = H_rank3[i];
                                }
                            }
                        }
                    }
                }
            }
        }
        // Force exact symmetry (each off-diagonal averaged with its mirror).
        for (int64_t rr = 0; rr < dim; ++rr) {
            for (int64_t cc = rr + 1; cc < dim; ++cc) {
                double avg = 0.5 * (H[rr * 4 + cc] + H[cc * 4 + rr]);
                H[rr * 4 + cc] = avg;
                H[cc * 4 + rr] = avg;
            }
        }

        // Write upper-tri (column-major: index k = cc·(cc+1)/2 + rr).
        int64_t k = 0;
        for (int64_t cc = 0; cc < dim; ++cc) {
            for (int64_t rr = 0; rr <= cc; ++rr) {
                double val = H[rr * 4 + cc];
                xcone_Hs[hs_b + hs_off + k] = val;
                if (fused)
                    kkt_values[kkt_off + H_xcone_hs_idx[hs_off + k]] =
                        xcone_px_baseline[hs_b + hs_off + k] + val;
                ++k;
            }
        }
        xcone_genpow_d2[b * numXGenPowerCones + gidx] = mu_b * d2;
        (void)secant_ok;
        (void)H;
    }
    } // end cone loop
}

void update_xcones_genpow_scaling(
    const double* x,
    const double* z_x,
    const double* mu,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    const int64_t* d_xcone_genpow_sparse_q_offsets,
    const int64_t* d_xcone_genpow_sparse_r_offsets,
    const double*  d_xcone_genpow_alphas,
    double* xcone_Hs,
    double* xcone_grad_primal,
    double* xcone_genpow_p,
    double* xcone_genpow_q,
    double* xcone_genpow_r,
    double* xcone_genpow_d1,
    double* xcone_genpow_d2,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    const int64_t* H_xcone_genpow_q_idx,
    const int64_t* H_xcone_genpow_r_idx,
    const int64_t* H_xcone_genpow_p_idx,
    const int64_t* H_xcone_genpow_pd_axis_idx_0,
    const int64_t* H_xcone_genpow_pd_axis_idx_1,
    const int64_t* H_xcone_genpow_pd_axis_idx_2,
    const int64_t* H_xcone_genpow_pd_axis_idx_3,
    const int64_t* H_xcone_genpow_pd_axis_idx_4,
    const int64_t* H_xcone_genpow_pd_axis_idx_5,
    const int64_t* H_xcone_genpow_exp_diag_idx,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    int64_t totalSparseXGenPowDim,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    update_xcones_genpow_scaling_kernel<<<grid, 256, 0, stream>>>(
        x, z_x, mu,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_dim_offsets,
        d_xcone_genpow_sparse_idx, d_xcone_genpow_sparse_offsets,
        d_xcone_genpow_sparse_q_offsets, d_xcone_genpow_sparse_r_offsets,
        d_xcone_genpow_alphas,
        xcone_Hs, xcone_grad_primal,
        xcone_genpow_p, xcone_genpow_q, xcone_genpow_r,
        xcone_genpow_d1, xcone_genpow_d2,
        kkt_values, xcone_px_baseline, H_xcone_hs_idx,
        H_xcone_genpow_q_idx, H_xcone_genpow_r_idx, H_xcone_genpow_p_idx,
        H_xcone_genpow_pd_axis_idx_0, H_xcone_genpow_pd_axis_idx_1,
        H_xcone_genpow_pd_axis_idx_2, H_xcone_genpow_pd_axis_idx_3,
        H_xcone_genpow_pd_axis_idx_4, H_xcone_genpow_pd_axis_idx_5,
        H_xcone_genpow_exp_diag_idx,
        xgenpow_pd_axes, xgenpow_pd_coefs, xgenpow_pd_signs, xgenpow_pd_active,
        nnzKKT, n, numXCones, numXGenPowerCones,
        totalXConeNumel, totalXConeHsEntries,
        totalXGenPowerDim, totalXGenPowerAlphas, totalXGenPowerDim2,
        numSparseXGenPow, totalSparseXGenPowDim);
}

// --------------------------------------------------------------------------
// fill_step_rhs_zx_genpow_affine
// Affine offset for GenPow direct-x: rhs_z_x = z_x (identity copy).
// --------------------------------------------------------------------------
__global__ void fill_step_rhs_zx_genpow_affine_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ var_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    int64_t totalXConeNumel, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t dim     = d_xcone_dims[c];
    const int64_t zx_off  = b * totalXConeNumel;

    for (int64_t i = 0; i < dim; ++i) {
        step_rhs_z_x[zx_off + num_off + i] = var_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void fill_step_rhs_zx_genpow_affine(
    double* step_rhs_z_x,
    const double* var_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    fill_step_rhs_zx_genpow_affine_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, var_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        totalXConeNumel, numXCones);
}

// --------------------------------------------------------------------------
// subtract_eta_primal_genpow
//
// 3rd-order Mehrotra η correction for direct-x GenPow:
//   step_rhs_z_x[i] -= (μ/2) · D³F(x)[step_x, step_x, ·]
// applied with K=7 ∞-norm cap against ‖step_z‖_∞.
//
// Math: F is the primal GenPower barrier
//   F(x) = -log ψ(x) - Σ_{i<n1} (1−α_i) log x_i,
//   ψ(x) = φ(x) − ‖w‖²,   φ(x) = Π_{i<n1} x_i^{2α_i},   w = x[n1..]
//
// Mirrors CPU `GenPowerCone::higher_correction_primal_direct`. Both arguments
// of D³F are the affine x-step (linearisation of `z_x + μ·∇F(x) = 0`
// gives a 3rd-order term `(μ/2)·D³F[δx, δx, ·]`); the explicit μ
// matters because direct-x stores `Hs = μ·H_primal + …` directly
// (slack absorbs μ via NT W-scaling).
// --------------------------------------------------------------------------
__global__ void subtract_eta_primal_genpow_kernel(
    double* __restrict__ step_rhs_z_x,           // (B, totalXConeNumel) — η subtracted here
    const double* __restrict__ var_x,            // (B, n) — current iterate
    const double* __restrict__ step_x,           // (B, n) — affine x-step
    const double* __restrict__ step_z,           // (B, totalXConeNumel) — affine z_x-step (for cap)
    const double* __restrict__ mu_per_batch,     // (B,)
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,        // (numXCones,) — index into genpow arrays
    const int64_t* __restrict__ d_xcone_genpow_dim1s,      // (numXGenPowerCones,)
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const double* __restrict__ d_xcone_genpow_alphas,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    if (b >= batchSize) return;
    const double mu = mu_per_batch[b];
    // η defaults: ∞-norm cap K=1.5 (constant), μ⁴ scaling. Mirrors CPU
    // `GenPowerCone::higher_correction_primal_direct`; K=1.5 μ⁴ chosen by
    // parameter sweep across dense-path and sparse-path problems.
    const double two = 2.0;
    const double half = 0.5;
    const double eps_sqrt = 1.4901161193847656e-8;

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
        if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

        const int64_t gidx = d_xcone_genpow_idx[c];
        const int64_t dim1 = d_xcone_genpow_dim1s[gidx];
        const int64_t dim  = d_xcone_dims[c];
        const int64_t dim2 = dim - dim1;
        const double K_cap = 1.5;
        (void)dim;  // K is now constant (was K=clip(dim/9.6,5,10) under μ²)
        const int64_t num_off = d_xcone_numel_offsets[c];
        const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];

        const double* alphas = &d_xcone_genpow_alphas[a_off];

        auto X = [&](int64_t i) -> double {
            return var_x[b * n + d_xcone_indices[num_off + i]];
        };
        auto SX = [&](int64_t i) -> double {
            return step_x[b * n + d_xcone_indices[num_off + i]];
        };
        auto SZ = [&](int64_t i) -> double {
            return step_z[b * totalXConeNumel + num_off + i];
        };

        // φ_p, ψ_p
        double log_phi = 0.0;
        for (int64_t i = 0; i < dim1; ++i) {
            log_phi += two * alphas[i] * log(X(i));
        }
        double phi = exp(log_phi);
        double norm2_w = 0.0;
        for (int64_t j = 0; j < dim2; ++j) {
            double wj = X(dim1 + j);
            norm2_w += wj * wj;
        }
        double psi = phi - norm2_w;
        if (psi <= 0.0 || psi < eps_sqrt * phi) continue;

        // u = v = step_x
        // σ_u, ρ_uu over i<dim1
        double sigma_u = 0.0, rho_uu = 0.0;
        for (int64_t i = 0; i < dim1; ++i) {
            double ai = two * alphas[i];
            double xi = X(i);
            double ui = SX(i);
            sigma_u += ai * ui / xi;
            rho_uu  += ai * ui * ui / (xi * xi);
        }
        // dot_u = g_ψ · u
        double dot_u = 0.0;
        for (int64_t i = 0; i < dim1; ++i) {
            double gpsi_i = two * alphas[i] * phi / X(i);
            dot_u += gpsi_i * SX(i);
        }
        for (int64_t j = 0; j < dim2; ++j) {
            double gpsi_j = -two * X(dim1 + j);
            dot_u += gpsi_j * SX(dim1 + j);
        }
        // u_w · u_w
        double uw_sq = 0.0;
        for (int64_t j = 0; j < dim2; ++j) {
            double uj = SX(dim1 + j);
            uw_sq += uj * uj;
        }
        double u_hpsi_u = phi * (sigma_u * sigma_u - rho_uu) - two * uw_sq;
        double coef = (u_hpsi_u * psi - two * dot_u * dot_u) / (psi * psi * psi);
        double inv_psi2 = 1.0 / (psi * psi);

        // Build η[i] per cone-index, then take ∞-norm for the cap.
        // First pass: compute η[i] and stash in step_rhs_z_x; track ‖η‖_∞.
        // We'll cap by rescaling in a second pass.
        double max_eta = 0.0;
        // Use scratch via step_rhs_z_x temporarily — but we need the
        // existing value too (which is `affine_dz_x + σμ·grad_primal`).
        // Strategy: compute η_i and the cap ratio in two passes
        // without overwriting step_rhs_z_x intermediate. We carry η in
        // a small register-array via per-i recomputation in the second
        // pass (since dim can be up to 64+, this is acceptable).
        for (int64_t i = 0; i < dim; ++i) {
            double xi = X(i);
            double ui = SX(i);
            double gpsi_i, hpsi_u_i, t_psi_i_val;
            if (i < dim1) {
                gpsi_i = two * alphas[i] * phi / xi;
                hpsi_u_i = gpsi_i * (sigma_u - ui / xi);
                double bracket = sigma_u * sigma_u - rho_uu
                    - 2.0 * sigma_u * ui / xi
                    + two * ui * ui / (xi * xi);
                t_psi_i_val = gpsi_i * bracket;
            } else {
                gpsi_i = -two * xi;
                hpsi_u_i = -two * ui;
                t_psi_i_val = 0.0;
            }
            double eta_i = -t_psi_i_val / psi
                         + coef * gpsi_i
                         + 2.0 * hpsi_u_i * dot_u * inv_psi2;
            if (i < dim1) {
                double beta = 1.0 - alphas[i];
                eta_i -= two * beta * ui * ui / (xi * xi * xi);
            }
            // Match pass-2 μ⁴ scaling so the cap derived from max_eta caps
            // the actual post-scaling η, not a μ²-scaled proxy.
            double mu_sq_p1 = mu * mu;
            eta_i *= half * mu_sq_p1 * mu_sq_p1;
            double abs_eta = fabs(eta_i);
            if (abs_eta > max_eta) max_eta = abs_eta;
        }

        // ‖step_z‖_∞ for cap.
        double max_sz = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double a = fabs(SZ(i));
            if (a > max_sz) max_sz = a;
        }
        double cap = K_cap * max_sz;
        double scale_cap = 1.0;
        if (max_eta > cap && max_sz > 0.0) {
            scale_cap = cap / max_eta;
        }

        // Second pass: recompute η[i] (same formula) and subtract from
        // step_rhs_z_x with the cap scale.
        for (int64_t i = 0; i < dim; ++i) {
            double xi = X(i);
            double ui = SX(i);
            double gpsi_i, hpsi_u_i, t_psi_i_val;
            if (i < dim1) {
                gpsi_i = two * alphas[i] * phi / xi;
                hpsi_u_i = gpsi_i * (sigma_u - ui / xi);
                double bracket = sigma_u * sigma_u - rho_uu
                    - 2.0 * sigma_u * ui / xi
                    + two * ui * ui / (xi * xi);
                t_psi_i_val = gpsi_i * bracket;
            } else {
                gpsi_i = -two * xi;
                hpsi_u_i = -two * ui;
                t_psi_i_val = 0.0;
            }
            double eta_i = -t_psi_i_val / psi
                         + coef * gpsi_i
                         + 2.0 * hpsi_u_i * dot_u * inv_psi2;
            if (i < dim1) {
                double beta = 1.0 - alphas[i];
                eta_i -= two * beta * ui * ui / (xi * xi * xi);
            }
            // μ⁴ scaling: keeps η in perturbation territory across the
            // IPM trajectory.
            const double mu_quad = mu * mu * mu * mu;
            eta_i *= half * mu_quad * scale_cap;
            // step_rhs_z_x[idx] -= eta_i (i.e., shift = grad·σμ - η)
            step_rhs_z_x[b * totalXConeNumel + num_off + i] -= eta_i;
        }
    } // cone loop
}

void subtract_eta_primal_genpow(
    double* step_rhs_z_x,
    const double* var_x,
    const double* step_x,
    const double* step_z,
    const double* mu_per_batch,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double* d_xcone_genpow_alphas,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    subtract_eta_primal_genpow_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, var_x, step_x, step_z, mu_per_batch,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s, d_xcone_genpow_alpha_offsets,
        d_xcone_genpow_alphas,
        batchSize, n, numXCones, totalXConeNumel);
}

// --------------------------------------------------------------------------
// add_combined_ds_shift_genpow
// Combined shift: shift[i] += sigma_mu * grad_primal[i]
// --------------------------------------------------------------------------
__global__ void add_combined_ds_shift_genpow_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ sigma_mu,
    const double* __restrict__ xcone_grad_primal,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    int64_t totalXConeNumel, int64_t numXCones)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t dim     = d_xcone_dims[c];
    const int64_t zx_off  = b * totalXConeNumel;
    const double sm = sigma_mu[b];

    for (int64_t i = 0; i < dim; ++i) {
        step_rhs_z_x[zx_off + num_off + i] +=
            sm * xcone_grad_primal[b * totalXConeNumel + num_off + i];
    }
    } // end cone loop
}

void add_combined_ds_shift_genpow(
    double* step_rhs_z_x,
    const double* sigma_mu,
    const double* xcone_grad_primal,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    add_combined_ds_shift_genpow_kernel<<<grid, 256, 0, stream>>>(
        step_rhs_z_x, sigma_mu, xcone_grad_primal,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        totalXConeNumel, numXCones);
}

// --------------------------------------------------------------------------
// xcone_step_length_genpow_reduce
// Backtracking line search for GenPow cones.
// Primal feasibility: ∏ (xᵢ + α·dxᵢ)^αᵢ ≥ ‖w + α·dw‖₂
// Dual feasibility: same cone with rescaled point.
// --------------------------------------------------------------------------
__global__ void xcone_step_length_genpow_reduce_kernel(
    double* __restrict__ alpha_s,
    double* __restrict__ alpha_z,
    const double* __restrict__ var_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ step_x,
    const double* __restrict__ step_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const double* __restrict__ d_xcone_genpow_alphas,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack)
{
    const int64_t b = blockIdx.x;

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t gidx    = d_xcone_genpow_idx[c];
    const int64_t dim1    = d_xcone_genpow_dim1s[gidx];
    const int64_t dim     = d_xcone_dims[c];
    const int64_t dim2    = dim - dim1;
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    // CPU parity: start the backtrack at the minimum of `max_step` and the
    // already-applied τ/κ + slack-cone bound stored in alpha_{z,s}[b].
    // CPU `direct_x_step_length` is called with `αmax = min(ατ, ακ, 1,
    // slack_bounds)`, and `backtrack_search` starts there. Starting only
    // from `max_step=1.0` causes a divergence: when α=1.0 is infeasible but
    // the τ/κ-bounded αmax (e.g. 0.893) is feasible, CUDA backtracks once
    // to 0.8 while CPU returns 0.893 — propagates through σ=(1−α)³ to a
    // 6× σμ mismatch in the combined-step shift, which knocks CUDA off
    // the central path within a few iterations.
    // CPU `backtrack_search` returns 0 when no feasible α ≥ α_min is found
    // (nonsymmetric_common.rs:1174-1176). Match that semantic so the IPM
    // sees the same infeasibility signal on CUDA.
    // Primal backtrack: check genpow_is_primal_feasible streaming (no stack).
    double alpha_primal = fmin(max_step, alpha_z[b]);
    bool primal_found = false;
    while (alpha_primal > min_step) {
        double log_phi = 0.0;
        bool ok = true;
        for (int64_t i = 0; i < dim1; ++i) {
            double xi = var_x[x_off + d_xcone_indices[num_off + i]]
                      + alpha_primal * step_x[x_off + d_xcone_indices[num_off + i]];
            if (xi <= 0.0) { ok = false; break; }
            log_phi += 2.0 * d_xcone_genpow_alphas[a_off + i] * log(xi);
        }
        if (ok) {
            double phi = exp(log_phi);
            double norm2w = 0.0;
            for (int64_t j = 0; j < dim2; ++j) {
                double wj = var_x[x_off + d_xcone_indices[num_off + dim1 + j]]
                          + alpha_primal * step_x[x_off + d_xcone_indices[num_off + dim1 + j]];
                norm2w += wj * wj;
            }
            if (phi > norm2w) { primal_found = true; break; }
        }
        alpha_primal *= backtrack;
    }
    if (!primal_found) alpha_primal = 0.0;

    double alpha_dual = fmin(max_step, alpha_s[b]);
    bool dual_found = false;
    while (alpha_dual > min_step) {
        double log_phi = 0.0;
        bool ok = true;
        for (int64_t i = 0; i < dim1; ++i) {
            double ai = d_xcone_genpow_alphas[a_off + i];
            double zi = var_z_x[zx_off + num_off + i]
                      + alpha_dual * step_z_x[zx_off + num_off + i];
            if (zi <= 0.0) { ok = false; break; }
            log_phi += 2.0 * ai * log(zi / ai);
        }
        if (ok) {
            double phi = exp(log_phi);
            double norm2w = 0.0;
            for (int64_t j = 0; j < dim2; ++j) {
                double zj = var_z_x[zx_off + num_off + dim1 + j]
                          + alpha_dual * step_z_x[zx_off + num_off + dim1 + j];
                norm2w += zj * zj;
            }
            if (phi > norm2w) { dual_found = true; break; }
        }
        alpha_dual *= backtrack;
    }
    if (!dual_found) alpha_dual = 0.0;

    cones::atomic_min_pos_double(&alpha_z[b], alpha_primal);
    cones::atomic_min_pos_double(&alpha_s[b], alpha_dual);
    } // end cone loop
}

void xcone_step_length_genpow_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    xcone_step_length_genpow_reduce_kernel<<<grid, 256, 0, stream>>>(
        alpha_s, alpha_z, var_x, var_z_x, step_x, step_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_alphas,
        n, numXCones, totalXConeNumel, max_step, min_step, backtrack);
}

// --------------------------------------------------------------------------
// subtract_xcone_combined_from_workx_genpow
// workx[J] -= rhs_z_x (identity direct_x_combined_offset for GenPow).
// --------------------------------------------------------------------------
__global__ void subtract_xcone_combined_from_workx_genpow_kernel(
    double* __restrict__ workx,
    const double* __restrict__ rhs_z_x,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n, int64_t numXCones,
    int64_t totalXConeNumel)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t dim     = d_xcone_dims[c];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;

    for (int64_t i = 0; i < dim; ++i) {
        workx[x_off + d_xcone_indices[num_off + i]] -=
            rhs_z_x[zx_off + num_off + i];
    }
    } // end cone loop
}

void subtract_xcone_combined_from_workx_genpow(
    double* workx,
    const double* rhs_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    subtract_xcone_combined_from_workx_genpow_kernel<<<grid, 256, 0, stream>>>(
        workx, rhs_z_x,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        n, numXCones, totalXConeNumel);
}

// --------------------------------------------------------------------------
// recover_dz_x_affine_genpow / recover_dz_x_combined_genpow
//
// mul_Hs(x): y = μ*(D·x + p*(p'*x) - q*(q'*x[:dim1]) - r*(r'*x[dim1:]))
// For dense (dim<=4): use packed upper-tri Hs directly.
// For sparse (dim>4): use stored p/q/r vectors via streaming dot-products.
//
// Affine:   dz_x = -Hs * dx
// Combined: dz_x = -Hs * dx - rhs_z_x
// --------------------------------------------------------------------------
__device__ void genpow_mul_Hs_sparse(
    double* out_local,          // dim entries of output (caller-allocated)
    const double* dx_local,     // dim entries of input
    const double* Hs_diag,      // dim diagonal entries
    const double* p_vec,        // dim entries of p (sqrt-scaled)
    const double* q_vec,        // dim1 entries of q (sqrt-scaled)
    const double* r_vec,        // dim2 entries of r (sqrt-scaled)
    int64_t dim1, int64_t dim2)
{
    int64_t dim = dim1 + dim2;
    // dot products
    double ptx = 0.0, qtx = 0.0, rtx = 0.0;
    for (int64_t i = 0; i < dim; ++i)  ptx += p_vec[i] * dx_local[i];
    for (int64_t i = 0; i < dim1; ++i) qtx += q_vec[i] * dx_local[i];
    for (int64_t j = 0; j < dim2; ++j) rtx += r_vec[j] * dx_local[dim1 + j];

    for (int64_t i = 0; i < dim; ++i) {
        out_local[i] = Hs_diag[i] * dx_local[i]
                     + p_vec[i] * ptx
                     - (i < dim1 ? q_vec[i] * qtx : 0.0)
                     - (i >= dim1 ? r_vec[i - dim1] * rtx : 0.0);
    }
}

// Rank-6 PD axes contribution to mul_Hs (mirror of CPU mul_Hs `pd_active`
// branch in genpowcone.rs). Caller must already have computed rank-3 into y.
__device__ inline void genpow_add_pd_rank6(
    double* y,
    const double* dx_local,
    const double* pd_axes_cone,   // 6 * dim (axes for this cone)
    const double* pd_coefs_cone,  // 6 entries
    const double* pd_signs_cone,  // 6 entries (±1 as double)
    double active_flag,
    int64_t dim)
{
    if (active_flag <= 0.5) return;
    for (int axk = 0; axk < 6; ++axk) {
        double coef = pd_coefs_cone[axk];
        if (coef == 0.0) continue;
        double sign = pd_signs_cone[axk];
        const double* axis = pd_axes_cone + axk * dim;
        double dot = 0.0;
        for (int64_t i = 0; i < dim; ++i) dot += axis[i] * dx_local[i];
        double scale = sign * coef * dot;
        for (int64_t i = 0; i < dim; ++i) y[i] += scale * axis[i];
    }
}

__global__ void recover_dz_x_affine_genpow_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ xcone_Hs,
    const double* __restrict__ xcone_genpow_p,
    const double* __restrict__ xcone_genpow_q,
    const double* __restrict__ xcone_genpow_r,
    const double* __restrict__ xgenpow_pd_axes,
    const double* __restrict__ xgenpow_pd_coefs,
    const double* __restrict__ xgenpow_pd_signs,
    const double* __restrict__ xgenpow_pd_active,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_dim2s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_idx,
    const int64_t* __restrict__ d_xcone_genpow_sparse_offsets,
    double* __restrict__ genpow_recover_workspace,
    int64_t n, int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t gidx    = d_xcone_genpow_idx[c];
    const int64_t dim1    = d_xcone_genpow_dim1s[gidx];
    const int64_t dim2    = d_xcone_genpow_dim2s[gidx];
    const int64_t dim     = dim1 + dim2;
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t hs_off  = d_xcone_hs_offsets[c];
    const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];
    const int64_t d_off   = d_xcone_genpow_dim_offsets[gidx];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t hs_b    = b * totalXConeHsEntries;
    const int64_t p_b     = b * totalXGenPowerDim;
    const int64_t q_b     = b * totalXGenPowerAlphas;
    const int64_t r_b     = b * totalXGenPowerDim2;
    const int64_t r_off   = d_off - a_off;

    if (dim > 4) {
        // Sparse: use p/q/r vectors + diagonal for rank-3 mul_Hs.
        // Scratch (dx_local, y) lives in a per-(batch,cone) slice of the
        // shared genpow workspace — 8·totalXGenPowerDim doubles per batch,
        // 8·d_off base per cone, so the two dim-sized regions stay disjoint
        // across the threads striding over cones. Replaces fixed
        // `double[128]` stack arrays, lifting the dim cap.
        double* ws_cone = genpow_recover_workspace
                          + b * 8 * totalXGenPowerDim + 8 * d_off;
        double* dx_local = ws_cone;          // region 0: [0, dim)
        double* y        = ws_cone + dim;    // region 1: [dim, 2·dim)
        for (int64_t i = 0; i < dim; ++i)
            dx_local[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];

        const double* Hs_diag = xcone_Hs + hs_b + hs_off;
        const double* p_vec   = xcone_genpow_p + p_b + d_off;
        const double* q_vec   = xcone_genpow_q + q_b + a_off;
        const double* r_vec   = xcone_genpow_r + r_b + r_off;

        genpow_mul_Hs_sparse(y, dx_local, Hs_diag, p_vec, q_vec, r_vec, dim1, dim2);

        // Add rank-6 PD axes contribution (mirror of CPU mul_Hs in genpowcone.rs).
        if (xgenpow_pd_axes != nullptr) {
            int64_t pd_axes_cone_base = b * 6 * totalXGenPowerDim + 6 * d_off;
            int64_t pd_state_cone_base = b * 6 * numXGenPowerCones + 6 * gidx;
            double active_flag = xgenpow_pd_active[b * numXGenPowerCones + gidx];
            genpow_add_pd_rank6(
                y, dx_local,
                xgenpow_pd_axes + pd_axes_cone_base,
                xgenpow_pd_coefs + pd_state_cone_base,
                xgenpow_pd_signs + pd_state_cone_base,
                active_flag, dim);
        }

        for (int64_t i = 0; i < dim; ++i)
            dz_x[zx_off + num_off + i] = -y[i] - var_z_x[zx_off + num_off + i];
    } else {
        // Dense path: use packed upper-tri Hs
        double dx_local[4];
        for (int64_t i = 0; i < dim; ++i)
            dx_local[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];

        const double* H = xcone_Hs + hs_b + hs_off;
        for (int64_t r = 0; r < dim; ++r) {
            double dot = 0.0;
            for (int64_t cc = 0; cc < dim; ++cc) {
                int64_t rr  = r < cc ? r : cc;
                int64_t ci  = cc > r ? cc : r;
                int64_t idx = ci * (ci + 1) / 2 + rr;
                dot += H[idx] * dx_local[cc];
            }
            dz_x[zx_off + num_off + r] = -dot - var_z_x[zx_off + num_off + r];
        }
    }
    } // end cone loop
}

void recover_dz_x_affine_genpow(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const double* xcone_genpow_p,
    const double* xcone_genpow_q,
    const double* xcone_genpow_r,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    double* genpow_recover_workspace,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_affine_genpow_kernel<<<grid, 256, 0, stream>>>(
        dz_x, lhs_x, var_z_x, xcone_Hs,
        xcone_genpow_p, xcone_genpow_q, xcone_genpow_r,
        xgenpow_pd_axes, xgenpow_pd_coefs, xgenpow_pd_signs, xgenpow_pd_active,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_dim_offsets,
        d_xcone_genpow_sparse_idx, d_xcone_genpow_sparse_offsets,
        genpow_recover_workspace,
        n, numXCones, numXGenPowerCones, totalXConeNumel, totalXConeHsEntries,
        totalXGenPowerDim, totalXGenPowerAlphas, totalXGenPowerDim2, numSparseXGenPow);
}

__global__ void recover_dz_x_combined_genpow_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ rhs_z_x,
    const double* __restrict__ xcone_Hs,
    const double* __restrict__ xcone_genpow_p,
    const double* __restrict__ xcone_genpow_q,
    const double* __restrict__ xcone_genpow_r,
    const double* __restrict__ xgenpow_pd_axes,
    const double* __restrict__ xgenpow_pd_coefs,
    const double* __restrict__ xgenpow_pd_signs,
    const double* __restrict__ xgenpow_pd_active,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_dim2s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_idx,
    const int64_t* __restrict__ d_xcone_genpow_sparse_offsets,
    double* __restrict__ genpow_recover_workspace,
    int64_t n, int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow)
{
    const int64_t b = blockIdx.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

    const int64_t gidx    = d_xcone_genpow_idx[c];
    const int64_t dim1    = d_xcone_genpow_dim1s[gidx];
    const int64_t dim2    = d_xcone_genpow_dim2s[gidx];
    const int64_t dim     = dim1 + dim2;
    const int64_t num_off = d_xcone_numel_offsets[c];
    const int64_t hs_off  = d_xcone_hs_offsets[c];
    const int64_t a_off   = d_xcone_genpow_alpha_offsets[gidx];
    const int64_t d_off   = d_xcone_genpow_dim_offsets[gidx];
    const int64_t x_off   = b * n;
    const int64_t zx_off  = b * totalXConeNumel;
    const int64_t hs_b    = b * totalXConeHsEntries;
    const int64_t p_b     = b * totalXGenPowerDim;
    const int64_t q_b     = b * totalXGenPowerAlphas;
    const int64_t r_b     = b * totalXGenPowerDim2;
    const int64_t r_off   = d_off - a_off;

    if (dim > 4) {
        // Sparse: use p/q/r vectors + diagonal for rank-3 mul_Hs.
        // Scratch (dx_local, y) lives in a per-(batch,cone) slice of the
        // shared genpow workspace — 8·totalXGenPowerDim doubles per batch,
        // 8·d_off base per cone, so the two dim-sized regions stay disjoint
        // across the threads striding over cones. Replaces fixed
        // `double[128]` stack arrays, lifting the dim cap.
        double* ws_cone = genpow_recover_workspace
                          + b * 8 * totalXGenPowerDim + 8 * d_off;
        double* dx_local = ws_cone;          // region 0: [0, dim)
        double* y        = ws_cone + dim;    // region 1: [dim, 2·dim)
        for (int64_t i = 0; i < dim; ++i)
            dx_local[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];

        const double* Hs_diag = xcone_Hs + hs_b + hs_off;
        const double* p_vec   = xcone_genpow_p + p_b + d_off;
        const double* q_vec   = xcone_genpow_q + q_b + a_off;
        const double* r_vec   = xcone_genpow_r + r_b + r_off;

        genpow_mul_Hs_sparse(y, dx_local, Hs_diag, p_vec, q_vec, r_vec, dim1, dim2);

        // Add rank-6 PD axes contribution (mirror of CPU mul_Hs in genpowcone.rs).
        if (xgenpow_pd_axes != nullptr) {
            int64_t pd_axes_cone_base = b * 6 * totalXGenPowerDim + 6 * d_off;
            int64_t pd_state_cone_base = b * 6 * numXGenPowerCones + 6 * gidx;
            double active_flag = xgenpow_pd_active[b * numXGenPowerCones + gidx];
            genpow_add_pd_rank6(
                y, dx_local,
                xgenpow_pd_axes + pd_axes_cone_base,
                xgenpow_pd_coefs + pd_state_cone_base,
                xgenpow_pd_signs + pd_state_cone_base,
                active_flag, dim);
        }

        for (int64_t i = 0; i < dim; ++i)
            dz_x[zx_off + num_off + i] = -y[i] - rhs_z_x[zx_off + num_off + i];
    } else {
        // Dense path: use packed upper-tri Hs
        double dx_local[4];
        for (int64_t i = 0; i < dim; ++i)
            dx_local[i] = lhs_x[x_off + d_xcone_indices[num_off + i]];

        const double* H = xcone_Hs + hs_b + hs_off;
        for (int64_t r = 0; r < dim; ++r) {
            double dot = 0.0;
            for (int64_t cc = 0; cc < dim; ++cc) {
                int64_t rr2 = r < cc ? r : cc;
                int64_t ci2 = cc > r ? cc : r;
                int64_t idx = ci2 * (ci2 + 1) / 2 + rr2;
                dot += H[idx] * dx_local[cc];
            }
            dz_x[zx_off + num_off + r] = -dot - rhs_z_x[zx_off + num_off + r];
        }
    }
    } // end cone loop
}

void recover_dz_x_combined_genpow(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* xcone_Hs,
    const double* xcone_genpow_p,
    const double* xcone_genpow_q,
    const double* xcone_genpow_r,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    double* genpow_recover_workspace,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    cudaStream_t stream)
{
    if (numXCones <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize));
    recover_dz_x_combined_genpow_kernel<<<grid, 256, 0, stream>>>(
        dz_x, lhs_x, rhs_z_x, xcone_Hs,
        xcone_genpow_p, xcone_genpow_q, xcone_genpow_r,
        xgenpow_pd_axes, xgenpow_pd_coefs, xgenpow_pd_signs, xgenpow_pd_active,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets,
        d_xcone_hs_offsets, d_xcone_indices,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_dim_offsets,
        d_xcone_genpow_sparse_idx, d_xcone_genpow_sparse_offsets,
        genpow_recover_workspace,
        n, numXCones, numXGenPowerCones, totalXConeNumel, totalXConeHsEntries,
        totalXGenPowerDim, totalXGenPowerAlphas, totalXGenPowerDim2, numSparseXGenPow);
}

// --------------------------------------------------------------------------
// refresh_xcone_genpow_expansion
//
// Scatters q/r/p column values and exp-diag entries into KKT.values for
// sparse x-GenPow cones. Called from refresh_xcone_hs (non-fused path).
//
// Storage layout:
//   xcone_genpow_p: [batchSize × totalXGenPowerDim] — offset by d_off = sparse_offsets[sc]
//   xcone_genpow_q: [batchSize × totalXGenPowerAlphas] — offset by a_off = sparse_alpha_offsets[sc]
//   xcone_genpow_r: [batchSize × totalXGenPowerDim2] — offset by r_off = d_off - a_off
// --------------------------------------------------------------------------
__global__ void refresh_xcone_genpow_expansion_kernel(
    double* __restrict__ kkt_values,
    const double* __restrict__ xcone_genpow_p,
    const double* __restrict__ xcone_genpow_q,
    const double* __restrict__ xcone_genpow_r,
    const int64_t* __restrict__ d_xcone_genpow_sparse_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_alpha_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_q_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_r_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_to_gidx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_dim2s,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,
    const int64_t* __restrict__ H_xcone_genpow_q_idx,
    const int64_t* __restrict__ H_xcone_genpow_r_idx,
    const int64_t* __restrict__ H_xcone_genpow_p_idx,
    const int64_t* __restrict__ H_xcone_genpow_exp_diag_idx,
    int64_t nnzKKT,
    int64_t numSparseXGenPow,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2)
{
    // One block per (batch, sparse_cone) with blockDim threads
    // cooperating: three parallel norm reductions (q_w, r_w, p_w) and
    // per-element parallel scatters for the q/r/p columns.
    const int64_t b  = blockIdx.x;
    const int64_t sc = blockIdx.y;  // sparse cone index — bounded by grid.y

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    extern __shared__ double smem[];

    const int64_t kkt_off = b * nnzKKT;
    const int64_t gidx = d_xcone_genpow_sparse_to_gidx[sc];
    const int64_t dim1 = d_xcone_genpow_dim1s[gidx];
    const int64_t dim2 = d_xcone_genpow_dim2s[gidx];
    const int64_t dim  = dim1 + dim2;

    // KKT index offsets (sparse-only prefix, for indexing into H_xcone_genpow_*_idx)
    const int64_t kkt_p_off = d_xcone_genpow_sparse_offsets[sc];   // sparse-only dim prefix
    const int64_t kkt_q_off = d_xcone_genpow_sparse_q_offsets[sc]; // sparse-only dim1 prefix
    const int64_t kkt_r_off = d_xcone_genpow_sparse_r_offsets[sc]; // sparse-only dim2 prefix

    // Data offsets (global prefix, for indexing into xcone_genpow_*[b][...])
    const int64_t global_a_off = d_xcone_genpow_sparse_alpha_offsets[sc]; // global dim1 prefix
    const int64_t global_d_off = d_xcone_genpow_dim_offsets[gidx];        // global dim prefix
    const int64_t global_r_off = global_d_off - global_a_off;             // global dim2 prefix

    const double* q_base = xcone_genpow_q + b * totalXGenPowerAlphas + global_a_off;
    const double* r_base = xcone_genpow_r + b * totalXGenPowerDim2   + global_r_off;
    const double* p_base = xcone_genpow_p + b * totalXGenPowerDim    + global_d_off;

    // Adaptive per-axis equilibration — mirrors CPU directldlkktsolver.rs:339
    // and the slack-side update_genpow_scaling_kernel logic. Stored values
    // xcone_genpow_{q,r,p}[i] = sqrt(μ)·{q,r,p}_raw[i], so ||stored||² = w
    // (the per-axis weight μ·||v||²) directly. Above threshold=1e12, swap the
    // off-diag column from `±sqrt(μ)·v` to `±v/||v||` and the sentinel from
    // ±1 to ±1/w. Schur contribution -c²/d is preserved exactly; only the
    // entry magnitudes change. Catches the catastrophic 1e22 regime that
    // dwarfs ±1 sentinels and confuses the LDL pivot strategy.
    const double EQ_THRESHOLD = 1.0e12;
    double my_q_w = 0.0, my_r_w = 0.0, my_p_w = 0.0;
    for (int64_t i = tid; i < dim1; i += blockDimX) {
        const double v = q_base[i];
        my_q_w += v * v;
    }
    for (int64_t j = tid; j < dim2; j += blockDimX) {
        const double v = r_base[j];
        my_r_w += v * v;
    }
    for (int64_t i = tid; i < dim; i += blockDimX) {
        const double v = p_base[i];
        my_p_w += v * v;
    }
    // Fused 3-value reduction over q_w, r_w, p_w norms.
    double valsW[3] = { my_q_w, my_r_w, my_p_w };
    cones::block_sum_reduce_N<3>(valsW, smem, tid);
    const double q_w = valsW[0];
    const double r_w = valsW[1];
    const double p_w = valsW[2];

    const bool q_eq = q_w > EQ_THRESHOLD;
    const bool r_eq = r_w > EQ_THRESHOLD;
    const bool p_eq = p_w > EQ_THRESHOLD;
    const double q_inv_norm = q_eq ? 1.0 / sqrt(q_w) : 1.0;
    const double r_inv_norm = r_eq ? 1.0 / sqrt(r_w) : 1.0;
    const double p_inv_norm = p_eq ? 1.0 / sqrt(p_w) : 1.0;

    // Parallel scatter: q column
    for (int64_t i = tid; i < dim1; i += blockDimX) {
        const double v = q_base[i];
        kkt_values[kkt_off + H_xcone_genpow_q_idx[kkt_q_off + i]] =
            q_eq ? v * q_inv_norm : v;
    }
    // Parallel scatter: r column
    for (int64_t j = tid; j < dim2; j += blockDimX) {
        const double v = r_base[j];
        kkt_values[kkt_off + H_xcone_genpow_r_idx[kkt_r_off + j]] =
            r_eq ? v * r_inv_norm : v;
    }
    // Parallel scatter: p column
    for (int64_t i = tid; i < dim; i += blockDimX) {
        const double v = p_base[i];
        kkt_values[kkt_off + H_xcone_genpow_p_idx[kkt_p_off + i]] =
            p_eq ? v * p_inv_norm : v;
    }
    // Expansion diagonals: direct-x sentinels are slack's negated. Slack uses
    // q-sent=-1, r-sent=-1, p-sent=+1 (low-w); direct-x flip is +1, +1, -1.
    // High-w equilibrated sentinels are ±1/w (sign preserved).
    // Layout: H_xcone_genpow_exp_diag_idx has 9 entries per sparse cone
    // (q,r,p, then 6 PD axes). PD axes are written by apply_xgenpow_pd_to_kkt.
    if (tid == 0) {
        kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sc + 0]] =
            q_eq ?  1.0 / q_w :  1.0; // q-diag
        kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sc + 1]] =
            r_eq ?  1.0 / r_w :  1.0; // r-diag
        kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sc + 2]] =
            p_eq ? -1.0 / p_w : -1.0; // p-diag
    }
}

void refresh_xcone_genpow_expansion(
    double* kkt_values,
    const double* xcone_genpow_p,
    const double* xcone_genpow_q,
    const double* xcone_genpow_r,
    const int64_t* d_xcone_genpow_sparse_offsets,
    const int64_t* d_xcone_genpow_sparse_alpha_offsets,
    const int64_t* d_xcone_genpow_sparse_q_offsets,
    const int64_t* d_xcone_genpow_sparse_r_offsets,
    const int64_t* d_xcone_genpow_sparse_to_gidx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* H_xcone_genpow_q_idx,
    const int64_t* H_xcone_genpow_r_idx,
    const int64_t* H_xcone_genpow_p_idx,
    const int64_t* H_xcone_genpow_exp_diag_idx,
    int64_t batchSize, int64_t nnzKKT,
    int64_t numSparseXGenPow,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    cudaStream_t stream)
{
    if (numSparseXGenPow <= 0 || batchSize <= 0) return;
    dim3 grid(static_cast<unsigned int>(batchSize),
              static_cast<unsigned int>(numSparseXGenPow));
    const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
    const size_t smem_bytes = sizeof(double) * block_size;
    refresh_xcone_genpow_expansion_kernel<<<grid, block_size, smem_bytes, stream>>>(
        kkt_values,
        xcone_genpow_p, xcone_genpow_q, xcone_genpow_r,
        d_xcone_genpow_sparse_offsets, d_xcone_genpow_sparse_alpha_offsets,
        d_xcone_genpow_sparse_q_offsets, d_xcone_genpow_sparse_r_offsets,
        d_xcone_genpow_sparse_to_gidx,
        d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_dim_offsets,
        H_xcone_genpow_q_idx, H_xcone_genpow_r_idx,
        H_xcone_genpow_p_idx, H_xcone_genpow_exp_diag_idx,
        nnzKKT, numSparseXGenPow,
        totalXGenPowerDim, totalXGenPowerAlphas, totalXGenPowerDim2);
}

} // namespace moreau
