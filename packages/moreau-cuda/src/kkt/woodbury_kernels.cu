/**
 * @file woodbury_kernels.cu
 * @brief CUDA kernels for the Woodbury KKT solver backend
 *
 * Provides GPU kernels for:
 * - dx recovery after Schur solve
 * - dlam/dy recovery
 * - pack/unpack for interleaved [dx; dz] layout
 * - sparse nonneg elimination (d_tilde_inv, rhs_adj, dlam)
 * - KKT matvec for iterative refinement
 * - backward pass (DiffWoodbury direct Schur complement)
 */

#include "moreau/kkt/woodbury_kernels.cuh"
#include "moreau/profiling/profiler.hpp"

namespace moreau {

static DiffWbLdltVariant g_diff_wb_ldlt_variant = DiffWbLdltVariant::Auto;

void set_diff_wb_ldlt_variant(DiffWbLdltVariant variant) {
    g_diff_wb_ldlt_variant = variant;
}

DiffWbLdltVariant get_diff_wb_ldlt_variant() {
    return g_diff_wb_ldlt_variant;
}

// ---------------------------------------------------------------------------
// dx recovery: dx = d_tilde_inv * (rhs_x_adj - Fdnu)
// ---------------------------------------------------------------------------
__global__ void woodbury_dx_recovery_kernel(
    double* __restrict__ dx,
    const double* __restrict__ d_tilde_inv,
    const double* __restrict__ rhs_x_adj,
    const double* __restrict__ Fdnu,
    int64_t count)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dx[idx] = d_tilde_inv[idx] * (rhs_x_adj[idx] - Fdnu[idx]);
    }
}

void launch_woodbury_dx_recovery(
    double* dx, const double* d_tilde_inv,
    const double* rhs_x_adj, const double* Fdnu,
    int64_t count, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_dx_recovery_kernel, blocks, threads, 0, stream,
        dx, d_tilde_inv, rhs_x_adj, Fdnu, count);
}

// ---------------------------------------------------------------------------
// Elementwise multiply: out = a * b
// ---------------------------------------------------------------------------
__global__ void woodbury_elementwise_mul_kernel(
    double* __restrict__ out, const double* __restrict__ a,
    const double* __restrict__ b, int64_t count)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = a[idx] * b[idx];
    }
}

void launch_woodbury_elementwise_mul(
    double* out, const double* a, const double* b,
    int64_t count, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_elementwise_mul_kernel, blocks, threads, 0, stream, out, a, b, count);
}

// ---------------------------------------------------------------------------
// Compute H_nonneg diagonal from cone w values: H_nonneg[i] = -w[i]^2
// For the KKT, H block for nonneg cones is -(w^2), so H_nonneg stores the
// positive value w^2 (we negate when building d_tilde).
// Actually in moreau, the KKT diagonal for nonneg is -(w^2 + eps).
// For Woodbury, we need the positive H_nonneg = w^2 (before negation).
// ---------------------------------------------------------------------------
__global__ void woodbury_compute_H_nonneg_kernel(
    double* __restrict__ H_nonneg,        // (B, numNonneg)
    const double* __restrict__ nonneg_w,  // (B, numNonneg)
    int64_t count)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double w = nonneg_w[idx];
        H_nonneg[idx] = w * w;
    }
}

void launch_woodbury_compute_H_nonneg(
    double* H_nonneg, const double* nonneg_w,
    int64_t count, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_compute_H_nonneg_kernel, blocks, threads, 0, stream,
        H_nonneg, nonneg_w, count);
}

// ---------------------------------------------------------------------------
// Gather direct-x nonneg (1,1)-block diagonal contribution per iteration.
// `cones.xcone_Hs` stores Hs_i = z_x/x[i] (NT scaling, post primal↔dual swap;
// see `update_xcones_nonneg_scaling_kernel` in `kernels.cu`). For Woodbury we
// pull those values into a contiguous (B, n_xcone_nonneg) buffer indexed
// by the cone-flat slot — `d_tilde_inv` then adds them to D[idx].
// ---------------------------------------------------------------------------
__global__ void woodbury_gather_xcone_nonneg_hs_kernel(
    double* __restrict__ xn_hs,            // (B, n_xn)
    const double* __restrict__ xcone_Hs,   // (B, totalXConeHsEntries)
    const int64_t* __restrict__ xn_hs_off, // (n_xn) offset into per-batch Hs
    int64_t totalXConeHsEntries,
    int64_t n_xn,
    int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n_xn;
        int64_t k = tid % n_xn;
        xn_hs[tid] = xcone_Hs[b * totalXConeHsEntries + xn_hs_off[k]];
    }
}

void launch_woodbury_gather_xcone_nonneg_hs(
    double* xn_hs,
    const double* xcone_Hs,
    const int64_t* xn_hs_off,
    int64_t totalXConeHsEntries,
    int64_t batch, int64_t n_xn,
    cudaStream_t stream)
{
    if (n_xn == 0) return;
    int threads = 256;
    int64_t total = batch * n_xn;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_gather_xcone_nonneg_hs_kernel, blocks, threads, 0, stream,
        xn_hs, xcone_Hs, xn_hs_off, totalXConeHsEntries, n_xn, total);
}

// ---------------------------------------------------------------------------
// Compute C_diag: inverse of zero-cone H diagonal + regularization
// For zero cones, H = 0, so C_diag = 1/eps (the regularization diagonal)
// C_diag[b,j] = 1/eps[b] for each zero-cone row j
// ---------------------------------------------------------------------------
__global__ void woodbury_compute_C_diag_kernel(
    double* __restrict__ C_diag,    // (B, k)
    const double* __restrict__ eps,  // (B)
    int64_t k, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / k;
        C_diag[tid] = 1.0 / eps[b];
    }
}

void launch_woodbury_compute_C_diag(
    double* C_diag, const double* eps,
    int64_t batch, int64_t k, cudaStream_t stream)
{
    int64_t total = batch * k;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_compute_C_diag_kernel, blocks, threads, 0, stream,
        C_diag, eps, k, total);
}

// ---------------------------------------------------------------------------
// Compute regularizer: eps[b] = eps_c + eps_p * max_diag[b]
// This mirrors compute_regularizer_per_batch but operates on Woodbury diagonals
// For Woodbury, max_diag is computed from P_diag + H_nonneg (the effective diagonal)
// ---------------------------------------------------------------------------
__global__ void woodbury_compute_eps_kernel(
    double* __restrict__ eps,
    double eps_c, double eps_p,
    int64_t batch)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        // eps already contains max_diag from a prior reduction
        eps[b] = eps_c + eps_p * eps[b];
    }
}

void launch_woodbury_compute_eps(
    double* eps, double eps_c, double eps_p,
    int64_t batch, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_compute_eps_kernel, blocks, threads, 0, stream,
        eps, eps_c, eps_p, batch);
}

// ---------------------------------------------------------------------------
// dy recovery: dy[i] = d_nu[i] - C_diag[i % (B*k)] * rhs_y[i]
// C_diag is (B, k); d_nu, rhs_y are (eB, k) where eB may be nrhs*B.
// We broadcast C_diag by taking i % (B*k).
// ---------------------------------------------------------------------------
__global__ void woodbury_dy_recovery_kernel(
    double* __restrict__ dy,
    const double* __restrict__ d_nu,
    const double* __restrict__ C_diag,
    const double* __restrict__ rhs_y,
    int64_t count)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dy[idx] = d_nu[idx] - C_diag[idx] * rhs_y[idx];
    }
}

void launch_woodbury_dy_recovery(
    double* dy, const double* d_nu, const double* C_diag,
    const double* rhs_y, int64_t batch, int64_t k, cudaStream_t stream)
{
    int64_t total = batch * k;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_dy_recovery_kernel, blocks, threads, 0, stream,
        dy, d_nu, C_diag, rhs_y, total);
}

// ---------------------------------------------------------------------------
// Pack solution: interleave dx(n) and dz(m) into sol(n+m) per batch
// dz is composed of: dy(k) for zero cones, dlam(n) for nonneg cones
// Layout: sol = [dx_0, dy_0, dlam_0, dx_1, dy_1, dlam_1, ...]
// Where dy covers zero-cone duals and dlam covers nonneg-cone duals
// ---------------------------------------------------------------------------
__global__ void woodbury_pack_sol_kernel(
    double* __restrict__ sol,          // (B, n+m) output
    const double* __restrict__ dx,     // (B, n)
    const double* __restrict__ dy,     // (B, k)
    const double* __restrict__ dlam,   // (B, n_nonneg)
    int64_t n, int64_t k, int64_t n_nonneg,
    int64_t N,  // n + m = n + k + n_nonneg
    int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / N;
        int64_t i = tid % N;
        if (i < n) {
            sol[tid] = dx[b * n + i];
        } else if (i < n + k) {
            sol[tid] = dy[b * k + (i - n)];
        } else {
            sol[tid] = dlam[b * n_nonneg + (i - n - k)];
        }
    }
}

void launch_woodbury_pack_sol(
    double* sol, const double* dx, const double* dy, const double* dlam,
    int64_t batch, int64_t n, int64_t k, int64_t n_nonneg, cudaStream_t stream)
{
    int64_t N = n + k + n_nonneg;
    int64_t total = batch * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_pack_sol_kernel, blocks, threads, 0, stream,
        sol, dx, dy, dlam, n, k, n_nonneg, N, total);
}

// Pack solution with k_total stride for dy_nu (Phase 2)
__global__ void woodbury_pack_sol_kt_kernel(
    double* __restrict__ sol,
    const double* __restrict__ dx,
    const double* __restrict__ dy_nu,    // (B, k_total) — read first k entries
    const double* __restrict__ dlam,
    int64_t n, int64_t k, int64_t k_total, int64_t n_nonneg,
    int64_t N, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / N;
        int64_t i = tid % N;
        if (i < n) {
            sol[tid] = dx[b * n + i];
        } else if (i < n + k) {
            sol[tid] = dy_nu[b * k_total + (i - n)];
        } else {
            sol[tid] = dlam[b * n_nonneg + (i - n - k)];
        }
    }
}

void launch_woodbury_pack_sol_kt(
    double* sol, const double* dx, const double* dy_nu, const double* dlam,
    int64_t batch, int64_t n, int64_t k, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream)
{
    int64_t N = n + k + n_nonneg;
    int64_t total = batch * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_pack_sol_kt_kernel, blocks, threads, 0, stream,
        sol, dx, dy_nu, dlam, n, k, k_total, n_nonneg, N, total);
}

// ---------------------------------------------------------------------------
// Unpack RHS: split rhs(n+m) into rhs_x(n), rhs_y(k), rhs_lam(n_nonneg)
// ---------------------------------------------------------------------------
__global__ void woodbury_unpack_rhs_kernel(
    const double* __restrict__ rhs,       // (B, n+m) input
    double* __restrict__ rhs_x,           // (B, n)
    double* __restrict__ rhs_y,           // (B, k)
    double* __restrict__ rhs_lam,         // (B, n_nonneg)
    int64_t n, int64_t k, int64_t n_nonneg,
    int64_t N,  // n + m
    int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / N;
        int64_t i = tid % N;
        double val = rhs[tid];
        if (i < n) {
            rhs_x[b * n + i] = val;
        } else if (i < n + k) {
            rhs_y[b * k + (i - n)] = val;
        } else {
            rhs_lam[b * n_nonneg + (i - n - k)] = val;
        }
    }
}

void launch_woodbury_unpack_rhs(
    const double* rhs, double* rhs_x, double* rhs_y, double* rhs_lam,
    int64_t batch, int64_t n, int64_t k, int64_t n_nonneg, cudaStream_t stream)
{
    int64_t N = n + k + n_nonneg;
    int64_t total = batch * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_unpack_rhs_kernel, blocks, threads, 0, stream,
        rhs, rhs_x, rhs_y, rhs_lam, n, k, n_nonneg, N, total);
}

// Unpack RHS with k_total stride for rhs_nu (Phase 2)
__global__ void woodbury_unpack_rhs_kt_kernel(
    const double* __restrict__ rhs,
    double* __restrict__ rhs_x,
    double* __restrict__ rhs_nu,      // (B, k_total) — write first k entries per row
    double* __restrict__ rhs_lam,
    int64_t n, int64_t k, int64_t k_total, int64_t n_nonneg,
    int64_t N, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / N;
        int64_t i = tid % N;
        double val = rhs[tid];
        if (i < n) {
            rhs_x[b * n + i] = val;
        } else if (i < n + k) {
            rhs_nu[b * k_total + (i - n)] = val;
        } else {
            rhs_lam[b * n_nonneg + (i - n - k)] = val;
        }
    }
}

void launch_woodbury_unpack_rhs_kt(
    const double* rhs, double* rhs_x, double* rhs_nu, double* rhs_lam,
    int64_t batch, int64_t n, int64_t k, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream)
{
    int64_t N = n + k + n_nonneg;
    int64_t total = batch * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_unpack_rhs_kt_kernel, blocks, threads, 0, stream,
        rhs, rhs_x, rhs_nu, rhs_lam, n, k, k_total, n_nonneg, N, total);
}

// ---------------------------------------------------------------------------
// F_scaled[b,i,j] = d_tilde_inv[b,i] * F[b,i,j]
// F is (B,n,k) row-major per batch, d_tilde_inv is (B,n), output is (B,n,k) row-major
// ---------------------------------------------------------------------------
__global__ void woodbury_scale_F_kernel(
    double* __restrict__ F_scaled,
    const double* __restrict__ F,
    const double* __restrict__ d_tilde_inv,
    int64_t n, int64_t k, bool f_shared, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t nk = n * k;
        int64_t b = tid / nk;
        int64_t rem = tid % nk;
        int64_t i = rem / k;
        double d = d_tilde_inv[b * n + i];
        int64_t f_idx = f_shared ? rem : tid;
        F_scaled[tid] = d * F[f_idx];
    }
}

void launch_woodbury_scale_F(
    double* F_scaled, const double* F, const double* d_tilde_inv,
    int64_t batch, int64_t n, int64_t k, bool f_shared, cudaStream_t stream)
{
    int64_t total = batch * n * k;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_scale_F_kernel, blocks, threads, 0, stream,
        F_scaled, F, d_tilde_inv, n, k, f_shared, total);
}

// ---------------------------------------------------------------------------
// Set S = eps*I (batched): S[b,i,j] = (i==j) ? eps[b] : 0
// ---------------------------------------------------------------------------
// S[b,i,i] += eps[b] for diagonal entries only
__global__ void woodbury_add_eps_diagonal_kernel(
    double* __restrict__ S,
    const double* __restrict__ eps,
    int64_t k, int64_t total_diag)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_diag) {
        int64_t b = tid / k;
        int64_t i = tid % k;
        S[b * k * k + i * k + i] += eps[b];
    }
}

void launch_woodbury_set_identity_eps(
    double* S, const double* eps, int64_t batch, int64_t k, cudaStream_t stream)
{
    int64_t total_diag = batch * k;
    int threads = 256;
    int blocks = (total_diag + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_add_eps_diagonal_kernel, blocks, threads, 0, stream,
        S, eps, k, total_diag);
}

// ===========================================================================
// Phase 1: Generalized sparse nonneg kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// h_tilde_inv for ALL nonneg rows: h_tilde_inv[b,i] = 1/(H_nonneg[b,i] + eps[b])
// Then d_tilde_inv per column using CSC of A_s:
//   D[b,j] = P_diag[b,j] + eps[b]
//   for each sparse row r touching col j:
//     nonneg_r = sparse_nonneg_idx[r]
//     a = A_s_vals[b * n_sparse + r]
//     h_inv = h_tilde_inv[b * n_nonneg + nonneg_r]
//     D[b,j] += a^2 * H_nonneg[b*n_nonneg+nonneg_r] * h_inv
//   d_tilde_inv[b,j] = 1/D[b,j]
// ---------------------------------------------------------------------------

// Step 1: h_tilde_inv for all nonneg rows (simple, 1:1)
__global__ void woodbury_h_tilde_inv_all_kernel(
    double* __restrict__ h_tilde_inv,
    const double* __restrict__ H_nonneg,
    const double* __restrict__ eps,
    int64_t n_nonneg, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n_nonneg;
        h_tilde_inv[tid] = 1.0 / (H_nonneg[tid] + eps[b]);
    }
}

// Step 2: d_tilde_inv per column, using CSC of A_s
//
// `xn_x_to_hs` is a length-n lookup table: for each x index j, holds the
// direct-x nonneg slot index k (so we read xn_hs[b*n_xn + k]) or -1 if j
// is not constrained by a direct-x nonneg cone. Built once at construction.
__global__ void woodbury_d_tilde_inv_sparse_kernel(
    double* __restrict__ d_tilde_inv,
    const double* __restrict__ P_diag,
    const double* __restrict__ H_nonneg,
    const double* __restrict__ h_tilde_inv,
    const double* __restrict__ A_s_vals,
    const double* __restrict__ eps,
    const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets,
    const int64_t* __restrict__ col_sparse_rows,
    const double* __restrict__ xn_hs,            // (B, n_xn) or nullptr
    const int64_t* __restrict__ xn_x_to_slot,    // (n) lookup or nullptr
    int64_t n_xn,
    int64_t n, int64_t n_nonneg, int64_t n_sparse, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n;
        int64_t j = tid % n;
        double D = P_diag[tid] + eps[b];
        // Accumulate sparse nonneg contributions for this column
        int64_t start = col_offsets[j];
        int64_t end = col_offsets[j + 1];
        for (int64_t p = start; p < end; ++p) {
            int64_t sr = col_sparse_rows[p];  // sparse row index
            int64_t ni = sparse_nonneg_idx[sr]; // nonneg index
            double a = A_s_vals[b * n_sparse + sr];
            double h_inv = h_tilde_inv[b * n_nonneg + ni];
            D += a * a * h_inv;  // = a^2 / (w^2 + eps)
        }
        // Direct-x nonneg: add Hs_i = z_x/x[i] from the xcone scaling.
        if (xn_x_to_slot != nullptr) {
            int64_t k = xn_x_to_slot[j];
            if (k >= 0) {
                D += xn_hs[b * n_xn + k];
            }
        }
        d_tilde_inv[tid] = 1.0 / D;
    }
}

void launch_woodbury_d_tilde_inv_sparse(
    double* d_tilde_inv, double* h_tilde_inv,
    const double* P_diag, const double* H_nonneg,
    const double* A_s_vals, const double* eps,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    const double* xn_hs,
    const int64_t* xn_x_to_slot,
    int64_t n_xn,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = 256;

    // Step 1: h_tilde_inv for all nonneg rows
    if (n_nonneg > 0) {
        int64_t total = batch * n_nonneg;
        int blocks = (total + threads - 1) / threads;
        MOREAU_KERNEL_LAUNCH(woodbury_h_tilde_inv_all_kernel, blocks, threads, 0, stream,
            h_tilde_inv, H_nonneg, eps, n_nonneg, total);
    }

    // Step 2: d_tilde_inv per column
    {
        int64_t total = batch * n;
        int blocks = (total + threads - 1) / threads;
        MOREAU_KERNEL_LAUNCH(woodbury_d_tilde_inv_sparse_kernel, blocks, threads, 0, stream,
            d_tilde_inv, P_diag, H_nonneg, h_tilde_inv, A_s_vals, eps,
            sparse_nonneg_idx, col_offsets, col_sparse_rows,
            xn_hs, xn_x_to_slot, n_xn,
            n, n_nonneg, n_sparse, total);
    }
}

// ---------------------------------------------------------------------------
// RHS adjustment with sparse nonneg rows
// rhs_x_adj[b,j] = rhs_x[b,j] + sum_{sparse r in col j} a_r * h_inv[nonneg_r] * rhs_lam[nonneg_r]
// ---------------------------------------------------------------------------
__global__ void woodbury_rhs_adj_sparse_kernel(
    double* __restrict__ rhs_x_adj,
    const double* __restrict__ rhs_x,
    const double* __restrict__ h_tilde_inv,
    const double* __restrict__ rhs_lam,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets,
    const int64_t* __restrict__ col_sparse_rows,
    int64_t n, int64_t n_nonneg, int64_t n_sparse, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n;
        int64_t j = tid % n;
        double val = rhs_x[tid];
        int64_t start = col_offsets[j];
        int64_t end = col_offsets[j + 1];
        for (int64_t p = start; p < end; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = A_s_vals[b * n_sparse + sr];
            val += a * h_tilde_inv[b * n_nonneg + ni] * rhs_lam[b * n_nonneg + ni];
        }
        rhs_x_adj[tid] = val;
    }
}

void launch_woodbury_rhs_adj_sparse(
    double* rhs_x_adj, const double* rhs_x,
    const double* h_tilde_inv, const double* rhs_lam,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int64_t total = batch * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_rhs_adj_sparse_kernel, blocks, threads, 0, stream,
        rhs_x_adj, rhs_x, h_tilde_inv, rhs_lam, A_s_vals,
        sparse_nonneg_idx, col_offsets, col_sparse_rows,
        n, n_nonneg, n_sparse, total);
}

// ---------------------------------------------------------------------------
// dlam recovery for sparse nonneg rows
// dlam[b, nonneg_idx] = h_inv[b, nonneg_idx] * (a * dx[b, col] - rhs_lam[b, nonneg_idx])
// One thread per (batch, sparse_row)
// ---------------------------------------------------------------------------
__global__ void woodbury_dlam_sparse_kernel(
    double* __restrict__ dlam,
    const double* __restrict__ h_tilde_inv,
    const double* __restrict__ dx,
    const double* __restrict__ rhs_lam,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ sparse_nonneg_idx,
    int64_t n, int64_t n_nonneg, int64_t n_sparse, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n_sparse;
        int64_t sr = tid % n_sparse;
        int64_t ni = sparse_nonneg_idx[sr];
        int64_t col = sparse_col[sr];
        double a = A_s_vals[b * n_sparse + sr];
        double h_inv = h_tilde_inv[b * n_nonneg + ni];
        dlam[b * n_nonneg + ni] = h_inv * (a * dx[b * n + col] - rhs_lam[b * n_nonneg + ni]);
    }
}

void launch_woodbury_dlam_sparse(
    double* dlam, const double* h_tilde_inv,
    const double* dx, const double* rhs_lam,
    const double* A_s_vals,
    const int64_t* sparse_col, const int64_t* sparse_nonneg_idx,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int64_t total = batch * n_sparse;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_dlam_sparse_kernel, blocks, threads, 0, stream,
        dlam, h_tilde_inv, dx, rhs_lam, A_s_vals,
        sparse_col, sparse_nonneg_idx,
        n, n_nonneg, n_sparse, total);
}

// ---------------------------------------------------------------------------
// dlam for empty nonneg rows (0 nonzeros)
// dlam[b, ni] = -h_inv[b, ni] * rhs_lam[b, ni]
// ---------------------------------------------------------------------------
__global__ void woodbury_dlam_empty_kernel(
    double* __restrict__ dlam,
    const double* __restrict__ h_tilde_inv,
    const double* __restrict__ rhs_lam,
    const int64_t* __restrict__ empty_nonneg_idx,
    int64_t n_nonneg, int64_t n_empty, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n_empty;
        int64_t ei = tid % n_empty;
        int64_t ni = empty_nonneg_idx[ei];
        dlam[b * n_nonneg + ni] = -h_tilde_inv[b * n_nonneg + ni] * rhs_lam[b * n_nonneg + ni];
    }
}

void launch_woodbury_dlam_empty(
    double* dlam, const double* h_tilde_inv, const double* rhs_lam,
    const int64_t* empty_nonneg_idx,
    int64_t batch, int64_t n_nonneg, int64_t n_empty,
    cudaStream_t stream)
{
    int64_t total = batch * n_empty;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_dlam_empty_kernel, blocks, threads, 0, stream,
        dlam, h_tilde_inv, rhs_lam, empty_nonneg_idx, n_nonneg, n_empty, total);
}

// ---------------------------------------------------------------------------
// Max diag with sparse nonneg: per column D_eff = P + sum a^2 * H
// Then reduce to max per batch.
// ---------------------------------------------------------------------------
__global__ void woodbury_max_diag_sparse_kernel(
    double* __restrict__ max_diag,
    const double* __restrict__ P_diag,
    const double* __restrict__ H_nonneg,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets,
    const int64_t* __restrict__ col_sparse_rows,
    int64_t n, int64_t n_nonneg, int64_t n_sparse, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n;
        int64_t j = tid % n;
        double D = P_diag[tid];
        int64_t start = col_offsets[j];
        int64_t end = col_offsets[j + 1];
        for (int64_t p = start; p < end; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = A_s_vals[b * n_sparse + sr];
            D += a * a * H_nonneg[b * n_nonneg + ni];
        }
        double val = fabs(D);
        // Atomic max for doubles
        unsigned long long int* addr = reinterpret_cast<unsigned long long int*>(&max_diag[b]);
        unsigned long long int old = *addr;
        unsigned long long int expected;
        do {
            expected = old;
            double old_val = __longlong_as_double(expected);
            if (val <= old_val) break;
            old = atomicCAS(addr, expected, __double_as_longlong(val));
        } while (old != expected);
    }
}

void launch_woodbury_max_diag_sparse(
    double* max_diag, const double* P_diag,
    const double* H_nonneg, const double* A_s_vals,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    cudaMemsetAsync(max_diag, 0, sizeof(double) * batch, stream);
    int64_t total = batch * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_max_diag_sparse_kernel, blocks, threads, 0, stream,
        max_diag, P_diag, H_nonneg, A_s_vals,
        sparse_nonneg_idx, col_offsets, col_sparse_rows,
        n, n_nonneg, n_sparse, total);
}

// ---------------------------------------------------------------------------
// Phase 2: Dense nonneg kernels
// ---------------------------------------------------------------------------

// Compute C_all: combined zero-cone + dense nonneg regularization diagonal
__global__ void woodbury_compute_C_all_kernel(
    double* __restrict__ C_all,
    const double* __restrict__ eps,
    const double* __restrict__ H_nonneg,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int64_t b = tid / k_total;
    int64_t i = tid % k_total;
    double e = eps[b];
    if (i < k) {
        C_all[tid] = 1.0 / e;
    } else {
        int64_t d = i - k;
        int64_t ni = dense_nonneg_idx[d];
        double H = H_nonneg[b * n_nonneg + ni];
        C_all[tid] = 1.0 / (H + e);
    }
}

void launch_woodbury_compute_C_all(
    double* C_all, const double* eps,
    const double* H_nonneg, const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream)
{
    int64_t total = batch * k_total;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_compute_C_all_kernel, blocks, threads, 0, stream,
        C_all, eps, H_nonneg, dense_nonneg_idx,
        k, k_d, k_total, n_nonneg, total);
}

// Add C_all^{-1} to S diagonal
// For zero-cone entries (i < k): S[b,i,i] += eps[b]
// For dense nonneg entries (i >= k): S[b,i,i] += H_nonneg[b, dense_nonneg_idx[i-k]] + eps[b]
__global__ void woodbury_set_C_all_inv_diagonal_kernel(
    double* __restrict__ S,
    const double* __restrict__ eps,
    const double* __restrict__ H_nonneg,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int64_t b = tid / k_total;
    int64_t i = tid % k_total;
    double e = eps[b];
    double diag_add;
    if (i < k) {
        diag_add = e;
    } else {
        int64_t d = i - k;
        int64_t ni = dense_nonneg_idx[d];
        diag_add = H_nonneg[b * n_nonneg + ni] + e;
    }
    S[b * k_total * k_total + i * k_total + i] += diag_add;
}

void launch_woodbury_set_C_all_inv_diagonal(
    double* S, const double* eps,
    const double* H_nonneg, const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream)
{
    int64_t total = batch * k_total;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_set_C_all_inv_diagonal_kernel, blocks, threads, 0, stream,
        S, eps, H_nonneg, dense_nonneg_idx,
        k, k_d, k_total, n_nonneg, total);
}

// Gather dense nonneg RHS into combined Schur RHS
__global__ void woodbury_gather_dense_rhs_kernel(
    double* __restrict__ rhs_nu,
    const double* __restrict__ rhs_lam,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t k, int64_t k_total, int64_t n_nonneg, int64_t k_d, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int64_t b = tid / k_d;
    int64_t d = tid % k_d;
    int64_t ni = dense_nonneg_idx[d];
    rhs_nu[b * k_total + k + d] = rhs_lam[b * n_nonneg + ni];
}

void launch_woodbury_gather_dense_rhs(
    double* rhs_nu, const double* rhs_lam,
    const int64_t* dense_nonneg_idx,
    int64_t eB, int64_t k, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream)
{
    int64_t k_d = k_total - k;
    int64_t total = eB * k_d;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_gather_dense_rhs_kernel, blocks, threads, 0, stream,
        rhs_nu, rhs_lam, dense_nonneg_idx,
        k, k_total, n_nonneg, k_d, total);
}

// Scatter dense nonneg duals from dy_nu[k..k_total] into dlam at dense positions
__global__ void woodbury_scatter_dlam_dense_kernel(
    double* __restrict__ dlam,
    const double* __restrict__ dy_nu,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t k, int64_t k_total, int64_t n_nonneg, int64_t k_d, int64_t total)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int64_t b = tid / k_d;
    int64_t d = tid % k_d;
    int64_t ni = dense_nonneg_idx[d];
    dlam[b * n_nonneg + ni] = dy_nu[b * k_total + k + d];
}

void launch_woodbury_scatter_dlam_dense(
    double* dlam, const double* dy_nu,
    int64_t eB, int64_t k, int64_t k_total, int64_t n_nonneg,
    const int64_t* dense_nonneg_idx, int64_t k_d,
    cudaStream_t stream)
{
    int64_t total = eB * k_d;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    MOREAU_KERNEL_LAUNCH(woodbury_scatter_dlam_dense_kernel, blocks, threads, 0, stream,
        dlam, dy_nu, dense_nonneg_idx,
        k, k_total, n_nonneg, k_d, total);
}

// =============================================================================
// KKT matvec kernels (for iterative refinement)
// =============================================================================

// out_x = (P_diag + eps) * dx, elementwise (B, n)
// P_diag is (B, n), eps is (B), dx is (eB, n). For eB > B, use b % B for data.
__global__ void woodbury_kkt_matvec_x_kernel(
    double* __restrict__ out_x,
    const double* __restrict__ dx,
    const double* __restrict__ P_diag,
    const double* __restrict__ eps,
    int64_t batch, int64_t n, int64_t total)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n;
        int64_t b_data = b % batch;
        out_x[tid] = (P_diag[b_data * n + tid % n] + eps[b_data]) * dx[tid];
    }
}

void launch_woodbury_kkt_matvec_x(
    double* out_x, const double* dx,
    const double* P_diag, const double* eps,
    int64_t batch, int64_t n, cudaStream_t stream)
{
    int64_t total = batch * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_kkt_matvec_x_kernel, blocks, threads, 0, stream,
        out_x, dx, P_diag, eps, batch, n, total);
}

// Scatter-accumulate A_s^T * dlam_s into out_x using CSC structure:
// out_x[b, col_j] += sum_{sparse r in col j} a_r * dlam[b, nonneg_r]
// One thread per (b, col_j). CSC gives which sparse rows touch each column.
__global__ void woodbury_kkt_As_T_dlam_kernel(
    double* __restrict__ out_x,
    const double* __restrict__ dlam,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets,
    const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse, int64_t total)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n;
        int64_t j = tid % n;
        int64_t start = col_offsets[j];
        int64_t end = col_offsets[j + 1];
        double acc = 0.0;
        for (int64_t p = start; p < end; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = A_s_vals[b * n_sparse + sr];
            acc += a * dlam[b * n_nonneg + ni];
        }
        out_x[tid] += acc;
    }
}

void launch_woodbury_kkt_As_T_dlam(
    double* out_x, const double* dlam,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int64_t total = batch * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_kkt_As_T_dlam_kernel, blocks, threads, 0, stream,
        out_x, dlam, A_s_vals,
        sparse_nonneg_idx, col_offsets, col_sparse_rows,
        batch, n, n_nonneg, n_sparse, total);
}

// out_lam_s[r] = a_r * dx[col_r] - (w_r^2 + eps) * dlam_r for sparse nonneg rows
// One thread per (batch, sparse_row).
__global__ void woodbury_kkt_matvec_sparse_z_kernel(
    double* __restrict__ out,
    const double* __restrict__ dx,
    const double* __restrict__ dlam,
    const double* __restrict__ H_nonneg,
    const double* __restrict__ eps,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ sparse_nonneg_idx,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse, int64_t total)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n_sparse;
        int64_t sr = tid % n_sparse;
        int64_t b_data = b % batch;
        int64_t ni = sparse_nonneg_idx[sr];
        int64_t col = sparse_col[sr];
        double a = A_s_vals[b_data * n_sparse + sr];
        double w2_eps = H_nonneg[b_data * n_nonneg + ni] + eps[b_data];
        out[b * n_nonneg + ni] = a * dx[b * n + col] - w2_eps * dlam[b * n_nonneg + ni];
    }
}

void launch_woodbury_kkt_matvec_sparse_z(
    double* out, const double* dx, const double* dlam,
    const double* H_nonneg, const double* eps,
    const double* A_s_vals,
    const int64_t* sparse_col, const int64_t* sparse_nonneg_idx,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int64_t total = batch * n_sparse;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_kkt_matvec_sparse_z_kernel, blocks, threads, 0, stream,
        out, dx, dlam, H_nonneg, eps, A_s_vals,
        sparse_col, sparse_nonneg_idx,
        batch, n, n_nonneg, n_sparse, total);
}

// out_lam_e[r] = -(w_r^2 + eps) * dlam_e for empty nonneg rows
// One thread per (batch, empty_row).
__global__ void woodbury_kkt_matvec_empty_z_kernel(
    double* __restrict__ out,
    const double* __restrict__ dlam,
    const double* __restrict__ H_nonneg,
    const double* __restrict__ eps,
    const int64_t* __restrict__ empty_nonneg_idx,
    int64_t batch, int64_t n_nonneg, int64_t n_empty, int64_t total)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid < total) {
        int64_t b = tid / n_empty;
        int64_t ei = tid % n_empty;
        int64_t b_data = b % batch;
        int64_t ni = empty_nonneg_idx[ei];
        double w2_eps = H_nonneg[b_data * n_nonneg + ni] + eps[b_data];
        out[b * n_nonneg + ni] = -w2_eps * dlam[b * n_nonneg + ni];
    }
}

void launch_woodbury_kkt_matvec_empty_z(
    double* out, const double* dlam,
    const double* H_nonneg, const double* eps,
    const int64_t* empty_nonneg_idx,
    int64_t batch, int64_t n_nonneg, int64_t n_empty,
    cudaStream_t stream)
{
    int64_t total = batch * n_empty;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_kkt_matvec_empty_z_kernel, blocks, threads, 0, stream,
        out, dlam, H_nonneg, eps,
        empty_nonneg_idx, batch, n_nonneg, n_empty, total);
}

// Subtract C_all_inv * nu from out_nu:
// out[b,i] -= C_all_inv[b,i] * nu[b,i]
// For i < k:  C_all_inv = eps[b]
// For i >= k: C_all_inv = H_nonneg[b, dense_nonneg_idx[i-k]] + eps[b]
// eB may exceed B (nrhs > 1); data arrays use b % B.
__global__ void woodbury_kkt_matvec_nu_sub_kernel(
    double* __restrict__ out,
    const double* __restrict__ nu,
    const double* __restrict__ eps,
    const double* __restrict__ H_nonneg,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t eB,
    int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    int64_t total)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int64_t b = tid / k_total;
    int64_t i = tid % k_total;
    int64_t b_data = b % batch;
    double e = eps[b_data];
    double c_inv;
    if (i < k) {
        c_inv = e;
    } else {
        int64_t d = i - k;
        int64_t ni = dense_nonneg_idx[d];
        c_inv = H_nonneg[b_data * n_nonneg + ni] + e;
    }
    out[tid] -= c_inv * nu[tid];
}

void launch_woodbury_kkt_matvec_nu_sub(
    double* out, const double* nu,
    const double* eps, const double* H_nonneg,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t eB,
    int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream)
{
    int64_t total = eB * k_total;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_kkt_matvec_nu_sub_kernel, blocks, threads, 0, stream,
        out, nu, eps, H_nonneg, dense_nonneg_idx,
        batch, eB, k, k_d, k_total, n_nonneg, total);
}

// Compute residual: r = rhs - y, in-place (r = rhs - y stored into r)
__global__ void woodbury_refine_residual_kernel(
    double* __restrict__ r,           // (count) output: rhs - y
    const double* __restrict__ rhs,   // (count) original RHS
    const double* __restrict__ y,     // (count) S * x
    int64_t count)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < count) r[i] = rhs[i] - y[i];
}

void launch_woodbury_refine_residual(
    double* r, const double* rhs, const double* y,
    int64_t count, cudaStream_t stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_refine_residual_kernel, blocks, threads, 0, stream, r, rhs, y, count);
}

// Fused x += delta (axpy)
__global__ void woodbury_axpy_kernel(
    double* __restrict__ x,
    const double* __restrict__ delta,
    int64_t count)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < count) x[i] += delta[i];
}

void launch_woodbury_axpy(
    double* x, const double* delta,
    int64_t count, cudaStream_t stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_axpy_kernel, blocks, threads, 0, stream, x, delta, count);
}

// ============================================================================
// Fill constant
// ============================================================================

__global__ void woodbury_fill_constant_kernel(
    double* __restrict__ out, double val, int64_t count)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < count) out[i] = val;
}

void launch_woodbury_fill_constant(
    double* out, double val,
    int64_t count, cudaStream_t stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(woodbury_fill_constant_kernel, blocks, threads, 0, stream, out, val, count);
}

// ============================================================================
// Fused Woodbury solve kernel (B=1 only)
//
// Performs the ENTIRE solve_core in a single kernel launch:
//   1. Unpack RHS
//   2. RHS adjustment (sparse nonneg)
//   3. Schur RHS = C_all * rhs_nu + F' @ rhs_x_adj (via shared-mem GEMV)
//   4. D^{-1} @ rhs_x_adj
//   5. F @ (D^{-1} rhs_x_adj) → Schur RHS (via shared-mem GEMV)
//   6. Cholesky solve (inline, k×k)
//   7. F' @ d_nu → Fdnu (via shared-mem GEMV)
//   8. dx = D^{-1} (rhs_x_adj - Fdnu)
//   9. dy = d_nu - C_all * rhs_nu
//   10. dlam recovery (sparse + empty + dense)
//   11. Pack solution
//
// Max k_total = 64 (limited by shared memory for Cholesky).
// Requires n threads (one per x-variable).
// ============================================================================

// Shared memory layout:
//   double F_col[k_total]     — one column of F_all at a time
//   double gemv_out[k_total]  — GEMV output accumulator
//   double S[k_total*k_total] — Schur complement (factored in-place)
//   double work[k_total]      — Cholesky solve workspace
// Total: k_total*(k_total + 3) doubles

__global__ void woodbury_fused_solve_kernel(
    double* __restrict__ sol,
    const double* __restrict__ rhs,
    const double* __restrict__ F_all,    // (n, k_total) row-major
    const double* __restrict__ d_tilde_inv, // (n)
    const double* __restrict__ h_tilde_inv, // (n_nonneg)
    const double* __restrict__ C_all,    // (k_total)
    const double* __restrict__ S_chol,   // (k_total, k_total) lower Cholesky
    const double* __restrict__ A_s_vals, // (n_sparse)
    const int64_t* __restrict__ sp_nn_idx,
    const int64_t* __restrict__ sp_col,
    const int64_t* __restrict__ col_off,
    const int64_t* __restrict__ col_sp_rows,
    const int64_t* __restrict__ dense_nn_idx,
    const int64_t* __restrict__ empty_nn_idx,
    int64_t n, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    int64_t n_nonneg, int64_t n_sparse, int64_t n_empty,
    int nrhs)
{
    // This kernel uses 1 block with n threads.
    // n must be <= 1024 (max threads per block).
    // For larger n, we'd need a multi-block approach.
    extern __shared__ double smem[];
    // Partition shared memory
    double* s_rhs_nu = smem;                         // [nrhs * k_total]
    double* s_rhs_lam = s_rhs_nu + nrhs * k_total;  // [nrhs * n_nonneg]
    double* s_d_nu = s_rhs_lam + nrhs * n_nonneg;   // [nrhs * k_total]
    double* s_gemv = s_d_nu + nrhs * k_total;        // [k_total] temp for GEMV reduction
    double* s_xvec = s_gemv + k_total;               // [n] temp for Dinv_rhs storage

    int64_t N = n + m;
    int tid = threadIdx.x;

    for (int rr = 0; rr < nrhs; ++rr) {
        const double* rhs_r = rhs + rr * N;
        double* sol_r = sol + rr * N;

        // ---- Step 1: Unpack RHS ----
        for (int64_t i = tid; i < k_total; i += blockDim.x)
            s_rhs_nu[rr * k_total + i] = (i < k) ? rhs_r[n + i] : 0.0;
        for (int64_t i = tid; i < n_nonneg; i += blockDim.x)
            s_rhs_lam[rr * n_nonneg + i] = rhs_r[n + k + i];
        __syncthreads();

        // Gather dense nonneg into rhs_nu[k..k_total]
        for (int64_t d = tid; d < k_d; d += blockDim.x) {
            int64_t ni = dense_nn_idx[d];
            s_rhs_nu[rr * k_total + k + d] = s_rhs_lam[rr * n_nonneg + ni];
        }
        __syncthreads();

        // ---- Step 2: RHS adjustment ----
        double rhs_x_adj = 0.0;
        if (tid < n) {
            rhs_x_adj = rhs_r[tid];
            for (int64_t p = col_off[tid]; p < col_off[tid + 1]; ++p) {
                int64_t sr = col_sp_rows[p];
                int64_t ni = sp_nn_idx[sr];
                rhs_x_adj += A_s_vals[sr] * h_tilde_inv[ni] * s_rhs_lam[rr * n_nonneg + ni];
            }
        }

        // ---- Step 3: rhs_x_adj += F_all' @ (C_all * rhs_nu) ----
        // Compute C_all * rhs_nu first
        if (tid < k_total) s_gemv[tid] = C_all[tid] * s_rhs_nu[rr * k_total + tid];
        __syncthreads();

        // F_all' @ s_gemv: each thread j computes dot(F_all[j,:], s_gemv)
        if (tid < n) {
            const double* F_row = F_all + tid * k_total;
            double dot = 0.0;
            for (int64_t i = 0; i < k_total; ++i) dot += F_row[i] * s_gemv[i];
            rhs_x_adj += dot;
        }
        __syncthreads();

        // ---- Step 4: u = D^{-1} @ rhs_x_adj ----
        double Dinv_rhs_j = 0.0;
        if (tid < n) Dinv_rhs_j = d_tilde_inv[tid] * rhs_x_adj;

        // ---- Step 5: schur_rhs = F_all @ u ----
        // Store Dinv_rhs in shared memory
        if (tid < n) s_xvec[tid] = Dinv_rhs_j;
        __syncthreads();
        // Threads 0..k_total-1 compute: s_gemv[i] = Σ_j F_all[j*k_total+i] * s_xvec[j]
        if (tid < k_total) {
            double dot = 0.0;
            for (int64_t j = 0; j < n; ++j)
                dot += F_all[j * k_total + tid] * s_xvec[j];
            s_gemv[tid] = dot;
        }
        __syncthreads();

        // Copy Schur RHS to d_nu workspace
        if (tid < k_total) s_d_nu[rr * k_total + tid] = s_gemv[tid];
        __syncthreads();

        // ---- Step 6: Cholesky solve S d_nu = schur_rhs ----
        // S_chol is pre-factored. Do forward then back substitution.
        // Only thread 0 does this (k_total is small, ~3-50).
        if (tid == 0) {
            double* d = s_d_nu + rr * k_total;
            // S_chol is stored column-major (cuSolver convention).
            // L[i,j] at S_chol[i + j * k_total] for i >= j.
            // Forward solve: L y = d
            for (int64_t i = 0; i < k_total; ++i) {
                double s = d[i];
                for (int64_t j = 0; j < i; ++j)
                    s -= S_chol[i + j * k_total] * d[j];
                d[i] = s / S_chol[i + i * k_total];
            }
            // Back solve: L' x = y
            for (int64_t i = k_total - 1; i >= 0; --i) {
                double s = d[i];
                for (int64_t j = i + 1; j < k_total; ++j)
                    s -= S_chol[j + i * k_total] * d[j];
                d[i] = s / S_chol[i + i * k_total];
            }
        }
        __syncthreads();

        // ---- Step 7: Fdnu = F_all' @ d_nu ----
        double Fdnu_j = 0.0;
        if (tid < n) {
            const double* F_row = F_all + tid * k_total;
            for (int64_t i = 0; i < k_total; ++i)
                Fdnu_j += F_row[i] * s_d_nu[rr * k_total + i];
        }

        // ---- Step 8: dx = d_tilde_inv * (rhs_x_adj - Fdnu) ----
        double dx_j = 0.0;
        if (tid < n) dx_j = d_tilde_inv[tid] * (rhs_x_adj - Fdnu_j);

        // ---- Step 9: dy_nu = d_nu - C_all * rhs_nu ----
        // (done in shared memory, only k_total entries)
        if (tid < k_total)
            s_gemv[tid] = s_d_nu[rr * k_total + tid] - C_all[tid] * s_rhs_nu[rr * k_total + tid];
        __syncthreads();

        // ---- Step 10: dlam recovery ----
        // Sparse: dlam[ni] = h_inv[ni] * (a_r * dx[col_r] - rhs_lam[ni])
        // Empty: dlam[ni] = -h_inv[ni] * rhs_lam[ni]
        // Dense: dlam[ni] = dy_nu[k+d]; written directly to sol below.

        // ---- Step 11: Pack [dx; dy; dlam] into sol ----
        // Store dx in shared memory so sparse nonneg recovery can read it
        if (tid < n) s_xvec[tid] = dx_j;
        __syncthreads();

        // Write dx to output
        if (tid < n) sol_r[tid] = dx_j;
        // Write dy (zero-cone part)
        if (tid < k) sol_r[n + tid] = s_gemv[tid];
        // Initialize nonneg dual to 0
        for (int64_t i = tid; i < n_nonneg; i += blockDim.x) sol_r[n + k + i] = 0.0;
        __syncthreads();

        // Sparse nonneg recovery: dlam[ni] = h_inv[ni] * (a_r * dx[col_r] - rhs_lam[ni])
        for (int64_t r = tid; r < n_sparse; r += blockDim.x) {
            int64_t ni = sp_nn_idx[r];
            int64_t col = sp_col[r];
            double dx_col = s_xvec[col];
            sol_r[n + k + ni] = h_tilde_inv[ni] * (A_s_vals[r] * dx_col - s_rhs_lam[rr * n_nonneg + ni]);
        }
        // Empty nonneg: dlam[ni] = -h_inv[ni] * rhs_lam[ni]
        for (int64_t r = tid; r < n_empty; r += blockDim.x) {
            int64_t ni = empty_nn_idx[r];
            sol_r[n + k + ni] = -h_tilde_inv[ni] * s_rhs_lam[rr * n_nonneg + ni];
        }
        // Dense nonneg: scatter from dy_nu[k..k_total]
        for (int64_t d = tid; d < k_d; d += blockDim.x) {
            int64_t ni = dense_nn_idx[d];
            sol_r[n + k + ni] = s_gemv[k + d];
        }
        __syncthreads();
    }
}

void launch_woodbury_fused_solve(
    double* sol, const double* rhs,
    const double* F_all, const double* d_tilde_inv,
    const double* h_tilde_inv, const double* C_all,
    const double* S_chol,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    const int64_t* dense_nonneg_idx, const int64_t* empty_nonneg_idx,
    int64_t n, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    int64_t n_nonneg, int64_t n_sparse, int64_t n_empty,
    int nrhs, cudaStream_t stream)
{
    if (n == 0) return;
    // Shared memory: nrhs*(k_total + n_nonneg + k_total) + k_total + n doubles
    size_t smem = sizeof(double) * (nrhs * (k_total + n_nonneg + k_total) + k_total + n);
    int threads = (int)n;  // One thread per x-variable
    if (threads > 1024) return;  // Fallback to non-fused path for large n
    MOREAU_KERNEL_LAUNCH(woodbury_fused_solve_kernel, 1, threads, smem, stream,
        sol, rhs, F_all, d_tilde_inv, h_tilde_inv, C_all, S_chol,
        A_s_vals, sparse_nonneg_idx, sparse_col,
        col_offsets, col_sparse_rows,
        dense_nonneg_idx, empty_nonneg_idx,
        n, m, k, k_d, k_total, n_nonneg, n_sparse, n_empty, nrhs);
}

// ============================================================
// Backward pass kernels (DiffWoodbury direct Schur complement)
// ============================================================

// Elementwise inverse
__global__ void elementwise_inv_kernel(double* __restrict__ out, const double* __restrict__ in, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = 1.0 / in[idx];
}

void launch_woodbury_elementwise_inv(double* out, const double* in, int64_t N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(elementwise_inv_kernel, blocks, threads, 0, stream, out, in, N);
}

// s-elimination diagonals
__global__ void diff_wb_s_elim_diagonals_kernel(
    double* __restrict__ g, double* __restrict__ Lambda, double* __restrict__ gh1,
    const double* __restrict__ H_nonneg, double eps,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* g_b = g + b * m;
    double* L_b = Lambda + b * m;
    double* gh_b = gh1 + b * m;
    const double* H_b = H_nonneg + b * n_nonneg;

    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        double denom = 2.0 + eps;
        g_b[i] = 1.0 / denom;
        L_b[i] = 1.0 - g_b[i];
        gh_b[i] = g_b[i] * 2.0;
    }
    for (int64_t i = threadIdx.x; i < n_nonneg; i += blockDim.x) {
        double h = H_b[i];
        double denom = 1.0 + h * h + eps;
        double gi = 1.0 / denom;
        g_b[k + i] = gi;
        L_b[k + i] = 1.0 - gi;
        gh_b[k + i] = gi * (1.0 + h);
    }
}

void launch_diff_wb_s_elim_diagonals(
    double* g, double* Lambda, double* gh1,
    const double* H_nonneg, double eps,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg,
    cudaStream_t stream)
{
    int threads = std::min((int)m, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_s_elim_diagonals_kernel, (int)batch, threads, 0, stream,
        g, Lambda, gh1, H_nonneg, eps, batch, m, k, n_nonneg);
}

// D_x computation with full cross-term correction from sparse nonneg elimination
// D_x[j] = P[j]^2 + eps + sum_sparse Lambda[ni]*a^2
//           - sum_sparse (P[j]*a + a*G[ni] + c1[j]*c2[k+ni])^2 / D_z[ni]
// where G[ni] = (h^2-h+eps)/(1+h^2+eps), D_z includes a^2
__global__ void diff_wb_compute_D_x_kernel(
    double* __restrict__ D_x,
    const double* __restrict__ P_diag, const double* __restrict__ Lambda,
    const double* __restrict__ A_s_vals, double eps,
    const double* __restrict__ H_nonneg, const double* __restrict__ c1, const double* __restrict__ c2,
    const double* __restrict__ D_z_inv, const double* __restrict__ rho,
    const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* D_b = D_x + b * n;
    const double* P_b = P_diag + b * n;
    const double* L_b = Lambda + b * m;
    const double* As_b = A_s_vals + b * n_sparse;
    const double* H_b = H_nonneg + b * n_nonneg;
    const double* c1_b = c1 + b * n;
    const double* c2_b = c2 + b * m;
    const double* Dzi_b = D_z_inv + b * n_nonneg;
    (void)rho;  // rho/c2 global SM unused: c2 is zeroed (gap handled globally)

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double pj = P_b[j];
        double val = pj * pj + eps;
        // λ-elimination for column j. The λ rows on column j form a small block
        // L = diag(D_zp) + a a' (D_zp = D_z − a²): the off-diagonal a_p·a_q from
        // multiple bounds on this column. Eliminate via per-column Sherman-Morrison:
        //   cross' L^{-1} cross = Σ cross²/D_zp − (Σ cross·a/D_zp)² / (1 + Σ a²/D_zp).
        double s_cc = 0.0, s_ca = 0.0, s_aa = 0.0;
        for (int64_t p = col_offsets[j]; p < col_offsets[j + 1]; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            double h = H_b[ni];
            double gh1_i = (1.0 + h) / (1.0 + h * h + eps);

            // s-elimination contribution (Lambda * a^2)
            val += L_b[k + ni] * a * a;

            if (Dzi_b) {
                double dzi = Dzi_b[ni];
                double dzp_inv = dzi / (1.0 - a * a * dzi);   // 1/(D_z − a²)
                double cross = a * (pj + 1.0 - gh1_i) + c1_b[j] * c2_b[k + ni];
                s_cc += cross * cross * dzp_inv;
                s_ca += cross * a * dzp_inv;
                s_aa += a * a * dzp_inv;
            }
        }
        val -= s_cc - s_ca * s_ca / (1.0 + s_aa);
        // Pivot-regularize the inner Woodbury diagonal. M'_xx = D_x + GG' is SPD
        // (a Schur complement of the SPD M), but D_x itself goes tiny/negative on
        // ill-conditioned instances, so the Woodbury identity through D_x⁻¹ loses
        // accuracy to cancellation. Flooring D_x ≥ δ keeps D_x⁻¹ bounded and
        // M'_xx_floored ≥ M'_xx, making the chain a contractive preconditioner;
        // iterative refinement recovers the accuracy this gives up. Large-P
        // (well-conditioned) columns have D_x ≈ P² ≫ δ and are untouched.
        constexpr double kInnerPivotFloor = 1e-3;
        D_b[j] = fmax(val, kInnerPivotFloor);
    }
}

void launch_diff_wb_compute_D_x(
    double* D_x,
    const double* P_diag, const double* Lambda,
    const double* A_s_vals, double eps,
    const double* H_nonneg, const double* c1, const double* c2,
    const double* D_z_inv, const double* rho,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_compute_D_x_kernel, (int)batch, threads, 0, stream,
        D_x, P_diag, Lambda, A_s_vals, eps,
        H_nonneg, c1, c2, D_z_inv, rho,
        sparse_nonneg_idx, col_offsets, col_sparse_rows,
        batch, n, m, k, n_nonneg, n_sparse);
}

// D_z computation with A_s contribution for sparse nonneg rows
// D_z[i] = a_i^2 + 2 + eps - (1+h)^2/(1+h^2+eps) + c2^2
// where a_i^2 is the sum of squares of row i of A_sparse (at most 1 nonzero per row)
__global__ void diff_wb_compute_D_z_kernel(
    double* __restrict__ D_z, double* __restrict__ D_z_inv,
    const double* __restrict__ H_nonneg, const double* __restrict__ c2,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx,
    double eps,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* Dz_b = D_z + b * n_nonneg;
    double* Dzi_b = D_z_inv + b * n_nonneg;
    const double* H_b = H_nonneg + b * n_nonneg;
    const double* c2_b = c2 + b * m;
    const double* As_b = A_s_vals + b * n_sparse;

    // Initialize all to the base value (for dense/empty nonneg rows without sparse A)
    for (int64_t i = threadIdx.x; i < n_nonneg; i += blockDim.x) {
        double h = H_b[i];
        double c2i = c2_b[k + i];
        double val = 2.0 + eps - (1.0 + h) * (1.0 + h) / (1.0 + h * h + eps) + c2i * c2i;
        Dz_b[i] = val;
    }

    __syncthreads();

    // Add a^2 contribution for sparse nonneg rows
    for (int64_t sr = threadIdx.x; sr < n_sparse; sr += blockDim.x) {
        int64_t ni = sparse_nonneg_idx[sr];
        double a = As_b[sr];
        atomicAdd(&Dz_b[ni], a * a);
    }

    __syncthreads();

    // Compute inverse
    for (int64_t i = threadIdx.x; i < n_nonneg; i += blockDim.x) {
        Dzi_b[i] = 1.0 / Dz_b[i];
    }
}

void launch_diff_wb_compute_D_z(
    double* D_z, double* D_z_inv,
    const double* H_nonneg, const double* c2,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx,
    double eps,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)std::max(n_nonneg, n_sparse), 256);
    if (threads == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_compute_D_z_kernel, (int)batch, threads, 0, stream,
        D_z, D_z_inv, H_nonneg, c2, A_s_vals, sparse_nonneg_idx, eps,
        batch, m, k, n_nonneg, n_sparse);
}

// Build F_ext = [F_all; c1']
// B_F = num batches to write. f_all_stride = 0 if shared, n*k_total if per-batch.
__global__ void diff_wb_build_F_ext_kernel(
    double* __restrict__ F_ext, const double* __restrict__ F_all, const double* __restrict__ c1,
    int64_t B_F, int64_t n, int64_t k_total, int64_t k_ext, long long f_all_stride)
{
    int64_t b = blockIdx.x;
    if (b >= B_F) return;
    const double* F_b = F_all + b * f_all_stride;
    double* Fe_b = F_ext + b * n * k_ext;
    const double* c1_b = c1 + b * n;

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        for (int64_t i = 0; i < k_total; ++i) {
            Fe_b[j * k_ext + i] = F_b[j * k_total + i];
        }
        Fe_b[j * k_ext + k_total] = c1_b[j];
    }
}

void launch_diff_wb_build_F_ext(
    double* F_ext, const double* F_all, const double* c1,
    int64_t B_F, int64_t n, int64_t k_total, int64_t k_ext,
    cudaStream_t stream)
{
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    // Default stride: per-batch (n*k_total)
    long long stride = (long long)(n * k_total);
    MOREAU_KERNEL_LAUNCH(diff_wb_build_F_ext_kernel, (int)B_F, threads, 0, stream,
        F_ext, F_all, c1, B_F, n, k_total, k_ext, stride);
}

// Build F_ext with backward cross-coupling: F_ext[j,i] = (P[j]-gh1[i])*A[i,j] + c1[j]*c2[i]
__global__ void diff_wb_build_F_ext_backward_kernel(
    double* __restrict__ F_ext, const double* __restrict__ F_all, long long f_all_stride,
    const double* __restrict__ P_diag, const double* __restrict__ gh1, const double* __restrict__ c1, const double* __restrict__ c2,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    const double* F_b = F_all + (f_all_stride > 0 ? b * f_all_stride : 0);
    double* Fe_b = F_ext + b * n * k_ext;
    const double* P_b = P_diag + b * n;
    const double* gh1_b = gh1 + b * m;
    const double* c1_b = c1 + b * n;
    const double* c2_b = c2 + b * m;

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double pj = P_b[j];
        double c1j = c1_b[j];
        for (int64_t i = 0; i < k; ++i)
            Fe_b[j * k_ext + i] = (pj + 1.0 - gh1_b[i]) * F_b[j * k_total + i] + c1j * c2_b[i];
        for (int64_t d = 0; d < k_d; ++d) {
            int64_t ni = dense_nonneg_idx[d];
            Fe_b[j * k_ext + k + d] = (pj + 1.0 - gh1_b[k + ni]) * F_b[j * k_total + k + d] + c1j * c2_b[k + ni];
        }
        Fe_b[j * k_ext + k_total] = c1j;
    }
}

void launch_diff_wb_build_F_ext_backward(
    double* F_ext, const double* F_all, long long f_all_stride,
    const double* P_diag, const double* gh1, const double* c1, const double* c2,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext, cudaStream_t stream)
{
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_build_F_ext_backward_kernel, (int)batch, threads, 0, stream,
        F_ext, F_all, f_all_stride, P_diag, gh1, c1, c2,
        dense_nonneg_idx, batch, n, m, k, k_d, k_total, k_ext);
}

// Tau-elimination diagonal parts
__global__ void diff_wb_tau_elim_diag_kernel(
    double* __restrict__ v_x, double* __restrict__ v_z, double* __restrict__ sigma,
    const double* __restrict__ P_diag, const double* __restrict__ q, const double* __restrict__ b,
    const double* __restrict__ c1, const double* __restrict__ c2, const double* __restrict__ c3,
    const double* __restrict__ Lambda, const double* __restrict__ g, const double* __restrict__ gh1,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    double eps,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    double* vx_b = v_x + bi * n;
    double* vz_b = v_z + bi * m;
    const double* P_b = P_diag + bi * n;
    const double* q_b = q + bi * n;
    const double* b_b = b + bi * m;
    const double* c1_b = c1 + bi * n;
    const double* c2_b = c2 + bi * m;
    double c3_b = c3[bi];
    const double* L_b = Lambda + bi * m;
    const double* gh1_b = gh1 + bi * m;
    const double* As_b = A_s_vals + bi * n_sparse;

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double val = P_b[j] * q_b[j] + c1_b[j] * c3_b;
        for (int64_t p = col_offsets[j]; p < col_offsets[j + 1]; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            val -= L_b[k + ni] * a * b_b[k + ni];
        }
        vx_b[j] = val;
    }

    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        vz_b[i] = c2_b[i] * c3_b - (1.0 - gh1_b[i]) * b_b[i];
    }

    __syncthreads();

    // Add sparse nonneg A*q contribution: v_z[k+ni] += a * q[col]
    for (int64_t sr = threadIdx.x; sr < n_sparse; sr += blockDim.x) {
        int64_t ni = sparse_nonneg_idx[sr];
        int64_t col = sparse_col[sr];
        double a = As_b[sr];
        atomicAdd(&vz_b[k + ni], a * q_b[col]);
    }

    // Parallel reduction for sigma = c3^2 + eps + sum(q^2) + sum(Lambda*b^2)
    __shared__ double sdata[256];
    double local_sum = 0.0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x)
        local_sum += q_b[j] * q_b[j];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x)
        local_sum += L_b[i] * b_b[i] * b_b[i];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sigma[bi] = sdata[0] + c3_b * c3_b + eps;
    }
}

void launch_diff_wb_tau_elim_diag(
    double* v_x, double* v_z, double* sigma,
    const double* P_diag, const double* q, const double* b,
    const double* c1, const double* c2, const double* c3,
    const double* Lambda, const double* g, const double* gh1,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    double eps,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)(n + m), 256);
    if (threads == 0) threads = 1;
    // Round up to power of 2 for correct parallel reduction
    { int t = 1; while (t < threads) t <<= 1; threads = std::min(t, 256); }
    MOREAU_KERNEL_LAUNCH(diff_wb_tau_elim_diag_kernel, (int)batch, threads, 0, stream,
        v_x, v_z, sigma, P_diag, q, b, c1, c2, c3, Lambda, g, gh1,
        A_s_vals, sparse_nonneg_idx, sparse_col, col_offsets, col_sparse_rows,
        eps, batch, n, m, k, n_nonneg, n_sparse);
}

// s-elimination RHS
__global__ void diff_wb_s_elim_rhs_kernel(
    const double* __restrict__ rhs_bar, double* __restrict__ rt_x, double* __restrict__ rt_z, double* __restrict__ rt_tau,
    const double* __restrict__ g, const double* __restrict__ gh1, const double* __restrict__ b,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    int64_t jdim = n + 2 * m + 1;
    const double* rhs_b = rhs_bar + bi * jdim;
    const double* r_x = rhs_b;
    const double* r_z = rhs_b + n;
    const double* r_s = rhs_b + n + m;
    double r_tau = rhs_b[jdim - 1];
    double* rtx_b = rt_x + bi * n;
    double* rtz_b = rt_z + bi * m;
    const double* g_b = g + bi * m;
    const double* gh1_b = gh1 + bi * m;
    const double* b_b = b + bi * m;
    const double* As_b = A_s_vals + bi * n_sparse;

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double val = r_x[j];
        for (int64_t p = col_offsets[j]; p < col_offsets[j + 1]; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            val += a * g_b[k + ni] * r_s[k + ni];
        }
        rtx_b[j] = val;
    }

    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        rtz_b[i] = r_z[i] + gh1_b[i] * r_s[i];
    }

    // Parallel reduction for rt_tau = r_tau - sum(b * g * r_s)
    __shared__ double sdata[256];
    double local_sum = 0.0;
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x)
        local_sum += b_b[i] * g_b[i] * r_s[i];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rt_tau[bi] = r_tau - sdata[0];
    }
}

void launch_diff_wb_s_elim_rhs(
    const double* rhs_bar, double* rt_x, double* rt_z, double* rt_tau,
    const double* g, const double* gh1, const double* b,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)(n + m), 256);
    if (threads == 0) threads = 1;
    { int t = 1; while (t < threads) t <<= 1; threads = std::min(t, 256); }
    MOREAU_KERNEL_LAUNCH(diff_wb_s_elim_rhs_kernel, (int)batch, threads, 0, stream,
        rhs_bar, rt_x, rt_z, rt_tau, g, gh1, b,
        A_s_vals, sparse_nonneg_idx, sparse_col, col_offsets, col_sparse_rows,
        batch, n, m, k, n_nonneg, n_sparse);
}

// Recover y_s
__global__ void diff_wb_recover_y_s_kernel(
    double* __restrict__ y_s,
    const double* __restrict__ rhs_bar, const double* __restrict__ y_x,
    const double* __restrict__ y_z, const double* __restrict__ y_tau,
    const double* __restrict__ g, const double* __restrict__ gh1, const double* __restrict__ b,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    int64_t jdim = n + 2 * m + 1;
    const double* r_s = rhs_bar + bi * jdim + n + m;
    double* ys_b = y_s + bi * m;
    const double* yx_b = y_x + bi * n;
    const double* yz_b = y_z + bi * m;
    double ytau = y_tau[bi];
    const double* g_b = g + bi * m;
    const double* gh1_b = gh1 + bi * m;
    const double* b_b = b + bi * m;

    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        ys_b[i] = g_b[i] * r_s[i] + gh1_b[i] * yz_b[i] - g_b[i] * b_b[i] * ytau;
    }

    __syncthreads();

    for (int64_t sr = threadIdx.x; sr < n_sparse; sr += blockDim.x) {
        int64_t ni = sparse_nonneg_idx[sr];
        int64_t col = sparse_col[sr];
        double a = A_s_vals[bi * n_sparse + sr];
        atomicAdd(&ys_b[k + ni], g_b[k + ni] * a * yx_b[col]);
    }
}

void launch_diff_wb_recover_y_s(
    double* y_s,
    const double* rhs_bar, const double* y_x,
    const double* y_z, const double* y_tau,
    const double* g, const double* gh1, const double* b,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)m, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_recover_y_s_kernel, (int)batch, threads, 0, stream,
        y_s, rhs_bar, y_x, y_z, y_tau, g, gh1, b,
        A_s_vals, sparse_nonneg_idx, sparse_col,
        batch, n, m, k, n_nonneg, n_sparse);
}

// J-matvec diagonal + sparse parts
__global__ void diff_wb_J_matvec_diag_kernel(
    double* __restrict__ y_out,
    const double* __restrict__ w_x, const double* __restrict__ w_z, const double* __restrict__ w_s,
    const double* __restrict__ w_tau,
    const double* __restrict__ P_diag, const double* __restrict__ H_nonneg,
    const double* __restrict__ q, const double* __restrict__ b,
    const double* __restrict__ c1, const double* __restrict__ c2, const double* __restrict__ c3,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    int64_t jdim = n + 2 * m + 1;
    double* out = y_out + bi * jdim;
    const double* wx = w_x + bi * n;
    const double* wz = w_z + bi * m;
    const double* ws = w_s + bi * m;
    double wtau = w_tau[bi];
    const double* P_b = P_diag + bi * n;
    const double* H_b = H_nonneg + bi * n_nonneg;
    const double* q_b = q + bi * n;
    const double* b_b = b + bi * m;
    const double* c1_b = c1 + bi * n;
    const double* c2_b = c2 + bi * m;
    double c3_b = c3[bi];
    const double* As_b = A_s_vals + bi * n_sparse;

    // Row 0: y_out[0:n] = P*w_x + A'_sparse*w_z + q*w_tau
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double val = P_b[j] * wx[j] + q_b[j] * wtau;
        for (int64_t p = col_offsets[j]; p < col_offsets[j + 1]; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            val += a * wz[k + ni];
        }
        out[j] = val;
    }

    // Row 1: y_out[n:n+m] = w_z - w_s - b*w_tau + A_sparse*w_x
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[n + i] = wz[i] - ws[i] - b_b[i] * wtau;
    }
    for (int64_t sr = threadIdx.x; sr < n_sparse; sr += blockDim.x) {
        int64_t ni = sparse_nonneg_idx[sr];
        int64_t col = sparse_col[sr];
        double a = As_b[sr];
        atomicAdd(&out[n + k + ni], a * wx[col]);
    }

    // Row 2: y_out[n+m:n+2m] = w_z - H*w_s
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        double h = (i < k) ? 1.0 : H_b[i - k];
        out[n + m + i] = wz[i] - h * ws[i];
    }

    // Row 3: y_out[jdim-1] = c1'*w_x + c2'*w_z + c3*w_tau
    // Parallel reduction for the dot products
    __shared__ double sdata[256];
    double local_sum = 0.0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x)
        local_sum += c1_b[j] * wx[j];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x)
        local_sum += c2_b[i] * wz[i];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[jdim - 1] = sdata[0] + c3_b * wtau;
    }
}

void launch_diff_wb_J_matvec_diag(
    double* y_out,
    const double* w_x, const double* w_z, const double* w_s,
    const double* w_tau,
    const double* P_diag, const double* H_nonneg,
    const double* q, const double* b,
    const double* c1, const double* c2, const double* c3,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)(n + m), 256);
    if (threads == 0) threads = 1;
    // Round up to power of 2 for correct parallel reduction
    { int t = 1; while (t < threads) t <<= 1; threads = std::min(t, 256); }
    MOREAU_KERNEL_LAUNCH(diff_wb_J_matvec_diag_kernel, (int)batch, threads, 0, stream,
        y_out, w_x, w_z, w_s, w_tau, P_diag, H_nonneg, q, b, c1, c2, c3,
        A_s_vals, sparse_nonneg_idx, sparse_col, col_offsets, col_sparse_rows,
        batch, n, m, k, n_nonneg, n_sparse);
}

// Backward RHS adjustment for sparse nonneg elimination
__global__ void diff_wb_rhs_adj_sparse_kernel(
    double* __restrict__ rhs_x_adj, const double* __restrict__ rhs_x,
    const double* __restrict__ D_z_inv, const double* __restrict__ rhs_lam,
    const double* __restrict__ P_diag, const double* __restrict__ A_s_vals,
    const double* __restrict__ H_nonneg, const double* __restrict__ c1, const double* __restrict__ c2,
    double eps,
    const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    const int64_t* __restrict__ sparse_col,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* out = rhs_x_adj + b * n;
    const double* rx = rhs_x + b * n;
    const double* Dzi = D_z_inv + b * n_nonneg;
    const double* lam = rhs_lam + b * n_nonneg;
    const double* P_b = P_diag + b * n;
    const double* As_b = A_s_vals + b * n_sparse;
    const double* H_b = H_nonneg + b * n_nonneg;
    const double* c1_b = c1 + b * n;
    const double* c2_b = c2 + b * m;

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double val = rx[j];
        // Eliminate λ from the x RHS for column j via per-column Sherman-Morrison
        // (λ-block = diag(D_zp) + a a', D_zp = D_z − a²):
        //   M'[x,λ] L^{-1} rhs_λ = Σ cross·rhs_λ/D_zp
        //                          − (Σ cross·a/D_zp)(Σ a·rhs_λ/D_zp)/(1+Σ a²/D_zp).
        double t_cl = 0.0, t_ca = 0.0, t_al = 0.0, s_aa = 0.0;
        for (int64_t p = col_offsets[j]; p < col_offsets[j + 1]; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            double h = H_b[ni];
            double gh1_i = (1.0 + h) / (1.0 + h * h + eps);
            double dzi = Dzi[ni];
            double dzp_inv = dzi / (1.0 - a * a * dzi);   // 1/(D_z − a²)
            double cross = a * (P_b[j] + 1.0 - gh1_i) + c1_b[j] * c2_b[k + ni];
            t_cl += cross * dzp_inv * lam[ni];
            t_ca += cross * a * dzp_inv;
            t_al += a * dzp_inv * lam[ni];
            s_aa += a * a * dzp_inv;
        }
        val += t_cl - t_ca * t_al / (1.0 + s_aa);
        out[j] = val;
    }
}

void launch_diff_wb_rhs_adj_sparse(
    double* rhs_x_adj, const double* rhs_x,
    const double* D_z_inv, const double* rhs_lam,
    const double* P_diag, const double* A_s_vals,
    const double* H_nonneg, const double* c1, const double* c2,
    double eps,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    const int64_t* sparse_col,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_rhs_adj_sparse_kernel, (int)batch, threads, 0, stream,
        rhs_x_adj, rhs_x, D_z_inv, rhs_lam, P_diag, A_s_vals,
        H_nonneg, c1, c2, eps,
        sparse_nonneg_idx, col_offsets, col_sparse_rows, sparse_col,
        batch, n, m, k, n_nonneg, n_sparse);
}

// Backward dlam recovery for sparse nonneg
__global__ void diff_wb_dlam_sparse_kernel(
    double* __restrict__ dlam, const double* __restrict__ D_z_inv,
    const double* __restrict__ y_x, const double* __restrict__ y_nu, const double* __restrict__ y_tau,
    const double* __restrict__ rhs_lam,
    const double* __restrict__ P_diag, const double* __restrict__ A_s_vals,
    const double* __restrict__ H_nonneg, const double* __restrict__ c1, const double* __restrict__ c2,
    const double* __restrict__ v_z,
    const double* __restrict__ F_all, long long f_all_stride,
    const int64_t* __restrict__ dense_nonneg_idx,
    double eps,
    const int64_t* __restrict__ sparse_col, const int64_t* __restrict__ sparse_nonneg_idx,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    if (threadIdx.x != 0) return;
    (void)c1; (void)c2; (void)dense_nonneg_idx; (void)sparse_col; (void)k_d;
    const double* F_b = F_all + (f_all_stride > 0 ? b * f_all_stride : 0);
    const double* ynu = y_nu + b * k_ext;
    double ytau = y_tau[b];

    // Recover λ per column: L y_λ = r with L = diag(D_zp)+a a' (per-column block).
    // The ν residual M'[λ_p,ν]·y_ν = a_p·Fy_j factors (Fy_j column-only). Solve
    // via Sherman-Morrison: y_λ_p = r_p/D_zp − (a_p/D_zp)·(Σ_q a_q r_q/D_zp)/β.
    for (int64_t j = 0; j < n; ++j) {
        int64_t p0 = col_offsets[j], p1 = col_offsets[j + 1];
        if (p0 == p1) continue;
        double Fy = 0.0;
        for (int64_t l = 0; l < k_total; ++l) Fy += F_b[j * k_total + l] * ynu[l];
        double yxj = y_x[b * n + j];
        double Pj = P_diag[b * n + j];

        double s_ar = 0.0, s_aa = 0.0;
        for (int64_t p = p0; p < p1; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = A_s_vals[b * n_sparse + sr];
            double h = H_nonneg[b * n_nonneg + ni];
            double gh1_i = (1.0 + h) / (1.0 + h * h + eps);
            double dzi = D_z_inv[b * n_nonneg + ni];
            double dzp_inv = dzi / (1.0 - a * a * dzi);
            double vz_i = v_z[b * m + k + ni];
            double cross_x = a * (Pj + 1.0 - gh1_i);
            double r_p = rhs_lam[b * n_nonneg + ni] - cross_x * yxj - a * Fy - vz_i * ytau;
            s_ar += a * dzp_inv * r_p;
            s_aa += a * a * dzp_inv;
        }
        double beta = 1.0 + s_aa;
        for (int64_t p = p0; p < p1; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = A_s_vals[b * n_sparse + sr];
            double h = H_nonneg[b * n_nonneg + ni];
            double gh1_i = (1.0 + h) / (1.0 + h * h + eps);
            double dzi = D_z_inv[b * n_nonneg + ni];
            double dzp_inv = dzi / (1.0 - a * a * dzi);
            double vz_i = v_z[b * m + k + ni];
            double cross_x = a * (Pj + 1.0 - gh1_i);
            double r_p = rhs_lam[b * n_nonneg + ni] - cross_x * yxj - a * Fy - vz_i * ytau;
            dlam[b * n_nonneg + ni] = dzp_inv * r_p - dzp_inv * a * s_ar / beta;
        }
    }
}

void launch_diff_wb_dlam_sparse(
    double* dlam, const double* D_z_inv,
    const double* y_x, const double* y_nu, const double* y_tau,
    const double* rhs_lam,
    const double* P_diag, const double* A_s_vals,
    const double* H_nonneg, const double* c1, const double* c2,
    const double* v_z,
    const double* F_all, long long f_all_stride,
    const int64_t* dense_nonneg_idx,
    double eps,
    const int64_t* sparse_col, const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    if (batch == 0 || n_sparse == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_dlam_sparse_kernel, (int)batch, 1, 0, stream,
        dlam, D_z_inv, y_x, y_nu, y_tau, rhs_lam, P_diag, A_s_vals,
        H_nonneg, c1, c2, v_z,
        F_all, f_all_stride, dense_nonneg_idx,
        eps, sparse_col, sparse_nonneg_idx,
        col_offsets, col_sparse_rows,
        batch, n, m, k, k_d, k_total, k_ext, n_nonneg, n_sparse);
}

// Gather weighted from m-vector
__global__ void diff_wb_gather_weighted_kernel(
    double* __restrict__ out, const double* __restrict__ in, const double* __restrict__ weight,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* o = out + b * k_total;
    const double* i_b = in + b * m;
    const double* w_b = weight ? (weight + b * m) : nullptr;

    for (int64_t idx = threadIdx.x; idx < k_total; idx += blockDim.x) {
        int64_t src;
        if (idx < k) {
            src = idx;
        } else {
            src = k + dense_nonneg_idx[idx - k];
        }
        o[idx] = w_b ? (w_b[src] * i_b[src]) : i_b[src];
    }
}

void launch_diff_wb_gather_weighted(
    double* out, const double* in, const double* weight,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    cudaStream_t stream)
{
    int threads = std::min((int)k_total, 256);
    if (threads == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_gather_weighted_kernel, (int)batch, threads, 0, stream,
        out, in, weight, dense_nonneg_idx, batch, m, k, k_d, k_total);
}

// Scatter-add weighted from k_total-vector into m-vector
__global__ void diff_wb_scatter_add_weighted_kernel(
    double* __restrict__ out, const double* __restrict__ in, const double* __restrict__ weight,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* o = out + b * m;
    const double* i_b = in + b * k_total;
    const double* w_b = weight ? (weight + b * m) : nullptr;

    for (int64_t idx = threadIdx.x; idx < k_total; idx += blockDim.x) {
        int64_t dst;
        if (idx < k) {
            dst = idx;
        } else {
            dst = k + dense_nonneg_idx[idx - k];
        }
        double val = w_b ? (w_b[dst] * i_b[idx]) : i_b[idx];
        o[dst] += val;
    }
}

void launch_diff_wb_scatter_add_weighted(
    double* out, const double* in, const double* weight,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    cudaStream_t stream)
{
    int threads = std::min((int)k_total, 256);
    if (threads == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_scatter_add_weighted_kernel, (int)batch, threads, 0, stream,
        out, in, weight, dense_nonneg_idx, batch, m, k, k_d, k_total);
}

// Unpack rhat_z into rhs_nu (k_ext) and rhs_lam (n_nonneg)
__global__ void diff_wb_unpack_z_kernel(
    double* __restrict__ rhs_nu, double* __restrict__ rhs_lam,
    const double* __restrict__ rhat_z,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    int64_t n_nonneg)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* nu = rhs_nu + b * k_ext;
    double* lam = rhs_lam + b * n_nonneg;
    const double* z = rhat_z + b * m;

    // Zero cone entries
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        nu[i] = z[i];
    }
    // Dense nonneg entries
    for (int64_t d = threadIdx.x; d < k_d; d += blockDim.x) {
        nu[k + d] = z[k + dense_nonneg_idx[d]];
    }
    // c1 row entry = 0
    if (threadIdx.x == 0) {
        nu[k_total] = 0.0;
    }
    // All nonneg entries
    for (int64_t i = threadIdx.x; i < n_nonneg; i += blockDim.x) {
        lam[i] = z[k + i];
    }
}

void launch_diff_wb_unpack_z(
    double* rhs_nu, double* rhs_lam,
    const double* rhat_z,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    int64_t n_nonneg,
    cudaStream_t stream)
{
    int threads = std::min((int)m, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_unpack_z_kernel, (int)batch, threads, 0, stream,
        rhs_nu, rhs_lam, rhat_z, dense_nonneg_idx,
        batch, m, k, k_d, k_total, k_ext, n_nonneg);
}

// Scatter zero-cone y_z from gemv_k
__global__ void diff_wb_scatter_y_z_zero_kernel(
    double* __restrict__ y_z, const double* __restrict__ gemv_k,
    int64_t batch, int64_t m, int64_t k, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* yz = y_z + b * m;
    const double* gk = gemv_k + b * k_ext;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        yz[i] = gk[i];
    }
}

void launch_diff_wb_scatter_y_z_zero(
    double* y_z, const double* gemv_k,
    int64_t batch, int64_t m, int64_t k, int64_t k_ext,
    cudaStream_t stream)
{
    int threads = std::min((int)k, 256);
    if (threads == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_scatter_y_z_zero_kernel, (int)batch, threads, 0, stream,
        y_z, gemv_k, batch, m, k, k_ext);
}

// Strided axpy: dst[b*dst_stride + dst_offset + i] += src[b*src_stride + i] for i < count
__global__ void diff_wb_strided_axpy_kernel(
    double* __restrict__ dst, const double* __restrict__ src,
    int64_t batch, int64_t count, int64_t dst_stride, int64_t dst_offset, int64_t src_stride)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* d = dst + b * dst_stride + dst_offset;
    const double* s = src + b * src_stride;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        d[i] += s[i];
    }
}

void launch_diff_wb_strided_axpy(
    double* dst, const double* src,
    int64_t batch, int64_t count, int64_t dst_stride, int64_t dst_offset, int64_t src_stride,
    cudaStream_t stream)
{
    if (count == 0 || batch == 0) return;
    int threads = std::min((int)count, 256);
    MOREAU_KERNEL_LAUNCH(diff_wb_strided_axpy_kernel, (int)batch, threads, 0, stream,
        dst, src, batch, count, dst_stride, dst_offset, src_stride);
}

// Extract from strided source, multiply element-wise, store contiguous:
// out[b*m + i] = a[b*m + i] * src[b*src_stride + src_offset + i]
__global__ void diff_wb_extract_mul_kernel(
    double* __restrict__ out, const double* __restrict__ a, const double* __restrict__ src,
    int64_t batch, int64_t m, int64_t src_stride, int64_t src_offset)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* o = out + b * m;
    const double* a_b = a + b * m;
    const double* s = src + b * src_stride + src_offset;
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        o[i] = a_b[i] * s[i];
    }
}

void launch_diff_wb_extract_mul(
    double* out, const double* a, const double* src,
    int64_t batch, int64_t m, int64_t src_stride, int64_t src_offset,
    cudaStream_t stream)
{
    if (m == 0 || batch == 0) return;
    int threads = std::min((int)m, 256);
    MOREAU_KERNEL_LAUNCH(diff_wb_extract_mul_kernel, (int)batch, threads, 0, stream,
        out, a, src, batch, m, src_stride, src_offset);
}

// Pack dlam_nonneg (B, n_nonneg stride) into y_z (B, m stride) at offset k:
// y_z[b*m + k + i] = src[b*n_nonneg + i] for i < n_nonneg
__global__ void diff_wb_pack_dlam_to_yz_kernel(
    double* __restrict__ y_z, const double* __restrict__ src,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* dst = y_z + b * m + k;
    const double* s = src + b * n_nonneg;
    for (int64_t i = threadIdx.x; i < n_nonneg; i += blockDim.x) {
        dst[i] = s[i];
    }
}

void launch_diff_wb_pack_dlam_to_yz(
    double* y_z, const double* src,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg,
    cudaStream_t stream)
{
    if (n_nonneg == 0 || batch == 0) return;
    int threads = std::min((int)n_nonneg, 256);
    MOREAU_KERNEL_LAUNCH(diff_wb_pack_dlam_to_yz_kernel, (int)batch, threads, 0, stream,
        y_z, src, batch, m, k, n_nonneg);
}

// ============================================================================
// New kernels for corrected backward pass (inner Woodbury + full Schur)
// ============================================================================

// Build G = [sqrt(Λ_zd) * F_all columns, c1]
// G[:,i] = sqrt(Λ[i]) * F_all[:,i] for i < k_total; G[:,k_total] = c1
__global__ void diff_wb_build_G_kernel(
    double* __restrict__ G, const double* __restrict__ F_all, long long f_all_stride,
    const double* __restrict__ Lambda, const double* __restrict__ c1,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    const double* F_b = F_all + (f_all_stride > 0 ? b * f_all_stride : 0);
    double* G_b = G + b * n * k_ext;
    const double* L_b = Lambda + b * m;
    const double* c1_b = c1 + b * n;

    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        for (int64_t i = 0; i < k; ++i) {
            double sqrtL = sqrt(L_b[i]);
            G_b[j * k_ext + i] = sqrtL * F_b[j * k_total + i];
        }
        for (int64_t d = 0; d < k_d; ++d) {
            int64_t ni = dense_nonneg_idx[d];
            double sqrtL = sqrt(L_b[k + ni]);
            G_b[j * k_ext + k + d] = sqrtL * F_b[j * k_total + k + d];
        }
        G_b[j * k_ext + k_total] = c1_b[j];
    }
}

void launch_diff_wb_build_G(
    double* G, const double* F_all, long long f_all_stride,
    const double* Lambda, const double* c1,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext, cudaStream_t stream)
{
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_build_G_kernel, (int)batch, threads, 0, stream,
        G, F_all, f_all_stride, Lambda, c1, dense_nonneg_idx,
        batch, n, m, k, k_d, k_total, k_ext);
}

// Add diagonal corrections + c2*c2' outer product to B_full (k_ext × k_ext).
// B is initialized to F_all'F_all (k_total×k_total block) via GEMM before this call.
// This kernel adds:
//   B[i,i] += 2+eps-(1+h)^2/(1+h^2+eps) for zero/dense rows
//   B[i,j] += c2_zd[i]*c2_zd[j] for i,j < k_total
//   B[k_total, 0:k_total] = v_nu[0:k_total]
//   B[0:k_total, k_total] = v_nu[0:k_total]
//   B[k_total, k_total] = sigma
__global__ void diff_wb_build_B_corrections_kernel(
    double* __restrict__ B, const double* __restrict__ v_nu, const double* __restrict__ sigma,
    const double* __restrict__ H_nonneg, const double* __restrict__ c2,
    const int64_t* __restrict__ dense_nonneg_idx, double eps,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* B_b = B + b * k_ext * k_ext;
    const double* H_b = H_nonneg + b * (m - k);
    const double* c2_b = c2 + b * m;
    const double* vnu_b = v_nu + b * k_ext;
    double sig = sigma[b];

    // Each thread handles a row i of the k_ext × k_ext matrix
    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x) {
        // Diagonal correction for zero/dense rows
        if (i < k) {
            B_b[i * k_ext + i] += 2.0 + eps - 4.0 / (2.0 + eps);
        } else if (i < k_total) {
            int64_t ni = dense_nonneg_idx[i - k];
            double h = H_b[ni];
            B_b[i * k_ext + i] += 2.0 + eps - (1.0 + h) * (1.0 + h) / (1.0 + h * h + eps);
        }

        // c2*c2' outer product contribution (only for i,j < k_total)
        if (i < k_total) {
            double c2_i;
            if (i < k) c2_i = c2_b[i];
            else c2_i = c2_b[k + dense_nonneg_idx[i - k]];

            for (int64_t j = 0; j < k_total; ++j) {
                double c2_j;
                if (j < k) c2_j = c2_b[j];
                else c2_j = c2_b[k + dense_nonneg_idx[j - k]];
                B_b[i * k_ext + j] += c2_i * c2_j;
            }
        }

        // Tau row and column
        if (i < k_total) {
            B_b[i * k_ext + k_total] = vnu_b[i];
            B_b[k_total * k_ext + i] = vnu_b[i];
        }
        if (i == k_total) {
            B_b[k_total * k_ext + k_total] = sig;
        }
    }
}

void launch_diff_wb_build_B_corrections(
    double* B, const double* v_nu, const double* sigma,
    const double* H_nonneg, const double* c2,
    const int64_t* dense_nonneg_idx, double eps,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    cudaStream_t stream)
{
    int threads = std::min((int)k_ext, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_build_B_corrections_kernel, (int)batch, threads, 0, stream,
        B, v_nu, sigma, H_nonneg, c2, dense_nonneg_idx, eps,
        batch, m, k, k_d, k_total, k_ext);
}

// Add identity to diagonal: A[b, i, i] += 1.0 for i < k_ext
__global__ void diff_wb_add_identity_kernel(
    double* __restrict__ A, int64_t batch, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* A_b = A + b * k_ext * k_ext;
    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x) {
        A_b[i * k_ext + i] += 1.0;
    }
}

void launch_diff_wb_add_identity(
    double* A, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    int threads = std::min((int)k_ext, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_add_identity_kernel, (int)batch, threads, 0, stream, A, batch, k_ext);
}

// Add scaled identity: A[i,i] += scale for each batch
__global__ void diff_wb_add_identity_scaled_kernel(
    double* __restrict__ A, double scale, int64_t batch, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* A_b = A + b * k_ext * k_ext;
    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x) {
        A_b[i * k_ext + i] += scale;
    }
}

void launch_diff_wb_add_identity_scaled(
    double* A, double scale, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    int threads = std::min((int)k_ext, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_add_identity_scaled_kernel, (int)batch, threads, 0, stream, A, scale, batch, k_ext);
}

static constexpr int DIFF_WB_WARP_SIZE = 32;
static constexpr int DIFF_WB_WARPS_PER_BLOCK = 4;
static constexpr int DIFF_WB_WARP_LDL_MAX = 32;
static constexpr int DIFF_WB_BLOCK_LDL_MAX = 64;

__device__ __forceinline__ double diff_wb_warp_sum(double v)
{
    for (int offset = DIFF_WB_WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    return v;
}

// Warp-cooperative LDL^T factorization for very small dense systems.
// One warp handles one batch item, parallelizing the row updates of each column.
__global__ void diff_wb_ldlt_factor_warp_kernel(
    double* __restrict__ S, int64_t batch, int64_t k_ext)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / DIFF_WB_WARP_SIZE;
    int lane = threadIdx.x % DIFF_WB_WARP_SIZE;
    if (warp_id >= batch) return;

    double* A = S + (int64_t)warp_id * k_ext * k_ext;

    for (int64_t j = 0; j < k_ext; ++j) {
        double dj_local = 0.0;
        if (lane < j) {
            double ljs = A[lane * k_ext + j];
            double ds = A[lane * k_ext + lane];
            dj_local = ljs * ljs * ds;
        }
        double dj_corr = diff_wb_warp_sum(dj_local);
        if (lane == 0) {
            A[j * k_ext + j] -= dj_corr;
        }
        double dj = __shfl_sync(0xFFFFFFFF, A[j * k_ext + j], 0);
        if (dj == 0.0) continue;
        double inv_dj = 1.0 / dj;

        int64_t i = j + 1 + lane;
        if (i < k_ext) {
            double aij = A[j * k_ext + i];
            for (int64_t s = 0; s < j; ++s) {
                double lis = A[s * k_ext + i];
                double ljs = A[s * k_ext + j];
                double ds = A[s * k_ext + s];
                aij -= lis * ljs * ds;
            }
            A[j * k_ext + i] = aij * inv_dj;
        }
        __syncwarp();
    }
}

// Warp-cooperative single-RHS LDL^T solve.
__global__ void diff_wb_ldlt_solve_warp_kernel(
    const double* __restrict__ S, double* __restrict__ b, int64_t batch, int64_t k_ext)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / DIFF_WB_WARP_SIZE;
    int lane = threadIdx.x % DIFF_WB_WARP_SIZE;
    if (warp_id >= batch) return;

    const double* A = S + (int64_t)warp_id * k_ext * k_ext;
    double* x = b + (int64_t)warp_id * k_ext;

    for (int64_t i = 0; i < k_ext; ++i) {
        double sum_local = 0.0;
        if (lane < i)
            sum_local = A[lane * k_ext + i] * x[lane];
        double sum = diff_wb_warp_sum(sum_local);
        if (lane == 0)
            x[i] -= sum;
        __syncwarp();
    }

    if (lane < k_ext) {
        double di = A[lane * k_ext + lane];
        x[lane] = (di != 0.0) ? x[lane] / di : 0.0;
    }
    __syncwarp();

    for (int64_t ii = k_ext; ii > 0; --ii) {
        int64_t i = ii - 1;
        double sum_local = 0.0;
        if (lane > i && lane < k_ext)
            sum_local = A[i * k_ext + lane] * x[lane];
        double sum = diff_wb_warp_sum(sum_local);
        if (lane == 0)
            x[i] -= sum;
        __syncwarp();
    }
}

// Warp-cooperative multi-RHS solve.
// Each warp handles one batch item and processes RHS columns in tiles of 32.
__global__ void diff_wb_ldlt_solve_mrhs_warp_kernel(
    const double* __restrict__ S, double* __restrict__ B, int64_t batch, int64_t k_ext, int64_t nrhs)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / DIFF_WB_WARP_SIZE;
    int lane = threadIdx.x % DIFF_WB_WARP_SIZE;
    int local_warp = threadIdx.x / DIFF_WB_WARP_SIZE;
    if (warp_id >= batch) return;

    const double* A = S + (int64_t)warp_id * k_ext * k_ext;
    double* B_b = B + (int64_t)warp_id * k_ext * nrhs;

    extern __shared__ double smem[];
    double* tile = smem + (size_t)local_warp * DIFF_WB_WARP_LDL_MAX * DIFF_WB_WARP_LDL_MAX;

    for (int64_t col0 = 0; col0 < nrhs; col0 += DIFF_WB_WARP_SIZE) {
        int64_t tile_rhs = (nrhs - col0 < DIFF_WB_WARP_SIZE) ? (nrhs - col0) : DIFF_WB_WARP_SIZE;

        if (lane < tile_rhs) {
            for (int64_t i = 0; i < k_ext; ++i)
                tile[i * DIFF_WB_WARP_SIZE + lane] = B_b[(col0 + lane) * k_ext + i];
        }
        __syncwarp();

        for (int64_t i = 0; i < k_ext; ++i) {
            if (lane < tile_rhs) {
                double val = tile[i * DIFF_WB_WARP_SIZE + lane];
                for (int64_t j = 0; j < i; ++j)
                    val -= A[j * k_ext + i] * tile[j * DIFF_WB_WARP_SIZE + lane];
                tile[i * DIFF_WB_WARP_SIZE + lane] = val;
            }
            __syncwarp();
        }

        if (lane < tile_rhs) {
            for (int64_t i = 0; i < k_ext; ++i) {
                double di = A[i * k_ext + i];
                double& val = tile[i * DIFF_WB_WARP_SIZE + lane];
                val = (di != 0.0) ? val / di : 0.0;
            }
        }
        __syncwarp();

        if (lane < tile_rhs) {
            for (int64_t ii = k_ext; ii > 0; --ii) {
                int64_t i = ii - 1;
                double val = tile[i * DIFF_WB_WARP_SIZE + lane];
                for (int64_t j = i + 1; j < k_ext; ++j)
                    val -= A[i * k_ext + j] * tile[j * DIFF_WB_WARP_SIZE + lane];
                tile[i * DIFF_WB_WARP_SIZE + lane] = val;
            }
        }
        __syncwarp();

        if (lane < tile_rhs) {
            for (int64_t i = 0; i < k_ext; ++i)
                B_b[(col0 + lane) * k_ext + i] = tile[i * DIFF_WB_WARP_SIZE + lane];
        }
        __syncwarp();
    }
}

// Block-cooperative LDL^T factorization for medium-size dense systems.
// One CUDA block handles one batch item using shared memory for the matrix.
__global__ void diff_wb_ldlt_factor_block_kernel(
    double* __restrict__ S, int64_t batch, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ double smem[];
    double* A = smem;
    double* S_b = S + b * k_ext * k_ext;
    __shared__ double diag_accum;

    for (int64_t idx = threadIdx.x; idx < k_ext * k_ext; idx += blockDim.x)
        A[idx] = S_b[idx];
    __syncthreads();

    for (int64_t j = 0; j < k_ext; ++j) {
        if (threadIdx.x == 0) diag_accum = 0.0;
        __syncthreads();
        double dj_local = 0.0;
        for (int64_t s = threadIdx.x; s < j; s += blockDim.x) {
            double ljs = A[s * k_ext + j];
            double ds = A[s * k_ext + s];
            dj_local += ljs * ljs * ds;
        }
        double dj_sum = dj_local;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            dj_sum += __shfl_down_sync(0xFFFFFFFF, dj_sum, offset);
        if ((threadIdx.x & 31) == 0)
            atomicAdd(&diag_accum, dj_sum);
        __syncthreads();

        if (threadIdx.x == 0) {
            A[j * k_ext + j] -= diag_accum;
            diag_accum = 0.0;
        }
        __syncthreads();

        double dj = A[j * k_ext + j];
        if (dj == 0.0) continue;
        double inv_dj = 1.0 / dj;

        for (int64_t i = j + 1 + threadIdx.x; i < k_ext; i += blockDim.x) {
            double aij = A[j * k_ext + i];
            for (int64_t s = 0; s < j; ++s) {
                double lis = A[s * k_ext + i];
                double ljs = A[s * k_ext + j];
                double ds = A[s * k_ext + s];
                aij -= lis * ljs * ds;
            }
            A[j * k_ext + i] = aij * inv_dj;
        }
        __syncthreads();
    }

    for (int64_t idx = threadIdx.x; idx < k_ext * k_ext; idx += blockDim.x)
        S_b[idx] = A[idx];
}

// Block-cooperative single-RHS solve. Uses shared memory for the RHS vector.
__global__ void diff_wb_ldlt_solve_block_kernel(
    const double* __restrict__ S, double* __restrict__ b, int64_t batch, int64_t k_ext)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;

    extern __shared__ double smem[];
    double* x = smem;
    const double* A = S + bi * k_ext * k_ext;
    double* b_b = b + bi * k_ext;
    __shared__ double accum;

    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x)
        x[i] = b_b[i];
    __syncthreads();

    for (int64_t i = 0; i < k_ext; ++i) {
        if (threadIdx.x == 0) accum = 0.0;
        __syncthreads();
        double sum_local = 0.0;
        for (int64_t j = threadIdx.x; j < i; j += blockDim.x)
            sum_local += A[j * k_ext + i] * x[j];
        double sum = sum_local;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if ((threadIdx.x & 31) == 0)
            atomicAdd(&accum, sum);
        __syncthreads();

        if (threadIdx.x == 0) {
            x[i] -= accum;
            accum = 0.0;
        }
        __syncthreads();
    }

    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x) {
        double di = A[i * k_ext + i];
        x[i] = (di != 0.0) ? x[i] / di : 0.0;
    }
    __syncthreads();

    for (int64_t ii = k_ext; ii > 0; --ii) {
        int64_t i = ii - 1;
        if (threadIdx.x == 0) accum = 0.0;
        __syncthreads();
        double sum_local = 0.0;
        for (int64_t j = i + 1 + threadIdx.x; j < k_ext; j += blockDim.x)
            sum_local += A[i * k_ext + j] * x[j];
        double sum = sum_local;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if ((threadIdx.x & 31) == 0)
            atomicAdd(&accum, sum);
        __syncthreads();

        if (threadIdx.x == 0) {
            x[i] -= accum;
            accum = 0.0;
        }
        __syncthreads();
    }

    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x)
        b_b[i] = x[i];
}

// Block-level multi-RHS solve. Parallelize across RHS columns within the block.
__global__ void diff_wb_ldlt_solve_mrhs_block_kernel(
    const double* __restrict__ S, double* __restrict__ B, int64_t batch, int64_t k_ext, int64_t nrhs)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;

    const double* A = S + bi * k_ext * k_ext;
    double* B_b = B + bi * k_ext * nrhs;

    for (int64_t r = threadIdx.x; r < nrhs; r += blockDim.x) {
        double* x = B_b + r * k_ext;

        for (int64_t i = 0; i < k_ext; ++i)
            for (int64_t j = 0; j < i; ++j)
                x[i] -= A[j * k_ext + i] * x[j];

        for (int64_t i = 0; i < k_ext; ++i) {
            double di = A[i * k_ext + i];
            x[i] = (di != 0.0) ? x[i] / di : 0.0;
        }

        for (int64_t ii = k_ext; ii > 0; --ii) {
            int64_t i = ii - 1;
            for (int64_t j = i + 1; j < k_ext; ++j)
                x[i] -= A[i * k_ext + j] * x[j];
        }
    }
}

// LDL^T factorization for small symmetric matrices (one per batch).
// Standard algorithm: for j = 0..k-1:
//   D[j] = A[j][j] - sum_{s<j} L[j][s]^2 * D[s]
//   L[i][j] = (A[i][j] - sum_{s<j} L[i][s] * L[j][s] * D[s]) / D[j]  for i > j
// Stores L in strict lower triangle, D on diagonal.  Upper triangle untouched.
// Unlike Cholesky, negative D[j] does not produce NaN.
__global__ void diff_wb_ldlt_factor_kernel(
    double* __restrict__ S, int64_t batch, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* A = S + b * k_ext * k_ext;

    // Column-major: A(i,j) = A[j * k_ext + i]
    for (int64_t j = 0; j < k_ext; ++j) {
        // D[j] = A[j][j] - sum_{s<j} L[j][s]^2 * D[s]
        double dj = A[j * k_ext + j];
        for (int64_t s = 0; s < j; ++s) {
            double ljs = A[s * k_ext + j];  // L[j][s]
            double ds = A[s * k_ext + s];   // D[s]
            dj -= ljs * ljs * ds;
        }
        A[j * k_ext + j] = dj;  // store D[j]

        // L[i][j] for i > j
        if (dj != 0.0) {
            double inv_dj = 1.0 / dj;
            for (int64_t i = j + 1; i < k_ext; ++i) {
                double aij = A[j * k_ext + i];  // A[i][j]
                for (int64_t s = 0; s < j; ++s) {
                    double lis = A[s * k_ext + i];
                    double ljs = A[s * k_ext + j];
                    double ds = A[s * k_ext + s];
                    aij -= lis * ljs * ds;
                }
                A[j * k_ext + i] = aij * inv_dj;  // store L[i][j]
            }
        }
    }
}

void launch_diff_wb_ldlt_factor(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    switch (g_diff_wb_ldlt_variant) {
        case DiffWbLdltVariant::Scalar:
            return launch_diff_wb_ldlt_factor_scalar(S, batch, k_ext, stream);
        case DiffWbLdltVariant::Warp:
            return launch_diff_wb_ldlt_factor_warp(S, batch, k_ext, stream);
        case DiffWbLdltVariant::Block:
            return launch_diff_wb_ldlt_factor_block(S, batch, k_ext, stream);
        case DiffWbLdltVariant::Auto:
        default:
            break;
    }
    if (k_ext <= DIFF_WB_WARP_LDL_MAX) return launch_diff_wb_ldlt_factor_warp(S, batch, k_ext, stream);
    if (k_ext <= DIFF_WB_BLOCK_LDL_MAX) return launch_diff_wb_ldlt_factor_block(S, batch, k_ext, stream);
    return launch_diff_wb_ldlt_factor_scalar(S, batch, k_ext, stream);
}

void launch_diff_wb_ldlt_factor_scalar(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_factor_kernel, (int)batch, 1, 0, stream, S, batch, k_ext);
}

void launch_diff_wb_ldlt_factor_warp(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    int threads = DIFF_WB_WARPS_PER_BLOCK * DIFF_WB_WARP_SIZE;
    int blocks = (int)((batch + DIFF_WB_WARPS_PER_BLOCK - 1) / DIFF_WB_WARPS_PER_BLOCK);
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_factor_warp_kernel, blocks, threads, 0, stream, S, batch, k_ext);
}

void launch_diff_wb_ldlt_factor_block(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    int threads = std::min(256, std::max(32, (int)k_ext));
    size_t shmem = (size_t)k_ext * k_ext * sizeof(double);
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_factor_block_kernel, (int)batch, threads, shmem, stream, S, batch, k_ext);
}

// LDL^T solve: given L (unit lower tri, strict lower of S) and D (diagonal of S),
// solve L D L^T x = b.  b is overwritten with x.
// Steps: L y = b (forward sub), D z = y (diagonal), L^T x = z (back sub).
__global__ void diff_wb_ldlt_solve_kernel(
    const double* __restrict__ S, double* __restrict__ b, int64_t batch, int64_t k_ext)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    const double* A = S + bi * k_ext * k_ext;
    double* x = b + bi * k_ext;

    // Forward substitution: L y = b  (L is unit lower tri)
    for (int64_t i = 0; i < k_ext; ++i) {
        for (int64_t j = 0; j < i; ++j)
            x[i] -= A[j * k_ext + i] * x[j];  // L[i][j] * y[j]
    }

    // Diagonal solve: z = D^{-1} y
    for (int64_t i = 0; i < k_ext; ++i) {
        double di = A[i * k_ext + i];
        x[i] = (di != 0.0) ? x[i] / di : 0.0;
    }

    // Back substitution: L^T x = z  (L^T is unit upper tri)
    for (int64_t i = k_ext - 1; i >= 0; --i) {
        for (int64_t j = i + 1; j < k_ext; ++j)
            x[i] -= A[i * k_ext + j] * x[j];  // L[j][i] = A[i*k_ext+j]
    }
}

void launch_diff_wb_ldlt_solve(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    switch (g_diff_wb_ldlt_variant) {
        case DiffWbLdltVariant::Scalar:
            return launch_diff_wb_ldlt_solve_scalar(S, b, batch, k_ext, stream);
        case DiffWbLdltVariant::Warp:
            return launch_diff_wb_ldlt_solve_warp(S, b, batch, k_ext, stream);
        case DiffWbLdltVariant::Block:
            return launch_diff_wb_ldlt_solve_block(S, b, batch, k_ext, stream);
        case DiffWbLdltVariant::Auto:
        default:
            break;
    }
    if (k_ext <= DIFF_WB_WARP_LDL_MAX) return launch_diff_wb_ldlt_solve_warp(S, b, batch, k_ext, stream);
    if (k_ext <= DIFF_WB_BLOCK_LDL_MAX) return launch_diff_wb_ldlt_solve_block(S, b, batch, k_ext, stream);
    return launch_diff_wb_ldlt_solve_scalar(S, b, batch, k_ext, stream);
}

void launch_diff_wb_ldlt_solve_scalar(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_solve_kernel, (int)batch, 1, 0, stream, S, b, batch, k_ext);
}

void launch_diff_wb_ldlt_solve_warp(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    int threads = DIFF_WB_WARPS_PER_BLOCK * DIFF_WB_WARP_SIZE;
    int blocks = (int)((batch + DIFF_WB_WARPS_PER_BLOCK - 1) / DIFF_WB_WARPS_PER_BLOCK);
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_solve_warp_kernel, blocks, threads, 0, stream, S, b, batch, k_ext);
}

void launch_diff_wb_ldlt_solve_block(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    int threads = std::min(256, std::max(32, (int)k_ext));
    size_t shmem = (size_t)k_ext * sizeof(double);
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_solve_block_kernel, (int)batch, threads, shmem, stream, S, b, batch, k_ext);
}

// LDL^T solve for multiple RHS: solve each column of B independently.
// B is (k_ext × nrhs) column-major per batch, stride = k_ext * nrhs.
__global__ void diff_wb_ldlt_solve_mrhs_kernel(
    const double* __restrict__ S, double* __restrict__ B, int64_t batch, int64_t k_ext, int64_t nrhs)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    const double* A = S + bi * k_ext * k_ext;
    double* B_b = B + bi * k_ext * nrhs;

    // Solve each column independently
    for (int64_t r = 0; r < nrhs; ++r) {
        double* x = B_b + r * k_ext;

        // Forward substitution: L y = b
        for (int64_t i = 0; i < k_ext; ++i)
            for (int64_t j = 0; j < i; ++j)
                x[i] -= A[j * k_ext + i] * x[j];

        // Diagonal solve: z = D^{-1} y
        for (int64_t i = 0; i < k_ext; ++i) {
            double di = A[i * k_ext + i];
            x[i] = (di != 0.0) ? x[i] / di : 0.0;
        }

        // Back substitution: L^T x = z
        for (int64_t i = k_ext - 1; i >= 0; --i)
            for (int64_t j = i + 1; j < k_ext; ++j)
                x[i] -= A[i * k_ext + j] * x[j];
    }
}

void launch_diff_wb_ldlt_solve_mrhs(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    switch (g_diff_wb_ldlt_variant) {
        case DiffWbLdltVariant::Scalar:
            return launch_diff_wb_ldlt_solve_mrhs_scalar(S, B, batch, k_ext, nrhs, stream);
        case DiffWbLdltVariant::Warp:
            return launch_diff_wb_ldlt_solve_mrhs_warp(S, B, batch, k_ext, nrhs, stream);
        case DiffWbLdltVariant::Block:
            return launch_diff_wb_ldlt_solve_mrhs_block(S, B, batch, k_ext, nrhs, stream);
        case DiffWbLdltVariant::Auto:
        default:
            break;
    }
    if (k_ext <= DIFF_WB_WARP_LDL_MAX && nrhs <= DIFF_WB_WARP_SIZE)
        return launch_diff_wb_ldlt_solve_mrhs_warp(S, B, batch, k_ext, nrhs, stream);
    if (k_ext <= DIFF_WB_BLOCK_LDL_MAX)
        return launch_diff_wb_ldlt_solve_mrhs_block(S, B, batch, k_ext, nrhs, stream);
    return launch_diff_wb_ldlt_solve_mrhs_scalar(S, B, batch, k_ext, nrhs, stream);
}

void launch_diff_wb_ldlt_solve_mrhs_scalar(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_solve_mrhs_kernel, (int)batch, 1, 0, stream, S, B, batch, k_ext, nrhs);
}

void launch_diff_wb_ldlt_solve_mrhs_warp(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    int threads = DIFF_WB_WARPS_PER_BLOCK * DIFF_WB_WARP_SIZE;
    int blocks = (int)((batch + DIFF_WB_WARPS_PER_BLOCK - 1) / DIFF_WB_WARPS_PER_BLOCK);
    size_t shmem = (size_t)DIFF_WB_WARPS_PER_BLOCK *
                   DIFF_WB_WARP_LDL_MAX * DIFF_WB_WARP_LDL_MAX * sizeof(double);
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_solve_mrhs_warp_kernel, blocks, threads, shmem, stream,
        S, B, batch, k_ext, nrhs);
}

void launch_diff_wb_ldlt_solve_mrhs_block(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream)
{
    if (batch == 0 || k_ext == 0) return;
    int threads = std::min(256, std::max(32, (int)nrhs));
    MOREAU_KERNEL_LAUNCH(diff_wb_ldlt_solve_mrhs_block_kernel, (int)batch, threads, 0, stream,
        S, B, batch, k_ext, nrhs);
}

// Gather v_nu from v_z at zero/dense positions (k_total entries),
// then append sigma at position k_total.
__global__ void diff_wb_gather_v_nu_kernel(
    double* __restrict__ v_nu, const double* __restrict__ v_z, const double* __restrict__ sigma,
    const int64_t* __restrict__ dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* vn = v_nu + b * k_ext;
    const double* vz = v_z + b * m;

    for (int64_t i = threadIdx.x; i < k_ext; i += blockDim.x) {
        if (i < k)
            vn[i] = vz[i];
        else if (i < k_total)
            vn[i] = vz[k + dense_nonneg_idx[i - k]];
        else
            vn[i] = sigma[b];
    }
}

void launch_diff_wb_gather_v_nu(
    double* v_nu, const double* v_z, const double* sigma,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    cudaStream_t stream)
{
    int threads = std::min((int)k_ext, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_gather_v_nu_kernel, (int)batch, threads, 0, stream,
        v_nu, v_z, sigma, dense_nonneg_idx, batch, m, k, k_d, k_total, k_ext);
}

// Compute Sherman-Morrison denominator for λ-block:
// rho[b] = 1 + sum_{i ∈ λ} c2[k+i]^2 / D_z[i]
// where λ = all nonneg rows NOT in the dense set (i.e., sparse + empty nonneg).
// Dense nonneg are part of ν and excluded from the λ block.
__global__ void diff_wb_compute_rho_kernel(
    double* __restrict__ rho, const double* __restrict__ D_z_inv, const double* __restrict__ c2,
    const int64_t* __restrict__ dense_nonneg_idx, int64_t k_d,
    int64_t batch, int64_t m, int64_t k,
    int64_t n_nonneg)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    const double* Dzi_b = D_z_inv + b * n_nonneg;
    const double* c2_b = c2 + b * m;

    // Build a flag for dense nonneg (excluded from λ)
    // Since k_d is typically very small, check membership by linear scan
    __shared__ double sdata[256];
    double local_sum = 0.0;
    for (int64_t i = threadIdx.x; i < n_nonneg; i += blockDim.x) {
        // Check if this nonneg index is dense (part of ν)
        bool is_dense = false;
        for (int64_t d = 0; d < k_d; ++d) {
            if (dense_nonneg_idx[d] == i) { is_dense = true; break; }
        }
        if (!is_dense) {
            double c2i = c2_b[k + i];
            local_sum += c2i * c2i * Dzi_b[i];
        }
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rho[b] = 1.0 + sdata[0];
    }
}

void launch_diff_wb_compute_rho(
    double* rho, const double* D_z_inv, const double* c2,
    const int64_t* dense_nonneg_idx, int64_t k_d,
    int64_t batch, int64_t m, int64_t k,
    int64_t n_nonneg,
    cudaStream_t stream)
{
    if (n_nonneg == 0) return;
    int threads = std::min((int)n_nonneg, 256);
    if (threads == 0) threads = 1;
    int t = 1; while (t < threads) t <<= 1; threads = std::min(t, 256);
    MOREAU_KERNEL_LAUNCH(diff_wb_compute_rho_kernel, (int)batch, threads, 0, stream,
        rho, D_z_inv, c2, dense_nonneg_idx, k_d, batch, m, k, n_nonneg);
}

// Sparse nonneg z-elimination corrections with Sherman-Morrison.
// The λ-block is M'[λ,λ] = D_z + c2_λ c2_λ'. Its inverse via SM is:
//   (D_z + c2c2')^{-1} = D_z^{-1} - (1/ρ) D_z^{-1} c2 c2' D_z^{-1}
// This kernel applies the exact λ-elimination to B_full, F_ext, v_x, v_nu, sigma:
//   1. Subtract diagonal-inverse part: -= M'[·,λ] D_z^{-1} M'[λ,·]
//   2. Add back SM correction: += (1/ρ) u u'  where u = M'[·,λ] D_z^{-1} c2_λ
__global__ void diff_wb_sparse_z_elim_corrections_kernel(
    double* __restrict__ B_full, double* __restrict__ F_ext, double* __restrict__ v_x, double* __restrict__ v_nu, double* __restrict__ sigma,
    const double* __restrict__ F_all, long long f_all_stride,
    const double* __restrict__ D_z_inv, const double* __restrict__ rho, const double* __restrict__ v_z,
    const double* __restrict__ P_diag, const double* __restrict__ gh1, const double* __restrict__ c1, const double* __restrict__ c2,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ dense_nonneg_idx,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    // Single-thread: per-column work is sequential and k_ext is small. (The old
    // multi-thread path only served the now-dead c2≠0 branch.)
    if (threadIdx.x != 0) return;
    (void)rho; (void)c1; (void)c2; (void)dense_nonneg_idx; (void)k_d;

    double* B_b = B_full + bi * k_ext * k_ext;
    double* Fe_b = F_ext + bi * n * k_ext;
    double* vx_b = v_x + bi * n;
    double* vnu_b = v_nu + bi * k_ext;
    const double* F_b = F_all + (f_all_stride > 0 ? bi * f_all_stride : 0);
    const double* Dzi_b = D_z_inv + bi * n_nonneg;
    const double* vz_b = v_z + bi * m;
    const double* P_b = P_diag + bi * n;
    const double* gh1_b = gh1 + bi * m;
    const double* As_b = A_s_vals + bi * n_sparse;

    // Eliminate the sparse λ block PER COLUMN. The λ rows on a column form a
    // small block L = diag(D_zp) + a a' (D_zp = D_z − a²); off-diagonal a_p·a_q
    // arises when a column has several bounds. Inverse via Sherman-Morrison:
    //   −M'[·,λ] L^{-1} M'[λ,·] = −Σ_p M'[·,λ_p] M'[λ_p,·]/D_zp
    //                            + (Σ_p M'[·,λ_p]a_p/D_zp)(Σ_q a_q M'[λ_q,·]/D_zp)/β.
    // u_nu/u_x/u_tau are the per-column SM accumulators; β = 1 + Σ a²/D_zp.
    extern __shared__ double diff_wb_sparse_z_smem[];
    double* u_nu = diff_wb_sparse_z_smem;          // [k_total]
    double* nu_lambda = diff_wb_sparse_z_smem + k_total;  // [k_total]

    for (int64_t j = 0; j < n; ++j) {
        int64_t p0 = col_offsets[j], p1 = col_offsets[j + 1];
        if (p0 == p1) continue;  // no sparse λ on this column

        for (int64_t l = 0; l < k_total; ++l) u_nu[l] = 0.0;
        double u_tau = 0.0, u_x = 0.0, beta = 1.0;

        // Pass 1: diagonal (D_zp) subtract + accumulate the rank-1 vectors.
        for (int64_t p = p0; p < p1; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            double dzi = Dzi_b[ni];
            double dzp_inv = dzi / (1.0 - a * a * dzi);   // 1/(D_z − a²)
            double vz_i = vz_b[k + ni];
            double gh1_i = gh1_b[k + ni];
            for (int64_t l = 0; l < k_total; ++l)
                nu_lambda[l] = F_b[j * k_total + l] * a;
            double cross_col = a * (P_b[j] + 1.0 - gh1_i);

            double vz_dzp = vz_i * dzp_inv;
            for (int64_t l1 = 0; l1 < k_total; ++l1) {
                double w1 = nu_lambda[l1] * dzp_inv;
                for (int64_t l2 = 0; l2 < k_total; ++l2)
                    B_b[l1 * k_ext + l2] -= w1 * nu_lambda[l2];
                B_b[l1 * k_ext + k_total] -= w1 * vz_i;
                B_b[k_total * k_ext + l1] -= vz_dzp * nu_lambda[l1];
                vnu_b[l1] -= nu_lambda[l1] * vz_dzp;
            }
            B_b[k_total * k_ext + k_total] -= vz_i * vz_dzp;
            sigma[bi] -= vz_i * vz_dzp;

            double w = cross_col * dzp_inv;
            for (int64_t l = 0; l < k_total; ++l)
                Fe_b[j * k_ext + l] -= w * nu_lambda[l];
            vx_b[j] -= w * vz_i;

            for (int64_t l = 0; l < k_total; ++l)
                u_nu[l] += nu_lambda[l] * a * dzp_inv;
            u_tau += vz_i * a * dzp_inv;
            u_x += cross_col * a * dzp_inv;
            beta += a * a * dzp_inv;
        }

        // Pass 2: per-column Sherman-Morrison rank-1 add-back (/β).
        for (int64_t l1 = 0; l1 < k_total; ++l1) {
            double w1 = u_nu[l1] / beta;
            for (int64_t l2 = 0; l2 < k_total; ++l2)
                B_b[l1 * k_ext + l2] += w1 * u_nu[l2];
            B_b[l1 * k_ext + k_total] += w1 * u_tau;
            B_b[k_total * k_ext + l1] += u_tau * u_nu[l1] / beta;
            vnu_b[l1] += u_nu[l1] * u_tau / beta;
            Fe_b[j * k_ext + l1] += u_x * u_nu[l1] / beta;
        }
        B_b[k_total * k_ext + k_total] += u_tau * u_tau / beta;
        sigma[bi] += u_tau * u_tau / beta;
        vx_b[j] += u_x * u_tau / beta;
    }
}

void launch_diff_wb_sparse_z_elim_corrections(
    double* B_full, double* F_ext, double* v_x, double* v_nu, double* sigma,
    const double* F_all, long long f_all_stride,
    const double* D_z_inv, const double* rho, const double* v_z,
    const double* P_diag, const double* gh1, const double* c1, const double* c2,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* dense_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    if (n_sparse == 0) return;
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    // Dynamic shared memory: u_nu[k_total] + nu_lambda[k_total]
    // Woodbury requires k_total < n, and k_total is small for the regime this
    // solver is designed for (portfolio factor dim, box constraints), so the
    // shared-memory budget is not a concern in practice.
    size_t shmem_bytes = 2 * (size_t)k_total * sizeof(double);
    MOREAU_KERNEL_LAUNCH(diff_wb_sparse_z_elim_corrections_kernel, (int)batch, threads, shmem_bytes, stream,
        B_full, F_ext, v_x, v_nu, sigma,
        F_all, f_all_stride,
        D_z_inv, rho, v_z,
        P_diag, gh1, c1, c2,
        A_s_vals,
        sparse_nonneg_idx, sparse_col,
        dense_nonneg_idx,
        col_offsets, col_sparse_rows,
        batch, n, m, k, k_d, k_total, k_ext,
        n_nonneg, n_sparse);
}

// λ-elimination RHS corrections for ν and τ.
// rhs_nu[l] -= Σ_i M'[ν_l, λ_i] * D_z_inv[i] * rhs_lam[i]
// rhs_nu[k_total] -= Σ_i v_z[k+ni] * D_z_inv[i] * rhs_lam[i]
// M'[ν_l, λ_i] = A[ν_l, col_i] * a_i + c2[ν_l] * c2[k+ni]
__global__ void diff_wb_rhs_lambda_elim_kernel(
    double* __restrict__ rhs_nu,
    const double* __restrict__ rhs_lam, const double* __restrict__ D_z_inv,
    const double* __restrict__ v_z,
    const double* __restrict__ F_all, long long f_all_stride,
    const double* __restrict__ c2,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ dense_nonneg_idx,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x;
    if (bi >= batch) return;
    if (threadIdx.x != 0) return;
    (void)c2; (void)dense_nonneg_idx; (void)sparse_col; (void)k_d;
    double* rnu = rhs_nu + bi * k_ext;
    const double* rlam = rhs_lam + bi * n_nonneg;
    const double* Dzi = D_z_inv + bi * n_nonneg;
    const double* vz = v_z + bi * m;
    const double* F_b = F_all + (f_all_stride > 0 ? bi * f_all_stride : 0);
    const double* As_b = A_s_vals + bi * n_sparse;

    // Eliminate λ from the ν/τ RHS, per column, via Sherman-Morrison. The ν-λ
    // coupling M'[ν_l,λ_p] = A[ν_l,j]·a_p factors (A[ν_l,j] = F_b[j*k_total+l] is
    // column-only), so a'L^{-1}rhs_λ = t_ar/β collapses the ν update to
    //   rnu[l] -= F_b[j*k_total+l] · t_ar / β.   (τ keeps the general SM form.)
    for (int64_t j = 0; j < n; ++j) {
        int64_t p0 = col_offsets[j], p1 = col_offsets[j + 1];
        if (p0 == p1) continue;
        double t_ar = 0.0, t_vr = 0.0, t_va = 0.0, s_aa = 0.0;
        for (int64_t p = p0; p < p1; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            double a = As_b[sr];
            double dzi = Dzi[ni];
            double dzp_inv = dzi / (1.0 - a * a * dzi);
            double vz_i = vz[k + ni];
            double rl = rlam[ni];
            t_ar += a * dzp_inv * rl;
            t_vr += vz_i * dzp_inv * rl;
            t_va += vz_i * a * dzp_inv;
            s_aa += a * a * dzp_inv;
        }
        double beta = 1.0 + s_aa;
        double scale = t_ar / beta;
        for (int64_t l = 0; l < k_total; ++l)
            rnu[l] -= F_b[j * k_total + l] * scale;
        rnu[k_total] -= t_vr - t_va * t_ar / beta;
    }
}

void launch_diff_wb_rhs_lambda_elim(
    double* rhs_nu,
    const double* rhs_lam, const double* D_z_inv,
    const double* v_z,
    const double* F_all, long long f_all_stride,
    const double* c2,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* dense_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream)
{
    if (n_sparse == 0) return;
    MOREAU_KERNEL_LAUNCH(diff_wb_rhs_lambda_elim_kernel, (int)batch, 1, 0, stream,
        rhs_nu, rhs_lam, D_z_inv, v_z,
        F_all, f_all_stride, c2, A_s_vals,
        sparse_nonneg_idx, sparse_col, dense_nonneg_idx,
        col_offsets, col_sparse_rows,
        batch, n, m, k, k_d, k_total, k_ext, n_nonneg, n_sparse);
}

// Copy v_x (B,n) into the last column of F_ext (B, n, k_ext):
// F_ext[b, j, k_total] = v_x[b, j]
__global__ void diff_wb_set_F_ext_last_col_kernel(
    double* __restrict__ F_ext, const double* __restrict__ v_x,
    int64_t batch, int64_t n, int64_t k_ext, int64_t k_total)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* Fe = F_ext + b * n * k_ext;
    const double* vx = v_x + b * n;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        Fe[j * k_ext + k_total] = vx[j];
    }
}

void launch_diff_wb_set_F_ext_last_col(
    double* F_ext, const double* v_x,
    int64_t batch, int64_t n, int64_t k_ext, int64_t k_total,
    cudaStream_t stream)
{
    int threads = std::min((int)n, 256);
    if (threads == 0) threads = 1;
    MOREAU_KERNEL_LAUNCH(diff_wb_set_F_ext_last_col_kernel, (int)batch, threads, 0, stream,
        F_ext, v_x, batch, n, k_ext, k_total);
}

// Woodbury vector subtraction: out = a - b (elementwise, B*n elements)
__global__ void diff_wb_vec_sub_kernel(
    double* __restrict__ out, const double* __restrict__ a, const double* __restrict__ b, int64_t count)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] - b[i];
}

void launch_diff_wb_vec_sub(
    double* out, const double* a, const double* b,
    int64_t count, cudaStream_t stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(diff_wb_vec_sub_kernel, blocks, threads, 0, stream, out, a, b, count);
}

// Copy one scalar per batch between strided arrays:
// dst[b * dst_stride + dst_offset] = src[b * src_stride + src_offset]
__global__ void diff_wb_strided_scalar_copy_kernel(
    double* __restrict__ dst, int64_t dst_stride, int64_t dst_offset,
    const double* __restrict__ src, int64_t src_stride, int64_t src_offset,
    int64_t batch)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    dst[b * dst_stride + dst_offset] = src[b * src_stride + src_offset];
}

void launch_diff_wb_strided_scalar_copy(
    double* dst, int64_t dst_stride, int64_t dst_offset,
    const double* src, int64_t src_stride, int64_t src_offset,
    int64_t batch, cudaStream_t stream)
{
    if (batch == 0) return;
    int threads = 256;
    int blocks = (int)((batch + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(diff_wb_strided_scalar_copy_kernel, blocks, threads, 0, stream,
        dst, dst_stride, dst_offset, src, src_stride, src_offset, batch);
}

} // namespace moreau
