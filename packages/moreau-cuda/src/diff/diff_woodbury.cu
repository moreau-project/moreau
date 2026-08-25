/**
 * @file diff_woodbury.cu
 * @brief Woodbury-specialized backward pass — two-level Schur complement solve
 *
 * Solves (J'J + eI)w = rhs_bar via block elimination, then y_out = J*w.
 *
 * Algorithm:
 *   1. Eliminate y_s (diagonal D_s = I + H^2 + eI)
 *   2. Solve reduced (n+m+1) system via two-level Woodbury:
 *      - M'_{xx} = D_x + G*G' where G = [sqrt(Λ)*F_all, c1]
 *      - Inner Woodbury: C = I + G'D^{-1}G, M'_{xx}^{-1} via Woodbury identity
 *      - Outer Schur: T = B - F_ext' K_hat where K_hat = M'_{xx}^{-1} F_ext
 *      - τ handled as part of the Schur complement (not separate Sherman-Morrison)
 *   3. Back-substitute for y_s
 *   4. J-matvec: y_out = J * w
 */

#include "moreau/diff/diff_woodbury.hpp"
#include "moreau/kkt/kkt_woodbury.hpp"
#include "moreau/kkt/woodbury_kernels.cuh"
#include "moreau/cuda/status_utils.hpp"

#include <stdexcept>


namespace moreau {

// ============================================================================
// File-local kernels for the global gap-row Woodbury correction
// ============================================================================
namespace {

// Pack the gap row g = [c1; c2; 0; c3] into a jdim RHS (z2 block is zero).
__global__ void diff_wb_build_g_rhs_kernel(
    double* __restrict__ g_rhs,
    const double* __restrict__ c1, const double* __restrict__ c2, const double* __restrict__ c3,
    int64_t n, int64_t m, int64_t jdim, int64_t batch)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    double* g = g_rhs + b * jdim;
    const double* c1_b = c1 + b * n;
    const double* c2_b = c2 + b * m;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) g[i] = c1_b[i];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) { g[n + i] = c2_b[i]; g[n + m + i] = 0.0; }
    if (threadIdx.x == 0) g[n + 2 * m] = c3[b];
}

// out[b] = g' w = Σ c1·wx + Σ c2·wz + c3·wtau  (gap weight on the z2 block is 0).
__global__ void diff_wb_gap_dot_kernel(
    double* __restrict__ out,
    const double* __restrict__ c1, const double* __restrict__ wx,
    const double* __restrict__ c2, const double* __restrict__ wz,
    const double* __restrict__ c3, const double* __restrict__ wtau,
    int64_t n, int64_t m, int64_t batch)
{
    int64_t b = blockIdx.x;
    if (b >= batch) return;
    __shared__ double sdata[256];
    double local = 0.0;
    const double* c1_b = c1 + b * n; const double* wx_b = wx + b * n;
    const double* c2_b = c2 + b * m; const double* wz_b = wz + b * m;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) local += c1_b[i] * wx_b[i];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) local += c2_b[i] * wz_b[i];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[b] = sdata[0] + c3[b] * wtau[b];
}

// coeff[b] = (g'·w0) / (1 + g'·wg)   (Woodbury denominator for the rank-1 gg').
__global__ void diff_wb_gap_coeff_kernel(
    double* __restrict__ coeff,
    const double* __restrict__ g_w0, const double* __restrict__ g_wg, int64_t batch)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    coeff[b] = g_w0[b] / (1.0 + g_wg[b]);
}

// In-place: w (holds wg on entry) ← w0 - coeff·wg.
__global__ void diff_wb_gap_combine_kernel(
    double* __restrict__ w, const double* __restrict__ w0, const double* __restrict__ coeff,
    int64_t len, int64_t batch)
{
    int64_t total = batch * len;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        int64_t b = t / len;
        w[t] = w0[t] - coeff[b] * w[t];
    }
}

// Per-batch y ← y − coeff·wg (gap combine with a cached wg).
__global__ void diff_wb_neg_axpy_kernel(
    double* __restrict__ y, const double* __restrict__ wg, const double* __restrict__ coeff,
    int64_t len, int64_t batch)
{
    int64_t total = batch * len;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        int64_t b = t / len;
        y[t] -= coeff[b] * wg[t];
    }
}

// Extract the prim block u[n:n+m] (jdim-strided) into a contiguous (B,m) buffer.
__global__ void diff_wb_extract_block_kernel(
    double* __restrict__ out, const double* __restrict__ u,
    int64_t off, int64_t len, int64_t jdim, int64_t batch)
{
    int64_t b = blockIdx.x; if (b >= batch) return;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x)
        out[b * len + i] = u[b * jdim + off + i];
}

// Transpose-matvec diagonal+sparse part: out = J'·u (variable layout). u is in
// row layout [stat(n); prim(m); comp(m); gap(1)]; the dense A·u_stat and A'·u_prim
// contributions are added separately via F_all GEMVs.
__global__ void diff_wb_Jt_matvec_diag_kernel(
    double* __restrict__ out, const double* __restrict__ u,
    const double* __restrict__ P, const double* __restrict__ H_nonneg,
    const double* __restrict__ q, const double* __restrict__ bvec,
    const double* __restrict__ c1, const double* __restrict__ c2, const double* __restrict__ c3,
    const double* __restrict__ A_s_vals,
    const int64_t* __restrict__ sparse_nonneg_idx, const int64_t* __restrict__ sparse_col,
    const int64_t* __restrict__ col_offsets, const int64_t* __restrict__ col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse)
{
    int64_t bi = blockIdx.x; if (bi >= batch) return;
    int64_t jdim = n + 2 * m + 1;
    const double* us = u + bi * jdim;            // stationarity
    const double* up = u + bi * jdim + n;        // primal-feas
    const double* uc = u + bi * jdim + n + m;    // complementarity
    double ug = u[bi * jdim + n + 2 * m];        // gap
    double* vx = out + bi * jdim;
    double* vz = out + bi * jdim + n;
    double* vs = out + bi * jdim + n + m;
    const double* P_b = P + bi * n; const double* H_b = H_nonneg + bi * n_nonneg;
    const double* q_b = q + bi * n; const double* b_b = bvec + bi * m;
    const double* c1_b = c1 + bi * n; const double* c2_b = c2 + bi * m; double c3_b = c3[bi];
    const double* As_b = A_s_vals + bi * n_sparse;

    // v[x_j] = P·us + c1·ug + Σ_sparse a·up[k+ni]   (dense A'·up added later)
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        double val = P_b[j] * us[j] + c1_b[j] * ug;
        for (int64_t p = col_offsets[j]; p < col_offsets[j + 1]; ++p) {
            int64_t sr = col_sparse_rows[p];
            int64_t ni = sparse_nonneg_idx[sr];
            val += As_b[sr] * up[k + ni];
        }
        vx[j] = val;
    }
    // v[z1_i] = up + uc + c2·ug   (dense+sparse A·us added next)
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x)
        vz[i] = up[i] + uc[i] + c2_b[i] * ug;
    // v[z2_i] = -up - H·uc   (H=1 on zero cones, like J_matvec_diag)
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        double h = (i < k) ? 1.0 : H_b[i - k];
        vs[i] = -up[i] - h * uc[i];
    }
    __syncthreads();
    for (int64_t sr = threadIdx.x; sr < n_sparse; sr += blockDim.x) {  // sparse A·us
        int64_t ni = sparse_nonneg_idx[sr]; int64_t col = sparse_col[sr];
        atomicAdd(&vz[k + ni], As_b[sr] * us[col]);
    }
    // v[tau] = Σ q·us − Σ b·up + c3·ug
    __shared__ double sdata[256];
    double local = 0.0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) local += q_b[j] * us[j];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) local -= b_b[i] * up[i];
    sdata[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) out[bi * jdim + n + 2 * m] = sdata[0] + c3_b * ug;
}

// Per-batch relative residual ‖Mv − rhs‖_inf / (‖rhs‖_inf + 1) — a conditioning
// signal used to fall back to cuDSS when the Woodbury chain loses accuracy.
__global__ void diff_wb_relresid_kernel(
    double* __restrict__ out, const double* __restrict__ Mv, const double* __restrict__ rhs,
    int64_t jdim, int64_t batch)
{
    int64_t b = blockIdx.x; if (b >= batch) return;
    __shared__ double sres[256]; __shared__ double srhs[256];
    double mres = 0.0, mrhs = 0.0;
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        double r = Mv[b * jdim + i] - rhs[b * jdim + i];
        mres = fmax(mres, fabs(r));
        mrhs = fmax(mrhs, fabs(rhs[b * jdim + i]));
    }
    sres[threadIdx.x] = mres; srhs[threadIdx.x] = mrhs; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sres[threadIdx.x] = fmax(sres[threadIdx.x], sres[threadIdx.x + s]);
                               srhs[threadIdx.x] = fmax(srhs[threadIdx.x], srhs[threadIdx.x + s]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[b] = sres[0] / (srhs[0] + 1.0);
}

// Pack the 4 solution blocks [x;z1;z2;tau] into a contiguous (B,jdim) buffer.
__global__ void diff_wb_pack_jdim_kernel(
    double* __restrict__ out, const double* __restrict__ x, const double* __restrict__ z,
    const double* __restrict__ s, const double* __restrict__ tau,
    int64_t n, int64_t m, int64_t jdim, int64_t batch)
{
    int64_t b = blockIdx.x; if (b >= batch) return;
    double* o = out + b * jdim;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = x[b * n + i];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) { o[n + i] = z[b * m + i]; o[n + m + i] = s[b * m + i]; }
    if (threadIdx.x == 0) o[n + 2 * m] = tau[b];
}

// Unpack a (B,jdim) buffer into the 4 solution blocks.
__global__ void diff_wb_unpack_jdim_kernel(
    double* __restrict__ x, double* __restrict__ z, double* __restrict__ s, double* __restrict__ tau,
    const double* __restrict__ in, int64_t n, int64_t m, int64_t jdim, int64_t batch)
{
    int64_t b = blockIdx.x; if (b >= batch) return;
    const double* o = in + b * jdim;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) x[b * n + i] = o[i];
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) { z[b * m + i] = o[n + i]; s[b * m + i] = o[n + m + i]; }
    if (threadIdx.x == 0) tau[b] = o[n + 2 * m];
}

// jdim ← a − b.
__global__ void diff_wb_vec_sub_jdim_kernel(
    double* __restrict__ out, const double* __restrict__ a, const double* __restrict__ b, int64_t total)
{
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x)
        out[t] = a[t] - b[t];
}

}  // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

DiffWoodbury::DiffWoodbury(int64_t n, int64_t m, int64_t batchSize,
                           const Cones& cones,
                           cublasHandle_t cublas, cusolverDnHandle_t cusolver,
                           cudaStream_t stream)
    : n_(n), m_(m), batchSize_(batchSize),
      k_(cones.numZeroCones), n_nonneg_(cones.numNonnegCones),
      jdim_(n + 2 * m + 1),
      cublas_(cublas), cusolver_(cusolver)
{
    // s-elimination diagonals
    g_ = make_device_unique<double>(batchSize_ * m_);
    Lambda_ = make_device_unique<double>(batchSize_ * m_);
    gh1_ = make_device_unique<double>(batchSize_ * m_);

    // Reduced RHS (s-elimination only, no tau elimination)
    rt_x_ = make_device_unique<double>(batchSize_ * n_);
    rt_z_ = make_device_unique<double>(batchSize_ * m_);
    rt_tau_ = make_device_unique<double>(batchSize_);

    // Tau coupling vectors
    v_x_ = make_device_unique<double>(batchSize_ * n_);
    v_z_ = make_device_unique<double>(batchSize_ * m_);
    sigma_ = make_device_unique<double>(batchSize_);

    // D_x, D_z
    D_x_ = make_device_unique<double>(batchSize_ * n_);
    D_x_inv_ = make_device_unique<double>(batchSize_ * n_);
    if (n_nonneg_ > 0) {
        D_z_ = make_device_unique<double>(batchSize_ * n_nonneg_);
        D_z_inv_ = make_device_unique<double>(batchSize_ * n_nonneg_);
        rho_ = make_device_unique<double>(batchSize_);
        dlam_nonneg_ = make_device_unique<double>(batchSize_ * n_nonneg_);
    }

    // Solution vectors
    y_x_ = make_device_unique<double>(batchSize_ * n_);
    y_z_ = make_device_unique<double>(batchSize_ * m_);
    y_s_ = make_device_unique<double>(batchSize_ * m_);
    y_tau_ = make_device_unique<double>(batchSize_);

    // GEMV workspace
    gemv_m_ = make_device_unique<double>(batchSize_ * m_);

    // Zeroed gap coefficients fed to the operator-assembly path so it factors
    // the base operator B (gap row applied globally via Woodbury instead).
    zero_c1_ = make_device_unique<double>(batchSize_ * n_);
    zero_c2_ = make_device_unique<double>(batchSize_ * m_);
    zero_c3_ = make_device_unique<double>(batchSize_);
    cudaMemsetAsync(zero_c1_.get(), 0, sizeof(double) * batchSize_ * n_, stream);
    cudaMemsetAsync(zero_c2_.get(), 0, sizeof(double) * batchSize_ * m_, stream);
    cudaMemsetAsync(zero_c3_.get(), 0, sizeof(double) * batchSize_, stream);

    // Global gap-row Woodbury workspace + iterative refinement.
    wg_x_ = make_device_unique<double>(batchSize_ * n_);
    wg_z_ = make_device_unique<double>(batchSize_ * m_);
    wg_s_ = make_device_unique<double>(batchSize_ * m_);
    wg_tau_ = make_device_unique<double>(batchSize_);
    g_rhs_ = make_device_unique<double>(batchSize_ * jdim_);
    gap_g_w0_ = make_device_unique<double>(batchSize_);
    gap_g_wg_ = make_device_unique<double>(batchSize_);
    gap_coeff_ = make_device_unique<double>(batchSize_);
    wacc_jdim_ = make_device_unique<double>(batchSize_ * jdim_);
    jw_ = make_device_unique<double>(batchSize_ * jdim_);
    Mv_ = make_device_unique<double>(batchSize_ * jdim_);
    mres_ = make_device_unique<double>(batchSize_ * jdim_);
    resid_norm_ = make_device_unique<double>(batchSize_);

    cudaStreamSynchronize(stream);
}

// ============================================================================
// Setup
// ============================================================================

void DiffWoodbury::setup(const WoodburyKKTData& wb,
    const double* H_nonneg, const double* q, const double* b,
    const double* c1, const double* c2, const double* c3, cudaStream_t stream)
{
    // Copy structural info from forward solver
    k_total_ = wb.k_total_; k_d_ = wb.k_d_;
    k_ext_ = k_total_ + 1;
    n_sparse_ = wb.n_sparse_; n_empty_ = wb.n_empty_;
    F_all_ = wb.F_all_.get(); f_all_stride_ = wb.f_all_stride_;
    f_all_shared_ = wb.f_all_shared_;
    P_diag_ = wb.P_diag_.get(); A_s_vals_ = wb.A_s_vals_.get();
    sparse_nonneg_idx_ = wb.d_sparse_nonneg_idx_.get();
    sparse_col_ = wb.d_sparse_col_.get();
    col_offsets_ = wb.d_col_offsets_.get();
    col_sparse_rows_ = wb.d_col_sparse_rows_.get();
    dense_nonneg_idx_ = wb.d_dense_nonneg_idx_.get();
    empty_nonneg_idx_ = wb.d_empty_nonneg_idx_.get();

    // Per-call HSDE data
    H_nonneg_ = H_nonneg; q_ = q; b_ = b;
    // Keep the real gap coefficients for J_matvec + the global gap Woodbury;
    // feed ZEROED c1/c2/c3 to the operator-assembly + base-solve kernels so they
    // factor the base operator B (gap row excluded).
    c1_real_ = c1; c2_real_ = c2; c3_real_ = c3;
    c1_ = zero_c1_.get(); c2_ = zero_c2_.get(); c3_ = zero_c3_.get();

    // Lazy-allocate k_ext-dependent buffers on first call
    if (!F_ext_) {
        // F_ext: coupling x → [ν, τ]
        F_ext_ = make_device_unique<double>(batchSize_ * n_ * k_ext_);
        f_ext_stride_ = (long long)(n_ * k_ext_);

        // Inner Woodbury
        G_ = make_device_unique<double>(batchSize_ * n_ * k_ext_);
        G_scaled_ = make_device_unique<double>(batchSize_ * n_ * k_ext_);
        C_inner_ = make_device_unique<double>(batchSize_ * k_ext_ * k_ext_);

        // Outer Schur
        B_full_ = make_device_unique<double>(batchSize_ * k_ext_ * k_ext_);
        S_schur_ = make_device_unique<double>(batchSize_ * k_ext_ * k_ext_);
        S_schur_orig_ = make_device_unique<double>(batchSize_ * k_ext_ * k_ext_);

        // Precomputed K_hat = M'_{xx}^{-1} * F_ext
        K_hat_ = make_device_unique<double>(batchSize_ * n_ * k_ext_);

        // v_nu (tau coupling at ν positions)
        v_nu_ = make_device_unique<double>(batchSize_ * k_ext_);

        // Workspace
        temp_kk_ = make_device_unique<double>(batchSize_ * k_ext_ * k_ext_);
        mx_scratch_ = make_device_unique<double>(batchSize_ * n_);

        // RHS workspace
        rhs_x_ = make_device_unique<double>(batchSize_ * n_);
        // rhs_nu_ is dual-purpose across solve_reduced():
        //   1. Initially holds the unpacked ν RHS (rt_z at zero-cone positions
        //      plus rt_tau at position k_total).
        //   2. After the F_ext' u_x subtraction it is overwritten with the
        //      Schur RHS (a snapshot of d_nu_ before LDL^T) and used as the
        //      reference right-hand side during iterative refinement.
        rhs_nu_ = make_device_unique<double>(batchSize_ * k_ext_);
        rhs_lam_ = make_device_unique<double>(batchSize_ * std::max(n_nonneg_, (int64_t)1));
        rhs_x_adj_ = make_device_unique<double>(batchSize_ * n_);
        u_x_ = make_device_unique<double>(batchSize_ * n_);
        d_nu_ = make_device_unique<double>(batchSize_ * k_ext_);
        schur_resid_ = make_device_unique<double>(batchSize_ * k_ext_);
        gemv_k_ = make_device_unique<double>(batchSize_ * k_ext_);

        cudaStreamSynchronize(stream);
    }
}

size_t DiffWoodbury::memoryUsage() const noexcept {
    size_t s = 0;
    s += 3 * batchSize_ * m_ * sizeof(double);  // g, Lambda, gh1
    s += 2 * batchSize_ * n_ * sizeof(double);  // D_x, D_x_inv
    s += 2 * batchSize_ * n_nonneg_ * sizeof(double);  // D_z, D_z_inv
    s += batchSize_ * (n_ + m_ + 1) * sizeof(double);  // v_x, v_z, sigma
    s += 4 * batchSize_ * n_ * k_ext_ * sizeof(double);  // F_ext, G, G_scaled, K_hat
    s += 4 * batchSize_ * k_ext_ * k_ext_ * sizeof(double);  // C_inner, B_full, S_schur, temp_kk
    s += 6 * batchSize_ * n_ * sizeof(double);  // rt_x, rhs_x, rhs_x_adj, u_x, mx_scratch, y_x
    s += 3 * batchSize_ * k_ext_ * sizeof(double);  // rhs_nu, d_nu, gemv_k
    s += batchSize_ * k_ext_ * sizeof(double);  // v_nu
    s += 2 * batchSize_ * m_ * sizeof(double);  // y_z, y_s
    return s;
}

// ============================================================================
// Precompute
// ============================================================================

void DiffWoodbury::build_F_ext(cudaStream_t stream) {
    // F_ext[j,i] = (P[j]+1-gh1[i])*A[i,j] + c1[j]*c2[i] for i < k_total
    // The last column (v_x) is set later after v_x is computed.
    launch_diff_wb_build_F_ext_backward(
        F_ext_.get(), F_all_, f_all_stride_,
        P_diag_, gh1_.get(), c1_, c2_,
        dense_nonneg_idx_,
        batchSize_, n_, m_, k_, k_d_, k_total_, k_ext_, stream);
}

void DiffWoodbury::build_G(cudaStream_t stream) {
    // G = [sqrt(Λ_zd) * F_all columns, c1]
    launch_diff_wb_build_G(
        G_.get(), F_all_, f_all_stride_,
        Lambda_.get(), c1_,
        dense_nonneg_idx_,
        batchSize_, n_, m_, k_, k_d_, k_total_, k_ext_, stream);
}

void DiffWoodbury::build_inner_woodbury(cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));

    // 1. G_scaled = D_x_inv * G (element-wise per row)
    launch_woodbury_scale_F(
        G_scaled_.get(), G_.get(), D_x_inv_.get(),
        batchSize_, n_, k_ext_, false, stream);

    // 2. C_inner = G_scaled' @ G = G' D^{-1} G
    double alpha = 1.0, beta = 0.0;
    CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)k_ext_, (int)k_ext_, (int)n_,
                &alpha,
                G_scaled_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                G_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                &beta,
                C_inner_.get(), (int)k_ext_, (long long)(k_ext_ * k_ext_),
                (int)batchSize_));

    // 3. C_inner += I
    launch_diff_wb_add_identity(C_inner_.get(), batchSize_, k_ext_, stream);

    // 4. LDL^T factorization of C_inner (same kernel as outer Schur)
    launch_diff_wb_ldlt_factor(
        C_inner_.get(), batchSize_, k_ext_, stream);
}

void DiffWoodbury::apply_Mx_inv(const double* rhs, double* out, cudaStream_t stream) {
    // Compute out = M'_{xx}^{-1} * rhs via Woodbury identity:
    // M'_{xx}^{-1} = D^{-1} - D^{-1} G C^{-1} G' D^{-1}
    //
    // Steps:
    // 1. out = D^{-1} * rhs
    // 2. temp = G' * out  (k_ext per batch)
    // 3. temp = C^{-1} * temp
    // 4. correction = D^{-1} * G * temp  (n per batch)
    // 5. out -= correction

    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));

    // Step 1: out = D^{-1} * rhs
    launch_woodbury_elementwise_mul(
        out, D_x_inv_.get(), rhs,
        batchSize_ * n_, stream);

    // Step 2: gemv_k = G' * out (k_ext per batch)
    double alpha = 1.0, beta_zero = 0.0;
    CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)k_ext_, 1, (int)n_,
                &alpha,
                G_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                out, (int)n_, (long long)n_,
                &beta_zero,
                gemv_k_.get(), (int)k_ext_, (long long)k_ext_,
                (int)batchSize_));

    // Step 3: gemv_k = C_inner^{-1} * gemv_k (single-vector LDL^T solve)
    launch_diff_wb_ldlt_solve(
        C_inner_.get(), gemv_k_.get(), batchSize_, k_ext_, stream);

    // Step 4: mx_scratch_ = G_scaled^T @ gemv_k = D^{-1}*G * C^{-1} * G' * D^{-1} * rhs
    CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)n_, 1, (int)k_ext_,
                &alpha,
                G_scaled_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                gemv_k_.get(), (int)k_ext_, (long long)k_ext_,
                &beta_zero,
                mx_scratch_.get(), (int)n_, (long long)n_,
                (int)batchSize_));

    // Step 5: out -= mx_scratch_
    launch_diff_wb_vec_sub(
        out, out, mx_scratch_.get(),
        batchSize_ * n_, stream);
}

void DiffWoodbury::build_B_full(cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));

    // B_full[0:k_total, 0:k_total] = F_all' @ F_all (AA' for zero/dense rows)
    cudaMemsetAsync(B_full_.get(), 0, sizeof(double) * batchSize_ * k_ext_ * k_ext_, stream);

    if (k_total_ > 0) {
        double alpha = 1.0, beta = 0.0;
        // Write k_total×k_total result into B_full with k_ext leading dim
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int)k_total_, (int)k_total_, (int)n_,
                    &alpha,
                    F_all_, (int)k_total_, f_all_stride_,
                    F_all_, (int)k_total_, f_all_stride_,
                    &beta,
                    B_full_.get(), (int)k_ext_, (long long)(k_ext_ * k_ext_),
                    (int)batchSize_));
    }

    // Add diagonal corrections + c2*c2' + tau coupling
    launch_diff_wb_build_B_corrections(
        B_full_.get(), v_nu_.get(), sigma_.get(),
        H_nonneg_, c2_, dense_nonneg_idx_, eps_,
        batchSize_, m_, k_, k_d_, k_total_, k_ext_, stream);
}

void DiffWoodbury::build_and_factor_outer_schur(cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));

    // Compute K_hat = M'_{xx}^{-1} * F_ext (n × k_ext per batch)
    // Do this column-by-column? No, apply_Mx_inv works on vectors.
    // Instead, batch it: treat F_ext as k_ext separate n-vectors.
    // But apply_Mx_inv uses gemv_k_ and u_x_ as scratch, so we can't call it k_ext times.
    // Instead, implement the batched version directly:

    // K_hat = D^{-1} F_ext - D^{-1} G C^{-1} (G' D^{-1} F_ext)

    // Step 1: temp_nk = D_x_inv * F_ext (element-wise per row)
    launch_woodbury_scale_F(
        K_hat_.get(), F_ext_.get(), D_x_inv_.get(),
        batchSize_, n_, k_ext_, false, stream);

    // Step 2: temp_kk = G' @ K_hat = G' D^{-1} F_ext  (k_ext × k_ext per batch)
    // G is (B, n, k_ext), K_hat is (B, n, k_ext)
    // G' @ K_hat: (k_ext × n) @ (n × k_ext) → k_ext × k_ext
    double alpha = 1.0, beta_zero = 0.0;
    CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)k_ext_, (int)k_ext_, (int)n_,
                &alpha,
                G_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                K_hat_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                &beta_zero,
                temp_kk_.get(), (int)k_ext_, (long long)(k_ext_ * k_ext_),
                (int)batchSize_));

    // Step 3: temp_kk = C_inner^{-1} @ temp_kk  (k_ext × k_ext, multi-RHS LDL^T solve)
    launch_diff_wb_ldlt_solve_mrhs(
        C_inner_.get(), temp_kk_.get(), batchSize_, k_ext_, k_ext_, stream);

    // Step 4: K_hat -= D^{-1}G @ C^{-1} @ G'D^{-1}F_ext
    // temp_kk_cb holds (C^{-1} G^T K_hat) in column-major. We need:
    //   K_hat_cb -= temp_kk_cb^T * G_scaled_cb
    // because the column-major storage transposes the real/cb relationship.
    double neg_alpha = -1.0, beta_one = 1.0;
    CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)k_ext_, (int)n_, (int)k_ext_,
                &neg_alpha,
                temp_kk_.get(), (int)k_ext_, (long long)(k_ext_ * k_ext_),
                G_scaled_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                &beta_one,
                K_hat_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                (int)batchSize_));

    // Now K_hat = D^{-1}F_ext - G_scaled @ C^{-1} @ G'D^{-1}F_ext = M'_{xx}^{-1} F_ext

    // Step 5: T = B_full - F_ext' @ K_hat  (k_ext × k_ext)
    CUDA_THROW(cudaMemcpyAsync(S_schur_.get(), B_full_.get(),
                    sizeof(double) * batchSize_ * k_ext_ * k_ext_,
                    cudaMemcpyDeviceToDevice, stream));

    // S_schur -= F_ext' @ K_hat
    CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)k_ext_, (int)k_ext_, (int)n_,
                &neg_alpha,
                F_ext_.get(), (int)k_ext_, f_ext_stride_,
                K_hat_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                &beta_one,
                S_schur_.get(), (int)k_ext_, (long long)(k_ext_ * k_ext_),
                (int)batchSize_));

    // Step 6: Save unfactored S_schur for iterative refinement
    CUDA_THROW(cudaMemcpyAsync(S_schur_orig_.get(), S_schur_.get(),
                    sizeof(double) * batchSize_ * k_ext_ * k_ext_,
                    cudaMemcpyDeviceToDevice, stream));

    // Step 7: LDL^T factorization (handles near-singular/indefinite matrices
    // without NaN, unlike Cholesky which fails on non-PD pivots)
    launch_diff_wb_ldlt_factor(
        S_schur_.get(), batchSize_, k_ext_, stream);
}

void DiffWoodbury::precompute(cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));

    // 1. s-elimination diagonals
    launch_diff_wb_s_elim_diagonals(
        g_.get(), Lambda_.get(), gh1_.get(),
        H_nonneg_, eps_,
        batchSize_, m_, k_, n_nonneg_, stream);

    // 2. D_z for sparse nonneg + Sherman-Morrison denominator ρ
    if (n_nonneg_ > 0) {
        launch_diff_wb_compute_D_z(
            D_z_.get(), D_z_inv_.get(),
            H_nonneg_, c2_,
            A_s_vals_, sparse_nonneg_idx_,
            eps_,
            batchSize_, m_, k_, n_nonneg_, n_sparse_, stream);

        launch_diff_wb_compute_rho(
            rho_.get(), D_z_inv_.get(), c2_,
            dense_nonneg_idx_, k_d_,
            batchSize_, m_, k_, n_nonneg_, stream);
    }

    // 3. D_x
    launch_diff_wb_compute_D_x(
        D_x_.get(), P_diag_, Lambda_.get(),
        A_s_vals_, eps_,
        H_nonneg_, c1_, c2_,
        (n_nonneg_ > 0) ? D_z_inv_.get() : nullptr,
        (n_nonneg_ > 0) ? rho_.get() : nullptr,
        sparse_nonneg_idx_, col_offsets_, col_sparse_rows_,
        batchSize_, n_, m_, k_, n_nonneg_, n_sparse_, stream);

    launch_woodbury_elementwise_inv(
        D_x_inv_.get(), D_x_.get(), batchSize_ * n_, stream);

    // 4. Build F_ext[:, 0:k_total]
    build_F_ext(stream);

    // 5. v_x, v_z, sigma
    launch_diff_wb_tau_elim_diag(
        v_x_.get(), v_z_.get(), sigma_.get(),
        P_diag_, q_, b_, c1_, c2_, c3_,
        Lambda_.get(), g_.get(), gh1_.get(),
        A_s_vals_, sparse_nonneg_idx_, sparse_col_,
        col_offsets_, col_sparse_rows_, eps_,
        batchSize_, n_, m_, k_, n_nonneg_, n_sparse_, stream);

    // 5b. GEMV contributions
    {
        launch_diff_wb_gather_weighted(
            gemv_k_.get(), b_, Lambda_.get(),
            dense_nonneg_idx_,
            batchSize_, m_, k_, k_d_, k_total_, stream);

        double alpha_neg = -1.0, beta_one = 1.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, 1, (int)k_total_,
                    &alpha_neg,
                    F_all_, (int)k_total_, f_all_stride_,
                    gemv_k_.get(), (int)k_total_, (long long)k_total_,
                    &beta_one,
                    v_x_.get(), (int)n_, (long long)n_,
                    (int)batchSize_));

        double alpha_one = 1.0, beta_zero = 0.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_total_, 1, (int)n_,
                    &alpha_one,
                    F_all_, (int)k_total_, f_all_stride_,
                    q_, (int)n_, (long long)n_,
                    &beta_zero,
                    gemv_k_.get(), (int)k_total_, (long long)k_total_,
                    (int)batchSize_));

        launch_diff_wb_scatter_add_weighted(
            v_z_.get(), gemv_k_.get(), nullptr,
            dense_nonneg_idx_,
            batchSize_, m_, k_, k_d_, k_total_, stream);
    }

    // 6. Gather v_nu
    launch_diff_wb_gather_v_nu(
        v_nu_.get(), v_z_.get(), sigma_.get(),
        dense_nonneg_idx_,
        batchSize_, m_, k_, k_d_, k_total_, k_ext_, stream);

    // 7. Build full B matrix
    build_B_full(stream);

    // 8. λ-corrections
    if (n_sparse_ > 0 && n_nonneg_ > 0) {
        launch_diff_wb_sparse_z_elim_corrections(
            B_full_.get(), F_ext_.get(), v_x_.get(), v_nu_.get(), sigma_.get(),
            F_all_, f_all_stride_,
            D_z_inv_.get(), rho_.get(), v_z_.get(),
            P_diag_, gh1_.get(), c1_, c2_,
            A_s_vals_, sparse_nonneg_idx_, sparse_col_,
            dense_nonneg_idx_,
            col_offsets_, col_sparse_rows_,
            batchSize_, n_, m_, k_, k_d_, k_total_, k_ext_,
            n_nonneg_, n_sparse_, stream);
    }

    // 9. Set F_ext last column = v_x
    launch_diff_wb_set_F_ext_last_col(
        F_ext_.get(), v_x_.get(),
        batchSize_, n_, k_ext_, k_total_, stream);

    // 10. Build G
    build_G(stream);

    // 11. Inner Woodbury
    build_inner_woodbury(stream);

    // 12. Outer Schur
    build_and_factor_outer_schur(stream);
}

// ============================================================================
// Eliminate s from RHS (no tau elimination — tau is in Schur)
// ============================================================================

void DiffWoodbury::eliminate_s_rhs(const double* rhs_bar, cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));

    // s-elimination on RHS (diagonal + sparse parts)
    launch_diff_wb_s_elim_rhs(
        rhs_bar, rt_x_.get(), rt_z_.get(), rt_tau_.get(),
        g_.get(), gh1_.get(), b_,
        A_s_vals_, sparse_nonneg_idx_, sparse_col_,
        col_offsets_, col_sparse_rows_,
        batchSize_, n_, m_, k_, n_nonneg_, n_sparse_, stream);

    // Add F_all' @ (g * r_s) at zero/dense positions to rt_x
    {
        launch_diff_wb_extract_mul(
            gemv_m_.get(), g_.get(), rhs_bar,
            batchSize_, m_, jdim_, n_ + m_, stream);

        launch_diff_wb_gather_weighted(
            gemv_k_.get(), gemv_m_.get(), nullptr,
            dense_nonneg_idx_,
            batchSize_, m_, k_, k_d_, k_total_, stream);

        double alpha = 1.0, beta = 1.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, 1, (int)k_total_,
                    &alpha,
                    F_all_, (int)k_total_, f_all_stride_,
                    gemv_k_.get(), (int)k_total_, (long long)k_total_,
                    &beta,
                    rt_x_.get(), (int)n_, (long long)n_,
                    (int)batchSize_));
    }
}

// ============================================================================
// Solve reduced (n+m+1) system via two-level Woodbury
// ============================================================================

void DiffWoodbury::solve_reduced(cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));

    // 1. Copy rt_x -> rhs_x, unpack rt_z -> rhs_nu + rhs_lam
    //    rhs_nu[k_total] = rt_tau (tau is in Schur, not eliminated separately)
    CUDA_THROW(cudaMemcpyAsync(rhs_x_.get(), rt_x_.get(),
                    sizeof(double) * batchSize_ * n_,
                    cudaMemcpyDeviceToDevice, stream));

    launch_diff_wb_unpack_z(
        rhs_nu_.get(), rhs_lam_.get(),
        rt_z_.get(), dense_nonneg_idx_,
        batchSize_, m_, k_, k_d_, k_total_, k_ext_, n_nonneg_, stream);

    // Set rhs_nu[k_total] = rt_tau (tau RHS goes into Schur)
    launch_diff_wb_strided_scalar_copy(
        rhs_nu_.get(), k_ext_, k_total_,
        rt_tau_.get(), 1, 0,
        batchSize_, stream);

    // 1b. Apply λ-elimination corrections to rhs_nu and rhs_tau
    // rhs_nu[l] -= M'[ν_l, λ_i] * D_z_inv[i] * rhs_lam[i]
    // rhs_nu[k_total] -= v_z[k+ni] * D_z_inv[i] * rhs_lam[i]
    if (n_sparse_ > 0 && n_nonneg_ > 0) {
        launch_diff_wb_rhs_lambda_elim(
            rhs_nu_.get(),
            rhs_lam_.get(), D_z_inv_.get(),
            v_z_.get(),
            F_all_, f_all_stride_,
            c2_,
            A_s_vals_, sparse_nonneg_idx_, sparse_col_,
            dense_nonneg_idx_,
            col_offsets_, col_sparse_rows_,
            batchSize_, n_, m_, k_, k_d_, k_total_, k_ext_,
            n_nonneg_, n_sparse_, stream);
    }

    // 2. RHS adjustment for sparse nonneg (backward cross-term, with corrected +1)
    if (n_sparse_ > 0) {
        launch_diff_wb_rhs_adj_sparse(
            rhs_x_adj_.get(), rhs_x_.get(),
            D_z_inv_.get(), rhs_lam_.get(),
            P_diag_, A_s_vals_,
            H_nonneg_, c1_, c2_, eps_,
            sparse_nonneg_idx_,
            col_offsets_, col_sparse_rows_, sparse_col_,
            batchSize_, n_, m_, k_,
            n_nonneg_, n_sparse_, stream);
    } else {
        CUDA_THROW(cudaMemcpyAsync(rhs_x_adj_.get(), rhs_x_.get(),
                        sizeof(double) * batchSize_ * n_,
                        cudaMemcpyDeviceToDevice, stream));
    }

    // 3. Apply M'_{xx}^{-1} to rhs_x_adj → u_x
    apply_Mx_inv(rhs_x_adj_.get(), u_x_.get(), stream);

    // 4. Schur RHS = rhs_nu - F_ext' @ u_x
    CUDA_THROW(cudaMemcpyAsync(d_nu_.get(), rhs_nu_.get(),
                    sizeof(double) * batchSize_ * k_ext_,
                    cudaMemcpyDeviceToDevice, stream));
    {
        double neg1 = -1.0, one = 1.0;
        // F_ext' @ u_x: (k_ext × n) @ (n × 1) → k_ext × 1
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_ext_, 1, (int)n_,
                    &neg1,
                    F_ext_.get(), (int)k_ext_, f_ext_stride_,
                    u_x_.get(), (int)n_, (long long)n_,
                    &one,
                    d_nu_.get(), (int)k_ext_, (long long)k_ext_,
                    (int)batchSize_));
    }

    // 5. Save Schur RHS for iterative refinement, then solve via LDL^T
    CUDA_THROW(cudaMemcpyAsync(rhs_nu_.get(), d_nu_.get(),
                    sizeof(double) * batchSize_ * k_ext_,
                    cudaMemcpyDeviceToDevice, stream));

    launch_diff_wb_ldlt_solve(
        S_schur_.get(), d_nu_.get(), batchSize_, k_ext_, stream);

    // 5b. Iterative refinement (3 steps).
    // Corrects rounding from small LDL^T pivots using the unfactored S_schur_orig.
    for (int refine_step = 0; refine_step < 3; ++refine_step) {
        // resid = schur_rhs_orig - S_orig * d_nu
        CUDA_THROW(cudaMemcpyAsync(schur_resid_.get(), rhs_nu_.get(),
                        sizeof(double) * batchSize_ * k_ext_,
                        cudaMemcpyDeviceToDevice, stream));
        double neg1 = -1.0, one = 1.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_ext_, 1, (int)k_ext_,
                    &neg1,
                    S_schur_orig_.get(), (int)k_ext_, (long long)(k_ext_ * k_ext_),
                    d_nu_.get(), (int)k_ext_, (long long)k_ext_,
                    &one,
                    schur_resid_.get(), (int)k_ext_, (long long)k_ext_,
                    (int)batchSize_));

        // Solve LDL^T * delta = resid
        launch_diff_wb_ldlt_solve(
            S_schur_.get(), schur_resid_.get(), batchSize_, k_ext_, stream);

        // d_nu += delta
        double alpha_one = 1.0;
        CUBLAS_THROW(cublasDaxpy(cublas_, (int)(batchSize_ * k_ext_), &alpha_one,
                    schur_resid_.get(), 1, d_nu_.get(), 1));
    }

    // 6. y_x = u_x - K_hat @ [w_ν; w_τ]
    CUDA_THROW(cudaMemcpyAsync(y_x_.get(), u_x_.get(),
                    sizeof(double) * batchSize_ * n_,
                    cudaMemcpyDeviceToDevice, stream));
    {
        double neg1 = -1.0, one = 1.0;
        // K_hat @ d_nu: K_hat is (B, n, k_ext) with k_ext leading dim
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, 1, (int)k_ext_,
                    &neg1,
                    K_hat_.get(), (int)k_ext_, (long long)(n_ * k_ext_),
                    d_nu_.get(), (int)k_ext_, (long long)k_ext_,
                    &one,
                    y_x_.get(), (int)n_, (long long)n_,
                    (int)batchSize_));
    }

    // 7. y_z: scatter zero-cone entries from d_nu
    launch_diff_wb_scatter_y_z_zero(
        y_z_.get(), d_nu_.get(),
        batchSize_, m_, k_, k_ext_, stream);

    // 8. Extract y_tau from d_nu[k_total] (must be before nonneg recovery which uses y_tau)
    launch_diff_wb_strided_scalar_copy(
        y_tau_.get(), 1, 0,
        d_nu_.get(), k_ext_, k_total_,
        batchSize_, stream);

    // 9. Nonneg y_z recovery
    if (n_nonneg_ > 0) {
        cudaMemsetAsync(dlam_nonneg_.get(), 0, sizeof(double) * batchSize_ * n_nonneg_, stream);

        if (k_d_ > 0) {
            launch_woodbury_scatter_dlam_dense(
                dlam_nonneg_.get(),
                d_nu_.get(),
                batchSize_, k_, k_ext_, n_nonneg_,
                dense_nonneg_idx_, k_d_, stream);
        }

        if (n_sparse_ > 0) {
            launch_diff_wb_dlam_sparse(
                dlam_nonneg_.get(),
                D_z_inv_.get(), y_x_.get(), d_nu_.get(), y_tau_.get(),
                rhs_lam_.get(),
                P_diag_, A_s_vals_,
                H_nonneg_, c1_, c2_,
                v_z_.get(),
                F_all_, f_all_stride_,
                dense_nonneg_idx_,
                eps_,
                sparse_col_, sparse_nonneg_idx_,
                col_offsets_, col_sparse_rows_,
                batchSize_, n_, m_, k_, k_d_, k_total_, k_ext_,
                n_nonneg_, n_sparse_, stream);
        }

        if (n_empty_ > 0) {
            launch_woodbury_dlam_empty(
                dlam_nonneg_.get(),
                D_z_inv_.get(), rhs_lam_.get(),
                empty_nonneg_idx_,
                batchSize_, n_nonneg_, n_empty_, stream);
        }

        launch_diff_wb_pack_dlam_to_yz(
            y_z_.get(), dlam_nonneg_.get(),
            batchSize_, m_, k_, n_nonneg_, stream);
    }

}

// ============================================================================
// Back-substitute y_s
// ============================================================================

void DiffWoodbury::back_substitute(const double* rhs_bar, cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));

    // y_tau is already available from the Schur solution (no back-sub needed)

    // Recover y_s (diagonal + sparse parts)
    launch_diff_wb_recover_y_s(
        y_s_.get(), rhs_bar, y_x_.get(), y_z_.get(), y_tau_.get(),
        g_.get(), gh1_.get(), b_,
        A_s_vals_, sparse_nonneg_idx_, sparse_col_,
        batchSize_, n_, m_, k_, n_nonneg_, n_sparse_, stream);

    // Add F_all GEMV contribution for zero/dense rows of A*y_x to y_s
    {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_total_, 1, (int)n_,
                    &alpha,
                    F_all_, (int)k_total_, f_all_stride_,
                    y_x_.get(), (int)n_, (long long)n_,
                    &beta,
                    gemv_k_.get(), (int)k_total_, (long long)k_total_,
                    (int)batchSize_));
    }

    launch_diff_wb_scatter_add_weighted(
        y_s_.get(), gemv_k_.get(), g_.get(),
        dense_nonneg_idx_,
        batchSize_, m_, k_, k_d_, k_total_, stream);
}

// ============================================================================
// J-matvec: y_out = J * w
// ============================================================================

void DiffWoodbury::J_matvec(double* y_out, cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));

    // Diagonal + sparse parts of J*w. Uses the REAL gap coefficients: J_matvec
    // applies the full Jacobian (including the gap row), unlike the operator
    // assembly which uses zeroed c1/c2/c3 to build the base B.
    launch_diff_wb_J_matvec_diag(
        y_out,
        y_x_.get(), y_z_.get(), y_s_.get(), y_tau_.get(),
        P_diag_, H_nonneg_, q_, b_, c1_real_, c2_real_, c3_real_,
        A_s_vals_, sparse_nonneg_idx_, sparse_col_,
        col_offsets_, col_sparse_rows_,
        batchSize_, n_, m_, k_, n_nonneg_, n_sparse_, stream);

    // Dense A' * w_z contribution to y_out[0:n]
    launch_diff_wb_gather_weighted(
        gemv_k_.get(), y_z_.get(), nullptr,
        dense_nonneg_idx_,
        batchSize_, m_, k_, k_d_, k_total_, stream);

    {
        double alpha = 1.0, beta = 1.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, 1, (int)k_total_,
                    &alpha,
                    F_all_, (int)k_total_, f_all_stride_,
                    gemv_k_.get(), (int)k_total_, (long long)k_total_,
                    &beta,
                    y_out, (int)jdim_, (long long)jdim_,
                    (int)batchSize_));
    }

    // Dense A * w_x contribution to y_out[n:n+m]
    {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_total_, 1, (int)n_,
                    &alpha,
                    F_all_, (int)k_total_, f_all_stride_,
                    y_x_.get(), (int)n_, (long long)n_,
                    &beta,
                    gemv_k_.get(), (int)k_total_, (long long)k_total_,
                    (int)batchSize_));
    }

    cudaMemsetAsync(gemv_m_.get(), 0, sizeof(double) * batchSize_ * m_, stream);
    launch_diff_wb_scatter_add_weighted(
        gemv_m_.get(), gemv_k_.get(), nullptr,
        dense_nonneg_idx_,
        batchSize_, m_, k_, k_d_, k_total_, stream);

    launch_diff_wb_strided_axpy(
        y_out, gemv_m_.get(),
        batchSize_, m_, jdim_, n_, m_, stream);
}

// ============================================================================
// solveAdjoint: main entry point
// ============================================================================


// Solve B w = rhs_bar for the base operator (gap row excluded; c1/c2/c3 are
// zero in the assembly). Result lands in y_x_/y_z_/y_s_/y_tau_.
void DiffWoodbury::base_solve(const double* rhs_bar, cudaStream_t stream) {
    eliminate_s_rhs(rhs_bar, stream);
    solve_reduced(stream);
    back_substitute(rhs_bar, stream);
}

// y_* = M^{-1} rhs_bar = B^{-1}rhs − wg·(g'B^{-1}rhs)/(1+g'wg). Requires wg_*
// (= B^{-1}g) and gap_g_wg_ (= g'wg) precomputed by solveAdjoint.
void DiffWoodbury::apply_Minv(const double* rhs_bar, cudaStream_t stream) {
    const int threads = 256;
    base_solve(rhs_bar, stream);
    diff_wb_gap_dot_kernel<<<(int)batchSize_, threads, 0, stream>>>(
        gap_g_w0_.get(), c1_real_, y_x_.get(), c2_real_, y_z_.get(),
        c3_real_, y_tau_.get(), n_, m_, batchSize_);
    diff_wb_gap_coeff_kernel<<<(int)((batchSize_ + threads - 1) / threads), threads, 0, stream>>>(
        gap_coeff_.get(), gap_g_w0_.get(), gap_g_wg_.get(), batchSize_);
    const int blocks_n = (int)((batchSize_ * n_ + threads - 1) / threads);
    const int blocks_m = (int)((batchSize_ * m_ + threads - 1) / threads);
    const int blocks_b = (int)((batchSize_ + threads - 1) / threads);
    diff_wb_neg_axpy_kernel<<<blocks_n, threads, 0, stream>>>(y_x_.get(), wg_x_.get(), gap_coeff_.get(), n_, batchSize_);
    diff_wb_neg_axpy_kernel<<<blocks_m, threads, 0, stream>>>(y_z_.get(), wg_z_.get(), gap_coeff_.get(), m_, batchSize_);
    diff_wb_neg_axpy_kernel<<<blocks_m, threads, 0, stream>>>(y_s_.get(), wg_s_.get(), gap_coeff_.get(), m_, batchSize_);
    diff_wb_neg_axpy_kernel<<<blocks_b, threads, 0, stream>>>(y_tau_.get(), wg_tau_.get(), gap_coeff_.get(), 1, batchSize_);
}

// out (variable layout) = J'·u, u in row layout. Mirrors J_matvec transposed:
// the diagonal/sparse part is one kernel; the dense A·u_stat and A'·u_prim use
// the same F_all GEMVs as J_matvec.
void DiffWoodbury::Jt_matvec(const double* u, double* out, cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    int threads = std::min((int)(n_ + m_), 256);
    if (threads == 0) threads = 1;
    { int t = 1; while (t < threads) t <<= 1; threads = std::min(t, 256); }
    diff_wb_Jt_matvec_diag_kernel<<<(int)batchSize_, threads, 0, stream>>>(
        out, u, P_diag_, H_nonneg_, q_, b_, c1_real_, c2_real_, c3_real_,
        A_s_vals_, sparse_nonneg_idx_, sparse_col_, col_offsets_, col_sparse_rows_,
        batchSize_, n_, m_, k_, n_nonneg_, n_sparse_);

    // Dense A·u_stat → out z1 block (u_stat = u[0:n])
    {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, (int)k_total_, 1, (int)n_,
            &alpha, F_all_, (int)k_total_, f_all_stride_,
            u, (int)n_, (long long)jdim_,
            &beta, gemv_k_.get(), (int)k_total_, (long long)k_total_, (int)batchSize_));
        cudaMemsetAsync(gemv_m_.get(), 0, sizeof(double) * batchSize_ * m_, stream);
        launch_diff_wb_scatter_add_weighted(gemv_m_.get(), gemv_k_.get(), nullptr,
            dense_nonneg_idx_, batchSize_, m_, k_, k_d_, k_total_, stream);
        launch_diff_wb_strided_axpy(out, gemv_m_.get(), batchSize_, m_, jdim_, n_, m_, stream);
    }
    // Dense A'·u_prim → out x block (u_prim = u[n:n+m])
    {
        diff_wb_extract_block_kernel<<<(int)batchSize_, 256, 0, stream>>>(
            gemv_m_.get(), u, n_, m_, jdim_, batchSize_);
        launch_diff_wb_gather_weighted(gemv_k_.get(), gemv_m_.get(), nullptr,
            dense_nonneg_idx_, batchSize_, m_, k_, k_d_, k_total_, stream);
        double alpha = 1.0, beta = 1.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, (int)n_, 1, (int)k_total_,
            &alpha, F_all_, (int)k_total_, f_all_stride_,
            gemv_k_.get(), (int)k_total_, (long long)k_total_,
            &beta, out, (int)jdim_, (long long)jdim_, (int)batchSize_));
    }
}

void DiffWoodbury::solveAdjoint(const double* rhs_bar, double* y_out, cudaStream_t stream) {
    // Factor the base operator B = J_nogap'J_nogap + εI (c1/c2/c3 zeroed).
    precompute(stream);
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    const int threads = 256;
    const int64_t N = batchSize_ * jdim_;

    // Cache wg = B^{-1}g and g'wg (RHS-independent); apply_Minv reuses them.
    diff_wb_build_g_rhs_kernel<<<(int)batchSize_, threads, 0, stream>>>(
        g_rhs_.get(), c1_real_, c2_real_, c3_real_, n_, m_, jdim_, batchSize_);
    base_solve(g_rhs_.get(), stream);
    CUDA_THROW(cudaMemcpyAsync(wg_x_.get(), y_x_.get(), sizeof(double) * batchSize_ * n_, cudaMemcpyDeviceToDevice, stream));
    CUDA_THROW(cudaMemcpyAsync(wg_z_.get(), y_z_.get(), sizeof(double) * batchSize_ * m_, cudaMemcpyDeviceToDevice, stream));
    CUDA_THROW(cudaMemcpyAsync(wg_s_.get(), y_s_.get(), sizeof(double) * batchSize_ * m_, cudaMemcpyDeviceToDevice, stream));
    CUDA_THROW(cudaMemcpyAsync(wg_tau_.get(), y_tau_.get(), sizeof(double) * batchSize_, cudaMemcpyDeviceToDevice, stream));
    diff_wb_gap_dot_kernel<<<(int)batchSize_, threads, 0, stream>>>(
        gap_g_wg_.get(), c1_real_, wg_x_.get(), c2_real_, wg_z_.get(),
        c3_real_, wg_tau_.get(), n_, m_, batchSize_);

    auto pack = [&](double* out) {
        diff_wb_pack_jdim_kernel<<<(int)batchSize_, threads, 0, stream>>>(
            out, y_x_.get(), y_z_.get(), y_s_.get(), y_tau_.get(), n_, m_, jdim_, batchSize_);
    };
    auto unpack = [&](const double* in) {
        diff_wb_unpack_jdim_kernel<<<(int)batchSize_, threads, 0, stream>>>(
            y_x_.get(), y_z_.get(), y_s_.get(), y_tau_.get(), in, n_, m_, jdim_, batchSize_);
    };
    // M·w into Mv_ = J'(J·w) + ε·w, where y_* must already hold w. Reuses jw_.
    auto Mw = [&](double* Mv) {
        J_matvec(jw_.get(), stream);          // jw = J·w
        Jt_matvec(jw_.get(), Mv, stream);     // Mv = J'(J·w)
        pack(jw_.get());                      // jw = w (packed)
        double eps = eps_;
        CUBLAS_THROW(cublasDaxpy(cublas_, (int)N, &eps, jw_.get(), 1, Mv, 1));  // Mv += ε·w
    };

    // Compute sol = J·wacc, the residual mres_ = rhs − M·wacc, its per-batch
    // relative norm (resid_norm_), and return the batch max (host). Sets y_* = wacc.
    std::vector<double> rn_host(batchSize_);
    const int blk_jdim = (int)((N + threads - 1) / threads);
    auto residual = [&]() -> double {
        unpack(wacc_jdim_.get());     // y_* = wacc
        J_matvec(y_out, stream);      // sol = J·wacc
        Mw(Mv_.get());                // Mv = M·wacc
        diff_wb_vec_sub_jdim_kernel<<<blk_jdim, threads, 0, stream>>>(
            mres_.get(), rhs_bar, Mv_.get(), N);   // mres = rhs − M·wacc
        diff_wb_relresid_kernel<<<(int)batchSize_, threads, 0, stream>>>(
            resid_norm_.get(), Mv_.get(), rhs_bar, jdim_, batchSize_);
        cudaStreamSynchronize(stream);
        CUDA_THROW(cudaMemcpy(rn_host.data(), resid_norm_.get(), sizeof(double) * batchSize_, cudaMemcpyDeviceToHost));
        double mx = 0.0; for (double v : rn_host) mx = std::max(mx, v); return mx;
    };

    // Solve M w = rhs via the gap Woodbury chain (inner D_x floored → contractive
    // but inaccurate), then ADAPTIVELY refine on the true M — like the forward
    // IPM's convergence loop. Well-conditioned problems converge with 0 steps (no
    // cost); moderate ones refine to tolerance; divergent ones (κ²-limited) break
    // early and backward() falls back to the κ-accurate cuDSS path.
    apply_Minv(rhs_bar, stream);   // y_* = w0
    pack(wacc_jdim_.get());        // wacc = w0
    double rn = residual();        // also sets sol = J·w0, mres_ = residual
    constexpr int kMaxRefine = 12;
    constexpr double kRefineTol = 1e-10;
    for (int it = 0; it < kMaxRefine && rn > kRefineTol; ++it) {
        apply_Minv(mres_.get(), stream);   // y_* = dw = M^{-1}·residual
        pack(jw_.get());                   // jw = dw
        double one = 1.0;
        CUBLAS_THROW(cublasDaxpy(cublas_, (int)N, &one, jw_.get(), 1, wacc_jdim_.get(), 1));  // wacc += dw
        double new_rn = residual();
        if (new_rn >= rn) break;           // stalled / diverging → leave to cuDSS fallback
        rn = new_rn;
    }
}

} // namespace moreau
