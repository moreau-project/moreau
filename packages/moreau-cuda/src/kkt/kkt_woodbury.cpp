/**
 * @file kkt_woodbury.cpp
 * @brief Woodbury/Phi-GEMM KKT solver implementation
 *
 * Solves the KKT system for problems with diagonal P + low-rank/sparse A:
 *
 *   [P + εI    A^T  ] [dx]   [rhs_x]
 *   [  A    -(H+εI) ] [dz] = [rhs_z]
 *
 * Problem structure:
 *   P = diag(D)            (n×n diagonal)
 *   A = [F; A_d; A_s]      (m×n, where m = k + n_nonneg)
 *   F = A_zero             (k×n, zero-cone rows = factor loadings)
 *   A_d = dense nonneg     (k_d rows, each with >1 nonzero, folded into F_all)
 *   A_s = sparse nonneg    (n_sparse rows, each with ≤1 nonzero)
 *   H = [0; diag(w^2)]     (m×m, zero for equality, w^2 for nonneg)
 *
 * Woodbury reduction:
 *   F_all = [A_zero; A_d]  (k_total×n, k_total = k + k_d)
 *   D = P + εI + A_s^T Γ_s A_s   (diagonal, sparse nonneg contribution)
 *   S = C_all^{-1} + F_all^T D^{-1} F_all  (k_total×k_total Schur complement)
 *
 * C_all diagonal:
 *   [0..k):       1/ε         (zero-cone regularization)
 *   [k..k_total): 1/(w_d²+ε)  (dense nonneg elimination)
 *
 * C_all^{-1} (added to S diagonal):
 *   [0..k):       ε
 *   [k..k_total): w_d² + ε
 */

#include "moreau/kkt/kkt_woodbury.hpp"
#include "moreau/cuda/status_utils.hpp"
#include "moreau/kkt/woodbury_kernels.cuh"
#include "moreau/kkt/kkt_kernels.cuh"
#include "moreau/vector/vector_kernels.cuh"
#include <cassert>
#include <cmath>
#include <stdexcept>


namespace moreau {

// ============================================================================
// Static detection
// ============================================================================

bool WoodburyKKTData::isCompatible(
    int64_t n, int64_t m,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones)
{
    // Slack cones: zero + nonneg only (diagonal H).
    if (cones.numExpCones > 0 || cones.numSocCones > 0 || cones.numPowerCones > 0
        || cones.numGenPowerCones > 0 || cones.numPsdCones > 0)
        return false;

    // Direct-x cones: nonneg only (diagonal H_primal). SOC, PSD, and the
    // asymmetric kinds break the diagonal-(1,1) structure that Woodbury
    // exploits.
    for (const auto& xc : cones.dir_cones) {
        if (xc.kind != XConeKind::Nonneg) return false;
    }

    int64_t k = cones.numZeroCones;
    int64_t n_nonneg = cones.numNonnegCones;

    if (k <= 0) return false;
    if (k + n_nonneg != m) return false;

    // P must be diagonal: each row has at most one entry, on the diagonal
    if (nnzP > n) return false;
    for (int64_t i = 0; i < n; ++i) {
        int64_t nnz_row = P_ro[i + 1] - P_ro[i];
        if (nnz_row > 1) return false;
        if (nnz_row == 1 && P_ci[P_ro[i]] != i) return false;
    }

    // Count dense nonneg rows (>1 nnz) to compute k_total
    int64_t k_d = 0;
    for (int64_t i = k; i < m; ++i) {
        if (A_ro[i + 1] - A_ro[i] > 1) k_d++;
    }
    int64_t k_total = k + k_d;

    // Schur complement must be smaller than n
    if (k_total >= n) return false;

    // Every column must be "covered" by either P (nonzero diagonal), a sparse
    // nonneg row (exactly 1 nnz), or a direct-x nonneg cone (which adds a
    // strictly-positive diagonal entry every iteration). Uncovered columns
    // have D[j] ≈ ε, making D^{-1}[j] ≈ 1/ε ≈ 1e7, which poisons the Schur
    // complement with catastrophic cancellation in the Cholesky factorization.
    std::vector<bool> covered(n, false);
    for (int64_t j = 0; j < n; ++j) {
        if (P_ro[j + 1] > P_ro[j]) covered[j] = true;
    }
    for (int64_t i = k; i < m; ++i) {
        if (A_ro[i + 1] - A_ro[i] == 1) {
            covered[A_ci[A_ro[i]]] = true;
        }
    }
    for (const auto& xc : cones.dir_cones) {
        for (int64_t idx : xc.indices) {
            if (idx >= 0 && idx < n) covered[idx] = true;
        }
    }
    for (int64_t j = 0; j < n; ++j) {
        if (!covered[j]) return false;
    }

    return true;
}

// ============================================================================
// Constructor
// ============================================================================

WoodburyKKTData::WoodburyKKTData(
    int64_t n, int64_t m, int64_t batchSize,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones,
    cudaStream_t stream)
    : n_(n), m_(m), k_(cones.numZeroCones), n_nonneg_(cones.numNonnegCones),
      batchSize_(batchSize)
{
    // Classify nonneg rows: sparse (1 nnz), empty (0 nnz), or dense (>1 nnz)
    for (int64_t i = 0; i < n_nonneg_; ++i) {
        int64_t row = k_ + i;
        int64_t nnz_row = A_ro[row + 1] - A_ro[row];
        if (nnz_row == 1) {
            int64_t col = A_ci[A_ro[row]];
            sparse_nonneg_idx_.push_back(i);
            sparse_col_.push_back(col);
        } else if (nnz_row == 0) {
            empty_nonneg_idx_.push_back(i);
        } else {
            dense_nonneg_idx_.push_back(i);
        }
    }
    n_sparse_ = static_cast<int64_t>(sparse_nonneg_idx_.size());
    n_empty_ = static_cast<int64_t>(empty_nonneg_idx_.size());
    k_d_ = static_cast<int64_t>(dense_nonneg_idx_.size());
    k_total_ = k_ + k_d_;

    // Build CSC of A_s: for each column j, which sparse rows touch it
    col_offsets_.resize(n_ + 1, 0);
    for (int64_t sr = 0; sr < n_sparse_; ++sr) {
        col_offsets_[sparse_col_[sr] + 1]++;
    }
    for (int64_t j = 0; j < n_; ++j) {
        col_offsets_[j + 1] += col_offsets_[j];
    }
    col_sparse_rows_.resize(n_sparse_);
    std::vector<int64_t> col_counts(n_, 0);
    for (int64_t sr = 0; sr < n_sparse_; ++sr) {
        int64_t j = sparse_col_[sr];
        int64_t pos = col_offsets_[j] + col_counts[j]++;
        col_sparse_rows_[pos] = sr;
    }

    // Upload sparse structure to device
    if (n_sparse_ > 0) {
        d_sparse_nonneg_idx_ = make_device_unique<int64_t>(n_sparse_);
        d_sparse_col_ = make_device_unique<int64_t>(n_sparse_);
        d_col_sparse_rows_ = make_device_unique<int64_t>(n_sparse_);
        CUDA_THROW(cudaMemcpy(d_sparse_nonneg_idx_.get(), sparse_nonneg_idx_.data(),
                   sizeof(int64_t) * n_sparse_, cudaMemcpyHostToDevice));
        CUDA_THROW(cudaMemcpy(d_sparse_col_.get(), sparse_col_.data(),
                   sizeof(int64_t) * n_sparse_, cudaMemcpyHostToDevice));
        CUDA_THROW(cudaMemcpy(d_col_sparse_rows_.get(), col_sparse_rows_.data(),
                   sizeof(int64_t) * n_sparse_, cudaMemcpyHostToDevice));
    }
    d_col_offsets_ = make_device_unique<int64_t>(n_ + 1);
    CUDA_THROW(cudaMemcpy(d_col_offsets_.get(), col_offsets_.data(),
               sizeof(int64_t) * (n_ + 1), cudaMemcpyHostToDevice));

    if (n_empty_ > 0) {
        d_empty_nonneg_idx_ = make_device_unique<int64_t>(n_empty_);
        CUDA_THROW(cudaMemcpy(d_empty_nonneg_idx_.get(), empty_nonneg_idx_.data(),
                   sizeof(int64_t) * n_empty_, cudaMemcpyHostToDevice));
    }

    // Direct-x nonneg cones: collapse all nonneg x-cones into a flat list of
    // (x_index, hs_offset) pairs. cones.xcone_Hs is laid out per cone block;
    // walk dir_cones in order and accumulate the per-cone offset.
    {
        int64_t hs_off = 0;
        for (const auto& xc : cones.dir_cones) {
            // isCompatible has already enforced kind == Nonneg.
            for (int64_t k = 0; k < static_cast<int64_t>(xc.indices.size()); ++k) {
                xn_idx_host_.push_back(xc.indices[k]);
                xn_hs_off_host_.push_back(hs_off + k);
            }
            hs_off += static_cast<int64_t>(xc.indices.size());
        }
    }
    n_xcone_nonneg_ = static_cast<int64_t>(xn_idx_host_.size());
    if (n_xcone_nonneg_ > 0) {
        d_xn_idx_ = make_device_unique<int64_t>(n_xcone_nonneg_);
        d_xn_hs_off_ = make_device_unique<int64_t>(n_xcone_nonneg_);
        xn_hs_ = make_device_unique<double>(batchSize_ * n_xcone_nonneg_);
        CUDA_THROW(cudaMemcpy(d_xn_idx_.get(), xn_idx_host_.data(),
                   sizeof(int64_t) * n_xcone_nonneg_, cudaMemcpyHostToDevice));
        CUDA_THROW(cudaMemcpy(d_xn_hs_off_.get(), xn_hs_off_host_.data(),
                   sizeof(int64_t) * n_xcone_nonneg_, cudaMemcpyHostToDevice));

        // Reverse lookup x_index -> xn slot (or -1).
        std::vector<int64_t> x_to_slot(n_, -1);
        for (int64_t k = 0; k < n_xcone_nonneg_; ++k) {
            int64_t j = xn_idx_host_[k];
            if (j >= 0 && j < n_) x_to_slot[j] = k;
        }
        d_xn_x_to_slot_ = make_device_unique<int64_t>(n_);
        CUDA_THROW(cudaMemcpy(d_xn_x_to_slot_.get(), x_to_slot.data(),
                   sizeof(int64_t) * n_, cudaMemcpyHostToDevice));
    }

    // Upload dense nonneg indices to device
    if (k_d_ > 0) {
        d_dense_nonneg_idx_ = make_device_unique<int64_t>(k_d_);
        CUDA_THROW(cudaMemcpy(d_dense_nonneg_idx_.get(), dense_nonneg_idx_.data(),
                   sizeof(int64_t) * k_d_, cudaMemcpyHostToDevice));
    }

    CUBLAS_THROW(cublasCreate(&cublas_));
    CUSOLVER_THROW(cusolverDnCreate(&cusolver_));
    handles_owned_ = true;

    // Per-batch arrays
    H_nonneg_ = make_device_unique<double>(batchSize_ * n_nonneg_);
    d_tilde_inv_ = make_device_unique<double>(batchSize_ * n_);
    h_tilde_inv_ = make_device_unique<double>(batchSize_ * n_nonneg_);
    eps_ = make_device_unique<double>(batchSize_);
    C_all_ = make_device_unique<double>(batchSize_ * k_total_);

    // Sparse nonneg values per batch
    if (n_sparse_ > 0) {
        A_s_vals_ = make_device_unique<double>(batchSize_ * n_sparse_);
    }

    // Dense Schur: full k_total x k_total Schur + F_scaled workspace
    S_f64_ = make_device_unique<double>(batchSize_ * k_total_ * k_total_);
    F_scaled_ = make_device_unique<double>(batchSize_ * n_ * k_total_);

    // Solve workspace — 2x for fused solve2
    rhs_x_ = make_device_unique<double>(2 * batchSize_ * n_);
    rhs_nu_ = make_device_unique<double>(2 * batchSize_ * k_total_);
    rhs_lam_ = make_device_unique<double>(2 * batchSize_ * n_nonneg_);
    rhs_x_adj_ = make_device_unique<double>(2 * batchSize_ * n_);
    Dinv_rhs_ = make_device_unique<double>(2 * batchSize_ * n_);
    schur_rhs_ = make_device_unique<double>(2 * batchSize_ * k_total_);
    d_nu_ = make_device_unique<double>(2 * batchSize_ * k_total_);
    Fdnu_ = make_device_unique<double>(2 * batchSize_ * n_);
    dx_ = make_device_unique<double>(2 * batchSize_ * n_);
    dy_nu_ = make_device_unique<double>(2 * batchSize_ * k_total_);
    dlam_ = make_device_unique<double>(2 * batchSize_ * n_nonneg_);
    cusolver_info_ = make_device_unique<int>(batchSize_);

    P_diag_ = make_device_unique<double>(batchSize_ * n_);
    F_all_ = make_device_unique<double>(batchSize_ * n_ * k_total_);

    // KKT-level iterative refinement workspace
    int64_t N = n_ + m_;
    refine_kkt_out_ = make_device_unique<double>(2 * batchSize_ * N);

    build_ptr_arrays();

    // For B==1, query non-batched Cholesky workspace size
    if (batchSize_ == 1) {
        CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));
        int lwork = 0;
        CUSOLVER_THROW(cusolverDnDpotrf_bufferSize(cusolver_, CUBLAS_FILL_MODE_LOWER,
                                     (int)k_total_, S_f64_.get(), (int)k_total_, &lwork));
        potrf_workspace_size_ = lwork;
        potrf_workspace_ = make_device_unique<double>(lwork);
    }

    cudaStreamSynchronize(stream);
}

WoodburyKKTData::~WoodburyKKTData() {
    if (handles_owned_) {
        if (cublas_) cublasDestroy(cublas_);
        if (cusolver_) cusolverDnDestroy(cusolver_);
    }
}

void WoodburyKKTData::build_ptr_arrays() {
    std::vector<double*> h_S_ptrs(batchSize_), h_dnu_ptrs(batchSize_), h_dnu_ptrs2(batchSize_);
    for (int64_t b = 0; b < batchSize_; ++b) {
        h_S_ptrs[b] = S_f64_.get() + b * k_total_ * k_total_;
        h_dnu_ptrs[b] = d_nu_.get() + b * k_total_;
        h_dnu_ptrs2[b] = d_nu_.get() + (batchSize_ + b) * k_total_;
    }
    S_f64_ptrs_ = make_device_unique<double*>(batchSize_);
    dnu_f64_ptrs_ = make_device_unique<double*>(batchSize_);
    dnu_f64_ptrs2_ = make_device_unique<double*>(batchSize_);
    CUDA_THROW(cudaMemcpy(S_f64_ptrs_.get(), h_S_ptrs.data(), sizeof(double*) * batchSize_, cudaMemcpyHostToDevice));
    CUDA_THROW(cudaMemcpy(dnu_f64_ptrs_.get(), h_dnu_ptrs.data(), sizeof(double*) * batchSize_, cudaMemcpyHostToDevice));
    CUDA_THROW(cudaMemcpy(dnu_f64_ptrs2_.get(), h_dnu_ptrs2.data(), sizeof(double*) * batchSize_, cudaMemcpyHostToDevice));
}

// ============================================================================
// populate: extract P diagonal, F_all, and A_s values from CSR
// ============================================================================

void WoodburyKKTData::populate(CSR& P, CSR& A, cudaStream_t stream) {
    const int64_t nnzA = A.nnz();
    const int64_t nnzP = P.nnz();

    // P is diagonal → extract per-batch diagonal values
    // Handle sparse P (rows with 0 entries get P_diag=0)
    if (nnzP == n_) {
        // Fast path: all rows have exactly one entry
        for (int64_t b = 0; b < batchSize_; ++b) {
            CUDA_THROW(cudaMemcpyAsync(P_diag_.get() + b * n_, P.values() + b * nnzP,
                            sizeof(double) * n_, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        // Sparse P: build P_diag on host from CSR structure
        std::vector<int64_t> h_P_ro(n_ + 1);
        CUDA_THROW(cudaMemcpyAsync(h_P_ro.data(), P.rowOffsets(), sizeof(int64_t) * (n_ + 1),
                        cudaMemcpyDeviceToHost, stream));
        std::vector<double> h_P_vals(nnzP * batchSize_);
        CUDA_THROW(cudaMemcpyAsync(h_P_vals.data(), P.values(), sizeof(double) * nnzP * batchSize_,
                        cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        std::vector<double> h_P_diag(batchSize_ * n_, 0.0);
        for (int64_t b = 0; b < batchSize_; ++b) {
            for (int64_t i = 0; i < n_; ++i) {
                if (h_P_ro[i + 1] > h_P_ro[i]) {
                    h_P_diag[b * n_ + i] = h_P_vals[b * nnzP + h_P_ro[i]];
                }
            }
        }
        CUDA_THROW(cudaMemcpyAsync(P_diag_.get(), h_P_diag.data(),
                        sizeof(double) * batchSize_ * n_,
                        cudaMemcpyHostToDevice, stream));
    }

    // Download A structure (shared) and all batch values
    std::vector<int64_t> h_A_ro(m_ + 1);
    std::vector<int64_t> h_A_ci(nnzA);
    std::vector<double> h_A_vals(nnzA * batchSize_);
    CUDA_THROW(cudaMemcpyAsync(h_A_ro.data(), A.rowOffsets(), sizeof(int64_t) * (m_ + 1),
                    cudaMemcpyDeviceToHost, stream));
    CUDA_THROW(cudaMemcpyAsync(h_A_ci.data(), A.colIndices(), sizeof(int64_t) * nnzA,
                    cudaMemcpyDeviceToHost, stream));
    CUDA_THROW(cudaMemcpyAsync(h_A_vals.data(), A.values(), sizeof(double) * nnzA * batchSize_,
                    cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Detect if A values are shared across batches (common in portfolio optimization)
    bool a_vals_shared = true;
    if (batchSize_ > 1) {
        const double* a0 = h_A_vals.data();
        for (int64_t b = 1; b < batchSize_ && a_vals_shared; ++b) {
            if (std::memcmp(a0, h_A_vals.data() + b * nnzA, sizeof(double) * nnzA) != 0) {
                a_vals_shared = false;
            }
        }
    }
    f_all_shared_ = a_vals_shared;
    a_s_shared_ = false;  // A_s_vals always per-batch (kernels index by batch)

    const int64_t B_F = f_all_shared_ ? 1 : batchSize_;
    f_all_stride_ = f_all_shared_ ? 0 : (long long)(n_ * k_total_);

    // Build F_all (B_F, n, k_total) row-major and A_s_vals (B_As, n_sparse)
    std::vector<double> h_F_all(B_F * n_ * k_total_, 0.0);
    const int64_t B_As = batchSize_;
    std::vector<double> h_A_s_vals(B_As * n_sparse_, 0.0);

    for (int64_t b = 0; b < B_F; ++b) {
        const double* A_vals_b = h_A_vals.data() + b * nnzA;
        double* h_F = h_F_all.data() + b * n_ * k_total_;

        // Zero-cone rows: F_all[j, i] = A_b[i, j] for i < k
        // (stored as (n, k_total) row-major so cuBLAS sees (k_total, n) col-major)
        for (int64_t i = 0; i < k_; ++i) {
            for (int64_t p = h_A_ro[i]; p < h_A_ro[i + 1]; ++p) {
                int64_t j = h_A_ci[p];
                h_F[j * k_total_ + i] = A_vals_b[p];
            }
        }

        // Dense nonneg rows: F_all[j, k+d] = A_b[k+dense_nonneg_idx_[d], j]
        for (int64_t d = 0; d < k_d_; ++d) {
            int64_t nonneg_i = dense_nonneg_idx_[d];
            int64_t row = k_ + nonneg_i;
            for (int64_t p = h_A_ro[row]; p < h_A_ro[row + 1]; ++p) {
                int64_t j = h_A_ci[p];
                h_F[j * k_total_ + (k_ + d)] = A_vals_b[p];
            }
        }
    }

    for (int64_t b = 0; b < B_As; ++b) {
        const double* A_vals_b = h_A_vals.data() + b * nnzA;
        double* h_As_b = h_A_s_vals.data() + b * n_sparse_;
        for (int64_t sr = 0; sr < n_sparse_; ++sr) {
            int64_t nonneg_i = sparse_nonneg_idx_[sr];
            int64_t row = k_ + nonneg_i;
            h_As_b[sr] = A_vals_b[h_A_ro[row]];
        }
    }

    // Upload F_all (shared: 1 copy, per-batch: B copies)
    // Reallocate if needed (first populate may have allocated for B, now we need 1)
    if (f_all_shared_ && B_F == 1) {
        F_all_ = make_device_unique<double>(n_ * k_total_);
    }
    CUDA_THROW(cudaMemcpyAsync(F_all_.get(), h_F_all.data(), sizeof(double) * B_F * n_ * k_total_,
                    cudaMemcpyHostToDevice, stream));

    // Upload A_s_vals
    if (n_sparse_ > 0) {
        CUDA_THROW(cudaMemcpyAsync(A_s_vals_.get(), h_A_s_vals.data(),
                        sizeof(double) * batchSize_ * n_sparse_,
                        cudaMemcpyHostToDevice, stream));
    }

    cudaStreamSynchronize(stream);
    populated_ = true;
}

// ============================================================================
// update_H: extract nonneg H diagonal from cone scaling
// ============================================================================

void WoodburyKKTData::update_H(const Cones& cones, const double* /*mu_data*/, cudaStream_t stream) {
    if (n_nonneg_ > 0) {
        launch_woodbury_compute_H_nonneg(
            H_nonneg_.get(), cones.nonneg_w.data(),
            batchSize_ * n_nonneg_, stream);
    }
    // Gather direct-x nonneg Hs from the xcone scaling output. The fwd
    // scaling kernel (`update_xcones_nonneg_scaling_kernel`) has already
    // populated `cones.xcone_Hs[]` with `Hs_i = z_x/x[i]` for each
    // direct-x nonneg slot; we just project onto the (B, n_xn) layout
    // d_tilde_inv expects.
    if (n_xcone_nonneg_ > 0) {
        launch_woodbury_gather_xcone_nonneg_hs(
            xn_hs_.get(), cones.xcone_Hs.data(), d_xn_hs_off_.get(),
            cones.totalXConeHsEntries, batchSize_, n_xcone_nonneg_, stream);
    }
}

// ============================================================================
// regularize_and_refactor: form Schur complement, Cholesky factor
// ============================================================================

bool WoodburyKKTData::regularize_and_refactor(
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    cudaStream_t stream)
{
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));

    // 1. Compute per-batch regularization eps
    if (static_regularization_enable) {
        launch_woodbury_max_diag_sparse(
            eps_.get(), P_diag_.get(), H_nonneg_.get(),
            A_s_vals_.get(),
            d_sparse_nonneg_idx_.get(),
            d_col_offsets_.get(), d_col_sparse_rows_.get(),
            batchSize_, n_, n_nonneg_, n_sparse_, stream);
        launch_woodbury_compute_eps(
            eps_.get(), static_regularization_constant,
            static_regularization_proportional, batchSize_, stream);
    } else {
        launch_woodbury_fill_constant(eps_.get(), 1e-13, batchSize_, stream);
    }

    // 2. Fused h_tilde_inv + d_tilde_inv using sparse CSC structure
    //    (dense nonneg rows do NOT contribute to D — they go into F_all).
    //    Direct-x nonneg cones contribute Hs_i to D[i] for i in their index
    //    set (gathered into xn_hs_ in update_H).
    launch_woodbury_d_tilde_inv_sparse(
        d_tilde_inv_.get(), h_tilde_inv_.get(),
        P_diag_.get(), H_nonneg_.get(),
        A_s_vals_.get(), eps_.get(),
        d_sparse_nonneg_idx_.get(),
        d_col_offsets_.get(), d_col_sparse_rows_.get(),
        n_xcone_nonneg_ > 0 ? xn_hs_.get() : nullptr,
        n_xcone_nonneg_ > 0 ? d_xn_x_to_slot_.get() : nullptr,
        n_xcone_nonneg_,
        batchSize_, n_, n_nonneg_, n_sparse_, stream);

    // 3. Form dense Schur complement and factor
    //    a) F_scaled = diag(d_tilde_inv) * F_all
    launch_woodbury_scale_F(
        F_scaled_.get(), F_all_.get(), d_tilde_inv_.get(),
        batchSize_, n_, k_total_, f_all_shared_, stream);

    //    b) S = F_scaled^T @ F_all via batched GEMM
    {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int)k_total_, (int)k_total_, (int)n_,
                    &alpha,
                    F_scaled_.get(), (int)k_total_, (long long)(n_ * k_total_),
                    F_all_.get(), (int)k_total_, f_all_stride_,
                    &beta,
                    S_f64_.get(), (int)k_total_, (long long)(k_total_ * k_total_),
                    (int)batchSize_));
    }

    //    c) Add C_all^{-1} on diagonal of S
    if (k_d_ > 0) {
        launch_woodbury_set_C_all_inv_diagonal(
            S_f64_.get(), eps_.get(), H_nonneg_.get(),
            d_dense_nonneg_idx_.get(),
            batchSize_, k_, k_d_, k_total_, n_nonneg_, stream);
    } else {
        launch_woodbury_set_identity_eps(
            S_f64_.get(), eps_.get(), batchSize_, k_total_, stream);
    }

    // 4. Cholesky factorization
    if (batchSize_ == 1) {
        CUSOLVER_THROW(cusolverDnDpotrf(cusolver_, CUBLAS_FILL_MODE_LOWER,
            (int)k_total_, S_f64_.get(), (int)k_total_,
            potrf_workspace_.get(), potrf_workspace_size_,
            cusolver_info_.get()));
    } else {
        CUSOLVER_THROW(cusolverDnDpotrfBatched(
            cusolver_, CUBLAS_FILL_MODE_LOWER,
            (int)k_total_, S_f64_ptrs_.get(), (int)k_total_,
            cusolver_info_.get(), (int)batchSize_));
    }

    // Check Cholesky success (info[b] > 0 means not positive definite)
    {
        std::vector<int> h_info(batchSize_);
        CUDA_THROW(cudaMemcpy(h_info.data(), cusolver_info_.get(),
                   sizeof(int) * batchSize_, cudaMemcpyDeviceToHost));
        for (int64_t b = 0; b < batchSize_; ++b) {
            if (h_info[b] != 0) return false;
        }
    }

    // C_all = [1/eps, ..., 1/(w_d^2+eps), ...] for dy/dlam_d recovery
    if (k_d_ > 0) {
        launch_woodbury_compute_C_all(
            C_all_.get(), eps_.get(), H_nonneg_.get(),
            d_dense_nonneg_idx_.get(),
            batchSize_, k_, k_d_, k_total_, n_nonneg_, stream);
    } else {
        launch_woodbury_compute_C_diag(
            C_all_.get(), eps_.get(), batchSize_, k_total_, stream);
    }

    return true;
}

// ============================================================================
// solve: given factored S, solve KKT system
// ============================================================================

void WoodburyKKTData::solve(const double* rhs, double* sol, cudaStream_t stream) {
    solve_impl(rhs, sol, 1, stream);
}

void WoodburyKKTData::solve2(const double* rhs, double* sol, cudaStream_t stream) {
    solve_impl(rhs, sol, 2, stream);
}

// ============================================================================
// solve_core: Woodbury solve without refinement
// ============================================================================

void WoodburyKKTData::solve_core(const double* rhs, double* sol, int nrhs, cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));
    CUSOLVER_THROW(cusolverDnSetStream(cusolver_, stream));

    // Fast path: fused single-kernel solve for B=1, n <= 1024
    if (batchSize_ == 1 && n_ <= 256) {
        launch_woodbury_fused_solve(
            sol, rhs, F_all_.get(), d_tilde_inv_.get(),
            h_tilde_inv_.get(), C_all_.get(), S_f64_.get(),
            A_s_vals_.get(),
            d_sparse_nonneg_idx_.get(), d_sparse_col_.get(),
            d_col_offsets_.get(), d_col_sparse_rows_.get(),
            d_dense_nonneg_idx_.get(), d_empty_nonneg_idx_.get(),
            n_, m_, k_, k_d_, k_total_,
            n_nonneg_, n_sparse_, n_empty_,
            nrhs, stream);
        return;
    }

    const int64_t eB = nrhs * batchSize_;
    const int64_t N = n_ + m_;

    // 1. Unpack [rhs_x(n); rhs_y(k); rhs_lam(n_nonneg)] from rhs(N) per batch
    if (k_d_ > 0) {
        // Phase 2: unpack rhs_y into rhs_nu_ with stride k_total (not k)
        launch_woodbury_unpack_rhs_kt(
            rhs, rhs_x_.get(), rhs_nu_.get(), rhs_lam_.get(),
            eB, n_, k_, k_total_, n_nonneg_, stream);
        // Gather dense nonneg RHS into rhs_nu_[k..k_total]
        launch_woodbury_gather_dense_rhs(
            rhs_nu_.get(), rhs_lam_.get(),
            d_dense_nonneg_idx_.get(),
            eB, k_, k_total_, n_nonneg_, stream);
    } else {
        // Phase 1: k_total == k, stride is k (standard)
        launch_woodbury_unpack_rhs(
            rhs, rhs_x_.get(), rhs_nu_.get(), rhs_lam_.get(),
            eB, n_, k_, n_nonneg_, stream);
    }

    // 2. rhs_adj = rhs_x + Σ a_r * h_inv_r * rhs_lam_r (per column via CSC)
    if (n_sparse_ > 0) {
        for (int r = 0; r < nrhs; ++r) {
            launch_woodbury_rhs_adj_sparse(
                rhs_x_adj_.get() + r * batchSize_ * n_,
                rhs_x_.get() + r * batchSize_ * n_,
                h_tilde_inv_.get(),
                rhs_lam_.get() + r * batchSize_ * n_nonneg_,
                A_s_vals_.get(),
                d_sparse_nonneg_idx_.get(),
                d_col_offsets_.get(), d_col_sparse_rows_.get(),
                batchSize_, n_, n_nonneg_, n_sparse_, stream);
        }
    } else {
        // No sparse nonneg rows — rhs_adj = rhs_x
        CUDA_THROW(cudaMemcpyAsync(rhs_x_adj_.get(), rhs_x_.get(), sizeof(double) * eB * n_,
                        cudaMemcpyDeviceToDevice, stream));
    }

    // 3. b = rhs_adj + F_all^T @ (C_all * rhs_nu)
    for (int r = 0; r < nrhs; ++r) {
        launch_woodbury_elementwise_mul(
            schur_rhs_.get() + r * batchSize_ * k_total_,
            C_all_.get(),
            rhs_nu_.get() + r * batchSize_ * k_total_,
            batchSize_ * k_total_, stream);
    }
    if (batchSize_ == 1 && nrhs == 2) {
        // B==1: both RHS are contiguous columns, use single GEMM with N=2
        double alpha = 1.0, beta = 1.0;
        CUBLAS_THROW(cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, nrhs, (int)k_total_,
                    &alpha,
                    F_all_.get(), (int)k_total_,
                    schur_rhs_.get(), (int)k_total_,
                    &beta,
                    rhs_x_adj_.get(), (int)n_));
    } else {
        for (int r = 0; r < nrhs; ++r) {
            double alpha = 1.0, beta = 1.0;
            CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)n_, 1, (int)k_total_,
                        &alpha,
                        F_all_.get(), (int)k_total_, f_all_stride_,
                        schur_rhs_.get() + r * batchSize_ * k_total_, (int)k_total_, (long long)k_total_,
                        &beta,
                        rhs_x_adj_.get() + r * batchSize_ * n_, (int)n_, (long long)n_,
                        (int)batchSize_));
        }
    }

    // 4. u = D^{-1} b
    for (int r = 0; r < nrhs; ++r) {
        launch_woodbury_elementwise_mul(
            Dinv_rhs_.get() + r * batchSize_ * n_,
            d_tilde_inv_.get(),
            rhs_x_adj_.get() + r * batchSize_ * n_,
            batchSize_ * n_, stream);
    }

    // 5. v = F_all @ u
    if (batchSize_ == 1 && nrhs == 2) {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_total_, nrhs, (int)n_,
                    &alpha,
                    F_all_.get(), (int)k_total_,
                    Dinv_rhs_.get(), (int)n_,
                    &beta,
                    schur_rhs_.get(), (int)k_total_));
    } else {
        for (int r = 0; r < nrhs; ++r) {
            double alpha = 1.0, beta = 0.0;
            CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)k_total_, 1, (int)n_,
                        &alpha,
                        F_all_.get(), (int)k_total_, f_all_stride_,
                        Dinv_rhs_.get() + r * batchSize_ * n_, (int)n_, (long long)n_,
                        &beta,
                        schur_rhs_.get() + r * batchSize_ * k_total_, (int)k_total_, (long long)k_total_,
                        (int)batchSize_));
        }
    }

    // 6. w = S^{-1} v (Cholesky solve)
    CUDA_THROW(cudaMemcpyAsync(d_nu_.get(), schur_rhs_.get(),
                    sizeof(double) * eB * k_total_, cudaMemcpyDeviceToDevice, stream));
    if (batchSize_ == 1) {
        CUSOLVER_THROW(cusolverDnDpotrs(cusolver_, CUBLAS_FILL_MODE_LOWER,
            (int)k_total_, nrhs, S_f64_.get(), (int)k_total_,
            d_nu_.get(), (int)k_total_, cusolver_info_.get()));
    } else {
        CUSOLVER_THROW(cusolverDnDpotrsBatched(cusolver_, CUBLAS_FILL_MODE_LOWER,
            (int)k_total_, 1, S_f64_ptrs_.get(), (int)k_total_,
            dnu_f64_ptrs_.get(), (int)k_total_,
            cusolver_info_.get(), (int)batchSize_));
        if (nrhs == 2) {
            CUSOLVER_THROW(cusolverDnDpotrsBatched(cusolver_, CUBLAS_FILL_MODE_LOWER,
                (int)k_total_, 1, S_f64_ptrs_.get(), (int)k_total_,
                dnu_f64_ptrs2_.get(), (int)k_total_,
                cusolver_info_.get(), (int)batchSize_));
        }
    }

    // 7. Fw = F_all^T @ w
    if (batchSize_ == 1 && nrhs == 2) {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, nrhs, (int)k_total_,
                    &alpha,
                    F_all_.get(), (int)k_total_,
                    d_nu_.get(), (int)k_total_,
                    &beta,
                    Fdnu_.get(), (int)n_));
    } else {
        for (int r = 0; r < nrhs; ++r) {
            double alpha = 1.0, beta = 0.0;
            CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)n_, 1, (int)k_total_,
                        &alpha,
                        F_all_.get(), (int)k_total_, f_all_stride_,
                        d_nu_.get() + r * batchSize_ * k_total_, (int)k_total_, (long long)k_total_,
                        &beta,
                        Fdnu_.get() + r * batchSize_ * n_, (int)n_, (long long)n_,
                        (int)batchSize_));
        }
    }

    // 8. dx = d_tilde_inv * (b - Fw)
    for (int r = 0; r < nrhs; ++r) {
        launch_woodbury_dx_recovery(
            dx_.get() + r * batchSize_ * n_,
            d_tilde_inv_.get(),
            rhs_x_adj_.get() + r * batchSize_ * n_,
            Fdnu_.get() + r * batchSize_ * n_,
            batchSize_ * n_, stream);
    }

    // 9. dy_nu = d_nu - C_all * rhs_nu (combined zero-cone dy + dense nonneg dlam_d)
    for (int r = 0; r < nrhs; ++r) {
        launch_woodbury_dy_recovery(
            dy_nu_.get() + r * batchSize_ * k_total_,
            d_nu_.get() + r * batchSize_ * k_total_,
            C_all_.get(),
            rhs_nu_.get() + r * batchSize_ * k_total_,
            batchSize_, k_total_, stream);
    }

    // 10. dlam recovery
    if (n_nonneg_ > 0) {
        for (int r = 0; r < nrhs; ++r) {
            // Sparse rows: dlam[ni] = h_inv[ni] * (a * dx[col] - rhs_lam[ni])
            if (n_sparse_ > 0) {
                launch_woodbury_dlam_sparse(
                    dlam_.get() + r * batchSize_ * n_nonneg_,
                    h_tilde_inv_.get(),
                    dx_.get() + r * batchSize_ * n_,
                    rhs_lam_.get() + r * batchSize_ * n_nonneg_,
                    A_s_vals_.get(),
                    d_sparse_col_.get(), d_sparse_nonneg_idx_.get(),
                    batchSize_, n_, n_nonneg_, n_sparse_, stream);
            }
            // Empty rows: dlam[ni] = -h_inv[ni] * rhs_lam[ni]
            if (n_empty_ > 0) {
                launch_woodbury_dlam_empty(
                    dlam_.get() + r * batchSize_ * n_nonneg_,
                    h_tilde_inv_.get(),
                    rhs_lam_.get() + r * batchSize_ * n_nonneg_,
                    d_empty_nonneg_idx_.get(),
                    batchSize_, n_nonneg_, n_empty_, stream);
            }
            // Dense rows: scatter from dy_nu_[k..k_total] into dlam at dense positions
            if (k_d_ > 0) {
                launch_woodbury_scatter_dlam_dense(
                    dlam_.get() + r * batchSize_ * n_nonneg_,
                    dy_nu_.get() + r * batchSize_ * k_total_,
                    batchSize_, k_, k_total_, n_nonneg_,
                    d_dense_nonneg_idx_.get(), k_d_, stream);
            }
        }
    }

    // 11. Pack [dx; dy; dlam] into sol
    //     dy is the first k entries of dy_nu_
    if (k_d_ > 0) {
        // Phase 2: dy_nu_ has stride k_total, need to read first k with k_total stride
        launch_woodbury_pack_sol_kt(
            sol, dx_.get(), dy_nu_.get(), dlam_.get(),
            eB, n_, k_, k_total_, n_nonneg_, stream);
    } else {
        // Phase 1: k_total == k, stride is k (standard)
        launch_woodbury_pack_sol(
            sol, dx_.get(), dy_nu_.get(), dlam_.get(),
            eB, n_, k_, n_nonneg_, stream);
    }
}

// ============================================================================
// kkt_matvec: compute out = KKT * sol using Woodbury structure
//
// KKT * sol decomposes as:
//   out_x     = (P_diag + eps) * dx + F_all^T * [dy; dlam_d] + A_s^T * dlam_s
//   out_nu    = F_all @ dx - C_all_inv * [dy; dlam_d]
//   out_lam_s = a_r * dx[col_r] - (w_r^2+eps) * dlam_r   (sparse nonneg)
//   out_lam_e = -(w_e^2+eps) * dlam_e                     (empty nonneg)
//
// where C_all_inv[i] = eps for i<k (zero-cone), (H+eps) for i>=k (dense nonneg).
// ============================================================================

void WoodburyKKTData::kkt_matvec(const double* sol, double* out, int nrhs, cudaStream_t stream) {
    CUBLAS_THROW(cublasSetStream(cublas_, stream));

    const int64_t eB = nrhs * batchSize_;

    // Unpack sol into dx_, dy_nu_ (k_total entries), dlam_ (n_nonneg entries)
    if (k_d_ > 0) {
        launch_woodbury_unpack_rhs_kt(
            sol, dx_.get(), dy_nu_.get(), dlam_.get(),
            eB, n_, k_, k_total_, n_nonneg_, stream);
        launch_woodbury_gather_dense_rhs(
            dy_nu_.get(), dlam_.get(),
            d_dense_nonneg_idx_.get(),
            eB, k_, k_total_, n_nonneg_, stream);
    } else {
        launch_woodbury_unpack_rhs(
            sol, dx_.get(), dy_nu_.get(), dlam_.get(),
            eB, n_, k_, n_nonneg_, stream);
    }

    // --- out_x = (P_diag + eps) * dx ---
    for (int r = 0; r < nrhs; ++r) {
        launch_woodbury_kkt_matvec_x(
            rhs_x_adj_.get() + r * batchSize_ * n_,
            dx_.get() + r * batchSize_ * n_,
            P_diag_.get(), eps_.get(),
            batchSize_, n_, stream);
    }

    // --- out_x += F_all^T @ [dy; dlam_d] ---
    if (batchSize_ == 1 && nrhs == 2) {
        double alpha = 1.0, beta = 1.0;
        CUBLAS_THROW(cublasDgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_, nrhs, (int)k_total_,
                    &alpha,
                    F_all_.get(), (int)k_total_,
                    dy_nu_.get(), (int)k_total_,
                    &beta,
                    rhs_x_adj_.get(), (int)n_));
    } else {
        for (int r = 0; r < nrhs; ++r) {
            double alpha = 1.0, beta = 1.0;
            CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)n_, 1, (int)k_total_,
                        &alpha,
                        F_all_.get(), (int)k_total_, f_all_stride_,
                        dy_nu_.get() + r * batchSize_ * k_total_, (int)k_total_, (long long)k_total_,
                        &beta,
                        rhs_x_adj_.get() + r * batchSize_ * n_, (int)n_, (long long)n_,
                        (int)batchSize_));
        }
    }

    // --- out_x += A_s^T * dlam_s ---
    if (n_sparse_ > 0) {
        for (int r = 0; r < nrhs; ++r) {
            launch_woodbury_kkt_As_T_dlam(
                rhs_x_adj_.get() + r * batchSize_ * n_,
                dlam_.get() + r * batchSize_ * n_nonneg_,
                A_s_vals_.get(),
                d_sparse_nonneg_idx_.get(),
                d_col_offsets_.get(), d_col_sparse_rows_.get(),
                batchSize_, n_, n_nonneg_, n_sparse_, stream);
        }
    }

    // --- out_nu = F_all @ dx ---
    if (batchSize_ == 1 && nrhs == 2) {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_THROW(cublasDgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)k_total_, nrhs, (int)n_,
                    &alpha,
                    F_all_.get(), (int)k_total_,
                    dx_.get(), (int)n_,
                    &beta,
                    schur_rhs_.get(), (int)k_total_));
    } else {
        for (int r = 0; r < nrhs; ++r) {
            double alpha = 1.0, beta = 0.0;
            CUBLAS_THROW(cublasDgemmStridedBatched(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)k_total_, 1, (int)n_,
                        &alpha,
                        F_all_.get(), (int)k_total_, f_all_stride_,
                        dx_.get() + r * batchSize_ * n_, (int)n_, (long long)n_,
                        &beta,
                        schur_rhs_.get() + r * batchSize_ * k_total_, (int)k_total_, (long long)k_total_,
                        (int)batchSize_));
        }
    }
    // Subtract C_all_inv * dy_nu from schur_rhs (= F_all @ dx) in-place.
    // C_all_inv[i] = eps for i<k, (H_nonneg[dense_ni]+eps) for i>=k.
    launch_woodbury_kkt_matvec_nu_sub(
        schur_rhs_.get(), dy_nu_.get(),
        eps_.get(), H_nonneg_.get(),
        d_dense_nonneg_idx_.get(),
        batchSize_, eB,
        k_, k_d_, k_total_, n_nonneg_, stream);
    // --- Sparse nonneg: out = a_r * dx[col_r] - (w_r^2 + eps) * dlam_r ---
    if (n_sparse_ > 0) {
        for (int r = 0; r < nrhs; ++r) {
            launch_woodbury_kkt_matvec_sparse_z(
                rhs_lam_.get() + r * batchSize_ * n_nonneg_,
                dx_.get() + r * batchSize_ * n_,
                dlam_.get() + r * batchSize_ * n_nonneg_,
                H_nonneg_.get(), eps_.get(),
                A_s_vals_.get(),
                d_sparse_col_.get(), d_sparse_nonneg_idx_.get(),
                batchSize_, n_, n_nonneg_, n_sparse_, stream);
        }
    }

    // --- Empty nonneg: out = -(w_r^2 + eps) * dlam_e ---
    if (n_empty_ > 0) {
        for (int r = 0; r < nrhs; ++r) {
            launch_woodbury_kkt_matvec_empty_z(
                rhs_lam_.get() + r * batchSize_ * n_nonneg_,
                dlam_.get() + r * batchSize_ * n_nonneg_,
                H_nonneg_.get(), eps_.get(),
                d_empty_nonneg_idx_.get(),
                batchSize_, n_nonneg_, n_empty_, stream);
        }
    }

    // --- Pack output: out = [out_x; out_y; out_lam] ---
    // out_x is in rhs_x_adj_
    // out_nu (= out_y + dense nonneg) is in schur_rhs_
    // out_lam (sparse/empty) is in rhs_lam_
    if (k_d_ > 0) {
        // Scatter dense nonneg outputs from schur_rhs[k..k_total] into rhs_lam_
        for (int r = 0; r < nrhs; ++r) {
            launch_woodbury_scatter_dlam_dense(
                rhs_lam_.get() + r * batchSize_ * n_nonneg_,
                schur_rhs_.get() + r * batchSize_ * k_total_,
                batchSize_, k_, k_total_, n_nonneg_,
                d_dense_nonneg_idx_.get(), k_d_, stream);
        }
        launch_woodbury_pack_sol_kt(
            out, rhs_x_adj_.get(), schur_rhs_.get(), rhs_lam_.get(),
            eB, n_, k_, k_total_, n_nonneg_, stream);
    } else {
        launch_woodbury_pack_sol(
            out, rhs_x_adj_.get(), schur_rhs_.get(), rhs_lam_.get(),
            eB, n_, k_, n_nonneg_, stream);
    }
}


// ============================================================================
// solve_impl: solve_core + 1 step of KKT-level iterative refinement
// ============================================================================

void WoodburyKKTData::solve_impl(const double* rhs, double* sol, int nrhs, cudaStream_t stream) {
    const int64_t eB = nrhs * batchSize_;
    const int64_t N = n_ + m_;

    // Step 1: Initial solve
    solve_core(rhs, sol, nrhs, stream);

    // Step 2: KKT-level iterative refinement (1 step)
    // Compute KKT residual: r = rhs - KKT * sol
    kkt_matvec(sol, refine_kkt_out_.get(), nrhs, stream);
    launch_woodbury_refine_residual(
        refine_kkt_out_.get(), rhs, refine_kkt_out_.get(),
        eB * N, stream);

    // Step 3: Solve KKT * delta = r using same factorization.
    // rhs == sol aliasing is safe: unpack reads rhs into internal buffers first,
    // pack writes sol last after all reads are done.
    solve_core(refine_kkt_out_.get(), refine_kkt_out_.get(), nrhs, stream);

    // Step 4: sol += delta
    launch_woodbury_axpy(sol, refine_kkt_out_.get(), eB * N, stream);
}

// ============================================================================
// Compound update methods
// ============================================================================

bool WoodburyKKTData::update(
    Cones& cones, const BatchedVector& s, const BatchedVector& z,
    const BatchedVector& mu, ScalingStrategy scaling,
    bool sreg_enable, double sreg_const, double sreg_prop,
    const BatchedVector& q, const BatchedVector& b,
    BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
    BatchedVector& x2, BatchedVector& z2, cudaStream_t stream)
{
    if (!populated_) throw std::logic_error("WoodburyKKTData::update() called before populate()");

    bool success = cones.update_scaling(s, z, mu, scaling, stream);
    if (!success) return false;

    update_H(cones, nullptr, stream);
    success = regularize_and_refactor(sreg_enable, sreg_const, sreg_prop, stream);
    if (!success) return false;

    axpby(workx, -1.0, q, 0.0, stream);
    pack_const_rhs(workx.data(), b.data(), const_rhs.data(), n_, m_, 0, batchSize_, stream);
    solve(const_rhs.data(), const_sol.data(), stream);
    unpack_const_sol(const_sol.data(), x2.data(), z2.data(), n_, m_, 0, batchSize_, stream);
    return true;
}

bool WoodburyKKTData::update(
    Cones& cones, const BatchedVector& s, const BatchedVector& z,
    const BatchedVector& mu, ScalingStrategy scaling,
    bool sreg_enable, double sreg_const, double sreg_prop,
    cudaStream_t stream)
{
    if (!populated_) throw std::logic_error("WoodburyKKTData::update() called before populate()");

    bool success = cones.update_scaling(s, z, mu, scaling, stream);
    if (!success) return false;

    update_H(cones, nullptr, stream);
    return regularize_and_refactor(sreg_enable, sreg_const, sreg_prop, stream);
}

bool WoodburyKKTData::updateFactorOnly(
    Cones& cones, const BatchedVector& s, const BatchedVector& z,
    const BatchedVector& mu, ScalingStrategy scaling,
    bool sreg_enable, double sreg_const, double sreg_prop,
    const BatchedVector& q, const BatchedVector& b,
    BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
    BatchedVector& x2, BatchedVector& z2, cudaStream_t stream)
{
    if (!populated_) throw std::logic_error("WoodburyKKTData::updateFactorOnly() called before populate()");

    update_H(cones, nullptr, stream);
    bool success = regularize_and_refactor(sreg_enable, sreg_const, sreg_prop, stream);
    if (!success) return false;

    axpby(workx, -1.0, q, 0.0, stream);
    pack_const_rhs(workx.data(), b.data(), const_rhs.data(), n_, m_, 0, batchSize_, stream);
    solve(const_rhs.data(), const_sol.data(), stream);
    unpack_const_sol(const_sol.data(), x2.data(), z2.data(), n_, m_, 0, batchSize_, stream);
    return true;
}

size_t WoodburyKKTData::memoryUsage() const noexcept {
    size_t mem = 0;
    int64_t B_F = f_all_shared_ ? 1 : batchSize_;
    mem += sizeof(double) * B_F * n_ * k_total_;               // F_all (shared or per-batch)
    mem += sizeof(double) * batchSize_ * n_;                   // P_diag (per-batch)
    mem += sizeof(double) * batchSize_ * n_nonneg_;           // H_nonneg
    mem += sizeof(double) * batchSize_ * n_;                   // d_tilde_inv
    mem += sizeof(double) * batchSize_ * n_nonneg_;           // h_tilde_inv
    mem += sizeof(double) * batchSize_ * n_ * k_total_;       // F_scaled
    mem += sizeof(double) * batchSize_ * k_total_ * k_total_; // S
    mem += sizeof(double) * batchSize_ * n_sparse_;            // A_s_vals
    // Solve workspace: 2x for solve2
    mem += sizeof(double) * 2 * batchSize_ * (n_ * 5 + k_total_ * 4 + n_nonneg_ * 2);
    // KKT refinement workspace
    mem += sizeof(double) * 2 * batchSize_ * (n_ + m_);
    // Sparse structure (device)
    mem += sizeof(int64_t) * (2 * n_sparse_ + n_ + 1 + n_sparse_ + n_empty_ + k_d_);
    return mem;
}

} // namespace moreau
