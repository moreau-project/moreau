/**
 * @file diff.cu
 * @brief GPU differentiation implementation for conic optimization
 */

#include "moreau/diff/diff.hpp"
#include "moreau/diff/diff_kkt.hpp"
#include "moreau/diff/diff_woodbury.hpp"
#include "moreau/diff/diff_kernels.cuh"
#include "moreau/diff/smoothed_kernels.cuh"
#include "moreau/solver/solver.hpp"
#include "moreau/kkt/kkt.hpp"
#include "moreau/kkt/kkt_solver.hpp"
#include "moreau/kkt/kkt_woodbury.hpp"
#include "moreau/vector/vector_kernels.cuh"
#include "moreau/residuals/residuals_kernels.cuh"

#include "moreau/debug.hpp"

#include <stdexcept>

namespace moreau {

// ============================================================================
// DiffState Implementation
// ============================================================================

DiffState::DiffState(
    int64_t n_, int64_t m_, int64_t batchSize_,
    int64_t nnzP, int64_t nnzA,
    cudaStream_t stream,
    int64_t totalXConeNumel_
)
    : n(n_), m(m_), batchSize(batchSize_),
      x(n_, batchSize_),
      z(m_, batchSize_),
      s(m_, batchSize_),
      tau(1, batchSize_),
      u(m_, batchSize_),
      pi_u(m_, batchSize_),
      z_x(totalXConeNumel_ > 0 ? totalXConeNumel_ : 1, batchSize_),
      totalXConeNumel(totalXConeNumel_),
      d(n_, batchSize_),
      dinv(n_, batchSize_),
      e(m_, batchSize_),
      einv(m_, batchSize_),
      c_scale(1, batchSize_),
      work_n(n_, batchSize_),
      work_m(m_, batchSize_),
      work_m2(m_, batchSize_),
      // HSDE size: n + 2m + xn + 1 (when direct-x cones present, the
      // augmented system adds xn extra rows/cols).
      rhs(n_ + 2 * m_ + totalXConeNumel_ + 1, batchSize_),
      sol(n_ + 2 * m_ + totalXConeNumel_ + 1, batchSize_),
      c1(n_, batchSize_),
      c2(m_, batchSize_),
      c3(1, batchSize_),
      smoothing_x(n_, batchSize_),
      smoothing_z(m_, batchSize_),
      smoothing_s(m_, batchSize_),
      smoothing_mu(1, batchSize_),
      dP_values(nnzP, batchSize_),
      dq(n_, batchSize_),
      dA_values(nnzA, batchSize_),
      db(m_, batchSize_)
{
    // Initialize all work vectors to zero
    x.setToConstant(0.0, stream);
    z.setToConstant(0.0, stream);
    s.setToConstant(0.0, stream);
    tau.setToConstant(1.0, stream);
    u.setToConstant(0.0, stream);
    pi_u.setToConstant(0.0, stream);
    d.setToConstant(1.0, stream);
    dinv.setToConstant(1.0, stream);
    e.setToConstant(1.0, stream);
    einv.setToConstant(1.0, stream);
    c_scale.setToConstant(1.0, stream);
    work_n.setToConstant(0.0, stream);
    work_m.setToConstant(0.0, stream);
    work_m2.setToConstant(0.0, stream);
    rhs.setToConstant(0.0, stream);
    sol.setToConstant(0.0, stream);
    c1.setToConstant(0.0, stream);
    c2.setToConstant(0.0, stream);
    c3.setToConstant(0.0, stream);
    smoothing_x.setToConstant(0.0, stream);
    smoothing_z.setToConstant(0.0, stream);
    smoothing_s.setToConstant(0.0, stream);
    smoothing_mu.setToConstant(0.0, stream);
    dP_values.setToConstant(0.0, stream);
    dq.setToConstant(0.0, stream);
    dA_values.setToConstant(0.0, stream);
    db.setToConstant(0.0, stream);
}

size_t DiffState::memoryUsage() const noexcept {
    return x.memoryUsage() + z.memoryUsage() + s.memoryUsage() + tau.memoryUsage() +
           u.memoryUsage() + pi_u.memoryUsage() +
           d.memoryUsage() + dinv.memoryUsage() + e.memoryUsage() + einv.memoryUsage() +
           c_scale.memoryUsage() +
           work_n.memoryUsage() + work_m.memoryUsage() + work_m2.memoryUsage() +
           rhs.memoryUsage() + sol.memoryUsage() +
           c1.memoryUsage() + c2.memoryUsage() + c3.memoryUsage() +
           smoothing_x.memoryUsage() + smoothing_z.memoryUsage() +
           smoothing_s.memoryUsage() + smoothing_mu.memoryUsage() +
           dP_values.memoryUsage() + dq.memoryUsage() +
           dA_values.memoryUsage() + db.memoryUsage();
}

// ============================================================================
// BackwardWorkspace Implementation
// ============================================================================

BackwardWorkspace::BackwardWorkspace(int64_t n_, int64_t m_, int64_t batchSize_, cudaStream_t stream)
    : n(n_), m(m_), batchSize(batchSize_),
      // HSDE path vectors
      x_eq(n_, batchSize_),
      z_eq(m_, batchSize_),
      s_eq(m_, batchSize_),
      u_eq(m_, batchSize_),
      pi_u_eq(m_, batchSize_),
      Px_eq(n_, batchSize_),
      // Equilibrated upstream gradient vectors
      dx_eq(n_, batchSize_),
      dz_eq(m_, batchSize_),
      ds_eq(m_, batchSize_),
      // Smoothed diff workspace
      smoothing_mu_comp(1, batchSize_)
{
    // Initialize all work vectors to zero
    x_eq.setToConstant(0.0, stream);
    z_eq.setToConstant(0.0, stream);
    s_eq.setToConstant(0.0, stream);
    u_eq.setToConstant(0.0, stream);
    pi_u_eq.setToConstant(0.0, stream);
    Px_eq.setToConstant(0.0, stream);
    dx_eq.setToConstant(0.0, stream);
    dz_eq.setToConstant(0.0, stream);
    ds_eq.setToConstant(0.0, stream);
    smoothing_mu_comp.setToConstant(0.0, stream);
}

// ============================================================================
// ConeDerivatives Implementation
// ============================================================================

ConeDerivatives::ConeDerivatives(
    const Cones& cones,
    int64_t batchSize_,
    cudaStream_t stream
)
    : batchSize(batchSize_),
      nonneg_H(cones.numNonnegCones > 0 ? cones.numNonnegCones : 1, batchSize_),
      // Dense SOC (dim <= 4): upper triangle = dim*(dim+1)/2 per cone
      soc_H(1, batchSize_),  // resized below
      // Sparse SOC (dim > 4): diagonal + rank-2
      soc_sparse_diag(1, batchSize_),  // resized below
      soc_sparse_v1(1, batchSize_),
      soc_sparse_v2(1, batchSize_),
      soc_sparse_c1(1, batchSize_),
      soc_sparse_c2(1, batchSize_),
      // Exp and power cones are NOT self-dual, so derivatives are NOT symmetric
      // We need full 9 elements per cone (not just upper triangle 6)
      exp_H(cones.numExpCones > 0 ? cones.numExpCones * 9 : 1, batchSize_),
      power_H(cones.numPowerCones > 0 ? cones.numPowerCones * 9 : 1, batchSize_),
      // PSD cones: dense upper triangle svec_dim×svec_dim per cone
      psd_H(cones.totalPsdHsEntries > 0 ? cones.totalPsdHsEntries : 1, batchSize_),
      // PSD eigendecomp cache
      psd_eigvals(cones.totalPsdMatDim > 0 ? cones.totalPsdMatDim : 1, batchSize_),
      psd_eigvecs(cones.totalPsdMatSqDim > 0 ? cones.totalPsdMatSqDim : 1, batchSize_),
      psd_omega(cones.totalPsdMatSqDim > 0 ? cones.totalPsdMatSqDim : 1, batchSize_),
      psd_work_mat(cones.totalPsdMatSqDim > 0 ? cones.totalPsdMatSqDim : 1, batchSize_),
      psd_work_mat2(cones.totalPsdMatSqDim > 0 ? cones.totalPsdMatSqDim : 1, batchSize_),
      psd_work_svec(cones.totalPsdSvecDim > 0 ? cones.totalPsdSvecDim : 1, batchSize_),
      genpow_sparse_diag(1, batchSize_),
      genpow_sparse_left1(1, batchSize_),
      genpow_sparse_right1(1, batchSize_),
      genpow_sparse_left2(1, batchSize_),
      genpow_sparse_right2(1, batchSize_),
      genpow_sparse_left3(1, batchSize_),
      genpow_sparse_c3(1, batchSize_),
      genpow_diff_work_vec(1, batchSize_),
      genpow_diff_work_dim1(1, batchSize_),
      // Direct-x cone derivative storage; resized below based on cones.x_cones.
      xcone_nonneg_H(1, batchSize_),
      xcone_soc_H(1, batchSize_),
      xcone_psd_H(1, batchSize_),
      xcone_exp_H(1, batchSize_),
      xcone_pow_H(1, batchSize_),
      xcone_genpow_H(1, batchSize_),
      // Direct-x GenPow backward workspaces and rank-3 scratch.
      xcone_genpow_diff_work_vec(1, batchSize_),
      xcone_genpow_diff_work_dim1(1, batchSize_),
      xcone_genpow_rank3_diag(1, batchSize_),
      xcone_genpow_rank3_left1(1, batchSize_),
      xcone_genpow_rank3_right1(1, batchSize_),
      xcone_genpow_rank3_left2(1, batchSize_),
      xcone_genpow_rank3_right2(1, batchSize_),
      xcone_genpow_rank3_left3(1, batchSize_),
      xcone_genpow_rank3_c3(1, batchSize_),
      xcone_soc_rank2_diag(1, batchSize_),
      xcone_soc_rank2_v1(1, batchSize_),
      xcone_soc_rank2_v2(1, batchSize_),
      xcone_soc_rank2_c1(1, batchSize_),
      xcone_soc_rank2_c2(1, batchSize_)
{
    totalPsdHsEntries = cones.totalPsdHsEntries;
    totalPsdMatDim = cones.totalPsdMatDim;
    totalPsdMatSqDim = cones.totalPsdMatSqDim;
    totalPsdSvecDim = cones.totalPsdSvecDim;
    // Compute dense-only and sparse-only SOC totals
    int64_t denseSocHs = 0;
    int64_t sparseSocDim = 0;
    for (int64_t i = 0; i < cones.numSocCones; ++i) {
        int64_t d = cones.socConeDims[i];
        if (d <= 4) {
            denseSocHs += d * (d + 1) / 2;
        } else {
            sparseSocDim += d;
        }
    }
    totalDenseSocHsEntries = denseSocHs;
    totalSparseSocDim = sparseSocDim;

    // Resize SOC buffers
    soc_H = BatchedVector(denseSocHs > 0 ? denseSocHs : 1, batchSize_);
    if (cones.numSparseSoc > 0) {
        soc_sparse_diag = BatchedVector(sparseSocDim, batchSize_);
        soc_sparse_v1 = BatchedVector(sparseSocDim, batchSize_);
        soc_sparse_v2 = BatchedVector(sparseSocDim, batchSize_);
        soc_sparse_c1 = BatchedVector(cones.numSparseSoc, batchSize_);
        soc_sparse_c2 = BatchedVector(cones.numSparseSoc, batchSize_);
    }

    // Build dense-only Hs offset array for the fused kernel
    // Always needed when there are SOC cones (even if all dense)
    if (cones.numSocCones > 0) {
        std::vector<int64_t> dense_hs_offsets(cones.numSocCones + 1, 0);
        int64_t acc = 0;
        for (int64_t i = 0; i < cones.numSocCones; ++i) {
            int64_t d = cones.socConeDims[i];
            dense_hs_offsets[i] = acc;
            if (d <= 4) {
                acc += d * (d + 1) / 2;
            }
            // Sparse cones don't contribute to this array
        }
        dense_hs_offsets[cones.numSocCones] = acc;
        cudaMalloc(&d_dense_soc_Hs_offsets, sizeof(int64_t) * (cones.numSocCones + 1));
        cudaMemcpyAsync(d_dense_soc_Hs_offsets, dense_hs_offsets.data(),
                        sizeof(int64_t) * (cones.numSocCones + 1),
                        cudaMemcpyHostToDevice, stream);
    }

    // GenPowerCone: sparse decomposition (diagonal + rank-3)
    if (cones.numGenPowerCones > 0) {
        int64_t gpTotalDim = 0;
        for (int64_t i = 0; i < cones.numGenPowerCones; ++i) {
            gpTotalDim += cones.genPowerDim1s[i] + cones.genPowerDim2s[i];
        }
        totalGenpowDim = gpTotalDim;

        genpow_sparse_diag = BatchedVector(gpTotalDim > 0 ? gpTotalDim : 1, batchSize_);
        genpow_sparse_left1 = BatchedVector(gpTotalDim > 0 ? gpTotalDim : 1, batchSize_);
        genpow_sparse_right1 = BatchedVector(gpTotalDim > 0 ? gpTotalDim : 1, batchSize_);
        genpow_sparse_left2 = BatchedVector(gpTotalDim > 0 ? gpTotalDim : 1, batchSize_);
        genpow_sparse_right2 = BatchedVector(gpTotalDim > 0 ? gpTotalDim : 1, batchSize_);
        genpow_sparse_left3 = BatchedVector(gpTotalDim > 0 ? gpTotalDim : 1, batchSize_);
        genpow_sparse_c3 = BatchedVector(cones.numGenPowerCones, batchSize_);

        // Workspace for projection and derivative kernels
        genpow_diff_work_vec = BatchedVector(cones.totalGenPowerDim, batchSize_);
        genpow_diff_work_dim1 = BatchedVector(7 * cones.totalGenPowerAlphas, batchSize_);
        genpow_diff_work_vec.setToConstant(0.0, stream);
        genpow_diff_work_dim1.setToConstant(0.0, stream);
    }

    // Direct-x cone derivative storage. Sum dims per cone kind to size
    // the per-cone-kind buffers; mirrors the Cones direct-x layout.
    {
        int64_t nn = 0, soc_kkt = 0, psd_kkt = 0;
        int64_t exp_kkt = 0, pow_kkt = 0, gp_kkt = 0;
        for (const auto& xc : cones.x_cones) {
            int64_t d = static_cast<int64_t>(xc.indices.size());
            switch (xc.kind) {
                case XConeKind::Nonneg:   nn      += d;       break;
                case XConeKind::SOC:      soc_kkt += d * d;   break;
                case XConeKind::PSD:      psd_kkt += d * d;   break;
                case XConeKind::Exp:      exp_kkt += 9;       break;
                case XConeKind::Power:    pow_kkt += 9;       break;
                case XConeKind::GenPower: gp_kkt  += d * d;   break;
            }
        }
        totalXNonneg = nn;
        totalXSocKkt = soc_kkt;
        totalXPsdKkt = psd_kkt;
        totalXExpKkt = exp_kkt;
        totalXPowKkt = pow_kkt;
        totalXGenPowKkt = gp_kkt;
        xcone_nonneg_H = BatchedVector(nn      > 0 ? nn      : 1, batchSize_);
        xcone_soc_H    = BatchedVector(soc_kkt > 0 ? soc_kkt : 1, batchSize_);
        xcone_psd_H    = BatchedVector(psd_kkt > 0 ? psd_kkt : 1, batchSize_);
        xcone_exp_H    = BatchedVector(exp_kkt > 0 ? exp_kkt : 1, batchSize_);
        xcone_pow_H    = BatchedVector(pow_kkt > 0 ? pow_kkt : 1, batchSize_);
        xcone_genpow_H = BatchedVector(gp_kkt  > 0 ? gp_kkt  : 1, batchSize_);
        xcone_nonneg_H.setToConstant(0.0, stream);
        xcone_soc_H.setToConstant(0.0, stream);
        xcone_psd_H.setToConstant(0.0, stream);
        xcone_exp_H.setToConstant(0.0, stream);
        xcone_pow_H.setToConstant(0.0, stream);
        xcone_genpow_H.setToConstant(0.0, stream);

        // Direct-x GenPow workspace + rank-3 scratch for backward.
        if (cones.numXGenPowerCones > 0) {
            int64_t total_xg_dim    = cones.totalXGenPowerDim;
            int64_t total_xg_alphas = cones.totalXGenPowerAlphas;
            xcone_genpow_diff_work_vec  = BatchedVector(total_xg_dim,    batchSize_);
            xcone_genpow_diff_work_dim1 = BatchedVector(7 * total_xg_alphas, batchSize_);
            xcone_genpow_rank3_diag   = BatchedVector(total_xg_dim, batchSize_);
            xcone_genpow_rank3_left1  = BatchedVector(total_xg_dim, batchSize_);
            xcone_genpow_rank3_right1 = BatchedVector(total_xg_dim, batchSize_);
            xcone_genpow_rank3_left2  = BatchedVector(total_xg_dim, batchSize_);
            xcone_genpow_rank3_right2 = BatchedVector(total_xg_dim, batchSize_);
            xcone_genpow_rank3_left3  = BatchedVector(total_xg_dim, batchSize_);
            xcone_genpow_rank3_c3     = BatchedVector(cones.numXGenPowerCones, batchSize_);
            xcone_genpow_diff_work_vec.setToConstant(0.0, stream);
            xcone_genpow_diff_work_dim1.setToConstant(0.0, stream);
            xcone_genpow_rank3_diag.setToConstant(0.0, stream);
            xcone_genpow_rank3_left1.setToConstant(0.0, stream);
            xcone_genpow_rank3_right1.setToConstant(0.0, stream);
            xcone_genpow_rank3_left2.setToConstant(0.0, stream);
            xcone_genpow_rank3_right2.setToConstant(0.0, stream);
            xcone_genpow_rank3_left3.setToConstant(0.0, stream);
            xcone_genpow_rank3_c3.setToConstant(0.0, stream);
        }

        // Direct-x SOC rank-2 stripe storage for sparse expansion path.
        int64_t total_xsoc_sparse_dim = 0;
        int64_t num_xsoc_sparse = 0;
        for (const auto& xc : cones.x_cones) {
            if (xc.kind == XConeKind::SOC && xc.indices.size() > 4) {
                total_xsoc_sparse_dim += static_cast<int64_t>(xc.indices.size());
                ++num_xsoc_sparse;
            }
        }
        totalSparseXSocDim = total_xsoc_sparse_dim;
        numSparseXSoc      = num_xsoc_sparse;
        if (num_xsoc_sparse > 0) {
            xcone_soc_rank2_diag = BatchedVector(total_xsoc_sparse_dim, batchSize_);
            xcone_soc_rank2_v1   = BatchedVector(total_xsoc_sparse_dim, batchSize_);
            xcone_soc_rank2_v2   = BatchedVector(total_xsoc_sparse_dim, batchSize_);
            xcone_soc_rank2_c1   = BatchedVector(num_xsoc_sparse, batchSize_);
            xcone_soc_rank2_c2   = BatchedVector(num_xsoc_sparse, batchSize_);
            xcone_soc_rank2_diag.setToConstant(0.0, stream);
            xcone_soc_rank2_v1.setToConstant(0.0, stream);
            xcone_soc_rank2_v2.setToConstant(0.0, stream);
            xcone_soc_rank2_c1.setToConstant(0.0, stream);
            xcone_soc_rank2_c2.setToConstant(0.0, stream);
        }
    }

    nonneg_H.setToConstant(0.0, stream);
    soc_H.setToConstant(0.0, stream);
    soc_sparse_diag.setToConstant(0.0, stream);
    soc_sparse_v1.setToConstant(0.0, stream);
    soc_sparse_v2.setToConstant(0.0, stream);
    soc_sparse_c1.setToConstant(0.0, stream);
    soc_sparse_c2.setToConstant(0.0, stream);
    exp_H.setToConstant(0.0, stream);
    power_H.setToConstant(0.0, stream);
    psd_H.setToConstant(0.0, stream);
    psd_eigvals.setToConstant(0.0, stream);
    psd_eigvecs.setToConstant(0.0, stream);
    psd_omega.setToConstant(0.0, stream);
    psd_work_mat.setToConstant(0.0, stream);
    psd_work_mat2.setToConstant(0.0, stream);
    psd_work_svec.setToConstant(0.0, stream);
    genpow_sparse_diag.setToConstant(0.0, stream);
    genpow_sparse_left1.setToConstant(0.0, stream);
    genpow_sparse_right1.setToConstant(0.0, stream);
    genpow_sparse_left2.setToConstant(0.0, stream);
    genpow_sparse_right2.setToConstant(0.0, stream);
    genpow_sparse_left3.setToConstant(0.0, stream);
    genpow_sparse_c3.setToConstant(0.0, stream);

    // Pre-allocate cuSOLVER workspace for eigendecomp (sized for largest PSD cone)
    if (cones.numPsdCones > 0) {
        int64_t max_psd_dim = 0;
        for (int64_t i = 0; i < cones.numPsdCones; ++i) {
            max_psd_dim = std::max(max_psd_dim, cones.psdConeDims[i]);
        }
        if (max_psd_dim > 1) {
            // Query workspace size for the largest cone
            cusolverDnHandle_t cusolver = cones.cusolverH_;
            cusolverDnSetStream(cusolver, stream);
            cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                         CUBLAS_FILL_MODE_LOWER,
                                         max_psd_dim, nullptr, max_psd_dim,
                                         nullptr, &d_psd_cusolver_work_size);
            cudaMalloc(&d_psd_cusolver_work, sizeof(double) * d_psd_cusolver_work_size);
            cudaMalloc(&d_psd_info, sizeof(int));
        }
    }
}

// ============================================================================
// Cone Projection
// ============================================================================

void compute_cone_projection(
    const BatchedVector& u,
    BatchedVector& pi_u,
    const Cones& cones,
    cudaStream_t stream,
    double* genpow_work_vec,
    int64_t totalGenPowerDim
) {
    int64_t m = u.n();
    int64_t batchSize = u.batchSize();
    int64_t offset = 0;

    // Zero cones - identity projection
    if (cones.numZeroCones > 0) {
        project_zero_cone_dual(
            pi_u.data(), u.data(),
            cones.numZeroCones, batchSize, m, stream
        );
        offset += cones.numZeroCones;
    }

    // Nonnegative cones - max(u, 0)
    if (cones.numNonnegCones > 0) {
        project_nonneg_cone_dual(
            pi_u.data(), u.data(),
            offset, cones.numNonnegCones, batchSize, m, stream
        );
        offset += cones.numNonnegCones;
    }

    // SOC cones (variable dimension)
    if (cones.numSocCones > 0) {
        project_soc_cone_dual(
            pi_u.data(), u.data(),
            offset, cones.numSocCones,
            cones.d_soc_dims, cones.d_soc_offsets, cones.totalSocDim,
            batchSize, m,
            cones.d_soc_sz_offsets, stream
        );
        offset += cones.totalSocDim;
    }

    // Exponential cones
    if (cones.numExpCones > 0) {
        project_exp_cone_dual(
            pi_u.data(), u.data(),
            offset, cones.numExpCones, batchSize, m, stream
        );
        offset += cones.numExpCones * 3;
    }

    // Power cones
    if (cones.numPowerCones > 0) {
        project_power_cone_dual(
            pi_u.data(), u.data(),
            cones.d_powerAlphas,
            offset, cones.numPowerCones, batchSize, m, stream
        );
        offset += cones.numPowerCones * 3;
    }

    // PSD cones - self-dual: project eigenvalues to max(0,λ)
    // Note: This standalone path doesn't cache eigendecomp for the derivative.
    // The backward path uses project_psd_cone_dual with ConeDerivatives for caching.
    if (cones.numPsdCones > 0) {
        // For standalone projection (forward path), create a temporary ConeDerivatives
        // just for workspace. This is only called from the forward differentiation path.
        ConeDerivatives temp_derivs(cones, batchSize, stream);
        project_psd_cone_dual(
            pi_u.data(), u.data(),
            offset, cones, temp_derivs,
            batchSize, m, stream
        );
        offset += cones.totalPsdSvecDim;
    }

    // GenPowerCones
    if (cones.numGenPowerCones > 0) {
        project_genpow_cone_dual(
            pi_u.data(), u.data(),
            cones.d_genPowerAlphas,
            cones.d_genPowerDim1s,
            cones.d_genPowerDim2s,
            cones.d_genPowerOffsets,
            cones.d_genPowerAlphaOffsets,
            offset, cones.numGenPowerCones, batchSize, m, stream,
            genpow_work_vec, totalGenPowerDim,
            cones.d_genPowerSzOffsets
        );
    }
}

// ============================================================================
// Cone Derivative
// ============================================================================

void compute_cone_derivative(
    const BatchedVector& u,
    ConeDerivatives& derivs,
    const Cones& cones,
    cudaStream_t stream,
    double* genpow_work_vec,
    int64_t totalGenPowerDim,
    double* genpow_work_dim1,
    int64_t totalGenPowerAlphas
) {
    int64_t m = u.n();
    int64_t batchSize = u.batchSize();
    int64_t offset = cones.numZeroCones;  // Zero cones have identity derivative

    // Nonnegative cones
    if (cones.numNonnegCones > 0) {
        compute_nonneg_derivative(
            derivs.nonneg_H.data(), u.data(),
            offset, cones.numNonnegCones, batchSize, m, stream
        );
        offset += cones.numNonnegCones;
    }

    // SOC cones (variable dimension)
    if (cones.numSocCones > 0) {
        if (cones.numSparseSoc > 0) {
            // Split: dense cones (dim<=4) get upper triangle, sparse cones (dim>4) get rank-2
            compute_soc_derivative_sparse(
                derivs.soc_H.data(), u.data(),
                offset, cones.numSocCones,
                cones.d_soc_dims, cones.d_soc_offsets,
                derivs.d_dense_soc_Hs_offsets,
                cones.d_soc_sparse_indices,
                derivs.totalDenseSocHsEntries,
                derivs.soc_sparse_diag.data(),
                derivs.soc_sparse_v1.data(),
                derivs.soc_sparse_v2.data(),
                derivs.soc_sparse_c1.data(),
                derivs.soc_sparse_c2.data(),
                cones.d_soc_sparse_offsets,
                derivs.totalSparseSocDim,
                cones.numSparseSoc,
                batchSize, m,
                cones.d_soc_sz_offsets, stream,
                cones.numLargeSoc
            );
        } else {
            // All cones are dense — use original path with dense offsets
            compute_soc_derivative(
                derivs.soc_H.data(), u.data(),
                offset, cones.numSocCones,
                cones.d_soc_dims, cones.d_soc_offsets, cones.d_soc_Hs_dense_offsets,
                cones.totalSocDim, cones.totalSocHsDenseEntries,
                batchSize, m,
                cones.d_soc_sz_offsets, stream
            );
        }
        offset += cones.totalSocDim;
    }

    // Exponential cones
    if (cones.numExpCones > 0) {
        compute_exp_derivative(
            derivs.exp_H.data(), u.data(),
            offset, cones.numExpCones, batchSize, m, stream
        );
        offset += cones.numExpCones * 3;
    }

    // Power cones
    if (cones.numPowerCones > 0) {
        compute_power_derivative(
            derivs.power_H.data(), u.data(),
            cones.d_powerAlphas,
            offset, cones.numPowerCones, batchSize, m, stream
        );
        offset += cones.numPowerCones * 3;
    }

    // PSD cones - self-dual: derivative uses Ω matrix
    if (cones.numPsdCones > 0) {
        compute_psd_derivative(
            derivs.psd_H.data(), u.data(),
            offset, cones, derivs,
            batchSize, m, stream
        );
        offset += cones.totalPsdSvecDim;
    }

    // GenPowerCones (sparse decomposition)
    if (cones.numGenPowerCones > 0) {
        compute_genpow_derivative_sparse(
            derivs.genpow_sparse_diag.data(),
            derivs.genpow_sparse_left1.data(),
            derivs.genpow_sparse_right1.data(),
            derivs.genpow_sparse_left2.data(),
            derivs.genpow_sparse_right2.data(),
            derivs.genpow_sparse_left3.data(),
            derivs.genpow_sparse_c3.data(),
            u.data(),
            cones.d_genPowerAlphas,
            cones.d_genPowerDim1s,
            cones.d_genPowerDim2s,
            cones.d_genPowerOffsets,
            cones.d_genPowerAlphaOffsets,
            offset, cones.numGenPowerCones,
            derivs.totalGenpowDim,
            batchSize, m, stream,
            genpow_work_vec,
            genpow_work_dim1, totalGenPowerAlphas,
            cones.d_genPowerSzOffsets
        );
    }
}

// ============================================================================
// Cache Solution for Backward
// ============================================================================

void DiffState::resize_for_xcones(int64_t xn, cudaStream_t stream) {
    if (xn == totalXConeNumel) return;
    int64_t new_dim = n + 2 * m + xn + 1;
    rhs = BatchedVector(new_dim, batchSize);
    sol = BatchedVector(new_dim, batchSize);
    z_x = BatchedVector(xn > 0 ? xn : 1, batchSize);
    totalXConeNumel = xn;
    if (stream) {
        rhs.setToConstant(0.0, stream);
        sol.setToConstant(0.0, stream);
        z_x.setToConstant(0.0, stream);
    }
}

void cache_solution_for_backward(
    DiffState& state,
    const CompiledSolver& solver,
    cudaStream_t stream
) {
    int64_t n = state.n;
    int64_t m = state.m;
    int64_t batchSize = state.batchSize;

    // Copy solution from solver
    cudaMemcpyAsync(
        state.x.data(), solver.solution.x.data(),
        sizeof(double) * n * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );
    cudaMemcpyAsync(
        state.z.data(), solver.solution.z.data(),
        sizeof(double) * m * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );
    cudaMemcpyAsync(
        state.s.data(), solver.solution.s.data(),
        sizeof(double) * m * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );

    // Cache direct-x dual `z_x` in equilibrated (HSDE-normalized) frame.
    // CPU's `Variables::unscale` divides z_x by τ + equilibration before
    // the backward path reads it; CUDA's `Solution::post_process` only
    // unscales x/z/s and leaves `variables.z_x` in IPM-internal frame
    // (un-normalized by τ). The equilibrated frame the IFT-direct math
    // expects is `z_x_eq = z_x_internal / τ` (with equilibration off,
    // d[J]/c factors drop out). Use solution.τ_raw — the saved τ at
    // convergence — since `variables.τ` may have been mutated by the
    // post-iter restore.
    int64_t total_xn = solver.variables.totalXConeNumel();
    if (total_xn != state.totalXConeNumel) {
        state.resize_for_xcones(total_xn, stream);
    }
    if (total_xn > 0) {
        div_per_batch(state.z_x, solver.variables.z_x,
                      solver.solution.τ_raw, stream);
    }

    // Set tau = 1.0 for differentiation
    // The solution (x, z, s) is already normalized (divided by tau from HSDE),
    // so for differentiation purposes we use tau = 1.0, matching the CPU implementation.
    state.tau.setToConstant(1.0, stream);

    // Copy equilibration factors from solver data
    cudaMemcpyAsync(
        state.d.data(), solver.data.equilibration.d.data(),
        sizeof(double) * n * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );
    cudaMemcpyAsync(
        state.dinv.data(), solver.data.equilibration.dinv.data(),
        sizeof(double) * n * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );
    cudaMemcpyAsync(
        state.e.data(), solver.data.equilibration.e.data(),
        sizeof(double) * m * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );
    cudaMemcpyAsync(
        state.einv.data(), solver.data.equilibration.einv.data(),
        sizeof(double) * m * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );
    cudaMemcpyAsync(
        state.c_scale.data(), solver.data.equilibration.c.data(),
        sizeof(double) * batchSize,
        cudaMemcpyDeviceToDevice, stream
    );

    // Fused: u = z - s + cone projection (no derivative needed for forward path)
    const Cones& cones_fwd = solver.data.cones;
    fused_u_and_cone_projection(
        state.u.data(), state.pi_u.data(),
        state.z.data(), state.s.data(),
        cones_fwd.d_powerAlphas,
        cones_fwd.numZeroCones, cones_fwd.numNonnegCones,
        cones_fwd.numSocCones, cones_fwd.d_soc_dims, cones_fwd.d_soc_offsets,
        cones_fwd.d_soc_sz_offsets,
        cones_fwd.totalSocDim,
        cones_fwd.numExpCones, cones_fwd.numPowerCones,
        m, batchSize, stream
    );

    // Compute cone projection Π_K*(u) for GenPowerCones (not covered by fused kernel)
    // Use work_m as scratch (totalGenPowerDim <= m)
    if (solver.data.cones.numGenPowerCones > 0) {
        compute_cone_projection(state.u, state.pi_u, solver.data.cones, stream,
            state.work_m.data(), solver.data.cones.totalGenPowerDim);
    }
}

// Forward declaration
static void create_diff_kkt(
    CompiledSolver& solver,
    int64_t n, int64_t m, int64_t batchSize,
    const CSR& P, const CSR& A, const Cones& cones,
    cudaStream_t stream
);

// ============================================================================
// Equilibration Helper
// ============================================================================

static void equilibrate_solution(
    BatchedVector& x_eq, BatchedVector& z_eq, BatchedVector& s_eq,
    const BatchedVector& x_src, const BatchedVector& z_src, const BatchedVector& s_src,
    const BatchedVector& dinv, const BatchedVector& einv, const BatchedVector& c_scale,
    BatchedVector& work_m,
    int64_t m, int64_t batchSize, cudaStream_t stream
) {
    // x_eq = dinv * x,  z_eq = c * einv * z,  s_eq = s / einv
    elementwise_mul(x_eq, x_src, dinv, stream);

    elementwise_mul(z_eq, z_src, einv, stream);
    mul_per_batch(work_m, z_eq, c_scale, stream);
    cudaMemcpyAsync(z_eq.data(), work_m.data(), sizeof(double) * m * batchSize,
                    cudaMemcpyDeviceToDevice, stream);

    elementwise_div(s_eq, s_src, einv, stream);
}

// ============================================================================
// Direct-x cone derivative kernels
// ============================================================================

// Compute the direct-x cone-projection Jacobian `H_x` for nonneg, SOC, and
// PSD direct-x cones. For nonneg the result is diagonal {0,1}; for SOC and
// PSD, dense `dim×dim` blocks per cone in row-major order.
//
// Inputs:
//   x_eq, z_x_eq             — equilibrated primal x (length n) and
//                              direct-x dual (length totalXConeNumel),
//                              both per batch.
//   xcone_kinds, xcone_dims  — per-cone metadata.
//   xcone_indices, xcone_numel_offsets — flattened J indices + cone start.
//   xcone_h_off              — per-cone offset into the kind-specific H.
//
// Self-dual cone projection Jacobian DΠ((s+z)/2). The slack convention
// uses u = z − s (Exact mode); for direct-x we use u = z_x − x[J] in eq
// frame, matching the CPU IFT-direct path.
//
// One block per batch.
// One block per batch; each thread handles one direct-x cone in a strided
// loop. For each cone, compute u = z_x − x[J] (gathered) and write the
// cone-projection Jacobian H_x to the kind-specific storage.
__global__ void compute_xcone_H_kernel(
    double* __restrict__ xcone_nonneg_H,            // [batchSize][totalXNonneg]
    double* __restrict__ xcone_soc_H,               // [batchSize][totalXSocKkt] (dim*dim per cone)
    double* /*xcone_psd_H*/,           // [batchSize][totalXPsdKkt] (svec_dim² per cone)
    const double* __restrict__ x_eq,                // [batchSize][n]
    const double* __restrict__ z_x_eq,              // [batchSize][totalXConeNumel]
    const int64_t* __restrict__ xcone_kinds,        // [numXCones] 0=nonneg, 1=SOC, 2=PSD
    const int64_t* __restrict__ xcone_dims,         // [numXCones]
    const int64_t* __restrict__ xcone_indices,      // [totalXConeNumel]
    const int64_t* __restrict__ xcone_numel_offsets,// [numXCones+1]
    const int64_t* __restrict__ xcone_h_off,        // [numXCones]
    int64_t numXCones,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXNonneg,
    int64_t totalXSocKkt,
    int64_t /*totalXPsdKkt*/,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* x_b  = x_eq    + batch * n;
    const double* zx_b = z_x_eq  + batch * totalXConeNumel;

    // Iterate direct-x cones. One thread per cone — fine since numXCones
    // is small. Each thread does its own per-cone loop.
    for (int64_t xc = threadIdx.x; xc < numXCones; xc += blockDim.x) {
        int64_t kind = xcone_kinds[xc];
        int64_t dim  = xcone_dims[xc];
        int64_t off  = xcone_h_off[xc];
        int64_t nu   = xcone_numel_offsets[xc];

        if (kind == 0) {
            // Nonneg direct-x: diagonal H[k] = 1 if u[k] >= 0 else 0.
            double* H = xcone_nonneg_H + batch * totalXNonneg + off;
            for (int64_t k = 0; k < dim; ++k) {
                double u = zx_b[nu + k] - x_b[xcone_indices[nu + k]];
                H[k] = (u >= 0.0) ? 1.0 : 0.0;
            }
        } else if (kind == 1) {
            // SOC direct-x: dense dim×dim row-major. Standard SOC
            // projection Jacobian with input u = z_x − x[J].
            //   t >= ||v||      → H = I
            //   t <= -||v||     → H = 0
            //   otherwise       → boundary formula (Clarabel/diffqcp).
            double* H = xcone_soc_H + batch * totalXSocKkt + off;
            int64_t i0 = xcone_indices[nu + 0];
            double t = zx_b[nu + 0] - x_b[i0];
            double norm_v_sq = 0.0;
            for (int64_t i = 1; i < dim; ++i) {
                double u_i = zx_b[nu + i] - x_b[xcone_indices[nu + i]];
                norm_v_sq += u_i * u_i;
            }
            double norm_v = sqrt(norm_v_sq);
            if (t >= norm_v) {
                for (int64_t r = 0; r < dim; ++r) {
                    for (int64_t c = 0; c < dim; ++c) {
                        H[r * dim + c] = (r == c) ? 1.0 : 0.0;
                    }
                }
            } else if (t <= -norm_v) {
                for (int64_t i = 0; i < dim * dim; ++i) H[i] = 0.0;
            } else {
                double inv_norm = (norm_v > 1e-300) ? (1.0 / norm_v) : 0.0;
                double t_over_norm = t * inv_norm;
                for (int64_t r = 0; r < dim; ++r) {
                    double u_r = (r == 0)
                        ? t
                        : (zx_b[nu + r] - x_b[xcone_indices[nu + r]]);
                    for (int64_t c = 0; c < dim; ++c) {
                        double u_c = (c == 0)
                            ? t
                            : (zx_b[nu + c] - x_b[xcone_indices[nu + c]]);
                        double val;
                        if (r == 0 && c == 0) {
                            val = 0.5;
                        } else if (r == 0) {
                            val = 0.5 * u_c * inv_norm;
                        } else if (c == 0) {
                            val = 0.5 * u_r * inv_norm;
                        } else {
                            double x_r = u_r * inv_norm;
                            double x_c = u_c * inv_norm;
                            double delta = (r == c) ? 1.0 : 0.0;
                            val = 0.5 * (delta + t_over_norm * (delta - x_r * x_c));
                        }
                        H[r * dim + c] = val;
                    }
                }
            }
        } else {
            // Other kinds:
            //   kind == 2 (PSD): handled by compute_xcone_psd_derivative.
            //   kind == 3 (Exp), kind == 4 (Power): handled by
            //              compute_xcone_asymm_H.
            //   kind == 5 (GenPow): backward not yet implemented; the host
            //              dispatch in backward_impl raises before launch.
            // No-op here for any of the above.
        }
    }
}

// ============================================================================
// Backward Differentiation
// ============================================================================

// Implementation accepting an optional `dz_x_bar`. Both the public
// `backward()` and `backward_with_dz_x()` route here.
static void backward_impl(
    DiffState& state,
    const BatchedVector& dx_bar,
    const BatchedVector& dz_bar,
    const BatchedVector& ds_bar,
    const BatchedVector* dz_x_bar,
    CompiledSolver& solver,
    cudaStream_t stream
);

void backward(
    DiffState& state,
    const BatchedVector& dx_bar,
    const BatchedVector& dz_bar,
    const BatchedVector& ds_bar,
    CompiledSolver& solver,
    cudaStream_t stream
) {
    backward_impl(state, dx_bar, dz_bar, ds_bar, /*dz_x_bar=*/nullptr,
                  solver, stream);
}

void backward_with_dz_x(
    DiffState& state,
    const BatchedVector& dx_bar,
    const BatchedVector& dz_bar,
    const BatchedVector& ds_bar,
    const BatchedVector* dz_x_bar,
    CompiledSolver& solver,
    cudaStream_t stream
) {
    backward_impl(state, dx_bar, dz_bar, ds_bar, dz_x_bar, solver, stream);
}

static void backward_impl(
    DiffState& state,
    const BatchedVector& dx_bar,
    const BatchedVector& dz_bar,
    const BatchedVector& ds_bar,
    const BatchedVector* dz_x_bar,
    CompiledSolver& solver,
    cudaStream_t stream
) {
    int64_t n = state.n;
    int64_t m = state.m;
    int64_t batchSize = state.batchSize;

    // Get references to solver components
    const Cones& cones = solver.data.cones;
    const CSR& P = solver.data.P;
    const CSR& A = solver.data.A;

    // Zero output gradients
    state.dP_values.setToConstant(0.0, stream);
    state.dq.setToConstant(0.0, stream);
    state.dA_values.setToConstant(0.0, stream);
    state.db.setToConstant(0.0, stream);

    // HSDE formulation for backward differentiation.
    // For Woodbury problems, uses PCG with J matvec (DiffWoodbury).
    // For general problems, uses cuDSS-based DiffKKT.
    //
    // The HSDE backward uses a different KKT system than the forward solve:
    //   [I   J ] [lam]   [0      ]
    //   [J' -εI] [y  ] = [rhs_bar]
    // where J is the HSDE Jacobian with dimension (n + 2m + 1).

    // Step 1: Lazily initialize DiffKKT if needed (recreate if batchSize changed)
    if (!solver.diff_kkt_ || solver.diff_kkt_->batchSize != batchSize) {
        create_diff_kkt(solver, n, m, batchSize, P, A, cones, stream);
    }

    // Lazily initialize backward workspace if needed (recreate if batchSize changed)
    if (!solver.backward_workspace_ || solver.backward_workspace_->batchSize != batchSize) {
        solver.backward_workspace_ = std::make_unique<BackwardWorkspace>(n, m, batchSize, stream);
    }

    // Get references to preallocated work vectors (avoids cudaMalloc in hot path)
    BackwardWorkspace& ws = *solver.backward_workspace_;
    BatchedVector& x_eq = ws.x_eq;
    BatchedVector& z_eq = ws.z_eq;
    BatchedVector& s_eq = ws.s_eq;
    BatchedVector& u_eq = ws.u_eq;
    BatchedVector& pi_u_eq = ws.pi_u_eq;
    BatchedVector& Px_eq = ws.Px_eq;

    // Steps 2-3: Compute equilibrated solution and cone derivatives
    // Branch based on diff_method: smoothed uses central-path iterate,
    // exact uses cone projection Jacobian
    bool use_smoothed = (solver.settings.ipm.diffMethod == DiffMethod::Smoothed
                         && state.smoothing_cached);

    ConeDerivatives derivs(cones, batchSize, stream);

    if (use_smoothed) {
        // Smoothed path: use cached central-path iterate.
        // The smoothing iterate is already in equilibrated HSDE space (with τ=1)
        // from refineSmoothingIterate(), so just copy directly — no re-equilibration.
        cudaMemcpyAsync(x_eq.data(), state.smoothing_x.data(),
                        sizeof(double) * n * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(z_eq.data(), state.smoothing_z.data(),
                        sizeof(double) * m * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(s_eq.data(), state.smoothing_s.data(),
                        sizeof(double) * m * batchSize,
                        cudaMemcpyDeviceToDevice, stream);

        // Compute mu = dot(s_eq, z_eq) / degree
        int64_t degree = cones.degree();
        dot_batched(ws.smoothing_mu_comp, s_eq, z_eq, stream);
        scale_by_scalar(ws.smoothing_mu_comp, 1.0 / (double)degree, stream);

        // Compute smoothed cone derivatives H = (I + mu*nabla^2*phi*(z))^{-1}
        compute_smoothed_cone_derivative(z_eq, s_eq, ws.smoothing_mu_comp,
                                          derivs, cones, stream);

    } else {
        // Exact path: equilibrate from cached solution
        equilibrate_solution(x_eq, z_eq, s_eq,
                             state.x, state.z, state.s,
                             state.dinv, state.einv, state.c_scale,
                             state.work_m, m, batchSize, stream);

        // Fused u_eq = z_eq - s_eq, cone projection, and cone derivative
        fused_cone_projection_and_derivative(
            u_eq.data(), pi_u_eq.data(),
            derivs.nonneg_H.data(),
            derivs.soc_H.data(),
            derivs.d_dense_soc_Hs_offsets, derivs.totalDenseSocHsEntries,
            derivs.soc_sparse_diag.data(), derivs.soc_sparse_v1.data(),
            derivs.soc_sparse_v2.data(), derivs.soc_sparse_c1.data(),
            derivs.soc_sparse_c2.data(),
            cones.d_soc_sparse_indices, cones.d_soc_sparse_offsets,
            derivs.totalSparseSocDim, cones.numSparseSoc,
            derivs.exp_H.data(), derivs.power_H.data(),
            z_eq.data(), s_eq.data(),
            cones.d_powerAlphas,
            cones.numZeroCones, cones.numNonnegCones,
            cones.numSocCones, cones.d_soc_dims, cones.d_soc_offsets,
            cones.d_soc_sz_offsets,
            cones.totalSocDim,
            cones.numExpCones, cones.numPowerCones,
            m, batchSize, stream
        );

        // GenPowerCone projection + derivative (not covered by fused kernel)
        // The fused kernel only computed u_eq = z_eq - s_eq for non-GenPow cones,
        // so we must compute u_eq for the GenPowerCone portion before projecting.
        if (cones.numGenPowerCones > 0) {
            compute_u_from_z_s(u_eq.data(), z_eq.data(), s_eq.data(), m, batchSize, stream);
            compute_cone_projection(u_eq, pi_u_eq, cones, stream,
                derivs.genpow_diff_work_vec.data(), cones.totalGenPowerDim);
            compute_cone_derivative(u_eq, derivs, cones, stream,
                derivs.genpow_diff_work_vec.data(), cones.totalGenPowerDim,
                derivs.genpow_diff_work_dim1.data(), cones.totalGenPowerAlphas);
        }
    }

    // PSD cones: separate projection + derivative (uses cuSOLVER, can't fuse)
    if (cones.numPsdCones > 0) {
        int64_t psd_offset = cones.numZeroCones + cones.numNonnegCones + cones.totalSocDim
                           + cones.numExpCones * 3 + cones.numPowerCones * 3;

        // Compute u_eq = z_eq - s_eq for the PSD region, per batch (stride m).
        for (int64_t b = 0; b < batchSize; b++) {
            double alpha_neg = -1.0;
            // u_eq[b*m + psd_offset : b*m + psd_offset + totalPsdSvecDim] =
            //   z_eq[same] - s_eq[same]
            cudaMemcpyAsync(u_eq.data() + b * m + psd_offset,
                           z_eq.data() + b * m + psd_offset,
                           sizeof(double) * cones.totalPsdSvecDim,
                           cudaMemcpyDeviceToDevice, stream);
            // u_eq -= s_eq (using cublas daxpy)
            cublasDaxpy_v2(cones.cublasH_, cones.totalPsdSvecDim,
                          &alpha_neg,
                          s_eq.data() + b * m + psd_offset, 1,
                          u_eq.data() + b * m + psd_offset, 1);
        }

        // Project PSD cones (caches eigendecomp in derivs)
        project_psd_cone_dual(
            pi_u_eq.data(), u_eq.data(),
            psd_offset, cones, derivs,
            batchSize, m, stream
        );

        // Compute PSD derivative using cached eigendecomp
        compute_psd_derivative(
            derivs.psd_H.data(), u_eq.data(),
            psd_offset, cones, derivs,
            batchSize, m, stream
        );
    }

    // Direct-x cone derivative computation (IFT-direct backward).
    // Native support: nonneg + SOC + PSD direct-x + asymmetric direct-x
    // (Exp, Power, GenPow). Each kind has its own dense H_x storage; the
    // populate kernel dispatches by kind. Asymmetric direct-x reuses the
    // slack-form projection Jacobian formulas — the asymmetric primal/
    // dual swap is a forward-only concern. GenPow direct-x backward
    // expands the rank-3 sparse Jacobian to dense (matches the CPU
    // path; see comment in `compute_xcone_genpow_H` and the CPU
    // `build_hsde_augmented_system_sparse_full` SOC/GenPow branch).
    if (cones.numXCones > 0) {
        derivs.xcone_nonneg_H.setToConstant(0.0, stream);
        derivs.xcone_soc_H.setToConstant(0.0, stream);
        derivs.xcone_psd_H.setToConstant(0.0, stream);
        derivs.xcone_exp_H.setToConstant(0.0, stream);
        derivs.xcone_pow_H.setToConstant(0.0, stream);
        derivs.xcone_genpow_H.setToConstant(0.0, stream);

        // Total per-kind sizes for kernel dispatch.
        int64_t total_xn = 0, total_x_nn = 0, total_x_soc_kkt = 0, total_x_psd_kkt = 0;
        int64_t total_x_exp_kkt = 0, total_x_pow_kkt = 0, total_x_genpow_kkt = 0;
        for (const auto& xc : cones.x_cones) {
            int64_t d = static_cast<int64_t>(xc.indices.size());
            total_xn += d;
            if (xc.kind == XConeKind::Nonneg)         total_x_nn         += d;
            else if (xc.kind == XConeKind::SOC)       total_x_soc_kkt    += d * d;
            else if (xc.kind == XConeKind::PSD)       total_x_psd_kkt    += d * d;
            else if (xc.kind == XConeKind::Exp)       total_x_exp_kkt    += 9;
            else if (xc.kind == XConeKind::Power)     total_x_pow_kkt    += 9;
            else if (xc.kind == XConeKind::GenPower)  total_x_genpow_kkt += d * d;
        }

        // Per-cone metadata. h_off is the kind-specific storage offset
        // (into nonneg array / SOC block / Exp block / Power block).
        // Recomputed each backward — small, ~4 bytes per cone, batch-agnostic.
        std::vector<int64_t> kinds, dims, numel_offsets(1, 0), h_off, indices_flat;
        int64_t nn_acc = 0, soc_acc = 0, exp_acc = 0, pow_acc = 0, gp_acc = 0;
        for (const auto& xc : cones.x_cones) {
            int64_t d = static_cast<int64_t>(xc.indices.size());
            dims.push_back(d);
            numel_offsets.push_back(numel_offsets.back() + d);
            switch (xc.kind) {
                case XConeKind::Nonneg:
                    kinds.push_back(0);
                    h_off.push_back(nn_acc);
                    nn_acc += d;
                    break;
                case XConeKind::SOC:
                    kinds.push_back(1);
                    h_off.push_back(soc_acc);
                    soc_acc += d * d;
                    break;
                case XConeKind::PSD:
                    kinds.push_back(2);
                    h_off.push_back(0);  // PSD path uses its own offsets
                    break;
                case XConeKind::Exp:
                    kinds.push_back(3);
                    h_off.push_back(exp_acc);
                    exp_acc += 9;
                    break;
                case XConeKind::Power:
                    kinds.push_back(4);
                    h_off.push_back(pow_acc);
                    pow_acc += 9;
                    break;
                case XConeKind::GenPower:
                    kinds.push_back(5);
                    h_off.push_back(gp_acc);
                    gp_acc += d * d;
                    break;
            }
            for (int64_t v : xc.indices) indices_flat.push_back(v);
        }
        auto upload = [&](const std::vector<int64_t>& host) {
            int64_t* p = nullptr;
            cudaMalloc(&p, sizeof(int64_t) * host.size());
            cudaMemcpyAsync(p, host.data(), sizeof(int64_t) * host.size(),
                            cudaMemcpyHostToDevice, stream);
            return p;
        };
        int64_t* d_kinds   = upload(kinds);
        int64_t* d_dims    = upload(dims);
        int64_t* d_numel_o = upload(numel_offsets);
        int64_t* d_h_off   = upload(h_off);
        int64_t* d_indices = upload(indices_flat);

        compute_xcone_H_kernel<<<batchSize, 256, 0, stream>>>(
            derivs.xcone_nonneg_H.data(),
            derivs.xcone_soc_H.data(),
            derivs.xcone_psd_H.data(),
            x_eq.data(), state.z_x.data(),
            d_kinds, d_dims, d_indices, d_numel_o, d_h_off,
            cones.numXCones, n, total_xn,
            total_x_nn, total_x_soc_kkt, total_x_psd_kkt,
            batchSize
        );

        // Asymmetric direct-x (Exp, Power): dense 3*3 H_x via the same
        // projection-Jacobian formulas as slack-form Exp/Power. Reuses
        // the Newton helpers from kernels.cu via the public wrapper.
        if (total_x_exp_kkt > 0 || total_x_pow_kkt > 0) {
            compute_xcone_asymm_H(
                derivs.xcone_exp_H.data(),
                derivs.xcone_pow_H.data(),
                x_eq.data(), state.z_x.data(),
                d_kinds, d_indices, d_numel_o, d_h_off,
                cones.d_xcone_pow_idx, cones.d_xcone_pow_alpha,
                cones.numXCones, n, total_xn,
                total_x_exp_kkt, total_x_pow_kkt,
                batchSize, stream
            );
        }

        // GenPow direct-x: Newton + rank-3 decomposition (mirrors slack
        // path) expanded to dense dim*dim per cone. CPU does the same
        // dense expansion for direct-x GenPow in
        // `build_hsde_augmented_system_sparse_full`.
        if (total_x_genpow_kkt > 0) {
            compute_xcone_genpow_H(
                derivs.xcone_genpow_H.data(),
                derivs.xcone_genpow_rank3_diag.data(),
                derivs.xcone_genpow_rank3_left1.data(),
                derivs.xcone_genpow_rank3_right1.data(),
                derivs.xcone_genpow_rank3_left2.data(),
                derivs.xcone_genpow_rank3_right2.data(),
                derivs.xcone_genpow_rank3_left3.data(),
                derivs.xcone_genpow_rank3_c3.data(),
                x_eq.data(), state.z_x.data(),
                d_kinds, d_indices, d_numel_o, d_h_off,
                cones.d_xcone_genpow_idx,
                cones.d_xcone_genpow_dim1s,
                cones.d_xcone_genpow_dim2s,
                cones.d_xcone_genpow_alpha_offsets,
                cones.d_xcone_genpow_dim_offsets,
                cones.d_xcone_genpow_alphas,
                derivs.xcone_genpow_diff_work_vec.data(),
                derivs.xcone_genpow_diff_work_dim1.data(),
                cones.numXCones,
                cones.numXGenPowerCones,
                cones.totalXGenPowerDim,
                cones.totalXGenPowerAlphas,
                n, total_xn, total_x_genpow_kkt,
                batchSize, stream
            );
        }

        // Direct-x SOC rank-2 sparse expansion (dim > 4). The KKT sparsity
        // emits 2 expansion vars per such cone; this kernel computes the
        // per-cone diag/v1/v2/c1/c2 stripes from u_x = z_x − x[J] using
        // the slack-form Moreau formulas.
        if (derivs.numSparseXSoc > 0 &&
            solver.diff_kkt_->getNumSparseXSoc() > 0) {
            compute_xcone_soc_rank2(
                derivs.xcone_soc_rank2_diag.data(),
                derivs.xcone_soc_rank2_v1.data(),
                derivs.xcone_soc_rank2_v2.data(),
                derivs.xcone_soc_rank2_c1.data(),
                derivs.xcone_soc_rank2_c2.data(),
                x_eq.data(), state.z_x.data(),
                d_indices, d_numel_o,
                solver.diff_kkt_->getXSocSparseDimOffsets(),
                solver.diff_kkt_->getXSocSparseToXc(),
                solver.diff_kkt_->getXSocSparseDims(),
                derivs.numSparseXSoc, derivs.totalSparseXSocDim,
                n, total_xn, batchSize, stream
            );
        }

        cudaStreamSynchronize(stream);
        cudaFree(d_kinds);
        cudaFree(d_dims);
        cudaFree(d_numel_o);
        cudaFree(d_h_off);
        cudaFree(d_indices);

        // PSD direct-x H_x is computed by a dedicated path that uses
        // cuSOLVER eigendecomp + Ω-matrix Jacobian construction (mirrors
        // the slack-form `compute_psd_derivative` but on `u_x = z_x −
        // x[J]`). compute_xcone_H_kernel is a no-op for PSD entries.
        if (total_x_psd_kkt > 0) {
            compute_xcone_psd_derivative(
                derivs.xcone_psd_H.data(),
                x_eq.data(), state.z_x.data(),
                cones, derivs,
                total_xn, total_x_psd_kkt, n, batchSize, stream
            );
        }
    }

    // Step 4: Compute HSDE coefficients c1, c2, c3 in equilibrated space
    // Need Px_eq first
    // P_eq * x_eq (using equilibrated P from solver data)
    if (P.nnz() > 0) {
        double alpha = 1.0, beta = 0.0;
        csrSpMVBatched(
            n, n, P.nnz(), batchSize,
            P.rowOffsets(), P.colIndices(), P.values(),
            x_eq.data(), Px_eq.data(),
            alpha, beta, stream
        );
    } else {
        Px_eq.setToConstant(0.0, stream);
    }

    // Get equilibrated q and b from solver data
    const double* q_eq = solver.data.q.data();
    const double* b_eq = solver.data.b.data();

    compute_hsde_coefficients(
        state.c1.data(), state.c2.data(), state.c3.data(),
        Px_eq.data(), x_eq.data(), q_eq, b_eq, state.tau.data(),
        n, m, batchSize, stream
    );

    // Step 5-6-7-8-9: Split into Woodbury PCG path vs DiffKKT (cuDSS) path
    //
    // For Woodbury problems (diagonal P, zero + nonneg cones only), use
    // DiffWoodbury which solves (J'J + εI)*lam = -rhs via PCG with
    // structure-exploiting J matvec. Avoids cuDSS entirely.

    // Steps 7-8 are shared (transform upstream grads, build RHS).
    // Step 7: Transform upstream gradients to equilibrated space.
    // The m-sized transforms are skipped when m=0 (direct-x-only problems
    // with no slack constraints): work_m / work_m2 / einv all have n=0
    // and the user typically passes placeholder dz_bar/ds_bar of size 1,
    // which would trip the elementwise_div size-equality assertion.
    elementwise_div(state.work_n, dx_bar, state.dinv, stream);  // work_n = dx_bar * D
    if (m > 0) {
        elementwise_div(state.work_m2, dz_bar, state.einv, stream);  // work_m2 = dz_bar * E (temp)
        div_per_batch(state.work_m, state.work_m2, state.c_scale, stream);  // work_m = dz_bar * E / c

        elementwise_mul(state.work_m2, ds_bar, state.einv, stream);  // work_m2 = ds_bar * E^{-1}
    }

    // Step 8: Build adjoint RHS. With direct-x cones the τ slot moves to
    // index n + 2m + xn, and any user-provided `dz_x_bar` flows positively
    // to BOTH the du_x slot and the corresponding primal-x slot (matching
    // the CPU IFT-direct convention `du_x = z_x − x[J]`).
    std::unique_ptr<BatchedVector> dz_x_eq_buf;
    const double* dz_x_eq_ptr = nullptr;
    int64_t xn = 0;
    for (const auto& xc : cones.x_cones) {
        xn += static_cast<int64_t>(xc.indices.size());
    }
    if (dz_x_bar != nullptr && xn > 0) {
        // Equilibrate dz_x_bar (user frame) → dz_x_eq, then pass through.
        dz_x_eq_buf = std::make_unique<BatchedVector>(xn, batchSize);
        equilibrate_dz_x(
            dz_x_eq_buf->data(), dz_x_bar->data(),
            state.dinv.data(), state.c_scale.data(),
            cones.d_xcone_indices,
            n, xn, batchSize, stream
        );
        dz_x_eq_ptr = dz_x_eq_buf->data();
    }
    {
        build_adjoint_rhs_hsde_with_xcones(
            state.rhs.data(),
            state.work_n.data(), state.work_m.data(), state.work_m2.data(),
            dz_x_eq_ptr,
            x_eq.data(), z_eq.data(), s_eq.data(),
            (xn > 0 ? state.z_x.data() : nullptr),
            cones.d_xcone_indices,
            n, m, xn, batchSize, stream
        );
    }

    // DiffWoodbury has no direct-x adjoint path (its setup() takes only the
    // slack rank-2 expansion, no x_cones). When the forward used Woodbury
    // *and* the problem has direct-x cones, route the backward through the
    // general DiffKKT/cuDSS path instead — the Woodbury forward solution is
    // still valid; only the adjoint solver differs. DiffKKT builds its own
    // augmented-HSDE structure from P/A and handles direct-x cones fully.
    // cuDSS (DiffKKT) backward — the κ-accurate path (solves the augmented KKT,
    // not the κ² normal equations). Used directly for non-Woodbury problems and
    // as a fallback when the Woodbury chain loses accuracy on an ill-conditioned
    // instance.
    auto run_cudss = [&]() {
        // Step 5: Lazily initialize DiffKKT
        if (!solver.diff_kkt_ || solver.diff_kkt_->batchSize != batchSize) {
            std::vector<int64_t> P_ro_host(n + 1);
            std::vector<int64_t> P_ci_host(P.nnz());
            std::vector<int64_t> A_ro_host(m + 1);
            std::vector<int64_t> A_ci_host(A.nnz());

            cudaStreamSynchronize(stream);
            cudaMemcpy(P_ro_host.data(), P.rowOffsets(), sizeof(int64_t) * (n + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(P_ci_host.data(), P.colIndices(), sizeof(int64_t) * P.nnz(), cudaMemcpyDeviceToHost);
            cudaMemcpy(A_ro_host.data(), A.rowOffsets(), sizeof(int64_t) * (m + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(A_ci_host.data(), A.colIndices(), sizeof(int64_t) * A.nnz(), cudaMemcpyDeviceToHost);

            KKTSolverType kkt_type = solver.kkt->actualSolverType();

            solver.diff_kkt_ = std::make_unique<DiffKKT>(
                n, m, batchSize,
                P_ro_host.data(), P_ci_host.data(), P.nnz(),
                A_ro_host.data(), A_ci_host.data(), A.nnz(),
                cones, kkt_type, stream
            );
        }

        // Step 5: Update J values
        solver.diff_kkt_->updateJ(
            P.values(), P.nnz(),
            A.values(), A.nnz(),
            q_eq, b_eq,
            state.c1.data(), state.c2.data(), state.c3.data(),
            derivs.nonneg_H.data(),
            derivs.soc_H.data(),
            derivs.soc_sparse_diag.data(),
            derivs.soc_sparse_v1.data(),
            derivs.soc_sparse_v2.data(),
            derivs.soc_sparse_c1.data(),
            derivs.soc_sparse_c2.data(),
            derivs.exp_H.data(),
            derivs.power_H.data(),
            derivs.psd_H.data(),
            derivs.genpow_sparse_diag.data(),
            derivs.genpow_sparse_left1.data(),
            derivs.genpow_sparse_right1.data(),
            derivs.genpow_sparse_left2.data(),
            derivs.genpow_sparse_right2.data(),
            derivs.genpow_sparse_left3.data(),
            derivs.genpow_sparse_c3.data(),
            // Direct-x cone projection Jacobians. nullptr when there are
            // no direct-x cones; the kernel only writes when sizes are >0.
            derivs.xcone_nonneg_H.data(),
            derivs.xcone_soc_H.data(),
            derivs.xcone_psd_H.data(),
            derivs.xcone_exp_H.data(),
            derivs.xcone_pow_H.data(),
            derivs.xcone_genpow_H.data(),
            derivs.xcone_genpow_rank3_diag.data(),
            derivs.xcone_genpow_rank3_left1.data(),
            derivs.xcone_genpow_rank3_right1.data(),
            derivs.xcone_genpow_rank3_left2.data(),
            derivs.xcone_genpow_rank3_right2.data(),
            derivs.xcone_genpow_rank3_left3.data(),
            derivs.xcone_genpow_rank3_c3.data(),
            derivs.xcone_soc_rank2_diag.data(),
            derivs.xcone_soc_rank2_v1.data(),
            derivs.xcone_soc_rank2_v2.data(),
            derivs.xcone_soc_rank2_c1.data(),
            derivs.xcone_soc_rank2_c2.data(),
            cones, stream
        );

        // Step 6: Factorize
        solver.diff_kkt_->factor(stream);

        // Step 9: Solve
        solver.diff_kkt_->solveAdjoint(state.rhs.data(), state.sol.data(), stream);
    };

    const bool use_woodbury_backward =
        solver.kkt->isWoodbury() && cones.numXCones == 0;
    if (use_woodbury_backward) {
        // Woodbury direct Schur complement backward path
        auto& wb = dynamic_cast<WoodburyKKTData&>(*solver.kkt);
        if (!solver.diff_woodbury_ || solver.diff_woodbury_->batchSize_ != batchSize) {
            solver.diff_woodbury_ = std::make_unique<DiffWoodbury>(
                n, m, batchSize, cones, wb.cublas_, wb.cusolver_, stream);
        }
        solver.diff_woodbury_->setup(
            wb, derivs.nonneg_H.data(),
            q_eq, b_eq,
            state.c1.data(), state.c2.data(), state.c3.data(),
            stream);
        solver.diff_woodbury_->solveAdjoint(state.rhs.data(), state.sol.data(), stream);

        // The Woodbury chain solves the normal equations (κ²); on near-singular
        // instances it loses accuracy. Detect via the relative solve residual and
        // fall back to the κ-accurate cuDSS path for the whole batch when any
        // element exceeds tolerance.
        constexpr double kWoodburyBackwardResidTol = 1e-6;
        cudaStreamSynchronize(stream);
        std::vector<double> rn(batchSize);
        cudaMemcpy(rn.data(), solver.diff_woodbury_->residualNorms(),
                   sizeof(double) * batchSize, cudaMemcpyDeviceToHost);
        double max_rn = 0.0;
        for (double v : rn) max_rn = std::max(max_rn, v);
        if (max_rn > kWoodburyBackwardResidTol) {
            run_cudss();  // re-solve into state.sol with the κ-accurate path
        }
    } else {
        run_cudss();
    }

    // Step 10: Extract gradients in equilibrated space.
    // τ slot of `state.sol` is at index n+2m+xn when direct-x cones are present.
    {
        int64_t xn = 0;
        for (const auto& xc : cones.x_cones) {
            xn += static_cast<int64_t>(xc.indices.size());
        }
        extract_gradients_hsde_with_xcones(
            state.dq.data(), state.db.data(),
            state.sol.data(),
            x_eq.data(), z_eq.data(), state.tau.data(),
            n, m, xn, batchSize, stream
        );
    }

    // Step 11: Transform dq from equilibrated space: dq_final = c * D * dq_eq = c * dq_eq / dinv
    mul_per_batch(state.work_n, state.dq, state.c_scale, stream);  // work_n = c * dq_eq
    elementwise_div(state.dq, state.work_n, state.dinv, stream);   // dq = c * dq_eq / dinv

    // Transform db from equilibrated space: db_final = E * db_eq = db_eq / einv.
    // Skipped when m=0 (direct-x-only): state.db / state.einv both have n=0
    // and the call would write nothing anyway.
    if (m > 0) {
        elementwise_div(state.db, state.db, state.einv, stream);
    }

    // Compute dP gradient in equilibrated space then transform.
    // λ_τ slot is at index n + 2m + xn (direct-x cones shift it).
    {
        int64_t xn_dp = 0;
        for (const auto& xc : cones.x_cones) {
            xn_dp += static_cast<int64_t>(xc.indices.size());
        }
        compute_dP_gradient_hsde_with_xcones(
            state.dP_values.data(),
            P.rowOffsets(),
            P.colIndices(),
            state.sol.data(),
            x_eq.data(), state.tau.data(),
            n, m, xn_dp, state.dP_values.n(), batchSize, stream
        );
    }

    // Transform dP_eq to dP: dP[i,j] = c * d[i] * d[j] * dP_eq = c / (dinv[i] * dinv[j]) * dP_eq
    transform_dP_from_equilibrated(
        state.dP_values.data(),
        P.rowOffsets(),
        P.colIndices(),
        state.dinv.data(), state.c_scale.data(),
        n, state.dP_values.n(), batchSize, stream
    );

    // Compute dA gradient in equilibrated space then transform
    // state.sol has stride n + 2*m + 1 (non-augmented HSDE dimension)
    // z_eq has stride m (non-augmented)
    // jdim accounts for direct-x rows: it's the stride between batches in
    // `state.sol`. Without this, lambda's tau slot is read at the wrong
    // offset on direct-x problems, corrupting both dA and dP gradients.
    int64_t xn_stride = 0;
    for (const auto& xc : cones.x_cones) {
        xn_stride += static_cast<int64_t>(xc.indices.size());
    }
    int64_t jdim = n + 2 * m + xn_stride + 1;
    compute_dA_gradient_hsde(
        state.dA_values.data(),
        A.rowOffsets(),
        A.colIndices(),
        state.sol.data(),
        x_eq.data(), z_eq.data(), state.tau.data(),
        n, m, jdim, m, state.dA_values.n(), batchSize, stream
    );

    // Transform dA_eq to dA: dA[i,j] = e[i] * d[j] * dA_eq = dA_eq / (einv[i] * dinv[j])
    transform_dA_from_equilibrated(
        state.dA_values.data(),
        A.rowOffsets(),
        A.colIndices(),
        state.dinv.data(), state.einv.data(),
        n, m, state.dA_values.n(), batchSize, stream
    );
}

// ============================================================================
// Forward Differentiation
// ============================================================================

void forward(
    DiffState& state,
    const BatchedVector& dP_values,
    const BatchedVector& dq,
    const BatchedVector& dA_values,
    const BatchedVector& db,
    BatchedVector& dx,
    BatchedVector& dz,
    BatchedVector& ds,
    CompiledSolver& solver,
    cudaStream_t stream
) {
    // Forward mode is disabled due to equilibration scaling issues.
    // The input perturbations dP/dA need to be transformed to equilibrated space
    // (dP_eq = c * D * dP * D, dA_eq = E * dA * D) but this is not implemented.
    // Use backward mode (which is correct) and finite differences for JVPs if needed.
    throw std::runtime_error(
        "Forward differentiation is not yet implemented correctly. "
        "Use backward() for gradient computation."
    );

}

// ============================================================================
// DiffKKT Construction Helper
// ============================================================================

static void create_diff_kkt(
    CompiledSolver& solver,
    int64_t n, int64_t m, int64_t batchSize,
    const CSR& P, const CSR& A, const Cones& cones,
    cudaStream_t stream
) {
    std::vector<int64_t> P_ro_host(n + 1);
    std::vector<int64_t> P_ci_host(P.nnz());
    std::vector<int64_t> A_ro_host(m + 1);
    std::vector<int64_t> A_ci_host(A.nnz());

    cudaStreamSynchronize(stream);
    cudaMemcpy(P_ro_host.data(), P.rowOffsets(), sizeof(int64_t) * (n + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(P_ci_host.data(), P.colIndices(), sizeof(int64_t) * P.nnz(), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_ro_host.data(), A.rowOffsets(), sizeof(int64_t) * (m + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_ci_host.data(), A.colIndices(), sizeof(int64_t) * A.nnz(), cudaMemcpyDeviceToHost);

    // Use the same solver type as the forward pass
    KKTSolverType kkt_type = solver.kkt->actualSolverType();

    solver.diff_kkt_ = std::make_unique<DiffKKT>(
        n, m, batchSize,
        P_ro_host.data(), P_ci_host.data(), P.nnz(),
        A_ro_host.data(), A_ci_host.data(), A.nnz(),
        cones, kkt_type, stream
    );
}

// ============================================================================
// Eager DiffKKT Initialization
// ============================================================================

void init_diff_kkt(
    CompiledSolver& solver,
    cudaStream_t stream
) {
    if (solver.diff_kkt_) {
        return;
    }

    create_diff_kkt(solver,
                    solver.data.n, solver.data.m, solver.data.batchSize,
                    solver.data.P, solver.data.A, solver.data.cones,
                    stream);
}

} // namespace moreau
