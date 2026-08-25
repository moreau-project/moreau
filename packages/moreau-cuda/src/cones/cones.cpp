/**
 * @file cones.cpp
 * @brief Implementation of cone projection and operations
 */

#include "moreau/cones/cones.hpp"
#include "moreau/solver/solver.hpp"  // For ScalingStrategy
#include "moreau/cones/cone_kernels.cuh"
#include "moreau/cones/psd_kernels.cuh"
#include "moreau/cuda/utils.hpp"
#include "moreau/cuda/status_utils.hpp"
#include <iostream>
#include <limits>

namespace moreau {

// Forward declaration of the PD-axes computation kernel launcher,
// implemented in src/cones/genpow_pd_kernels.cu.
void compute_genpow_pd_axes(
    const double* s, const double* z,
    const double* genpow_grad,
    const double* genpow_p, const double* genpow_q, const double* genpow_r,
    const double* genpow_d1, const double* genpow_d2,
    const double* genpow_z_copy,
    const double* mu_per_batch,
    const double* d_genPowerAlphas,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerAlphaOffsets,
    const int64_t* d_genPowerSzOffsets,
    int64_t numGenPowerCones,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerDim2,
    int64_t m_total,
    int64_t offset_genpow,
    int64_t batchSize,
    double* pd_axes, double* pd_coefs, double* pd_signs,
    double* pd_active, double* pd_workspace,
    const int8_t* pd_enabled_per_batch,
    cudaStream_t stream
);

bool Cones::update_scaling(const BatchedVector& s, const BatchedVector& z, const BatchedVector& mu, ScalingStrategy scaling, cudaStream_t stream, const int8_t* pd_enabled_per_batch) {
    // Use fused kernel for all cone types (replaces 5 separate kernel launches)
    int64_t m_total = totalConstraints();
    int scaling_int = (scaling == ScalingStrategy::PrimalDual) ? 0 : 1;

    bool success = update_scaling_all_cones(
        // Nonneg outputs
        nonneg_w.data(),
        nonneg_lambda.data(),
        // SOC outputs
        soc_u.data(),
        soc_v.data(),
        soc_d.data(),
        soc_w.data(),
        soc_eta.data(),
        soc_lambda.data(),
        soc_Hs.data(),
        // Exp outputs
        exp_grad.data(),
        exp_H_dual.data(),
        exp_Hs.data(),
        exp_z.data(),
        // Power outputs
        power_grad.data(),
        power_H_dual.data(),
        power_Hs.data(),
        power_z.data(),
        // Inputs
        s.data(),
        z.data(),
        mu.data(),
        d_powerAlphas,
        // Cone structure
        numZeroCones,
        numNonnegCones,
        numSocCones,
        numExpCones,
        numPowerCones,
        m_total,
        batchSize,
        scaling_int,
        stream,
        // Variable-dim SOC params
        d_soc_dims,
        d_soc_offsets,
        d_soc_Hs_offsets,
        totalSocDim,
        totalSocHsEntries,
        d_scaling_success,
        d_soc_sz_offsets,
        h_scaling_success_pinned,
        // GenPowerCone params
        genpow_grad.data(),
        genpow_z.data(),
        genpow_Hs.data(),
        genpow_p.data(),
        genpow_q.data(),
        genpow_r.data(),
        genpow_d1.data(),
        genpow_d2.data(),
        d_genPowerAlphas,
        d_genPowerDim1s,
        d_genPowerDim2s,
        d_genPowerOffsets,
        d_genPowerAlphaOffsets,
        d_genPowerHsOffsets,
        d_genPowerSzOffsets,
        numGenPowerCones,
        totalGenPowerDim,
        totalGenPowerAlphas,
        totalGenPowerGradEntries,
        totalGenPowerHsEntries,
        numLargeSoc,
        numLargeGenPow
    );

    if (!success) return false;

    // PSD cones: separate scaling (uses cuSOLVER, can't fuse into composite kernel).
    // update_psd_scaling reports per-batch numerical failure (non-finite Hs from a
    // degenerate s/z) by writing d_scaling_success[b] = 0 — the same per-batch flag
    // the composite kernel sets for SOC/GenPow. The IPM loop picks these up via its
    // deferred per-batch check at the next iteration; sync once here so the
    // synchronous caller (smoothing-iterate refinement) also observes the result.
    if (numPsdCones > 0) {
        update_psd_scaling(*this, s.data(), z.data(), m_total, d_scaling_success, stream);
        if (h_scaling_success_pinned) {
            cudaStreamSynchronize(stream);
            for (int64_t b = 0; b < batchSize; b++) {
                if (h_scaling_success_pinned[b] == 0) { success = false; break; }
            }
        }
    }

    // Store mu for later use in mul_Hs (sparse GenPowerCone needs mu for rank-3 multiply)
    if (numSparseGenPow > 0) {
        CUDA_THROW(cudaMemcpyAsync(genpow_mu.data(), mu.data(), sizeof(double) * batchSize,
                        cudaMemcpyDeviceToDevice, stream));
    }

    // Rank-6 sparse PD scaling for GenPow. When PrimalDual scaling is
    // requested, attempt to compute the 6 PD axes; on numerical failure
    // (secant verification trips) fall back to dual-only with pd_active=0
    // for that cone in that batch. Mirrors CPU `try_compute_pd_axes`.
    // The kernel also gates on `pd_enabled_per_batch[b]` (CPU mirror of
    // `strategy_checkpoint_small_step`): batches where that flag is 0
    // skip rank-6 entirely and write the inactive defaults.
    if (numGenPowerCones > 0 && scaling == ScalingStrategy::PrimalDual) {
        const int64_t m_total_int = totalConstraints();
        const int64_t offset_genpow = numZeroCones + numNonnegCones + totalSocDim
                                    + numExpCones * 3 + numPowerCones * 3;
        compute_genpow_pd_axes(
            s.data(), z.data(),
            genpow_grad.data(),
            genpow_p.data(), genpow_q.data(), genpow_r.data(),
            genpow_d1.data(), genpow_d2.data(),
            genpow_z.data(),
            mu.data(),
            d_genPowerAlphas,
            d_genPowerDim1s, d_genPowerDim2s,
            d_genPowerOffsets, d_genPowerAlphaOffsets, d_genPowerSzOffsets,
            numGenPowerCones, totalGenPowerDim, totalGenPowerAlphas,
            totalGenPowerDim2, m_total_int, offset_genpow, batchSize,
            genpow_pd_axes.data(), genpow_pd_coefs.data(), genpow_pd_signs.data(),
            genpow_pd_active.data(), genpow_pd_workspace.data(),
            pd_enabled_per_batch,
            stream
        );
    } else if (numGenPowerCones > 0) {
        // Dual scaling: clear pd_active so KKT/mul_Hs see no PD contribution.
        genpow_pd_active.setToConstant(0.0, stream);
        genpow_pd_coefs.setToConstant(0.0, stream);
    }

    return success;
}

void Cones::smoothing(BatchedVector& z, const BatchedVector& work, const BatchedVector& mu, cudaStream_t stream) {
    int64_t m_total = totalConstraints();

    smoothing_all_cones(
        z.data(),
        work.data(),
        mu.data(),
        d_powerAlphas,
        numZeroCones,
        numNonnegCones,
        numSocCones,
        numExpCones,
        numPowerCones,
        m_total,
        batchSize,
        stream,
        d_soc_dims,
        d_soc_offsets,
        totalSocDim,
        d_soc_sz_offsets,
        // GenPowerCone params
        numGenPowerCones,
        d_genPowerAlphas,
        d_genPowerDim1s,
        d_genPowerDim2s,
        d_genPowerOffsets,
        d_genPowerAlphaOffsets,
        d_genPowerSzOffsets,
        totalGenPowerDim,
        // GenPowerCone workspace
        genpow_smooth_zlocal.data(),
        genpow_smooth_wlocal.data(),
        genpow_smooth_res.data(),
        genpow_smooth_delta.data(),
        genpow_smooth_hmat.data(),
        genpow_smooth_lmat.data(),
        d_genPowerDimSqOffsets,
        totalGenPowerDimSq,
        genpow_grad.data(),
        genpow_d1.data(),
        genpow_p.data(),
        genpow_q.data(),
        genpow_r.data(),
        totalGenPowerAlphas,
        totalGenPowerDim2
    );

    // PSD cones: eigendecomp smoothing
    if (numPsdCones > 0) {
        int64_t psd_offset = numZeroCones + numNonnegCones + totalSocDim
                           + numExpCones * 3 + numPowerCones * 3;
        psd_smoothing(*this, z.data(), work.data(), mu.data(),
                      psd_offset, m_total, stream);
    }
}

void Cones::margins_batched(const BatchedVector& z, BatchedVector& min_margin_out, BatchedVector& pos_margin_out,
                            bool is_primal_cone, cudaStream_t stream) {
    int64_t m_total = totalConstraints();

    // Call GPU kernel to compute per-batch margins
    // This version does NOT reduce across batches, keeping results independent
    compute_margins_batched_kernel(
        z.data(),
        min_margin_out.data(),
        pos_margin_out.data(),
        d_batch_margin_results,
        numZeroCones,
        numNonnegCones,
        numSocCones,
        numExpCones,
        numPowerCones,
        m_total,
        batchSize,
        stream,
        d_soc_dims,
        d_soc_offsets,
        totalSocDim,
        d_soc_sz_offsets
    );
}

void Cones::scaled_unit_shift_batched(BatchedVector& z, const BatchedVector& alpha, bool is_primal_cone, cudaStream_t stream) {
    int64_t m_total = totalConstraints();

    // Call GPU kernel to apply per-batch scaled unit shift
    // Each batch has its own alpha value
    scaled_unit_shift_batched_kernel(
        z.data(),
        alpha.data(),
        is_primal_cone,
        numZeroCones,
        numNonnegCones,
        numSocCones,
        numExpCones,
        numPowerCones,
        m_total,
        batchSize,
        stream,
        d_soc_offsets,
        totalSocDim,
        d_soc_sz_offsets
    );
}

void Cones::fused_margins_and_shift(BatchedVector& z, bool is_primal_cone, double total_degree, cudaStream_t stream) {
    int64_t m_total = totalConstraints();

    fused_margins_and_shift_kernel(
        z.data(),
        is_primal_cone,
        total_degree,
        numZeroCones,
        numNonnegCones,
        numSocCones,
        numExpCones,
        numPowerCones,
        m_total,
        batchSize,
        stream,
        d_soc_dims,
        d_soc_offsets,
        totalSocDim,
        d_soc_sz_offsets
    );
}

void Cones::psd_shift_to_interior(BatchedVector& z, double total_degree, cudaStream_t stream) {
    if (numPsdCones == 0) return;

    int64_t m_total = totalConstraints();
    int64_t psd_offset = numZeroCones + numNonnegCones + totalSocDim
                       + numExpCones * 3 + numPowerCones * 3;
    cudaStreamSynchronize(stream);

    // Reconstruct host-side sz_offsets
    std::vector<int64_t> orig_off(psdConeDimsOriginal.size() + 1, 0);
    for (size_t i = 0; i < psdConeDimsOriginal.size(); i++)
        orig_off[i+1] = orig_off[i] + psdConeDimsOriginal[i]*(psdConeDimsOriginal[i]+1)/2;
    std::vector<int64_t> sz_off(numPsdCones);
    for (int64_t i = 0; i < numPsdCones; i++)
        sz_off[i] = orig_off[psdSortPerm[i]];

    // Compute PSD min_margin and pos_margin via eigendecomp
    double min_margin = 1e30, pos_margin = 0.0;

    for (int64_t b = 0; b < batchSize; b++) {
        int64_t matsq_off = 0;
        for (int64_t c = 0; c < numPsdCones; c++) {
            int64_t n = psdConeDims[c];
            int64_t n2 = n * n;
            int64_t svec_dim = n*(n+1)/2;

            // Read svec, convert to dense matrix
            std::vector<double> svec_h(svec_dim);
            CUDA_THROW(cudaMemcpy(svec_h.data(),
                      z.data() + b * m_total + psd_offset + sz_off[c],
                      sizeof(double) * svec_dim, cudaMemcpyDeviceToHost));

            std::vector<double> mat(n2, 0.0);
            int64_t idx = 0;
            for (int64_t col = 0; col < n; col++) {
                for (int64_t row = 0; row <= col; row++) {
                    double v = svec_h[idx++];
                    if (row == col) {
                        mat[col*n+row] = v;
                    } else {
                        double u = v * 0.7071067811865476;
                        mat[col*n+row] = u;
                        mat[row*n+col] = u;
                    }
                }
            }

            // Eigendecompose using this cone's workspace
            double* d_mat = psd_work_mat1.data() + b * totalPsdMatSqDim + matsq_off;
            double* d_eig = psd_work_mat3.data() + b * totalPsdMatSqDim + matsq_off;
            CUDA_THROW(cudaMemcpy(d_mat, mat.data(), sizeof(double) * n2, cudaMemcpyHostToDevice));
            CUDA_THROW(cudaMemset(d_psd_info_, 0, sizeof(int)));
            CUSOLVER_THROW(cusolverDnSetStream(cusolverH_, stream));
            CUSOLVER_THROW(cusolverDnDsyevd(cusolverH_,
                            CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                            n, d_mat, n, d_eig,
                            d_psd_syevd_work_, psd_syevd_work_size_,
                            d_psd_info_));
            CUDA_THROW(cudaDeviceSynchronize());

            std::vector<double> eigs(n);
            CUDA_THROW(cudaMemcpy(eigs.data(), d_eig, sizeof(double) * n, cudaMemcpyDeviceToHost));

            for (int64_t i = 0; i < n; i++) {
                if (eigs[i] < min_margin) min_margin = eigs[i];
                if (eigs[i] > 0) pos_margin += eigs[i];
            }
            matsq_off += n2;
        }
    }

    // CPU _shift_to_cone_interior algorithm
    double target = std::max(1.0, pos_margin * 0.1 / total_degree);

    if (min_margin <= 0.0) {
        psd_scaled_unit_shift(z.data(), -min_margin, psd_offset, m_total, *this, stream);
        psd_scaled_unit_shift(z.data(), target, psd_offset, m_total, *this, stream);
    } else if (min_margin < target) {
        psd_scaled_unit_shift(z.data(), target - min_margin, psd_offset, m_total, *this, stream);
    }
}

} // namespace moreau
