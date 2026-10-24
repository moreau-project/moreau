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
        totalPsdSvecDim,
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
        const int64_t offset_genpow = genPowerOffset();
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
        totalPsdSvecDim,
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
        int64_t psd_offset = psdOffset();
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
        totalPsdSvecDim,
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
        totalPsdSvecDim,
        d_soc_sz_offsets
    );
}

void Cones::fused_margins_and_shift(BatchedVector& z, bool is_primal_cone, double total_degree, cudaStream_t stream) {
    int64_t m_total = totalConstraints();
    if (numPsdCones > 0) psd_eigenvalues(*this, z.data(), psdOffset(), m_total, stream);

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
        totalPsdSvecDim,
        d_soc_sz_offsets,
        numPsdCones,
        d_psd_dims,
        d_psd_sz_offsets,
        d_psd_mat_offsets,
        psd_work_mat3.data(),
        totalPsdMatSqDim
    );
}

} // namespace moreau
