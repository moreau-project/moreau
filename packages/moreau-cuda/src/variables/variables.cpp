/**
 * @file variables.cpp
 * @brief Implementation of variables methods
 */

#include "moreau/variables/variables.hpp"
#include "moreau/residuals/residuals.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/solver/solver.hpp"  // For ScalingStrategy
#include "moreau/variables/variables_kernels.cuh"
#include "moreau/vector/vector_kernels.cuh"
#include "moreau/kkt/kkt_kernels.cuh"    // update_xcones_nonneg_scaling
#include "moreau/kkt/kkt_solver.hpp" // KKTSolver::xcone_scatter_targets
#include "moreau/cones/psd_kernels.cuh" // update_xcones_psd_scaling
#include <cmath>
#include <iostream>
#include <limits>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>

namespace moreau {

// Forward declarations for the direct-x rank-6 PD scaling helpers
// implemented in cones/genpow_pd_kernels.cu.
void compute_xgenpow_pd_axes(
    const double* var_x, const double* var_z_x,
    const double* xcone_grad_primal,
    const double* xgenpow_p, const double* xgenpow_q, const double* xgenpow_r,
    const double* xgenpow_d1, const double* xgenpow_d2,
    const double* mu_per_batch,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t n, int64_t batchSize,
    double* pd_axes, double* pd_coefs, double* pd_signs,
    double* pd_active, double* pd_workspace,
    const int8_t* pd_enabled_per_batch,
    cudaStream_t stream);

void apply_xgenpow_pd_to_kkt(
    double* kkt_values,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    const int64_t* H_xcone_genpow_pd_axis_idx_0,
    const int64_t* H_xcone_genpow_pd_axis_idx_1,
    const int64_t* H_xcone_genpow_pd_axis_idx_2,
    const int64_t* H_xcone_genpow_pd_axis_idx_3,
    const int64_t* H_xcone_genpow_pd_axis_idx_4,
    const int64_t* H_xcone_genpow_pd_axis_idx_5,
    const int64_t* H_xcone_genpow_exp_diag_idx,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowerDim,
    int64_t nnzKKT,
    int64_t batchSize,
    cudaStream_t stream);

void Variables::calc_mu(BatchedVector& mu, const Residuals& residuals, const Cones& cones,
                        cudaStream_t stream) {
    int64_t degree = cones.degree();

    // Call kernel: mu = (dot_sz + τ*κ) / (degree + 1)
    calc_mu_kernel(
        mu.data(),
        residuals.dot_sz.data(),
        τ.data(),
        κ.data(),
        degree,
        mu.batchSize(),
        stream
    );
}

bool Variables::scale_cones(Cones& cones, const BatchedVector& mu,
                             ScalingStrategy scaling, cudaStream_t stream,
                             KKTSolver* kkt_solver,
                             const int8_t* pd_enabled_per_batch) {
    // Update slack cone scaling (zero/nonneg/SOC/exp/power/genpow).
    // `pd_enabled_per_batch` propagates the CPU `strategy_checkpoint_small_step`
    // per-batch fallback to the slack GenPow rank-6 PD-axes computation.
    const bool ok = cones.update_scaling(s, z, mu, scaling, stream, pd_enabled_per_batch);

    // Direct-x NT scaling. CPU trait: `direct_x_update_scaling(x, z)`
    // applies the primal↔dual swap at the slack boundary so that the
    // stored w/λ/η/Hs carry the dual-barrier interpretation the direct-x
    // (1,1) block needs (see kernels.cu block comment above
    // `_nt_scaling_soc_dense_slack`). Two kernels — one for nonneg,
    // one for dense SOC (dim ≤ 4); each skips cones of the other
    // kind via d_xcone_kinds[cone]. Sparse SOC (dim > 4) uses a rank-2
    // u/v refresh inside the same kernel.
    if (total_xcone_numel_ > 0 && ok) {
        // PSD direct-x scaling: NT factorization with primal↔dual swap,
        // populating xcone_psd_R/Rinv/lambda/Lambdaisqrt/Hs. The Hs values
        // get scattered into KKT by `refresh_xcone_hs` (PSD branch).
        if (cones.numXPsdCones > 0) {
            update_xcones_psd_scaling(cones, x.data(), z_x.data(),
                                       x.n(), stream);
        }
        // Pull the fused KKT scatter targets from the KKT solver (when
        // available). Default-constructed targets have null pointers and
        // cause the kernels to fall back to the classic two-step path.
        KKTSolver::XConeScatterTargets t;
        if (kkt_solver != nullptr) {
            t = kkt_solver->xcone_scatter_targets();
        }
        update_xcones_nonneg_scaling(
            x.data(), z_x.data(),
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
            cones.d_xcone_indices,
            cones.xcone_w.data(), cones.xcone_lambda.data(),
            cones.xcone_Hs.data(),
            t.kkt_values, t.xcone_px_baseline, t.H_xcone_hs_idx, t.nnzKKT,
            x.batchSize(), x.n(),
            cones.numXCones, total_xcone_numel_,
            cones.totalXConeHsEntries,
            stream);
        update_xcones_soc_scaling(
            x.data(), z_x.data(),
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
            cones.d_xcone_indices,
            cones.d_xcone_sparse_indices, cones.d_xcone_sparse_offsets,
            cones.d_xcone_cone_pos_for_sorted,
            cones.xcone_w.data(), cones.xcone_lambda.data(),
            cones.xcone_eta.data(), cones.xcone_Hs.data(),
            cones.xcone_u.data(), cones.xcone_v.data(), cones.xcone_d.data(),
            t.kkt_values, t.xcone_px_baseline,
            t.H_xcone_hs_idx, t.H_xcone_u_idx,
            t.H_xcone_v_idx, t.H_xcone_exp_diag_idx,
            t.nnzKKT,
            x.batchSize(), x.n(),
            cones.numXCones, total_xcone_numel_,
            cones.totalXConeHsEntries,
            cones.totalSparseXSocDim, cones.numSparseXSoc,
            stream);
        if (cones.numXExpCones > 0) {
            update_xcones_exp_scaling(
                x.data(), z_x.data(), mu.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
                cones.d_xcone_indices,
                cones.xcone_Hs.data(), cones.xcone_grad_primal.data(),
                t.kkt_values, t.xcone_px_baseline, t.H_xcone_hs_idx, t.nnzKKT,
                x.batchSize(), x.n(),
                cones.numXCones, total_xcone_numel_,
                cones.totalXConeHsEntries,
                stream);
        }
        if (cones.numXPowerCones > 0) {
            update_xcones_pow_scaling(
                x.data(), z_x.data(), mu.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
                cones.d_xcone_indices,
                cones.d_xcone_pow_idx, cones.d_xcone_pow_alpha,
                cones.xcone_Hs.data(), cones.xcone_grad_primal.data(),
                t.kkt_values, t.xcone_px_baseline, t.H_xcone_hs_idx, t.nnzKKT,
                x.batchSize(), x.n(),
                cones.numXCones, total_xcone_numel_,
                cones.totalXConeHsEntries,
                stream);
        }
        if (cones.numXGenPowerCones > 0) {
            update_xcones_genpow_scaling(
                x.data(), z_x.data(), mu.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
                cones.d_xcone_indices,
                cones.d_xcone_genpow_idx,
                cones.d_xcone_genpow_dim1s,
                cones.d_xcone_genpow_dim2s,
                cones.d_xcone_genpow_alpha_offsets,
                cones.d_xcone_genpow_dim_offsets,
                cones.d_xcone_genpow_sparse_idx,
                cones.d_xcone_genpow_sparse_offsets,
                cones.d_xcone_genpow_sparse_q_offsets,
                cones.d_xcone_genpow_sparse_r_offsets,
                cones.d_xcone_genpow_alphas,
                cones.xcone_Hs.data(),
                cones.xcone_grad_primal.data(),
                cones.xcone_genpow_p.data(),
                cones.xcone_genpow_q.data(),
                cones.xcone_genpow_r.data(),
                cones.xcone_genpow_d1.data(),
                cones.xcone_genpow_d2.data(),
                t.kkt_values,
                t.xcone_px_baseline,
                t.H_xcone_hs_idx,
                t.H_xcone_genpow_q_idx,
                t.H_xcone_genpow_r_idx,
                t.H_xcone_genpow_p_idx,
                t.H_xcone_genpow_pd_axis_idx[0],
                t.H_xcone_genpow_pd_axis_idx[1],
                t.H_xcone_genpow_pd_axis_idx[2],
                t.H_xcone_genpow_pd_axis_idx[3],
                t.H_xcone_genpow_pd_axis_idx[4],
                t.H_xcone_genpow_pd_axis_idx[5],
                t.H_xcone_exp_diag_idx
                    ? t.H_xcone_exp_diag_idx + 2 * cones.numSparseXSoc
                    : nullptr,
                // PD-axis state: pass-through from cones (nullptr until rank-6
                // PD compute kernel is wired). Inactive everywhere → kernel writes
                // tiny ε + sign·1 sentinel for cuDSS structural soundness.
                cones.xcone_genpow_pd_axes.data(),
                cones.xcone_genpow_pd_coefs.data(),
                cones.xcone_genpow_pd_signs.data(),
                cones.xcone_genpow_pd_active.data(),
                t.nnzKKT,
                x.batchSize(), x.n(),
                cones.numXCones,
                cones.numXGenPowerCones,
                total_xcone_numel_,
                cones.totalXConeHsEntries,
                cones.totalXGenPowerDim,
                cones.totalXGenPowerAlphas,
                cones.totalXGenPowerDim2,
                cones.numSparseXGenPow,
                cones.totalSparseXGenPowDim,
                stream);

            // Rank-6 PD scaling. Per-batch gate via `pd_enabled_per_batch`
            // mirrors CPU `direct_x_update_scaling` PrimalDual/Dual branches.
            const bool any_pd =
                (scaling == ScalingStrategy::PrimalDual) ||
                (pd_enabled_per_batch != nullptr);
            if (!any_pd) {
                cones.xcone_genpow_pd_active.setToConstant(0.0, stream);
            } else {
            compute_xgenpow_pd_axes(
                x.data(), z_x.data(),
                cones.xcone_grad_primal.data(),
                cones.xcone_genpow_p.data(),
                cones.xcone_genpow_q.data(),
                cones.xcone_genpow_r.data(),
                cones.xcone_genpow_d1.data(),
                cones.xcone_genpow_d2.data(),
                mu.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_indices,
                cones.d_xcone_genpow_idx,
                cones.d_xcone_genpow_dim1s, cones.d_xcone_genpow_dim2s,
                cones.d_xcone_genpow_alpha_offsets,
                cones.d_xcone_genpow_dim_offsets,
                cones.d_xcone_genpow_alphas,
                cones.numXCones, cones.numXGenPowerCones,
                total_xcone_numel_, cones.totalXGenPowerDim,
                cones.totalXGenPowerAlphas, cones.totalXGenPowerDim2,
                x.n(), x.batchSize(),
                cones.xcone_genpow_pd_axes.data(),
                cones.xcone_genpow_pd_coefs.data(),
                cones.xcone_genpow_pd_signs.data(),
                cones.xcone_genpow_pd_active.data(),
                cones.xcone_genpow_pd_workspace.data(),
                pd_enabled_per_batch,
                stream);
            } // end if (any_pd)

            // Step 3: rank-9 PD axes scatter into KKT happens inside
            // KKTData::refresh_xcone_hs (called from solver.cpp after
            // scale_cones), where the KKT-internal index arrays live.
        }
    }
    return ok;
}

void Variables::affine_step_rhs(const Residuals& residuals, const Variables& variables, Cones& cones, cudaStream_t stream) {
    // Fused: x=rx, z=rz, τ=rτ, κ=var_τ*var_κ (3 memcpys + 1 kernel → 1 kernel)
    fused_affine_step_rhs_kernel(
        x.data(), z.data(), τ.data(), κ.data(),
        residuals.rx.data(), residuals.rz.data(), residuals.rτ.data(),
        variables.τ.data(), variables.κ.data(),
        x.n(), z.n(), x.batchSize(), stream
    );

    // Compute affine ds using cone-specific operations
    cones.affine_ds(s, variables.s, stream);

    // Direct-x affine_ds.
    //   Nonneg: step_rhs.z_x[k] = x[J[k]] · variables.z_x[k]  (= λ²).
    //   SOC:    step_rhs.z_x[off..off+dim] = λ∘λ  (Jordan square of λ).
    // Nonneg kernel skips SOC entries via d_xcone_kind_per_entry; SOC
    // kernel overwrites SOC entries per (batch, cone) block.
    if (total_xcone_numel_ > 0) {
        fill_step_rhs_zx_nonneg_affine(
            z_x.data(), variables.x.data(), variables.z_x.data(),
            cones.d_xcone_indices,
            cones.d_xcone_kind_per_entry,
            z_x.batchSize(), variables.x.n(), total_xcone_numel_, stream);
        fill_step_rhs_zx_soc_affine(
            z_x.data(), cones.xcone_lambda.data(),
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets,
            z_x.batchSize(), cones.numXCones, total_xcone_numel_, stream);
        if (cones.numXExpCones > 0) {
            fill_step_rhs_zx_exp_affine(
                z_x.data(), variables.z_x.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                z_x.batchSize(), cones.numXCones, total_xcone_numel_, stream);
        }
        if (cones.numXPowerCones > 0) {
            fill_step_rhs_zx_pow_affine(
                z_x.data(), variables.z_x.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                z_x.batchSize(), cones.numXCones, total_xcone_numel_, stream);
        }
        if (cones.numXPsdCones > 0) {
            fill_step_rhs_zx_psd_affine(cones, z_x.data(), stream);
        }
        if (cones.numXGenPowerCones > 0) {
            fill_step_rhs_zx_genpow_affine(
                z_x.data(), variables.z_x.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                z_x.batchSize(), cones.numXCones, total_xcone_numel_, stream);
        }
    }
}

void Variables::combined_step_rhs(
    const Residuals& residuals,
    const Variables& variables,
    Cones& cones,
    Variables& affine_step,
    const BatchedVector& sigma,
    const BatchedVector& mu,
    const BatchedVector& m,
    cudaStream_t stream
) {
    // Fused: compute (1-σ), x=(1-σ)*rx, τ=(1-σ)*rτ  (3→1 kernel)
    fused_scale_residuals_kernel(
        one_minus_sigma.data(),
        x.data(),
        τ.data(),
        sigma.data(),
        residuals.rx.data(),
        residuals.rτ.data(),
        x.n(),
        x.batchSize(),
        stream
    );

    // Fused: kappa_rhs, scaled_affine_z=m*aff.z, sigma_mu=σ*μ  (3→1 kernel)
    fused_scalar_prep_kernel(
        κ.data(),
        scaled_affine_z.data(),
        sigma_mu_vec.data(),
        sigma.data(),
        mu.data(),
        m.data(),
        affine_step.τ.data(),
        affine_step.κ.data(),
        variables.τ.data(),
        variables.κ.data(),
        affine_step.z.data(),
        scaled_affine_z.n(),
        scaled_affine_z.batchSize(),
        stream
    );

    // Call combined_ds_shift to compute the shift term into z (as work vector)
    // NOTE: combined_ds_shift writes the result into the first parameter (z)
    cones.combined_ds_shift(z, scaled_affine_z, affine_step.s, sigma_mu_vec, stream);

    // Fused: s += z (shift result); z = (1-σ) * rz  (2 → 1 kernel)
    fused_axpby_and_scale_kernel(
        s.data(), z.data(),
        residuals.rz.data(), one_minus_sigma.data(),
        z.n(), z.batchSize(), stream
    );

    // Direct-x combined shift. Mirror of slack `_combined_ds_shift_symmetric`
    // with primal/dual swapped:
    //   Nonneg: z_x += step_aff.x[J] · (m · step_aff.z_x) − σμ
    //   SOC:    z_x += (W·step_aff.x[J]) ∘ (W⁻¹·m·step_aff.z_x) − σμ·e
    if (total_xcone_numel_ > 0) {
        add_combined_ds_shift_nonneg(
            z_x.data(),
            affine_step.x.data(), affine_step.z_x.data(),
            sigma_mu_vec.data(), m.data(),
            cones.d_xcone_indices,
            cones.d_xcone_kind_per_entry,
            z_x.batchSize(), x.n(), total_xcone_numel_, stream);
        add_combined_ds_shift_soc(
            z_x.data(),
            affine_step.x.data(), affine_step.z_x.data(),
            sigma_mu_vec.data(), m.data(),
            cones.xcone_w.data(), cones.xcone_eta.data(),
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets, cones.d_xcone_indices,
            z_x.batchSize(), x.n(), cones.numXCones,
            total_xcone_numel_, stream);
        if (cones.numXExpCones > 0) {
            add_combined_ds_shift_exp(
                z_x.data(),
                sigma_mu_vec.data(), m.data(), mu.data(),
                variables.x.data(),
                affine_step.x.data(), affine_step.z_x.data(),
                cones.xcone_Hs.data(),
                cones.xcone_grad_primal.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                cones.d_xcone_hs_offsets, cones.d_xcone_indices,
                z_x.batchSize(), x.n(),
                cones.numXCones,
                total_xcone_numel_, cones.totalXConeHsEntries,
                stream);
        }
        if (cones.numXPowerCones > 0) {
            add_combined_ds_shift_pow(
                z_x.data(),
                sigma_mu_vec.data(), m.data(), mu.data(),
                variables.x.data(),
                affine_step.x.data(), affine_step.z_x.data(),
                cones.xcone_Hs.data(),
                cones.xcone_grad_primal.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                cones.d_xcone_hs_offsets, cones.d_xcone_indices,
                cones.d_xcone_pow_idx, cones.d_xcone_pow_alpha,
                z_x.batchSize(), x.n(),
                cones.numXCones,
                total_xcone_numel_, cones.totalXConeHsEntries,
                stream);
        }
        if (cones.numXPsdCones > 0) {
            add_combined_ds_shift_psd(
                cones, z_x.data(),
                affine_step.x.data(), affine_step.z_x.data(),
                sigma_mu_vec.data(), m.data(),
                x.n(), stream);
        }
        if (cones.numXGenPowerCones > 0) {
            add_combined_ds_shift_genpow(
                z_x.data(),
                sigma_mu_vec.data(),
                cones.xcone_grad_primal.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                z_x.batchSize(), cones.numXCones,
                total_xcone_numel_, stream);
            // 3rd-order Mehrotra η correction (CPU `direct_x_combined_ds_shift`).
            subtract_eta_primal_genpow(
                z_x.data(),
                variables.x.data(),
                affine_step.x.data(),
                affine_step.z_x.data(),
                mu.data(),
                cones.d_xcone_kinds,
                cones.d_xcone_dims,
                cones.d_xcone_numel_offsets,
                cones.d_xcone_indices,
                cones.d_xcone_genpow_idx,
                cones.d_xcone_genpow_dim1s,
                cones.d_xcone_genpow_alpha_offsets,
                cones.d_xcone_genpow_alphas,
                z_x.batchSize(), x.n(), cones.numXCones,
                total_xcone_numel_, stream);
        }
    }
}

void Variables::add_step(const Variables& step, const BatchedVector& alpha, cudaStream_t stream) {
    // Fused update: x+=alpha*dx, s+=alpha*ds, z+=alpha*dz, tau+=alpha*dtau, kappa+=alpha*dkappa
    apply_step(x, step.x, s, step.s, z, step.z, τ, step.τ, κ, step.κ, alpha, stream);

    // Direct-x cone dual step: z_x += α · step.z_x. No-op when no
    // direct-x cones are present (z_x is placeholder size).
    if (total_xcone_numel_ > 0) {
        apply_step_z_x(z_x, step.z_x, alpha, total_xcone_numel_, stream);
    }
}

void Variables::calc_step_length(
    const Variables& step,
    Cones& cones,
    double max_step_fraction,
    bool is_combined_step,
    BatchedVector& alpha,
    BatchedVector& work1,
    BatchedVector& work2,
    double backtrack_step,
    double min_step_length,
    cudaStream_t stream
) {
    // Fused kernel: initializes alpha to 1.0, computes step length constraints
    // for both τ and κ, and also initializes alpha_z (work1) and alpha_s (work2).
    // This eliminates 2 D2D memcpys that Cones::step_length previously needed.
    calc_step_length_tau_kappa_init_kernel(
        alpha.data(),
        τ.data(),
        step.τ.data(),
        κ.data(),
        step.κ.data(),
        alpha.batchSize(),
        stream,
        work1.data(),
        work2.data()
    );

    // Get step length from cones (alpha_z=work1, alpha_s=work2 are pre-initialized)
    cones.step_length(step.z, step.s, z, s, alpha, work1, work2, backtrack_step, min_step_length, stream);

    // Direct-x min-ratio step length: fold into work1 (alpha_z) and
    // work2 (alpha_s). Nonneg entries: scalar min-ratio. SOC entries:
    // quadratic boundary solve. Both no-op when no x-cones present.
    if (total_xcone_numel_ > 0) {
        // Match CPU asymmetric-cone ceiling `1 - √ε`
        // (compositecone.rs::step_length).
        const double x_cone_max_step =
            1.0 - std::sqrt(std::numeric_limits<double>::epsilon());
        xcone_step_length_nonneg_reduce(
            work2.data(), work1.data(),
            x.data(), z_x.data(),
            step.x.data(), step.z_x.data(),
            cones.d_xcone_indices,
            cones.d_xcone_kind_per_entry,
            alpha.batchSize(), x.n(), total_xcone_numel_,
            x_cone_max_step, stream);
        xcone_step_length_soc_reduce(
            work2.data(), work1.data(),
            x.data(), z_x.data(),
            step.x.data(), step.z_x.data(),
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets, cones.d_xcone_indices,
            alpha.batchSize(), x.n(), cones.numXCones,
            total_xcone_numel_, x_cone_max_step, stream);
        if (cones.numXExpCones > 0) {
            xcone_step_length_exp_reduce(
                work2.data(), work1.data(),
                x.data(), z_x.data(),
                step.x.data(), step.z_x.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_indices,
                alpha.batchSize(), x.n(), cones.numXCones,
                total_xcone_numel_, x_cone_max_step,
                min_step_length, backtrack_step, stream);
        }
        if (cones.numXPowerCones > 0) {
            xcone_step_length_pow_reduce(
                work2.data(), work1.data(),
                x.data(), z_x.data(),
                step.x.data(), step.z_x.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_indices,
                cones.d_xcone_pow_idx, cones.d_xcone_pow_alpha,
                alpha.batchSize(), x.n(), cones.numXCones,
                total_xcone_numel_, x_cone_max_step,
                min_step_length, backtrack_step, stream);
        }
        if (cones.numXPsdCones > 0) {
            xcone_step_length_psd_reduce(
                cones, work2.data(), work1.data(),
                x.data(), z_x.data(),
                step.x.data(), step.z_x.data(),
                x.n(), x_cone_max_step, stream);
        }
        if (cones.numXGenPowerCones > 0) {
            xcone_step_length_genpow_reduce(
                work2.data(), work1.data(),
                x.data(), z_x.data(),
                step.x.data(), step.z_x.data(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_numel_offsets, cones.d_xcone_indices,
                cones.d_xcone_genpow_idx, cones.d_xcone_genpow_dim1s,
                cones.d_xcone_genpow_alpha_offsets,
                cones.d_xcone_genpow_alphas,
                alpha.batchSize(), x.n(), cones.numXCones,
                total_xcone_numel_, x_cone_max_step,
                min_step_length, backtrack_step, stream);
        }
    }

    // Fused: alpha = min(alpha_z, alpha_s) * scale  (2→1 kernel)
    double scale = is_combined_step ? max_step_fraction : 1.0;
    fused_step_length_finalize_kernel(alpha.data(), work1.data(), work2.data(), scale, alpha.batchSize(), stream);
}

double Variables::barrier(
    const Variables& step,
    double alpha,
    Cones& cones,
    cudaStream_t stream
) {
    // NOTE: The scalar tau/kappa/mu computation below reads only the first
    // batch element. cones.computeBarrier is fully batched and sums across
    // batches. For batchSize>1, the tau/kappa/mu terms would need per-batch
    // handling as well. Currently unused (the solver uses backtrackStepToBarrier).
    int64_t bs = z.batchSize();

    int64_t degree = cones.degree();
    double central_coef = static_cast<double>(degree + 1);

    // Get current τ and κ (per-batch values summed)
    std::vector<double> tau_vals(bs), kappa_vals(bs), step_tau_vals(bs), step_kappa_vals(bs);
    cudaMemcpy(tau_vals.data(), τ.data(), sizeof(double) * bs, cudaMemcpyDeviceToHost);
    cudaMemcpy(kappa_vals.data(), κ.data(), sizeof(double) * bs, cudaMemcpyDeviceToHost);
    cudaMemcpy(step_tau_vals.data(), step.τ.data(), sizeof(double) * bs, cudaMemcpyDeviceToHost);
    cudaMemcpy(step_kappa_vals.data(), step.κ.data(), sizeof(double) * bs, cudaMemcpyDeviceToHost);

    // Compute mu = (s'z + τ*κ) / (degree + 1)
    // Main cones: cur_s = s + α*step.s, cur_z = z + α*step.z
    axpby(cur_s, alpha, step.s, 0.0, stream);
    axpby(cur_s, 1.0, s, 1.0, stream);
    axpby(cur_z, alpha, step.z, 0.0, stream);
    axpby(cur_z, 1.0, z, 1.0, stream);

    // Compute dot product for main cones (per-batch)
    dot_batched(sz_dot, cur_s, cur_z, stream);
    std::vector<double> sz_vals(bs);
    cudaMemcpy(sz_vals.data(), sz_dot.data(), sizeof(double) * bs, cudaMemcpyDeviceToHost);

    // Barrier terms from gap and scalars (summed across batches)
    // barrier = central_coef * log(mu) - log(τ) - log(κ)
    double barrier_val = 0.0;

    // Safe log: if x <= 0, return -infinity
    auto logsafe = [](double x) {
        return (x <= 0.0) ? -std::numeric_limits<double>::infinity() : std::log(x);
    };

    for (int64_t i = 0; i < bs; i++) {
        double cur_tau = tau_vals[i] + alpha * step_tau_vals[i];
        double cur_kappa = kappa_vals[i] + alpha * step_kappa_vals[i];
        double mu = (sz_vals[i] + cur_tau * cur_kappa) / central_coef;
        barrier_val += central_coef * logsafe(mu);
        barrier_val -= logsafe(cur_tau);
        barrier_val -= logsafe(cur_kappa);
    }

    // Add cone barriers from main cones (fully batched)
    barrier_val += cones.computeBarrier(z, s, step.z, step.s, alpha, stream);

    return barrier_val;
}

} // namespace moreau
