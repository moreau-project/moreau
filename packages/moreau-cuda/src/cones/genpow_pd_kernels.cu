/**
 * @file genpow_pd_kernels.cu
 * @brief Kernel implementing the rank-6 sparse Mosek-Tunçel primal-dual
 * scaling for GenPowerCone on GPU.
 *
 * Mirrors the CPU `try_compute_pd_axes` (genpowcone.rs) +
 * `pd_scaling_nd_qr6` (nonsymmetric_common.rs). One thread per cone.
 * On success, sets pd_active=1.0 and stashes 6 axes/coefs/signs. On
 * failure (numerical guards trip), falls back to dual-only scaling
 * with pd_active=0.0 and zero axes/coefs.
 *
 * Workspace layout (per batch, per cone, dim-sized):
 *   workspace = [gs | δs | δz | e_z | e_dzp | q3 | q4 | h_z]
 *               (8 dim-sized regions; reused for h_dzp where possible)
 */

#include "moreau/cones/genpow_pd_kernels.cuh"
#include "moreau/cones/cones.hpp"
#include "moreau/cuda/utils.hpp"
#include <cuda_runtime.h>
#include <math.h>

namespace moreau {

namespace {

// Inline pd_scaling_nd_qr6 — see the CPU implementation in
// nonsymmetric_common.rs for the math derivation. Returns true on
// success, false if any numerical guard fires (all axes zeroed by caller
// in that case via memset of pd_axes before the kernel runs).
__device__ __forceinline__ bool pd_scaling_nd_qr6_device(
    const double* s,            // length dim
    const double* z,            // length dim
    const double* gz,           // length dim (= dual gradient)
    const double* gs,           // length dim (= primal gradient at s)
    double mu_local,
    int64_t dim,
    // Rank-3 H_dual data:
    const double* p_axis,       // length dim
    const double* q_axis,       // length dim1
    const double* r_axis,       // length dim2
    const double* d1,           // length dim1
    double d2,
    double mu_ipm,
    int64_t dim1,
    int64_t dim2,
    // Workspace (8 × dim doubles):
    double* delta_s, double* delta_z,
    double* e_z, double* e_dzp,
    double* q3, double* q4,
    double* h_z, double* h_dzp_scratch,
    // Output (6 axes, 6 coefs, 6 signs):
    double* axes_out,           // (6, dim) — flattened by axis-then-i
    double* coefs_out,          // 6 entries
    double* signs_out           // 6 entries (±1.0)
) {
    const double EPS_SQRT = 1.49011611938476562e-8;  // sqrt(DBL_EPSILON)
    const double EPS_TINY = 2.22044604925031e-16;    // DBL_EPSILON

    // Zero out outputs first (guard return paths leave them zero).
    for (int k = 0; k < 6; ++k) {
        coefs_out[k] = 0.0;
        signs_out[k] = 1.0;
        for (int64_t i = 0; i < dim; ++i) {
            axes_out[k * dim + i] = 0.0;
        }
    }

    // sz, z_norm guards.
    double sz = 0.0, z_norm_sq = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        sz += s[i] * z[i];
        z_norm_sq += z[i] * z[i];
    }
    if (sz <= EPS_SQRT) return false;
    double z_norm = sqrt(z_norm_sq);
    if (z_norm <= EPS_SQRT) return false;

    // δs = s + μ·gz, δz = z + μ·gs
    for (int64_t i = 0; i < dim; ++i) {
        delta_s[i] = s[i] + mu_local * gz[i];
        delta_z[i] = z[i] + mu_local * gs[i];
    }
    double ds_dz = 0.0;
    for (int64_t i = 0; i < dim; ++i) ds_dz += delta_s[i] * delta_z[i];
    if (ds_dz <= EPS_SQRT) return false;

    // Write secant axes (k=4,5) BEFORE Step 2 clobbers s. The direct-x caller
    // passes h_z = ws_x_local (= s); Step 2 writes μH·e_z into h_z, which would
    // overwrite s before Step 7 reads it. Capture s and δs into axes_out now.
    for (int64_t i = 0; i < dim; ++i) {
        axes_out[4 * dim + i] = s[i];
        axes_out[5 * dim + i] = delta_s[i];
    }
    coefs_out[4] = 1.0 / sz;
    coefs_out[5] = 1.0 / ds_dz;
    signs_out[4] = 1.0;
    signs_out[5] = 1.0;

    // Step 1: orthonormal basis e_z = z/||z||, e_dzp = (δz − ⟨δz,e_z⟩ e_z) / ||·||
    for (int64_t i = 0; i < dim; ++i) e_z[i] = z[i] / z_norm;
    double dot1 = 0.0;
    for (int64_t i = 0; i < dim; ++i) dot1 += delta_z[i] * e_z[i];
    for (int64_t i = 0; i < dim; ++i) e_dzp[i] = delta_z[i] - dot1 * e_z[i];
    // Reorthogonalisation pass (Kahan: "twice is enough")
    double dot2 = 0.0;
    for (int64_t i = 0; i < dim; ++i) dot2 += e_dzp[i] * e_z[i];
    for (int64_t i = 0; i < dim; ++i) e_dzp[i] -= dot2 * e_z[i];
    double dzp_norm_sq = 0.0;
    for (int64_t i = 0; i < dim; ++i) dzp_norm_sq += e_dzp[i] * e_dzp[i];
    double dzp_norm = sqrt(dzp_norm_sq);
    if (dzp_norm <= EPS_SQRT) return false;
    for (int64_t i = 0; i < dim; ++i) e_dzp[i] /= dzp_norm;

    // Step 2: h_z = μH·e_z, h_dzp_scratch = μH·e_dzp; q11, q12, q22
    mul_mu_h_rank3_genpow(h_z,           e_z,   p_axis, q_axis, r_axis, d1, d2, mu_ipm, dim1, dim2);
    mul_mu_h_rank3_genpow(h_dzp_scratch, e_dzp, p_axis, q_axis, r_axis, d1, d2, mu_ipm, dim1, dim2);
    double q11 = 0.0, q12 = 0.0, q22 = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        q11 += e_z[i] * h_z[i];
        q12 += e_z[i] * h_dzp_scratch[i];
        q22 += e_dzp[i] * h_dzp_scratch[i];
    }

    // Step 3: q3 = Gram-Schmidt residual of h_z against e_z, e_dzp
    for (int64_t i = 0; i < dim; ++i) {
        q3[i] = h_z[i] - q11 * e_z[i] - q12 * e_dzp[i];
    }
    double r1 = 0.0, r2 = 0.0;
    for (int64_t i = 0; i < dim; ++i) { r1 += q3[i] * e_z[i]; r2 += q3[i] * e_dzp[i]; }
    for (int64_t i = 0; i < dim; ++i) q3[i] -= r1 * e_z[i] + r2 * e_dzp[i];
    double alpha_sq = 0.0;
    for (int64_t i = 0; i < dim; ++i) alpha_sq += q3[i] * q3[i];
    double alpha_norm = sqrt(alpha_sq);
    bool q3_ok = alpha_norm > EPS_SQRT;
    if (q3_ok) {
        for (int64_t i = 0; i < dim; ++i) q3[i] /= alpha_norm;
    }

    // q4 = Gram-Schmidt residual of h_dzp against e_z, e_dzp, q3
    double beta = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        q4[i] = h_dzp_scratch[i] - q12 * e_z[i] - q22 * e_dzp[i];
    }
    if (q3_ok) {
        for (int64_t i = 0; i < dim; ++i) beta += q4[i] * q3[i];
        for (int64_t i = 0; i < dim; ++i) q4[i] -= beta * q3[i];
    }
    double r3 = 0.0, r4 = 0.0;
    for (int64_t i = 0; i < dim; ++i) { r3 += q4[i] * e_z[i]; r4 += q4[i] * e_dzp[i]; }
    for (int64_t i = 0; i < dim; ++i) q4[i] -= r3 * e_z[i] + r4 * e_dzp[i];
    if (q3_ok) {
        double r5 = 0.0;
        for (int64_t i = 0; i < dim; ++i) r5 += q4[i] * q3[i];
        for (int64_t i = 0; i < dim; ++i) q4[i] -= r5 * q3[i];
    }
    double gamma_sq = 0.0;
    for (int64_t i = 0; i < dim; ++i) gamma_sq += q4[i] * q4[i];
    double gamma_norm = sqrt(gamma_sq);
    bool q4_ok = gamma_norm > EPS_SQRT;
    if (q4_ok) {
        for (int64_t i = 0; i < dim; ++i) q4[i] /= gamma_norm;
    }

    // Step 4: 4×4 M matrix
    double m_mat[4][4] = {{0.0}};
    m_mat[0][0] = -q11;
    m_mat[0][1] = -q12;
    m_mat[1][0] = -q12;
    m_mat[1][1] = -q22;
    if (q3_ok) {
        m_mat[0][2] = -alpha_norm;
        m_mat[2][0] = -alpha_norm;
        m_mat[1][2] = -beta;
        m_mat[2][1] = -beta;
    }
    if (q4_ok) {
        m_mat[1][3] = -gamma_norm;
        m_mat[3][1] = -gamma_norm;
    }

    // Step 5: Jacobi eigendecomp
    double v_mat[4][4] = {{0.0}};
    for (int i = 0; i < 4; ++i) v_mat[i][i] = 1.0;
    jacobi_4x4(m_mat, v_mat);
    double lam[4] = {m_mat[0][0], m_mat[1][1], m_mat[2][2], m_mat[3][3]};

    // Step 6: 4 projector axes ax_k = Q · v[:,k]
    for (int k = 0; k < 4; ++k) {
        double l = lam[k];
        double mag = fabs(l);
        if (mag <= EPS_TINY) {
            coefs_out[k] = 0.0;
            // axis already zeroed
            continue;
        }
        coefs_out[k] = mag;
        signs_out[k] = (l >= 0.0) ? 1.0 : -1.0;
        double v0 = v_mat[0][k];
        double v1 = v_mat[1][k];
        double v2 = q3_ok ? v_mat[2][k] : 0.0;
        double v3 = q4_ok ? v_mat[3][k] : 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            axes_out[k * dim + i] = v0 * e_z[i] + v1 * e_dzp[i]
                                  + v2 * q3[i] + v3 * q4[i];
        }
    }

    // Step 7: secant axes (k=4,5) already written before Step 2 to avoid
    // s-aliasing clobber in the direct-x caller. Nothing to do here.

    return true;
}

}  // anon namespace

// Per-cone PD axis computation kernel. One thread per (batch, cone).
__global__ void try_compute_pd_axes_genpow_kernel(
    // Inputs
    const double* __restrict__ s_global,         // (B, m_total) — slack vector
    const double* __restrict__ z_global,         // (B, m_total) — dual vector
    const double* __restrict__ genpow_grad,      // (B, totalGenPowerDim) — gz
    const double* __restrict__ genpow_p,         // (B, totalGenPowerDim) — rank-3 axis
    const double* __restrict__ genpow_q,         // (B, totalGenPowerAlphas) — rank-3 axis
    const double* __restrict__ genpow_r,         // (B, totalGenPowerDim2) — rank-3 axis
    const double* __restrict__ genpow_d1,        // (B, totalGenPowerAlphas)
    const double* __restrict__ genpow_d2,        // (B, numGenPowerCones)
    const double* __restrict__ genpow_z_copy,    // (B, totalGenPowerDim) — z at scaling pt
    const double* __restrict__ mu_per_batch,     // (B,) — IPM μ
    const double* __restrict__ d_genPowerAlphas, // (totalGenPowerAlphas,)
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    int64_t numGenPowerCones,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerDim2,
    int64_t m_total,
    int64_t offset_genpow,
    int64_t batchSize,
    // Outputs (per batch, per cone)
    double* __restrict__ pd_axes,        // (B, 6 * totalGenPowerDim)
    double* __restrict__ pd_coefs,       // (B, 6 * numGenPowerCones)
    double* __restrict__ pd_signs,       // (B, 6 * numGenPowerCones)
    double* __restrict__ pd_active,      // (B, numGenPowerCones)
    // Workspace (per batch, dim-sized × 8)
    double* __restrict__ pd_workspace,   // (B, 8 * totalGenPowerDim)
    // Per-batch PrimalDual gate. null = all batches enabled.
    // 1 = PrimalDual, 0 = Dual.
    const int8_t* pd_enabled_per_batch
) {
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batchSize) return;

    // Dual branch: zero all outputs and skip.
    if (pd_enabled_per_batch != nullptr && pd_enabled_per_batch[batch_idx] == 0) {
        int64_t stride0 = blockDim.x;
        for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += stride0) {
            int64_t dim1 = d_genPowerDim1s[cone_idx];
            int64_t dim2 = d_genPowerDim2s[cone_idx];
            int64_t dim = dim1 + dim2;
            int64_t gp_off = d_genPowerOffsets[cone_idx];
            double* axes_out  = &pd_axes[batch_idx * 6 * totalGenPowerDim + 6 * gp_off];
            double* coefs_out = &pd_coefs[batch_idx * 6 * numGenPowerCones + 6 * cone_idx];
            double* signs_out = &pd_signs[batch_idx * 6 * numGenPowerCones + 6 * cone_idx];
            double* active_out = &pd_active[batch_idx * numGenPowerCones + cone_idx];
            *active_out = 0.0;
            for (int k = 0; k < 6; ++k) {
                coefs_out[k] = 0.0; signs_out[k] = 1.0;
                for (int64_t i = 0; i < dim; ++i) axes_out[k * dim + i] = 0.0;
            }
        }
        return;
    }

    double mu_ipm = mu_per_batch[batch_idx];

    int64_t stride = blockDim.x;
    for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += stride) {
        int64_t dim1 = d_genPowerDim1s[cone_idx];
        int64_t dim2 = d_genPowerDim2s[cone_idx];
        int64_t dim = dim1 + dim2;
        int64_t gp_off = d_genPowerOffsets[cone_idx];
        int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
        int64_t r_off = gp_off - alpha_off;
        int64_t sz_off = d_genPowerSzOffsets[cone_idx];

        const double* s_cone = &s_global[batch_idx * m_total + offset_genpow + sz_off];
        const double* z_cone = &genpow_z_copy[batch_idx * totalGenPowerDim + gp_off];
        const double* gz_cone = &genpow_grad[batch_idx * totalGenPowerDim + gp_off];
        const double* p_cone = &genpow_p[batch_idx * totalGenPowerDim + gp_off];
        const double* q_cone = &genpow_q[batch_idx * totalGenPowerAlphas + alpha_off];
        const double* r_cone = &genpow_r[batch_idx * totalGenPowerDim2 + r_off];
        const double* d1_cone = &genpow_d1[batch_idx * totalGenPowerAlphas + alpha_off];
        double d2_cone = genpow_d2[batch_idx * numGenPowerCones + cone_idx];
        const double* alphas = &d_genPowerAlphas[alpha_off];

        // Per-cone workspace base. 8 * totalGenPowerDim per batch; we use
        // 8 * dim of it for this cone, packed contiguously by cone offset.
        double* ws_base = &pd_workspace[batch_idx * 8 * totalGenPowerDim + 8 * gp_off];
        double* ws_gs    = ws_base + 0 * dim;
        double* ws_ds    = ws_base + 1 * dim;
        double* ws_dz    = ws_base + 2 * dim;
        double* ws_ez    = ws_base + 3 * dim;
        double* ws_edzp  = ws_base + 4 * dim;
        double* ws_q3    = ws_base + 5 * dim;
        double* ws_q4    = ws_base + 6 * dim;
        double* ws_hz    = ws_base + 7 * dim;
        // h_dzp uses ws_gs as scratch (gs already consumed by then).

        // Output base for this cone.
        double* axes_out  = &pd_axes[batch_idx * 6 * totalGenPowerDim + 6 * gp_off];
        double* coefs_out = &pd_coefs[batch_idx * 6 * numGenPowerCones + 6 * cone_idx];
        double* signs_out = &pd_signs[batch_idx * 6 * numGenPowerCones + 6 * cone_idx];
        double* active_out = &pd_active[batch_idx * numGenPowerCones + cone_idx];

        // 1. Compute gs = ∇F_primal(s)
        double psi_init = 0.0;  // 1/Σα²
        for (int64_t i = 0; i < dim1; ++i) psi_init += alphas[i] * alphas[i];
        psi_init = (psi_init > 0.0) ? 1.0 / psi_init : 1.0;
        gradient_primal_genpow(ws_gs, s_cone, alphas, psi_init, dim1, dim2);

        // 2. μ_local = ⟨s, z⟩ / ν, where ν = degree = dim1 + 1
        double sz = 0.0;
        for (int64_t i = 0; i < dim; ++i) sz += s_cone[i] * z_cone[i];
        if (sz <= 0.0) {
            *active_out = 0.0;
            for (int k = 0; k < 6; ++k) {
                coefs_out[k] = 0.0; signs_out[k] = 1.0;
                for (int64_t i = 0; i < dim; ++i) axes_out[k * dim + i] = 0.0;
            }
            continue;
        }
        double nu = (double)(dim1 + 1);
        double mu_local = sz / nu;

        // 3. pd_scaling_nd_qr6
        bool ok = pd_scaling_nd_qr6_device(
            s_cone, z_cone, gz_cone, ws_gs, mu_local, dim,
            p_cone, q_cone, r_cone, d1_cone, d2_cone, mu_ipm, dim1, dim2,
            ws_ds, ws_dz, ws_ez, ws_edzp, ws_q3, ws_q4, ws_hz, ws_gs,
            axes_out, coefs_out, signs_out
        );

        if (!ok) {
            *active_out = 0.0;
            continue;
        }

        // 4. Verify Hs·z ≈ s and Hs·δz ≈ δs (mirror CPU secant check).
        // Compute hs_z = μH·z + Σ sign_k · coef_k · ⟨axis_k, z⟩ · axis_k,
        // then ||hs_z − s|| / max(||s||, 1).
        double s_norm_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) s_norm_sq += s_cone[i] * s_cone[i];
        double s_norm = sqrt(s_norm_sq);

        double ds_norm_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) ds_norm_sq += ws_ds[i] * ws_ds[i];
        double ds_norm = sqrt(ds_norm_sq);

        // hs_z = μH · z (reuse ws_hz as buffer)
        mul_mu_h_rank3_genpow(ws_hz, z_cone, p_cone, q_cone, r_cone, d1_cone, d2_cone, mu_ipm, dim1, dim2);
        double max_coef = 0.0;
        for (int k = 0; k < 6; ++k) {
            double coef = coefs_out[k];
            if (coef > max_coef) max_coef = coef;
            if (coef == 0.0) continue;
            double dot = 0.0;
            for (int64_t i = 0; i < dim; ++i) dot += axes_out[k * dim + i] * z_cone[i];
            double scale = signs_out[k] * coef * dot;
            for (int64_t i = 0; i < dim; ++i) ws_hz[i] += scale * axes_out[k * dim + i];
        }
        double err_z_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double d = ws_hz[i] - s_cone[i];
            err_z_sq += d * d;
        }

        // hs_dz = μH · δz
        mul_mu_h_rank3_genpow(ws_hz, ws_dz, p_cone, q_cone, r_cone, d1_cone, d2_cone, mu_ipm, dim1, dim2);
        for (int k = 0; k < 6; ++k) {
            double coef = coefs_out[k];
            if (coef == 0.0) continue;
            double dot = 0.0;
            for (int64_t i = 0; i < dim; ++i) dot += axes_out[k * dim + i] * ws_dz[i];
            double scale = signs_out[k] * coef * dot;
            for (int64_t i = 0; i < dim; ++i) ws_hz[i] += scale * axes_out[k * dim + i];
        }
        double err_dz_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double d = ws_hz[i] - ws_ds[i];
            err_dz_sq += d * d;
        }

        // Tolerance: max(1e-7, √dim · ulp · max|coef|) — see CPU comment.
        double n_t = (double)dim;
        double fp_floor = sqrt(n_t) * 2.220446049250313e-16 * max_coef;
        double tol = (fp_floor > 1e-7) ? fp_floor : 1e-7;
        double r_z = sqrt(err_z_sq) / fmax(s_norm, 1.0);
        double r_dz = sqrt(err_dz_sq) / fmax(ds_norm, 1.0);
        if (r_z > tol || r_dz > tol) {
            *active_out = 0.0;
            for (int k = 0; k < 6; ++k) {
                coefs_out[k] = 0.0; signs_out[k] = 1.0;
                for (int64_t i = 0; i < dim; ++i) axes_out[k * dim + i] = 0.0;
            }
            continue;
        }

        *active_out = 1.0;
    }
}

// Host-side launcher.
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
) {
    int threadsPerBlock = (numGenPowerCones < 32) ? 32 : 256;
    if (numGenPowerCones < threadsPerBlock) {
        threadsPerBlock = (numGenPowerCones + 31) / 32 * 32;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
    }
    try_compute_pd_axes_genpow_kernel<<<(unsigned int)batchSize, threadsPerBlock, 0, stream>>>(
        s, z, genpow_grad, genpow_p, genpow_q, genpow_r,
        genpow_d1, genpow_d2, genpow_z_copy, mu_per_batch,
        d_genPowerAlphas, d_genPowerDim1s, d_genPowerDim2s,
        d_genPowerOffsets, d_genPowerAlphaOffsets, d_genPowerSzOffsets,
        numGenPowerCones, totalGenPowerDim, totalGenPowerAlphas,
        totalGenPowerDim2, m_total, offset_genpow, batchSize,
        pd_axes, pd_coefs, pd_signs, pd_active, pd_workspace,
        pd_enabled_per_batch
    );
}

// =====================================================================
// Direct-x rank-6 PD scaling for GenPowerCone.
//
// Mirror of `try_compute_pd_axes_genpow_kernel` (slack) with the
// direct-x argument swap from CPU `try_compute_primal_pd_axes`:
//
//   slack:    pd_scaling_nd_qr6(s,   z,    gz=∇F*(z),   gs=∇F(s),   μ·H_dual)
//             builds Hs satisfying Hs·z = s (so Hs⁻¹·s = z, the
//             primal→dual direction the slack KKT (1,1) block needs
//             via A^T·Hs⁻¹·A).
//
//   direct-x: pd_scaling_nd_qr6(z_x, x,    gz=g_x,      gs=g_zx,    μ·H_primal)
//             builds Hs satisfying Hs·x = z_x (no inverse — the
//             direct-x (1,1) block uses Hs directly, and the
//             linearisation δz_x + μ·H_F(x)·δx = … makes μ·H_F(x)
//             the right Newton coefficient).
//
// Inputs:
// - var_x: (B, n) — full primal vector; cone slice gathered via d_xcone_indices
// - var_z_x: (B, totalXConeNumel) — direct-x dual (flat, offset via numel_offsets)
// - xcone_grad_primal: (B, totalXConeNumel) — ∇F_primal(x) (already populated
//   by update_xcones_genpow_scaling_kernel). For genpow rows this is `g_x`.
// - xcone_genpow_p / q / r / d1 / d2: rank-3 fields populated by scaling kernel.
//
// Outputs: pd_axes / pd_coefs / pd_signs / pd_active (writes per cone).
// =====================================================================
__global__ void try_compute_pd_axes_xgenpow_kernel(
    const double* __restrict__ var_x,
    const double* __restrict__ var_z_x,
    const double* __restrict__ xcone_grad_primal,
    const double* __restrict__ xgenpow_p,    // (B, totalXGenPowerDim)
    const double* __restrict__ xgenpow_q,    // (B, totalXGenPowerAlphas)
    const double* __restrict__ xgenpow_r,    // (B, totalXGenPowerDim2)
    const double* __restrict__ xgenpow_d1,   // (B, totalXGenPowerAlphas)
    const double* __restrict__ xgenpow_d2,   // (B, numXGenPowerCones)
    const double* __restrict__ mu_per_batch, // (B,)
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_dim2s,
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,
    const double* __restrict__ d_xcone_genpow_alphas,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t n,
    int64_t batchSize,
    // Outputs
    double* __restrict__ pd_axes,        // (B, 6 * totalXGenPowerDim)
    double* __restrict__ pd_coefs,       // (B, 6 * numXGenPowerCones)
    double* __restrict__ pd_signs,       // (B, 6 * numXGenPowerCones)
    double* __restrict__ pd_active,      // (B, numXGenPowerCones)
    double* __restrict__ pd_workspace,   // (B, 8 * totalXGenPowerDim)
    // Per-batch PrimalDual gate. null = all batches enabled.
    // 1 = PrimalDual, 0 = Dual.
    const int8_t* pd_enabled_per_batch
) {
    int64_t b = blockIdx.x;
    if (b >= batchSize) return;
    // Dual branch: write zero_pd_axes equivalent and skip rank-6.
    if (pd_enabled_per_batch != nullptr && pd_enabled_per_batch[b] == 0) {
        for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
            if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;
            int64_t gidx = d_xcone_genpow_idx[c];
            int64_t d_off = d_xcone_genpow_dim_offsets[gidx];
            int64_t dim = d_xcone_dims[c];
            double* axes_out = &pd_axes[b * 6 * totalXGenPowerDim + 6 * d_off];
            double* coefs_out = &pd_coefs[b * 6 * numXGenPowerCones + 6 * gidx];
            double* signs_out = &pd_signs[b * 6 * numXGenPowerCones + 6 * gidx];
            double* active_out = &pd_active[b * numXGenPowerCones + gidx];
            *active_out = 0.0;
            for (int k = 0; k < 6; ++k) {
                coefs_out[k] = 0.0;
                signs_out[k] = 1.0;
                for (int64_t i = 0; i < dim; ++i) axes_out[k * dim + i] = 0.0;
            }
        }
        return;
    }
    double mu_ipm = mu_per_batch[b];

    int64_t stride = blockDim.x;
    for (int64_t c = threadIdx.x; c < numXCones; c += stride) {
        if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;

        int64_t gidx     = d_xcone_genpow_idx[c];
        int64_t dim1     = d_xcone_genpow_dim1s[gidx];
        int64_t dim2     = d_xcone_genpow_dim2s[gidx];
        int64_t dim      = dim1 + dim2;
        int64_t num_off  = d_xcone_numel_offsets[c];
        int64_t a_off    = d_xcone_genpow_alpha_offsets[gidx];
        int64_t d_off    = d_xcone_genpow_dim_offsets[gidx];
        int64_t r_off    = d_off - a_off;

        const double* alphas = &d_xcone_genpow_alphas[a_off];

        // Workspace base for this cone (8 × dim doubles).
        double* ws_base = &pd_workspace[b * 8 * totalXGenPowerDim + 8 * d_off];
        double* ws_x_local = ws_base + 0 * dim;  // gathered x[J]
        double* ws_gzx     = ws_base + 1 * dim;  // ∇F*(z_x)
        double* ws_dx      = ws_base + 2 * dim;  // δx = x + μ·gzx
        double* ws_dzx     = ws_base + 3 * dim;  // δz_x = z_x + μ·g_x
        double* ws_ez      = ws_base + 4 * dim;
        double* ws_edzp    = ws_base + 5 * dim;
        double* ws_q3      = ws_base + 6 * dim;
        double* ws_q4      = ws_base + 7 * dim;
        // Scratch for h_z and h_dzp_scratch reuses ws_dx / ws_dzx after their
        // single-pass use in the orthogonalisation step. We reuse explicitly
        // below to avoid an extra allocation.

        // Output pointers for this cone.
        double* axes_out  = &pd_axes[b * 6 * totalXGenPowerDim + 6 * d_off];
        double* coefs_out = &pd_coefs[b * 6 * numXGenPowerCones + 6 * gidx];
        double* signs_out = &pd_signs[b * 6 * numXGenPowerCones + 6 * gidx];
        double* active_out = &pd_active[b * numXGenPowerCones + gidx];

        // Default to inactive on any early-return path. We seed defaults
        // up-front; later steps overwrite axes/coefs/signs on success.
        *active_out = 0.0;
        for (int k = 0; k < 6; ++k) {
            coefs_out[k] = 0.0;
            signs_out[k] = 1.0;
            for (int64_t i = 0; i < dim; ++i) axes_out[k * dim + i] = 0.0;
        }

        // Gather x[J] into ws_x_local.
        for (int64_t i = 0; i < dim; ++i) {
            ws_x_local[i] = var_x[b * n + d_xcone_indices[num_off + i]];
        }
        const double* z_x_cone = &var_z_x[b * totalXConeNumel + num_off];
        const double* g_x_cone = &xcone_grad_primal[b * totalXConeNumel + num_off];

        // Step A: g_zx = ∇F*(z_x) (closed-form). Bail on dual-infeasible.
        double zeta_d = gradient_dual_genpow(ws_gzx, z_x_cone, alphas, dim1, dim2);
        if (!(zeta_d > 0.0)) continue;

        // Step C: μ_local = ⟨x, z_x⟩ / ν, with ν = degree = dim1 + 1.
        double xz = 0.0;
        for (int64_t i = 0; i < dim; ++i) xz += ws_x_local[i] * z_x_cone[i];
        if (!(xz > 0.0)) continue;
        double nu = (double)(dim1 + 1);
        double mu_local = xz / nu;

        // Pull rank-3 cone data.
        const double* p_cone  = &xgenpow_p [b * totalXGenPowerDim    + d_off];
        const double* q_cone  = &xgenpow_q [b * totalXGenPowerAlphas + a_off];
        const double* r_cone  = &xgenpow_r [b * totalXGenPowerDim2   + r_off];
        const double* d1_cone = &xgenpow_d1[b * totalXGenPowerAlphas + a_off];
        double d2_cone        =  xgenpow_d2[b * numXGenPowerCones    + gidx];

        // Step D: pd_scaling_nd_qr6 with the direct-x argument swap.
        // Helper signature is (s, z, gz, gs); we feed (z_x, x, g_x, g_zx)
        // so it produces Hs satisfying Hs·x = z_x (the right Newton-
        // coefficient direction for the direct-x (1,1) block). Workspace
        // re-use note: pd_scaling_nd_qr6_device uses the FOUR basis slots
        // ws_ez/edzp/q3/q4, plus h_z and h_dzp_scratch; we re-use
        // ws_x_local and ws_gzx for the latter two (their contents are
        // no longer needed once the helper has its own copies via its
        // arg list).
        // Direct-x storage convention: p,q,r already scaled by sqrt(μ);
        // d1,d2 already scaled by μ (see update_xcones_genpow_scaling_kernel).
        // mul_mu_h_rank3_genpow's `mu` arg therefore must be 1.0 (not mu_ipm)
        // to avoid double-scaling.
        bool ok = pd_scaling_nd_qr6_device(
            z_x_cone, ws_x_local, g_x_cone, ws_gzx, mu_local, dim,
            p_cone, q_cone, r_cone, d1_cone, d2_cone, /*mu=*/1.0, dim1, dim2,
            ws_dzx, ws_dx, ws_ez, ws_edzp, ws_q3, ws_q4,
            /*h_z=*/ws_x_local, /*h_dzp_scratch=*/ws_gzx,
            axes_out, coefs_out, signs_out
        );
        if (!ok) continue;  // bail; defaults seeded above

        // Step E: secant verification — H·x ≈ z_x and H·δx ≈ δz_x.
        // Re-gather x into ws_x_local (was clobbered by qr6 internal writes).
        for (int64_t i = 0; i < dim; ++i) {
            ws_x_local[i] = var_x[b * n + d_xcone_indices[num_off + i]];
        }
        // Recompute g_zx into ws_gzx (was clobbered).
        gradient_dual_genpow(ws_gzx, z_x_cone, alphas, dim1, dim2);

        double z_norm_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) z_norm_sq += z_x_cone[i] * z_x_cone[i];
        double z_norm = sqrt(z_norm_sq);

        // δx = x + μ_local·g_zx, δz_x = z_x + μ_local·g_x.
        for (int64_t i = 0; i < dim; ++i) {
            ws_dx [i] = ws_x_local[i] + mu_local * ws_gzx[i];
            ws_dzx[i] = z_x_cone[i]   + mu_local * g_x_cone[i];
        }
        double dz_norm_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) dz_norm_sq += ws_dzx[i] * ws_dzx[i];
        double dz_norm = sqrt(dz_norm_sq);

        // h_x = μH·x → reuse ws_q3 as scratch. Pass mu=1.0; storage
        // already has μ baked into d1/d2 and sqrt(μ) into p/q/r.
        mul_mu_h_rank3_genpow(ws_q3, ws_x_local, p_cone, q_cone, r_cone,
                              d1_cone, d2_cone, /*mu=*/1.0, dim1, dim2);
        double max_coef = 0.0;
        for (int k = 0; k < 6; ++k) {
            double coef = coefs_out[k];
            if (coef > max_coef) max_coef = coef;
            if (coef == 0.0) continue;
            double dot = 0.0;
            for (int64_t i = 0; i < dim; ++i) dot += axes_out[k * dim + i] * ws_x_local[i];
            double scale = signs_out[k] * coef * dot;
            for (int64_t i = 0; i < dim; ++i) ws_q3[i] += scale * axes_out[k * dim + i];
        }
        double err_x_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double d = ws_q3[i] - z_x_cone[i];
            err_x_sq += d * d;
        }

        // h_dx = μH·δx → reuse ws_q4. Pass mu=1.0 (see above).
        mul_mu_h_rank3_genpow(ws_q4, ws_dx, p_cone, q_cone, r_cone,
                              d1_cone, d2_cone, /*mu=*/1.0, dim1, dim2);
        for (int k = 0; k < 6; ++k) {
            double coef = coefs_out[k];
            if (coef == 0.0) continue;
            double dot = 0.0;
            for (int64_t i = 0; i < dim; ++i) dot += axes_out[k * dim + i] * ws_dx[i];
            double scale = signs_out[k] * coef * dot;
            for (int64_t i = 0; i < dim; ++i) ws_q4[i] += scale * axes_out[k * dim + i];
        }
        double err_dx_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double d = ws_q4[i] - ws_dzx[i];
            err_dx_sq += d * d;
        }

        // Tolerance: max(1e-7, √dim · ulp · max|coef|).
        double n_t = (double)dim;
        double fp_floor = sqrt(n_t) * 2.220446049250313e-16 * max_coef;
        double tol = (fp_floor > 1e-7) ? fp_floor : 1e-7;
        double r_x  = sqrt(err_x_sq)  / fmax(z_norm,  1.0);
        double r_dx = sqrt(err_dx_sq) / fmax(dz_norm, 1.0);
        if (r_x > tol || r_dx > tol) {
            // Roll back axes/coefs to inactive defaults.
            for (int k = 0; k < 6; ++k) {
                coefs_out[k] = 0.0; signs_out[k] = 1.0;
                for (int64_t i = 0; i < dim; ++i) axes_out[k * dim + i] = 0.0;
            }
            continue;
        }

        *active_out = 1.0;
    }
}

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
    cudaStream_t stream
) {
    if (numXGenPowerCones <= 0 || batchSize <= 0) return;
    int threadsPerBlock = (numXGenPowerCones < 32) ? 32 : 256;
    if (numXGenPowerCones < threadsPerBlock) {
        threadsPerBlock = (int)((numXGenPowerCones + 31) / 32 * 32);
        if (threadsPerBlock < 32) threadsPerBlock = 32;
    }
    try_compute_pd_axes_xgenpow_kernel<<<(unsigned int)batchSize,
                                         threadsPerBlock, 0, stream>>>(
        var_x, var_z_x, xcone_grad_primal,
        xgenpow_p, xgenpow_q, xgenpow_r, xgenpow_d1, xgenpow_d2,
        mu_per_batch,
        d_xcone_kinds, d_xcone_dims, d_xcone_numel_offsets, d_xcone_indices,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_dim_offsets,
        d_xcone_genpow_alphas,
        numXCones, numXGenPowerCones, totalXConeNumel,
        totalXGenPowerDim, totalXGenPowerAlphas, totalXGenPowerDim2,
        n, batchSize,
        pd_axes, pd_coefs, pd_signs, pd_active, pd_workspace,
        pd_enabled_per_batch
    );
}

// =====================================================================
// Apply rank-6 PD axes to direct-x KKT (overwrite the inactive defaults
// that update_xcones_genpow_scaling wrote, when pd_active=1.0).
//
// Schur convention (matches direct-x rank-3): off-diag c = -sqrt(coef)·a
// and diag d = sign·1 (active) or d = sign/w (equilibrated). Together
// these contribute -c·c'/d = +sign·coef·a·a' to (1,1) block.
// =====================================================================
__global__ void apply_xgenpow_pd_to_kkt_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_genpow_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim1s,
    const int64_t* __restrict__ d_xcone_genpow_dim2s,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,
    const int64_t* __restrict__ d_xcone_genpow_sparse_idx,     // gidx → sparse cone idx (-1 if dense)
    const int64_t* __restrict__ d_xcone_genpow_sparse_offsets, // sparse-only dim prefix
    const double* __restrict__ xgenpow_pd_axes,    // (B, 6 * totalXGenPowerDim)
    const double* __restrict__ xgenpow_pd_coefs,   // (B, 6 * numXGenPowerCones)
    const double* __restrict__ xgenpow_pd_signs,   // (B, 6 * numXGenPowerCones)
    const double* __restrict__ xgenpow_pd_active,  // (B, numXGenPowerCones)
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_0,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_1,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_2,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_3,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_4,
    const int64_t* __restrict__ H_xcone_genpow_pd_axis_idx_5,
    const int64_t* __restrict__ H_xcone_genpow_exp_diag_idx,   // genpow-only portion
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowerDim,
    int64_t nnzKKT
) {
    const int64_t b = blockIdx.x;
    const int64_t* xpd_axis_idx_arr[6] = {
        H_xcone_genpow_pd_axis_idx_0, H_xcone_genpow_pd_axis_idx_1,
        H_xcone_genpow_pd_axis_idx_2, H_xcone_genpow_pd_axis_idx_3,
        H_xcone_genpow_pd_axis_idx_4, H_xcone_genpow_pd_axis_idx_5
    };
    const int64_t kkt_off = b * nnzKKT;

    for (int64_t c = threadIdx.x; c < numXCones; c += blockDim.x) {
        if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) continue;
        int64_t gidx = d_xcone_genpow_idx[c];
        int64_t sidx = d_xcone_genpow_sparse_idx[gidx];
        if (sidx < 0) continue;  // Dense path: no expansion cols.

        int64_t dim1 = d_xcone_genpow_dim1s[gidx];
        int64_t dim2 = d_xcone_genpow_dim2s[gidx];
        int64_t dim  = dim1 + dim2;
        int64_t d_off  = d_xcone_genpow_dim_offsets[gidx];
        int64_t sp_off = d_xcone_genpow_sparse_offsets[sidx];

        double active_flag = xgenpow_pd_active[b * numXGenPowerCones + gidx];
        bool active = active_flag > 0.5;

        int64_t pd_axes_cone_base = b * 6 * totalXGenPowerDim + 6 * d_off;
        int64_t pd_state_cone_base = b * 6 * numXGenPowerCones + 6 * gidx;

        // Inactive sentinel: tiny ε on basis position + sign·1 diagonal,
        // matching update_xcones_genpow_scaling_kernel's fused inactive path.
        const double XPD_INACTIVE_EPS = 1.0e-8;

        for (int axk = 0; axk < 6; ++axk) {
            double sign = xgenpow_pd_signs[pd_state_cone_base + axk];
            double coef = xgenpow_pd_coefs[pd_state_cone_base + axk];

            // Direct-x sentinel = -(slack sentinel) = -sign[k] (low-w) or
            // -sign[k]/w (high-w). Schur -(c²/d)·v·v' = +sign·coef·v·v' to
            // (1,1) requires d = -sign with c² = coef.
            if (active && coef > 0.0) {
                double n_sq = 0.0;
                for (int64_t i = 0; i < dim; ++i) {
                    double a_val = xgenpow_pd_axes[pd_axes_cone_base + axk * dim + i];
                    n_sq += a_val * a_val;
                }
                double w = coef * n_sq;
                double off_scale, sent;
                if (w > 1.0e12 && n_sq > 0.0) {
                    off_scale = -1.0 / sqrt(n_sq);
                    sent = -sign / w;
                } else {
                    off_scale = -sqrt(coef);
                    sent = -sign;
                }
                for (int64_t i = 0; i < dim; ++i) {
                    double a_val = xgenpow_pd_axes[pd_axes_cone_base + axk * dim + i];
                    kkt_values[kkt_off + xpd_axis_idx_arr[axk][sp_off + i]]
                        = off_scale * a_val;
                }
                kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 3 + axk]]
                    = sent;
            } else {
                // Inactive: tiny ε on a single basis position keeps the PD-axis
                // column structurally distinct so cuDSS doesn't EXECUTION_FAIL
                // on identical all-zero columns (mirrors slack INACTIVE_EPS).
                // Schur contribution is ε² ≈ 1e-16 — negligible numerically.
                // Sentinel = -sign[k] (direct-x inactive convention).
                for (int64_t i = 0; i < dim; ++i) {
                    double v = (i == (int64_t)axk % dim) ? XPD_INACTIVE_EPS : 0.0;
                    kkt_values[kkt_off + xpd_axis_idx_arr[axk][sp_off + i]] = v;
                }
                double sent_sign = (sign >= 0.0) ? -1.0 : 1.0;
                kkt_values[kkt_off + H_xcone_genpow_exp_diag_idx[9 * sidx + 3 + axk]]
                    = sent_sign;
            }
        }
    }
}

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
    cudaStream_t stream
) {
    if (numXGenPowerCones <= 0 || batchSize <= 0) return;
    int threadsPerBlock = (numXGenPowerCones < 32) ? 32 : 256;
    if (numXGenPowerCones < threadsPerBlock) {
        threadsPerBlock = (int)((numXGenPowerCones + 31) / 32 * 32);
        if (threadsPerBlock < 32) threadsPerBlock = 32;
    }
    apply_xgenpow_pd_to_kkt_kernel<<<(unsigned int)batchSize,
                                     threadsPerBlock, 0, stream>>>(
        kkt_values,
        d_xcone_kinds, d_xcone_dims, d_xcone_genpow_idx,
        d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_dim_offsets, d_xcone_genpow_sparse_idx,
        d_xcone_genpow_sparse_offsets,
        xgenpow_pd_axes, xgenpow_pd_coefs, xgenpow_pd_signs, xgenpow_pd_active,
        H_xcone_genpow_pd_axis_idx_0, H_xcone_genpow_pd_axis_idx_1,
        H_xcone_genpow_pd_axis_idx_2, H_xcone_genpow_pd_axis_idx_3,
        H_xcone_genpow_pd_axis_idx_4, H_xcone_genpow_pd_axis_idx_5,
        H_xcone_genpow_exp_diag_idx,
        numXCones, numXGenPowerCones, totalXGenPowerDim, nnzKKT
    );
}

// =====================================================================
// refresh_xgenpow_pd_dsigns_kernel: update the static dsigns array for
// direct-x GenPow PD-axis diagonal slots to match the runtime sign of
// `apply_xgenpow_pd_to_kkt`'s sentinel.
//
// dsigns is a single-batch int8_t array of length N (column count) that
// `backup_and_regularize_diagonal_kernel` reads to decide whether to add
// `+eps` or `-eps` to each diagonal. The PD-axis diagonal sentinel
// written by apply_xgenpow is:
//   active branch:  sent =  -sign   (low-w) or -sign/w (high-w)
//   inactive:       sent = (sign >= 0) ? -1 : +1   (= -sign sign)
// In every branch the *sign* of `sent` is `-sign of pd_signs[k]`, so
// dsigns[col_for_axis_k] should be -1 when pd_signs[k] > 0 and +1
// otherwise.
//
// pd_signs is per-batch but dsigns is single-batch — we refresh from
// batch 0 (the common path: signs are uniform across batches that
// share cone-active state). Batches that diverge fall back to cuDSS
// pivoting for the mismatch.
// =====================================================================
__global__ void refresh_xgenpow_pd_dsigns_kernel(
    int8_t* dsigns,                               // (N,)
    const double* __restrict__ pd_signs,                       // (B, 6 * numXGenPowerCones)
    const int64_t* __restrict__ d_xcone_kinds,                 // (numXCones,)
    const int64_t* __restrict__ d_xcone_genpow_idx,            // (numXCones,)
    const int64_t* __restrict__ d_xcone_genpow_sparse_idx,     // (numXGenPowerCones,)
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t pd_axis_base_col)
{
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= numXCones) return;
    if (static_cast<XConeKind>(d_xcone_kinds[c]) != XConeKind::GenPower) return;

    int64_t gidx = d_xcone_genpow_idx[c];
    int64_t sidx = d_xcone_genpow_sparse_idx[gidx];
    if (sidx < 0) return;  // dense — no expansion cols, no PD axes in KKT

    // First sparse GenPow PD-axis-0 col is at pd_axis_base_col +
    // 9*sidx + 3 (q, r, p occupy +0/+1/+2; PD axes occupy +3..+8).
    int64_t base = pd_axis_base_col + 9 * sidx + 3;
    int64_t pd_state_base = 6 * gidx;  // batch 0

    for (int axk = 0; axk < 6; ++axk) {
        double sign = pd_signs[pd_state_base + axk];
        dsigns[base + axk] = (sign > 0.0) ? (int8_t)(-1) : (int8_t)(+1);
    }
}

void refresh_xgenpow_pd_dsigns(
    int8_t* dsigns,
    const double* pd_signs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_sparse_idx,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t pd_axis_base_col,
    cudaStream_t stream)
{
    if (numXCones <= 0 || numXGenPowerCones <= 0) return;
    int threadsPerBlock = (numXCones < 32) ? 32 : 256;
    int64_t blocks = (numXCones + threadsPerBlock - 1) / threadsPerBlock;
    refresh_xgenpow_pd_dsigns_kernel<<<(unsigned int)blocks, threadsPerBlock, 0, stream>>>(
        dsigns, pd_signs,
        d_xcone_kinds, d_xcone_genpow_idx, d_xcone_genpow_sparse_idx,
        numXCones, numXGenPowerCones, pd_axis_base_col);
}

// =====================================================================
// refresh_genpow_pd_dsigns_kernel: slack GenPow analogue. The slack
// sentinel written by the slack GenPow KKT scaling kernel is +sign[k]
// (active low-w), +sign[k]/w (high-w), or (sign>=0 ? +1 : -1) (inactive)
// — in every branch the sentinel's sign tracks `+sign[k]`. dsigns must
// match the sentinel sign, so we write +1 when pd_signs[k] > 0 and -1
// otherwise (exactly opposite of the direct-x mapping).
// =====================================================================
__global__ void refresh_genpow_pd_dsigns_kernel(
    int8_t* dsigns,                                // (N,)
    const double* __restrict__ pd_signs,                        // (B, 6 * numGenPowerCones)
    const int64_t* __restrict__ d_genPowerSparseIndices,        // (numGenPowerCones,)
    int64_t numGenPowerCones,
    int64_t pd_axis_base_col)
{
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= numGenPowerCones) return;

    int64_t sidx = d_genPowerSparseIndices[c];
    if (sidx < 0) return;  // dense — no expansion cols, no PD axes in KKT

    int64_t base = pd_axis_base_col + 9 * sidx + 3;
    int64_t pd_state_base = 6 * c;  // batch 0

    for (int axk = 0; axk < 6; ++axk) {
        double sign = pd_signs[pd_state_base + axk];
        dsigns[base + axk] = (sign > 0.0) ? (int8_t)(+1) : (int8_t)(-1);
    }
}

void refresh_genpow_pd_dsigns(
    int8_t* dsigns,
    const double* pd_signs,
    const int64_t* d_genPowerSparseIndices,
    int64_t numGenPowerCones,
    int64_t pd_axis_base_col,
    cudaStream_t stream)
{
    if (numGenPowerCones <= 0) return;
    int threadsPerBlock = (numGenPowerCones < 32) ? 32 : 256;
    int64_t blocks = (numGenPowerCones + threadsPerBlock - 1) / threadsPerBlock;
    refresh_genpow_pd_dsigns_kernel<<<(unsigned int)blocks, threadsPerBlock, 0, stream>>>(
        dsigns, pd_signs,
        d_genPowerSparseIndices, numGenPowerCones, pd_axis_base_col);
}

}  // namespace moreau
