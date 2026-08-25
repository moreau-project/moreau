/**
 * @file genpow_pd_kernels.cuh
 * @brief Device functions for the rank-6 sparse Mosek-Tunçel primal-dual
 * scaling on GenPowerCone.
 *
 * Mirrors `nonsymmetric_common::pd_scaling_nd_qr6` (CPU) — see that
 * function for the math derivation. Single-thread per (batch, cone)
 * implementation. Workspace arrays are passed in pre-allocated.
 */

#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace moreau {

// 1D Newton-Raphson root-finder used by gradient_primal_genpow.
// Solves f(x) = 0 starting from x0 < x* with f(x0) > 0; converges
// quadratically. Returns x*.
__device__ __forceinline__ double newton_raphson_genpow_1d(
    double x0,
    double norm_r,
    const double* p,
    double phi,
    const double* alphas,
    int64_t dim1,
    double psi
) {
    double x = x0;
    const double TOL = 1e-12;
    const int MAX_ITER = 50;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // f(x) = -log(2x/norm_r + x²) + Σ 2α_i (log(x·norm_r + (1+α_i)/α_i) - log p_i)
        double inner = 2.0 * x / norm_r + x * x;
        if (inner <= 0.0) {
            // Past the root region; back off.
            x = 0.5 * x;
            continue;
        }
        double f = -log(inner);
        double f_prime = -(2.0 * x + 2.0 / norm_r) / (x * x + 2.0 * x / norm_r);
        for (int64_t i = 0; i < dim1; ++i) {
            double ai = alphas[i];
            double inner2 = x * norm_r + (1.0 + ai) / ai;
            f += 2.0 * ai * (log(inner2) - log(p[i]));
            f_prime += 2.0 * ai * norm_r / (norm_r * x + (1.0 + ai) / ai);
        }
        if (fabs(f) < TOL) break;
        if (fabs(f_prime) < 1e-300) break;
        double dx = f / f_prime;
        x = x - dx;
        if (fabs(dx) < TOL * (1.0 + fabs(x))) break;
    }
    return x;
}

// gradient_primal for GenPowerCone: g = ∇F(s) where F is the primal
// barrier. Closed form via 1D Newton-Raphson on a barrier-derivative
// equation. Mirrors CPU `gradient_primal` in genpowcone.rs.
//
//   g[i<dim1] = -(1 + α + α·g1·||r||) / p[i]
//   g[i≥dim1] = (g1/||r||) · r[i]
//
// where g1 = newton_raphson_genpow_1d(...) and (p, r) = (s[..dim1], s[dim1..]).
// When ||r|| ≈ 0: g1 → 0 falls back to g[i<dim1] = -(1+α)/p[i], gr = 0.
__device__ __forceinline__ void gradient_primal_genpow(
    double* g,                  // out, length dim1+dim2
    const double* s,            // in, length dim1+dim2
    const double* alphas,       // length dim1
    double psi_init,            // 1/Σα_i² (initial-point constant from cone)
    int64_t dim1,
    int64_t dim2
) {
    const double EPS = 1e-300;
    // log_phi = Σ 2α_i log(s_i)
    double log_phi = 0.0;
    for (int64_t i = 0; i < dim1; ++i) {
        log_phi += 2.0 * alphas[i] * log(s[i]);
    }
    double phi = exp(log_phi);

    double norm_r2 = 0.0;
    for (int64_t i = 0; i < dim2; ++i) {
        norm_r2 += s[dim1 + i] * s[dim1 + i];
    }
    double norm_r = sqrt(norm_r2);

    if (norm_r > 1e-15) {
        // Initial point: x0 such that f(x0) > 0.
        double phi_minus_r2 = phi - norm_r * norm_r;
        double inner = (phi / (norm_r * norm_r) + psi_init * psi_init - 1.0) * phi;
        if (inner < 0.0) inner = 0.0;
        double x0 = -1.0 / norm_r
                  + (psi_init * norm_r + sqrt(inner)) / fmax(phi_minus_r2, EPS);
        double g1 = newton_raphson_genpow_1d(x0, norm_r, s, phi, alphas, dim1, psi_init);

        for (int64_t i = 0; i < dim2; ++i) {
            g[dim1 + i] = (g1 / norm_r) * s[dim1 + i];
        }
        for (int64_t i = 0; i < dim1; ++i) {
            double ai = alphas[i];
            g[i] = -(1.0 + ai + ai * g1 * norm_r) / s[i];
        }
    } else {
        for (int64_t i = 0; i < dim2; ++i) {
            g[dim1 + i] = 0.0;
        }
        for (int64_t i = 0; i < dim1; ++i) {
            double ai = alphas[i];
            g[i] = -(1.0 + ai) / s[i];
        }
    }
}

// Apply rank-3 H_dual to vector x:  out = μ·H_dual·x
//   H_dual = D + p p' − q q' − r r',  D = diag(d1, d2 I_{dim2})
__device__ __forceinline__ void mul_mu_h_rank3_genpow(
    double* out,                // out, length dim
    const double* x,            // in, length dim
    const double* p,            // length dim
    const double* q,            // length dim1
    const double* r,            // length dim2
    const double* d1,           // length dim1
    double d2,
    double mu,
    int64_t dim1,
    int64_t dim2
) {
    int64_t dim = dim1 + dim2;
    // Rank-3 dot products
    double dot_p = 0.0, dot_q = 0.0, dot_r = 0.0;
    for (int64_t i = 0; i < dim; ++i) dot_p += p[i] * x[i];
    for (int64_t i = 0; i < dim1; ++i) dot_q += q[i] * x[i];
    for (int64_t i = 0; i < dim2; ++i) dot_r += r[i] * x[dim1 + i];
    // Diagonal + rank-3 contributions, scaled by μ
    for (int64_t i = 0; i < dim1; ++i) {
        out[i] = mu * (d1[i] * x[i] + dot_p * p[i] - dot_q * q[i]);
    }
    for (int64_t i = 0; i < dim2; ++i) {
        out[dim1 + i] = mu * (d2 * x[dim1 + i] + dot_p * p[dim1 + i] - dot_r * r[i]);
    }
}

// gradient of the *dual* barrier F* at z (closed form, no Newton).
//   F*(z) = -log ψ_d(z) - Σ (1-α_i) log z_i,
//   ψ_d(z) = ∏ (z_i/α_i)^(2α_i) - ‖w‖²
// For i < dim1: g[i] = -τ_d_i·phi_d/ζ_d - (1-α_i)/z_i,  τ_d_i = 2α_i/z_i
// For i ≥ dim1: g[i] = (2/ζ_d)·z_i
// Returns ζ_d (≤0 means dual-infeasible — caller should bail).
__device__ __forceinline__ double gradient_dual_genpow(
    double* g,                  // out, length dim1+dim2
    const double* z,            // in, length dim1+dim2
    const double* alphas,       // length dim1
    int64_t dim1,
    int64_t dim2
) {
    double log_phi = 0.0;
    for (int64_t i = 0; i < dim1; ++i) {
        log_phi += 2.0 * alphas[i] * log(z[i] / alphas[i]);
    }
    double phi = exp(log_phi);
    double norm2w = 0.0;
    for (int64_t j = 0; j < dim2; ++j) {
        norm2w += z[dim1 + j] * z[dim1 + j];
    }
    double zeta = phi - norm2w;
    for (int64_t i = 0; i < dim1; ++i) {
        double ai = alphas[i];
        double zi = z[i];
        double tau = 2.0 * ai / zi;
        g[i] = -tau * phi / zeta - (1.0 - ai) / zi;
    }
    for (int64_t j = 0; j < dim2; ++j) {
        g[dim1 + j] = (2.0 / zeta) * z[dim1 + j];
    }
    return zeta;
}

// In-place Jacobi eigendecomposition of a 4×4 symmetric matrix. On
// exit, `m` is diagonal and `v` is the orthogonal eigenvector matrix.
// `v` must be initialised to identity by caller.
__device__ __forceinline__ void jacobi_4x4(double m[4][4], double v[4][4]) {
    const double TOL = 1e-15;
    const int MAX_SWEEPS = 30;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double max_off = 0.0;
        for (int p = 0; p < 4; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                double a = fabs(m[p][q]);
                if (a > max_off) max_off = a;
            }
        }
        if (max_off < TOL) break;
        for (int p = 0; p < 4; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                double mpq = m[p][q];
                if (fabs(mpq) < TOL) continue;
                double mpp = m[p][p];
                double mqq = m[q][q];
                double theta = (mqq - mpp) / (2.0 * mpq);
                double t;
                if (theta >= 0.0) {
                    t = 1.0 / (theta + sqrt(1.0 + theta * theta));
                } else {
                    t = 1.0 / (theta - sqrt(1.0 + theta * theta));
                }
                double c = 1.0 / sqrt(1.0 + t * t);
                double sn = t * c;
                m[p][p] = mpp - t * mpq;
                m[q][q] = mqq + t * mpq;
                m[p][q] = 0.0;
                m[q][p] = 0.0;
                for (int i = 0; i < 4; ++i) {
                    if (i != p && i != q) {
                        double mip = m[i][p];
                        double miq = m[i][q];
                        m[i][p] = c * mip - sn * miq;
                        m[p][i] = m[i][p];
                        m[i][q] = sn * mip + c * miq;
                        m[q][i] = m[i][q];
                    }
                }
                for (int i = 0; i < 4; ++i) {
                    double vip = v[i][p];
                    double viq = v[i][q];
                    v[i][p] = c * vip - sn * viq;
                    v[i][q] = sn * vip + c * viq;
                }
            }
        }
    }
}

} // namespace moreau
