/**
 * @file common.cuh
 * @brief Common device functions and constants for cone operations
 *
 * Shared utilities used across all cone type implementations.
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace moreau {
namespace cones {

// Machine epsilon constants for device code
constexpr double DEVICE_EPSILON = 2.22044604925031308085e-16;
constexpr double DEVICE_SQRT_EPSILON = 1.49011611938476562500e-08;
constexpr double DEVICE_SQRT2 = 1.4142135623730951;

/**
 * @brief Safe logarithm that returns -infinity for x <= 0
 *
 * Matches CPU implementation in scalarmath.rs for numerical consistency.
 */
__device__ __forceinline__ double logsafe(double x) {
    return (x > 0.0) ? log(x) : -INFINITY;
}

/**
 * @brief Compute dot product of two 3D vectors
 */
__device__ __forceinline__ double dot3(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * @brief Matrix-vector multiply for symmetric 3x3 (packed upper triangle)
 *
 * H is stored as [H00, H01, H02, H11, H12, H22]
 */
__device__ __forceinline__ void matvec3_sym(double* y, const double* H, const double* x) {
    y[0] = H[0]*x[0] + H[1]*x[1] + H[2]*x[2];
    y[1] = H[1]*x[0] + H[3]*x[1] + H[4]*x[2];
    y[2] = H[2]*x[0] + H[4]*x[1] + H[5]*x[2];
}

/**
 * @brief Compute quadratic form x'*H*x for symmetric 3x3
 */
__device__ __forceinline__ double quad_form3(const double* H, const double* x) {
    double Hx[3];
    matvec3_sym(Hx, H, x);
    return dot3(x, Hx);
}

/**
 * @brief Frobenius norm of symmetric 3x3 matrix (packed upper triangle)
 */
__device__ __forceinline__ double norm_fro3(const double* H) {
    return sqrt(H[0]*H[0] + H[3]*H[3] + H[5]*H[5] +
                2.0*(H[1]*H[1] + H[2]*H[2] + H[4]*H[4]));
}

/**
 * @brief Normalize a 3D vector in place
 *
 * Uses DEVICE_EPSILON for consistency with other tolerance checks.
 */
__device__ __forceinline__ void normalize3(double* v) {
    double norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm > DEVICE_EPSILON) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

/**
 * @brief Compute symmetric matrix index from row/col
 * Returns packed upper triangle index for (i, j) where i <= j
 */
__device__ __forceinline__ int sym_idx(int i, int j) {
    if (i == 0) return j;
    if (i == 1 && j == 1) return 3;
    if (i == 1 && j == 2) return 4;
    return 5;  // i == 2 && j == 2
}

/**
 * @brief Block-wide sum reduction with broadcast.
 *
 * Each thread contributes `my_val`; all threads receive the sum of
 * `my_val` across the block. Uses `shared` as scratch (size blockDim.x
 * doubles). Safe to call repeatedly with the same scratch buffer — the
 * trailing __syncthreads ensures all threads finish reading shared[0]
 * before the caller reuses the buffer.
 */
__device__ __forceinline__ double block_sum_reduce(
    double my_val,
    double* shared,
    int tid
) {
    // Warp-level reduction first via __shfl_down_sync — no shared mem
    // and no __syncthreads. Reduces each warp's 32 values to lane 0.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_val += __shfl_down_sync(0xffffffff, my_val, off);
    }
    const int lane = tid & 31;
    const int warp = tid >> 5;
    // Each warp's lane-0 stores its partial into the first num_warps
    // slots of `shared`. Caller-supplied `shared` is sized blockDim.x;
    // we only use blockDim.x/32 slots here.
    if (lane == 0) shared[warp] = my_val;
    __syncthreads();
    // Warp 0 reduces the per-warp partials. Inactive lanes contribute 0.
    if (warp == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        my_val = (lane < num_warps) ? shared[lane] : 0.0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            my_val += __shfl_down_sync(0xffffffff, my_val, off);
        }
        if (lane == 0) shared[0] = my_val;
    }
    __syncthreads();
    const double result = shared[0];
    __syncthreads();
    return result;
}

/**
 * @brief Block-wide N-value sum reduction with broadcast.
 *
 * Same shape as `block_sum_reduce` but fuses N reductions into a single
 * shared-memory round-trip — saves N-1 `__syncthreads` calls relative
 * to N sequential calls. The two warp-shuffle stages reduce all N
 * values in parallel (one inner shfl loop per value, all sharing the
 * same warp-comm cycles). Caller-supplied `shared` must hold at least
 * `(num_warps + 1) * N` doubles where `num_warps = (blockDim.x + 31) / 32`.
 * For blockDim=256, N=6, that's 54 doubles (432 B) — well under the
 * existing per-block 2 KB scratch allocation.
 */
template <int N>
__device__ __forceinline__ void block_sum_reduce_N(
    double (&vals)[N],
    double* shared,
    int tid
) {
    // Stage 1: warp-level reduce each of the N values via shfl_down.
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vals[k] += __shfl_down_sync(0xffffffff, vals[k], off);
        }
    }
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            shared[warp * N + k] = vals[k];
        }
    }
    __syncthreads();
    // Stage 2: warp 0 reduces the per-warp partials.
    if (warp == 0) {
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            vals[k] = (lane < num_warps) ? shared[lane * N + k] : 0.0;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vals[k] += __shfl_down_sync(0xffffffff, vals[k], off);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                shared[num_warps * N + k] = vals[k];  // broadcast slot
            }
        }
    }
    __syncthreads();
    // Broadcast result back to every thread.
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        vals[k] = shared[num_warps * N + k];
    }
    __syncthreads();
}

/**
 * @brief Atomic min on a strictly-positive double.
 *
 * Uses `atomicCAS` on the underlying uint64 view. Safe only when both
 * the stored value and `val` are >= 0 — positive double bit patterns are
 * monotone in floating value, so the int compare implements the double
 * compare. Used by the per-batch step-length reductions where multiple
 * cones race to lower `alpha_{s,z}[batch]`.
 */
__device__ __forceinline__ void atomic_min_pos_double(double* addr, double val) {
    unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *a, assumed;
    do {
        double cur = __longlong_as_double(static_cast<long long>(old));
        if (cur <= val) break;
        assumed = old;
        old = atomicCAS(a, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(val)));
    } while (assumed != old);
}

/**
 * @brief Block-wide min reduction with broadcast.
 *
 * Sibling of `block_sum_reduce` — same contract, returns the minimum
 * across the block instead of the sum.
 */
__device__ __forceinline__ double block_min_reduce(
    double my_val,
    double* shared,
    int tid
) {
    // Warp-shuffle fast path; mirrors block_sum_reduce above.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double other = __shfl_down_sync(0xffffffff, my_val, off);
        if (other < my_val) my_val = other;
    }
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) shared[warp] = my_val;
    __syncthreads();
    if (warp == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        my_val = (lane < num_warps) ? shared[lane] : INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const double other = __shfl_down_sync(0xffffffff, my_val, off);
            if (other < my_val) my_val = other;
        }
        if (lane == 0) shared[0] = my_val;
    }
    __syncthreads();
    const double result = shared[0];
    __syncthreads();
    return result;
}

// ============================================================================
// Cone-specific helper functions used by multiple kernels
// These must be in the header to be visible across translation units. They
// live in `namespace cones` alongside `logsafe` / `block_sum_reduce` etc.;
// callers either qualify (`cones::wright_omega`) or `using namespace cones;`.
// ============================================================================

/**
 * @brief Wright omega function - used for computing primal gradient
 *
 * Guard against invalid input - return a safe default instead of producing NaN.
 * This can happen during IPM iterations when numerical instability pushes
 * variables slightly outside the cone. Returning 1.0 (omega(1) = 1) provides
 * a reasonable approximation that allows the solver to continue.
 */
__device__ __forceinline__ double wright_omega(double z) {
    if (z < 0.0) return 1.0;  // Match CPU: return 1.0 for negative inputs

    double p, w;
    const double PI = 3.14159265358979323846;

    if (z < 1.0 + PI) {
        // Initialize with Taylor series
        double zm1 = z - 1.0;
        p = zm1;
        w = 1.0 + p * 0.5;
        p *= zm1;
        w += p * (1.0 / 16.0);
        p *= zm1;
        w -= p * (1.0 / 192.0);
        p *= zm1;
        w -= p * (1.0 / 3072.0);
        p *= zm1;
        w += p * (13.0 / 61440.0);
    } else {
        // Initialize with asymptotic expansion
        double logz = log(z);
        double zinv = 1.0 / z;
        w = z - logz;

        double q = logz * zinv;
        w += q;

        q *= zinv;
        w += q * (logz / 2.0 - 1.0);

        q *= zinv;
        w += q * (logz * logz / 3.0 - logz * 1.5 + 1.0);
    }

    // Refinement iterations
    double r = z - w - logsafe(w);

    for (int iter = 0; iter < 2; iter++) {
        double wp1 = w + 1.0;
        double t = wp1 * (wp1 + (r * 2.0) / 3.0);
        w *= 1.0 + (r / wp1) * (t - r * 0.5) / (t - r);

        double r_4th = r * r * r * r;
        double wp1_6th = wp1 * wp1 * wp1 * wp1 * wp1 * wp1;
        r = (w * w * 2.0 - w * 8.0 - 1.0) / (wp1_6th * 72.0) * r_4th;
    }

    return w;
}

/**
 * @brief Compute primal gradient for exponential cone
 */
__device__ __forceinline__ void gradient_primal_exp(double* g, const double* s) {
    double arg = 1.0 - s[0] / s[1] - logsafe(s[1] / s[2]);
    double omega = wright_omega(arg);

    g[0] = 1.0 / ((omega - 1.0) * s[1]);
    g[1] = g[0] + g[0] * logsafe(omega * s[1] / s[2]) - 1.0 / s[1];
    g[2] = omega / ((1.0 - omega) * s[2]);
}

/**
 * @brief Newton-Raphson solver for power cone primal gradient
 *
 * Solves for x in the nonlinear equation arising from the power cone barrier.
 * Matches the upstream Clarabel newton_raphson_onesided implementation:
 * starts from a point left of the root and stops when step becomes negative.
 */
__device__ __forceinline__ double newton_raphson_powcone(double s3, double phi, double alpha) {
    const double two = 2.0;
    const double three = 3.0;

    // Initial point x0
    double x0 = -1.0/s3 + (s3 * two + sqrt((phi * phi) / (s3 * s3) + phi * three)) / (phi - s3 * s3);

    // Additional shift due to dual barrier choice
    double t0 = -two * alpha * logsafe(alpha) - two * (1.0 - alpha) * logsafe(1.0 - alpha);

    // Newton-Raphson iterations (one-sided: stop when step becomes negative)
    double x = x0;

    for (int iter = 0; iter < 100; iter++) {
        double t1 = x * x;
        double t2 = (two * x) / s3;

        double f_val = two * alpha * logsafe(two * alpha * t1 + (1.0 + alpha) * t2)
                     + two * (1.0 - alpha) * logsafe(two * (1.0 - alpha) * t1 + (two - alpha) * t2)
                     - logsafe(phi)
                     - logsafe(t1 + t2)
                     - two * logsafe(t2)
                     + t0;

        double f_deriv = (alpha * alpha * two) / (alpha * x + (1.0 + alpha) / s3)
                       + ((1.0 - alpha) * two) * (1.0 - alpha) / ((1.0 - alpha) * x + (two - alpha) / s3)
                       - ((x + 1.0/s3) * two) / (t1 + t2);

        double dx = -f_val / f_deriv;

        if ((dx < DEVICE_EPSILON) ||
            (fabs(dx / x) < DEVICE_SQRT_EPSILON) ||
            (fabs(f_deriv) < DEVICE_EPSILON)) {
            break;
        }

        x += dx;
    }

    return x;
}

/**
 * @brief Compute primal gradient for power cone
 */
__device__ __forceinline__ void gradient_primal_power(double* g, const double* s, double alpha) {
    const double two = 2.0;

    double phi = pow(s[0], two * alpha) * pow(s[1], two * (1.0 - alpha));

    double abs_s = fabs(s[2]);
    if (abs_s > DEVICE_EPSILON) {
        g[2] = newton_raphson_powcone(abs_s, phi, alpha);
        if (s[2] < 0.0) {
            g[2] = -g[2];
        }
        g[0] = -(alpha * g[2] * s[2] + 1.0 + alpha) / s[0];
        g[1] = -((1.0 - alpha) * g[2] * s[2] + two - alpha) / s[1];
    } else {
        g[2] = 0.0;
        g[0] = -(1.0 + alpha) / s[0];
        g[1] = -(two - alpha) / s[1];
    }
}

} // namespace cones

} // namespace moreau
