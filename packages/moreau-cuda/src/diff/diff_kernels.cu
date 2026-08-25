/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for differentiation
 */

#include "moreau/diff/diff_kernels.cuh"
#include "moreau/cones/common.cuh"
#include "moreau/cones/cones.hpp"
#include "moreau/profiling/profiler.hpp"
#include <cmath>
#include <cfloat>
#include <vector>

namespace moreau {

// ============================================================================
// Helper device functions
// ============================================================================

__device__ inline double safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

__device__ inline double safe_div(double num, double den, double fallback = 0.0) {
    return (fabs(den) > DBL_EPSILON) ? (num / den) : fallback;
}

// ============================================================================
// Cone Projection Kernels
// ============================================================================

__global__ void project_zero_cone_dual_kernel(
    double* __restrict__ pi_u,
    const double* __restrict__ u,
    int64_t numZeroCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // Zero cone dual is the whole space, so projection is identity
    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        pi_u[batch * m + i] = u[batch * m + i];
    }
}

void project_zero_cone_dual(
    double* pi_u,
    const double* u,
    int64_t numZeroCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numZeroCones == 0) return;
    int threads = min((int)numZeroCones, 256);
    MOREAU_KERNEL_LAUNCH(project_zero_cone_dual_kernel, batchSize, threads, 0, stream,
        pi_u, u, numZeroCones, batchSize, m
    );
}

__global__ void project_nonneg_cone_dual_kernel(
    double* __restrict__ pi_u,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // Dual of nonnegative cone is nonnegative cone: max(u, 0)
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t idx = batch * m + offset + i;
        pi_u[idx] = fmax(u[idx], 0.0);
    }
}

void project_nonneg_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numNonnegCones == 0) return;
    int threads = min((int)numNonnegCones, 256);
    MOREAU_KERNEL_LAUNCH(project_nonneg_cone_dual_kernel, batchSize, threads, 0, stream,
        pi_u, u, offset, numNonnegCones, batchSize, m
    );
}

__global__ void project_soc_cone_dual_kernel(
    double* __restrict__ pi_u,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_sz_offsets,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // Each thread handles one SOC cone (variable dimension)
    for (int64_t cone = threadIdx.x; cone < numSocCones; cone += blockDim.x) {
        int64_t dim = d_soc_dims[cone];
        int64_t sz_off = d_soc_sz_offsets[cone];
        int64_t base = batch * m + offset + sz_off;

        double t = u[base + 0];

        // Compute ||x[1:]||
        double norm_x_sq = 0.0;
        for (int64_t i = 1; i < dim; ++i) {
            norm_x_sq += u[base + i] * u[base + i];
        }
        double norm_x = safe_sqrt(norm_x_sq);

        // SOC dual is SOC: project (t, x) onto {(t, x) : t >= ||x||}
        if (t >= norm_x) {
            // Already in cone - copy all elements
            for (int64_t i = 0; i < dim; ++i) {
                pi_u[base + i] = u[base + i];
            }
        } else if (t <= -norm_x) {
            // In negative cone, project to zero
            for (int64_t i = 0; i < dim; ++i) {
                pi_u[base + i] = 0.0;
            }
        } else {
            // On boundary: project to (t + ||x||)/2 * (1, x/||x||)
            double scale = 0.5 * (t + norm_x);
            double inv_norm = safe_div(1.0, norm_x, 0.0);
            pi_u[base + 0] = scale;
            for (int64_t i = 1; i < dim; ++i) {
                pi_u[base + i] = scale * u[base + i] * inv_norm;
            }
        }
    }
}

void project_soc_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    int64_t totalSocDim,
    int64_t batchSize,
    int64_t m,
    const int64_t* d_soc_sz_offsets,
    cudaStream_t stream
) {
    if (numSocCones == 0) return;
    int threads = min((int)numSocCones, 256);
    // Use d_soc_sz_offsets if provided (sorted cones), otherwise fall back to d_soc_offsets
    const int64_t* offsets_for_sz = d_soc_sz_offsets ? d_soc_sz_offsets : d_soc_offsets;
    MOREAU_KERNEL_LAUNCH(project_soc_cone_dual_kernel, batchSize, threads, 0, stream,
        pi_u, u, offset, numSocCones, d_soc_dims, offsets_for_sz, batchSize, m
    );
}


// Forward declaration of primal projection (defined later in the file)
__device__ void project_exp_cone_primal(const double* v, double* out);

// Exponential cone dual projection - uses Moreau decomposition
// Π_{K*}(z) = z - Π_K(z) where K is the primal cone
__global__ void project_exp_cone_dual_kernel(
    double* __restrict__ pi_u,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numExpCones; cone += blockDim.x) {
        int64_t base = batch * m + offset + cone * 3;

        // Get input point
        double v[3] = {u[base + 0], u[base + 1], u[base + 2]};

        // Dual cone projection via Moreau: Π_{K*}(z) = z + Π_K(-z)
        // Note: K* (dual) ≠ K° (polar). The polar formula would be z - Π_K(z).
        double neg_v[3] = {-v[0], -v[1], -v[2]};
        double primal_proj[3];
        project_exp_cone_primal(neg_v, primal_proj);

        pi_u[base + 0] = v[0] + primal_proj[0];
        pi_u[base + 1] = v[1] + primal_proj[1];
        pi_u[base + 2] = v[2] + primal_proj[2];
    }
}

void project_exp_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numExpCones == 0) return;
    int threads = min((int)numExpCones, 256);
    MOREAU_KERNEL_LAUNCH(project_exp_cone_dual_kernel, batchSize, threads, 0, stream,
        pi_u, u, offset, numExpCones, batchSize, m
    );
}

// Forward declaration of power cone primal projection (defined later with derivative code)
__device__ void project_pow_cone_primal(double x, double y, double z, double alpha, double* out);

// Power cone dual projection - uses Moreau decomposition
// Π_{K*}(z) = z + Π_K(-z) where K is the primal power cone
// Note: K* (dual) ≠ K° (polar). The polar formula would be z - Π_K(z).
__global__ void project_power_cone_dual_kernel(
    double* __restrict__ pi_u,
    const double* __restrict__ u,
    const double* __restrict__ alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numPowerCones; cone += blockDim.x) {
        int64_t base = batch * m + offset + cone * 3;
        double x = u[base + 0];
        double y = u[base + 1];
        double z = u[base + 2];
        double alpha = alphas[cone];

        // Dual cone projection via Moreau: Π_{K*}(z) = z + Π_K(-z)
        double primal_proj[3];
        project_pow_cone_primal(-x, -y, -z, alpha, primal_proj);

        pi_u[base + 0] = x + primal_proj[0];
        pi_u[base + 1] = y + primal_proj[1];
        pi_u[base + 2] = z + primal_proj[2];
    }
}

void project_power_cone_dual(
    double* pi_u,
    const double* u,
    const double* alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numPowerCones == 0) return;
    int threads = min((int)numPowerCones, 256);
    MOREAU_KERNEL_LAUNCH(project_power_cone_dual_kernel, batchSize, threads, 0, stream,
        pi_u, u, alphas, offset, numPowerCones, batchSize, m
    );
}

// ============================================================================
// Cone Derivative Kernels
// ============================================================================

__global__ void compute_nonneg_derivative_kernel(
    double* __restrict__ H_diag,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // DΠ_K*(u) = diag(u >= 0) for nonnegative cone
    // Note: At u=0, the subgradient is [0,1]. We use 1 for consistency with CPU.
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t u_idx = batch * m + offset + i;
        int64_t h_idx = batch * numNonnegCones + i;
        H_diag[h_idx] = (u[u_idx] >= 0.0) ? 1.0 : 0.0;
    }
}

void compute_nonneg_derivative(
    double* H_diag,
    const double* u,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numNonnegCones == 0) return;
    int threads = min((int)numNonnegCones, 256);
    MOREAU_KERNEL_LAUNCH(compute_nonneg_derivative_kernel, batchSize, threads, 0, stream,
        H_diag, u, offset, numNonnegCones, batchSize, m
    );
}

// Kernel for dense SOC cones (dim <= 4): stores full upper triangle
__global__ void compute_soc_derivative_dense_kernel(
    double* __restrict__ H,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_Hs_offsets,   // dense-only prefix sum
    const int64_t* __restrict__ d_soc_sparse_indices,
    int64_t totalDenseSocHsEntries,
    int64_t batchSize,
    int64_t m,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // For variable-dim SOC: compute DΠ_K*(u)
    // Stored as row-major upper triangle: dim*(dim+1)/2 entries per cone
    for (int64_t cone = threadIdx.x; cone < numSocCones; cone += blockDim.x) {
        // Skip sparse cones (when d_soc_sparse_indices is provided)
        if (d_soc_sparse_indices != nullptr && d_soc_sparse_indices[cone] >= 0) continue;

        int64_t dim = d_soc_dims[cone];
        int64_t sz_off = d_soc_sz_offsets[cone];
        int64_t hs_off = d_soc_Hs_offsets[cone];
        int64_t u_base = batch * m + offset + sz_off;
        int64_t h_base = batch * totalDenseSocHsEntries + hs_off;
        int64_t hs_count = dim * (dim + 1) / 2;

        double t = u[u_base + 0];

        // Compute ||x[1:]||
        double norm_x_sq = 0.0;
        for (int64_t i = 1; i < dim; ++i) {
            norm_x_sq += u[u_base + i] * u[u_base + i];
        }
        double norm_x = safe_sqrt(norm_x_sq);

        if (t >= norm_x) {
            // Interior: DΠ = I (identity upper triangle)
            int64_t idx = 0;
            for (int64_t r = 0; r < dim; ++r) {
                for (int64_t c = r; c < dim; ++c) {
                    H[h_base + idx] = (r == c) ? 1.0 : 0.0;
                    ++idx;
                }
            }
        } else if (t <= -norm_x) {
            // Negative cone: DΠ = 0
            for (int64_t i = 0; i < hs_count; ++i) {
                H[h_base + i] = 0.0;
            }
        } else {
            // Boundary case:
            // DΠ = 0.5 * [1, x'/||x||; x/||x||, I + (t/||x||)(I - xx'/||x||^2)]
            double inv_norm = safe_div(1.0, norm_x, 0.0);
            double t_over_norm = t * inv_norm;

            int64_t idx = 0;
            for (int64_t r = 0; r < dim; ++r) {
                for (int64_t c = r; c < dim; ++c) {
                    double val;
                    if (r == 0 && c == 0) {
                        val = 0.5;
                    } else if (r == 0) {
                        val = 0.5 * u[u_base + c] * inv_norm;
                    } else {
                        double x_r = u[u_base + r] * inv_norm;
                        double x_c = u[u_base + c] * inv_norm;
                        double delta = (r == c) ? 1.0 : 0.0;
                        val = 0.5 * (delta + t_over_norm * (delta - x_r * x_c));
                    }
                    H[h_base + idx] = val;
                    ++idx;
                }
            }
        }
    }
}

// Kernel for sparse SOC cones (dim > 4): stores diagonal + rank-2 decomposition
// H = diag(d) + c1 * v1 * v1^T + c2 * v2 * v2^T
// Boundary: d=(0,α,...,α), v1=(1,û), c1=0.5, v2=(0,û), c2=-α
// Small-sparse-cone path. Processes cones in [0, numSmallSoc); dense cones
// (sparse_idx < 0) within this range are skipped. Large cones (dim >
// SOC_PARALLEL_THRESHOLD) are handled by compute_soc_derivative_sparse_large_kernel.
__global__ void compute_soc_derivative_sparse_kernel(
    double* __restrict__ sparse_diag,     // [batch][totalSparseSocDim]
    double* __restrict__ sparse_v1,       // [batch][totalSparseSocDim]
    double* __restrict__ sparse_v2,       // [batch][totalSparseSocDim]
    double* __restrict__ sparse_c1,       // [batch][numSparseSoc]
    double* __restrict__ sparse_c2,       // [batch][numSparseSoc]
    const double* __restrict__ u,
    int64_t offset,
    int64_t numSmallSoc,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_sparse_indices,
    const int64_t* __restrict__ d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    int64_t batchSize,
    int64_t m,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numSmallSoc; cone += blockDim.x) {
        int64_t sparse_idx = d_soc_sparse_indices[cone];
        if (sparse_idx < 0) continue;  // Skip dense cones

        int64_t dim = d_soc_dims[cone];
        int64_t sz_off = d_soc_sz_offsets[cone];
        int64_t sparse_off = d_soc_sparse_offsets[cone];
        int64_t u_base = batch * m + offset + sz_off;
        int64_t s_base = batch * totalSparseSocDim + sparse_off;

        double t = u[u_base + 0];

        // Compute ||x[1:]||
        double norm_x_sq = 0.0;
        for (int64_t i = 1; i < dim; ++i) {
            norm_x_sq += u[u_base + i] * u[u_base + i];
        }
        double norm_x = safe_sqrt(norm_x_sq);

        if (t >= norm_x) {
            // Interior: H = I → diag = 1, c1 = c2 = 0 (consistent sparsity)
            for (int64_t i = 0; i < dim; ++i) {
                sparse_diag[s_base + i] = 1.0;
                sparse_v1[s_base + i] = 0.0;
                sparse_v2[s_base + i] = 0.0;
            }
            sparse_c1[batch * numSparseSoc + sparse_idx] = 0.0;
            sparse_c2[batch * numSparseSoc + sparse_idx] = 0.0;
        } else if (t <= -norm_x) {
            // Polar: H = 0 → all zeros (consistent sparsity)
            for (int64_t i = 0; i < dim; ++i) {
                sparse_diag[s_base + i] = 0.0;
                sparse_v1[s_base + i] = 0.0;
                sparse_v2[s_base + i] = 0.0;
            }
            sparse_c1[batch * numSparseSoc + sparse_idx] = 0.0;
            sparse_c2[batch * numSparseSoc + sparse_idx] = 0.0;
        } else {
            // Boundary: H = diag(d) + c1*v1*v1^T + c2*v2*v2^T
            // d = (0, α, ..., α), α = (t + ||x||) / (2*||x||)
            // v1 = (1, û), c1 = 0.5
            // v2 = (0, û), c2 = -α
            // û = x / ||x||
            double inv_norm = safe_div(1.0, norm_x, 0.0);
            double alpha = 0.5 * (t + norm_x) * inv_norm;

            // diag[0] = 0, diag[i] = α for i >= 1
            sparse_diag[s_base] = 0.0;
            for (int64_t i = 1; i < dim; ++i) {
                sparse_diag[s_base + i] = alpha;
            }

            // v1 = (1, û)
            sparse_v1[s_base] = 1.0;
            for (int64_t i = 1; i < dim; ++i) {
                sparse_v1[s_base + i] = u[u_base + i] * inv_norm;
            }

            // v2 = (0, û)
            sparse_v2[s_base] = 0.0;
            for (int64_t i = 1; i < dim; ++i) {
                sparse_v2[s_base + i] = u[u_base + i] * inv_norm;
            }

            sparse_c1[batch * numSparseSoc + sparse_idx] = 0.5;
            sparse_c2[batch * numSparseSoc + sparse_idx] = -alpha;
        }
    }
}

// Block-per-cone variant for large sparse SOCs (dim > SOC_PARALLEL_THRESHOLD).
// Matches compute_soc_derivative_sparse_kernel semantics exactly; just uses a
// shared-memory block reduction for the tail-norm and parallel writes for the
// per-entry outputs. Sorted-ascending layout places large cones in the
// contiguous suffix [numSmallSoc, numSocCones).
__global__ void compute_soc_derivative_sparse_large_kernel(
    double* __restrict__ sparse_diag,
    double* __restrict__ sparse_v1,
    double* __restrict__ sparse_v2,
    double* __restrict__ sparse_c1,
    double* __restrict__ sparse_c2,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numSmallSoc,
    int64_t numLargeSoc,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_sparse_indices,
    const int64_t* __restrict__ d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    int64_t batchSize,
    int64_t m,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t large_idx = blockIdx.x;
    int64_t batch = blockIdx.y;
    if (large_idx >= numLargeSoc || batch >= batchSize) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    int64_t cone = numSmallSoc + large_idx;
    int64_t sparse_idx = d_soc_sparse_indices[cone];
    // Large cones are guaranteed sparse (dim > SOC_PARALLEL_THRESHOLD > 4),
    // but the defensive check matches the small-kernel semantics.
    if (sparse_idx < 0) return;

    int64_t dim = d_soc_dims[cone];
    int64_t sz_off = d_soc_sz_offsets[cone];
    int64_t sparse_off = d_soc_sparse_offsets[cone];
    int64_t u_base = batch * m + offset + sz_off;
    int64_t s_base = batch * totalSparseSocDim + sparse_off;
    int64_t c_idx = batch * numSparseSoc + sparse_idx;

    extern __shared__ double smem[];

    double t = u[u_base];

    // Reduce ||x||² over the tail in parallel.
    double my_sq = 0.0;
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double ui = u[u_base + i];
        my_sq += ui * ui;
    }
    double norm_x_sq = cones::block_sum_reduce(my_sq, smem, tid);
    double norm_x = safe_sqrt(norm_x_sq);

    if (t >= norm_x) {
        // Interior: H = I → diag = 1, c1 = c2 = 0
        for (int64_t i = tid; i < dim; i += blockDimX) {
            sparse_diag[s_base + i] = 1.0;
            sparse_v1[s_base + i] = 0.0;
            sparse_v2[s_base + i] = 0.0;
        }
        if (tid == 0) {
            sparse_c1[c_idx] = 0.0;
            sparse_c2[c_idx] = 0.0;
        }
        return;
    }

    if (t <= -norm_x) {
        // Polar: H = 0
        for (int64_t i = tid; i < dim; i += blockDimX) {
            sparse_diag[s_base + i] = 0.0;
            sparse_v1[s_base + i] = 0.0;
            sparse_v2[s_base + i] = 0.0;
        }
        if (tid == 0) {
            sparse_c1[c_idx] = 0.0;
            sparse_c2[c_idx] = 0.0;
        }
        return;
    }

    // Boundary: H = diag(d) + c1*v1*v1^T + c2*v2*v2^T
    double inv_norm = safe_div(1.0, norm_x, 0.0);
    double alpha = 0.5 * (t + norm_x) * inv_norm;

    if (tid == 0) {
        sparse_diag[s_base] = 0.0;
        sparse_v1[s_base] = 1.0;
        sparse_v2[s_base] = 0.0;
        sparse_c1[c_idx] = 0.5;
        sparse_c2[c_idx] = -alpha;
    }
    for (int64_t i = 1 + tid; i < dim; i += blockDimX) {
        double ui_normalized = u[u_base + i] * inv_norm;
        sparse_diag[s_base + i] = alpha;
        sparse_v1[s_base + i] = ui_normalized;
        sparse_v2[s_base + i] = ui_normalized;
    }
}

void compute_soc_derivative(
    double* H,
    const double* u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries,
    int64_t batchSize,
    int64_t m,
    const int64_t* d_soc_sz_offsets,
    cudaStream_t stream
) {
    if (numSocCones == 0) return;
    int threads = min((int)numSocCones, 256);
    const int64_t* offsets_for_sz = d_soc_sz_offsets ? d_soc_sz_offsets : d_soc_offsets;
    MOREAU_KERNEL_LAUNCH(compute_soc_derivative_dense_kernel, batchSize, threads, 0, stream,
        H, u, offset, numSocCones, d_soc_dims, d_soc_offsets,
        d_soc_Hs_offsets, nullptr, totalSocHsEntries, batchSize, m, offsets_for_sz
    );
}

void compute_soc_derivative_sparse(
    double* H,
    const double* u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_dense_offsets,
    const int64_t* d_soc_sparse_indices,
    int64_t totalDenseSocHsEntries,
    double* sparse_diag,
    double* sparse_v1,
    double* sparse_v2,
    double* sparse_c1,
    double* sparse_c2,
    const int64_t* d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    int64_t batchSize,
    int64_t m,
    const int64_t* d_soc_sz_offsets,
    cudaStream_t stream,
    int64_t numLargeSoc
) {
    if (numSocCones == 0) return;
    int64_t numSmallSoc = numSocCones - numLargeSoc;
    int threads = min((int)numSocCones, 256);
    const int64_t* offsets_for_sz = d_soc_sz_offsets ? d_soc_sz_offsets : d_soc_offsets;

    // Dense cones (dim <= 4): upper triangle. Dense cones are always small, so
    // the existing kernel's numSocCones bound is fine — large cones have
    // sparse_idx >= 0 so the inner `if (sparse_idx < 0) continue` skips them
    // naturally from this dense-only output path.
    if (totalDenseSocHsEntries > 0) {
        MOREAU_KERNEL_LAUNCH(compute_soc_derivative_dense_kernel, batchSize, threads, 0, stream,
            H, u, offset, numSocCones, d_soc_dims, d_soc_offsets,
            d_soc_Hs_dense_offsets, d_soc_sparse_indices,
            totalDenseSocHsEntries, batchSize, m, offsets_for_sz
        );
    }

    // Small sparse cones (4 < dim <= SOC_PARALLEL_THRESHOLD) use the existing
    // thread-per-cone kernel; loop bound restricted to the small prefix.
    if (numSparseSoc > 0 && numSmallSoc > 0) {
        int small_threads = min((int)numSmallSoc, 256);
        MOREAU_KERNEL_LAUNCH(compute_soc_derivative_sparse_kernel, batchSize, small_threads, 0, stream,
            sparse_diag, sparse_v1, sparse_v2, sparse_c1, sparse_c2,
            u, offset, numSmallSoc, d_soc_dims, d_soc_offsets,
            d_soc_sparse_indices, d_soc_sparse_offsets,
            totalSparseSocDim, numSparseSoc, batchSize, m, offsets_for_sz
        );
    }

    // Large sparse cones (dim > SOC_PARALLEL_THRESHOLD) use block-per-cone.
    if (numLargeSoc > 0) {
        const int block_size = cones::SOC_PARALLEL_BLOCK_SIZE;
        dim3 grid(static_cast<unsigned int>(numLargeSoc),
                  static_cast<unsigned int>(batchSize));
        dim3 block(block_size);
        size_t smem_bytes = sizeof(double) * block_size;
        MOREAU_KERNEL_LAUNCH(compute_soc_derivative_sparse_large_kernel, grid, block, smem_bytes, stream,
            sparse_diag, sparse_v1, sparse_v2, sparse_c1, sparse_c2,
            u, offset, numSmallSoc, numLargeSoc,
            d_soc_dims, d_soc_offsets,
            d_soc_sparse_indices, d_soc_sparse_offsets,
            totalSparseSocDim, numSparseSoc, batchSize, m, offsets_for_sz
        );
    }
}

// ============================================================================
// Exponential Cone Derivative Implementation
// ============================================================================

constexpr double EXP_CONE_TOL = 1e-8;
constexpr int EXP_MAX_ITERS = 200;

// Check if point is in exp cone interior: s > margin and s*exp(r/s) + margin < t
__device__ bool in_exp_interior(double r, double s, double t, double margin) {
    if (s <= margin) return false;
    double ratio = fmin(r / s, 500.0);  // Prevent exp overflow
    return s * exp(ratio) + margin < t;
}

// Check if point is in dual exp cone interior
// Dual exp cone: r < 0 and -r * exp(s/r - 1) <= e * t
// Interior: strict inequality with margin
__device__ bool in_exp_dual_interior(double r, double s, double t, double margin) {
    constexpr double E = 2.718281828459045;
    if (r >= -margin) return false;
    double ratio = fmin(s / r - 1.0, 500.0);  // Prevent exp overflow
    return -r * exp(ratio) + margin < E * t;
}

// Newton's method to find t in exp cone projection
__device__ double exp_newton_one_d(double rho, double y_hat, double z_hat) {
    double t = fmax(-z_hat, 1e-6);
    double rho_safe = fmax(fabs(rho), 1e-10);

    for (int iter = 0; iter < EXP_MAX_ITERS; iter++) {
        double t_safe = fmax(t, 1e-12);
        double f = t_safe * (t_safe + z_hat) / (rho_safe * rho_safe) - y_hat / rho_safe + log(t_safe / rho_safe) + 1.0;
        double fp = (2.0 * t_safe + z_hat) / (rho_safe * rho_safe) + 1.0 / t_safe;
        if (fabs(fp) < 1e-15) break;
        t = t - f / fp;

        if (t <= -z_hat) return 0.0;
        if (t <= 0.0) return z_hat;
        if (fabs(f) < EXP_CONE_TOL) break;
    }
    return t + z_hat;
}

// Solve for x given rho in exp cone projection
__device__ void exp_solve_for_x_with_rho(const double* v, double rho, double* x) {
    double x2 = exp_newton_one_d(rho, v[1], v[2]);
    x[2] = x2;
    x[1] = (x2 - v[2]) * x2 / rho;
    x[0] = v[0] - rho;
}

// Calculate gradient for bisection
__device__ double exp_calc_grad(const double* v, double rho) {
    double x[3];
    exp_solve_for_x_with_rho(v, rho, x);
    if (x[1] <= 1e-12 || x[2] <= 1e-12) {
        return x[0];
    }
    return x[0] + x[1] * log(x[1] / x[2]);
}

// Get upper bound for rho via doubling
__device__ void exp_get_rho_ub(const double* v, double* lb, double* ub) {
    *lb = 0.0;
    *ub = 0.125;

    while (exp_calc_grad(v, *ub) > 0.0) {
        *lb = *ub;
        *ub = *ub * 2.0;
    }
}

// Project onto primal exponential cone
__device__ void project_exp_cone_primal(const double* v, double* out) {
    double r = v[0], s = v[1], t = v[2];

    // Check if already in cone
    if ((r <= 0.0 && fabs(s) <= EXP_CONE_TOL && t >= 0.0) ||
        (s > 0.0 && s * exp(fmin(r / s, 500.0)) - t <= EXP_CONE_TOL)) {
        out[0] = r; out[1] = s; out[2] = t;
        return;
    }

    // Check if -v is in the dual exp cone: {(r,s,t) : r<=0, -r*exp(s/r-1) <= e*t}
    // If -v in K*, then Π_K(v) = 0 by Moreau decomposition
    constexpr double E = 2.718281828459045;
    double neg_v[3] = {-r, -s, -t};
    if ((fabs(neg_v[0]) <= EXP_CONE_TOL && neg_v[1] >= 0.0 && neg_v[2] >= 0.0) ||
        (neg_v[0] < 0.0 && -neg_v[0] * exp(fmin(neg_v[1] / neg_v[0] - 1.0, 500.0)) - E * neg_v[2] <= EXP_CONE_TOL)) {
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;
        return;
    }

    // Special case: r < 0, s < 0
    if (r < 0.0 && s < 0.0) {
        out[0] = r;
        out[1] = 0.0;
        out[2] = fmax(t, 0.0);
        return;
    }

    // Bisection for rho
    double lb, ub;
    exp_get_rho_ub(v, &lb, &ub);
    double rho = 0.0;

    for (int iter = 0; iter < EXP_MAX_ITERS; iter++) {
        rho = (ub + lb) / 2.0;
        double g = exp_calc_grad(v, rho);
        if (g > 0.0) {
            lb = rho;
        } else {
            ub = rho;
        }
        if (ub - lb < EXP_CONE_TOL) break;
    }

    exp_solve_for_x_with_rho(v, rho, out);
}

// Invert 4x4 matrix using Gaussian elimination with partial pivoting
__device__ void invert_4x4(double m[4][4], double inv[4][4]) {
    // Augment with identity
    double a[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            a[i][j] = m[i][j];
            a[i][j + 4] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < 4; col++) {
        // Find pivot
        int max_row = col;
        double max_val = fabs(a[col][col]);
        for (int row = col + 1; row < 4; row++) {
            if (fabs(a[row][col]) > max_val) {
                max_val = fabs(a[row][col]);
                max_row = row;
            }
        }

        // Swap rows
        if (max_row != col) {
            for (int j = 0; j < 8; j++) {
                double tmp = a[col][j];
                a[col][j] = a[max_row][j];
                a[max_row][j] = tmp;
            }
        }

        // Regularize if needed. Threshold matches the CPU implementation in
        // packages/moreau-cpu/src/.../diff/cones.rs:1947 (1e-12); both backends
        // must use the same pivot floor so exp-cone backward Jacobians agree
        // near the boundary. #175
        if (fabs(a[col][col]) < 1e-12) {
            a[col][col] = copysign(1e-12, a[col][col]);
        }

        // Eliminate
        double pivot = a[col][col];
        for (int j = 0; j < 8; j++) {
            a[col][j] /= pivot;
        }

        for (int row = 0; row < 4; row++) {
            if (row != col) {
                double factor = a[row][col];
                for (int j = 0; j < 8; j++) {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }
    }

    // Extract inverse
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            inv[i][j] = a[i][j + 4];
        }
    }
}

// Compute exp cone derivative (Jacobian of projection)
__global__ void compute_exp_derivative_kernel(
    double* __restrict__ H,
    const double* __restrict__ u,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numExpCones; cone += blockDim.x) {
        int64_t u_base = batch * m + offset + cone * 3;
        // Exp cone is NOT self-dual, so derivative is NOT symmetric
        // Store full 9 elements per cone (not just upper triangle 6)
        int64_t h_base = batch * numExpCones * 9 + cone * 9;

        // Input is u = z - s (dual space), we need derivative of Π_{K*}
        // Following diffclarabel: for dual derivative, evaluate at -u
        double xi[3] = {-u[u_base + 0], -u[u_base + 1], -u[u_base + 2]};
        double margin = 1e-6;

        double block[3][3];

        if (in_exp_interior(xi[0], xi[1], xi[2], margin)) {
            // xi in primal interior: D = I
            block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
            block[1][0] = 0.0; block[1][1] = 1.0; block[1][2] = 0.0;
            block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 1.0;
        } else if (in_exp_dual_interior(-xi[0], -xi[1], -xi[2], margin)) {
            // -xi in dual interior => Π_K(xi) = 0, so D = 0
            block[0][0] = 0.0; block[0][1] = 0.0; block[0][2] = 0.0;
            block[1][0] = 0.0; block[1][1] = 0.0; block[1][2] = 0.0;
            block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 0.0;
        } else if (xi[0] < -margin && xi[1] < -margin) {
            // Special case: r < 0, s < 0 (with tolerance to avoid discontinuity at xi ≈ 0)
            block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
            block[1][0] = 0.0; block[1][1] = 0.0; block[1][2] = 0.0;
            block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = (xi[2] >= 0.0) ? 1.0 : 0.0;
        } else {
            // Boundary case: compute Jacobian via 4x4 system
            double rs[3];
            project_exp_cone_primal(xi, rs);

            double r = rs[0], s = rs[1], t = rs[2];
            // s_min must match the CPU s_eff floor in cones.rs:1493 (1e-10).
            // The CPU implementation is the reference; reducing CUDA from 1e-6
            // to 1e-10 closes a 4-order-of-magnitude gap that produced O(1)
            // divergence in alpha = exp(r/s_eff) for s in [1e-10, 1e-6]. #175
            double s_min = 1e-10;
            double s_eff = fmax(s, s_min);
            double l = t - xi[2];
            // Clamp r/s_eff to prevent exp() overflow (exp(500) ~ 1e217, safe in f64)
            double r_over_s = fmin(r / s_eff, 500.0);
            double alpha = exp(r_over_s);
            double beta = l * r / (s_eff * s_eff) * alpha;

            // Build 4x4 system J^{-1}
            double j_inv[4][4] = {{0}};
            j_inv[0][0] = alpha;
            j_inv[0][1] = (-r + s_eff) / s_eff * alpha;
            j_inv[0][2] = -1.0;
            j_inv[1][0] = 1.0 + l / s_eff * alpha;
            j_inv[1][1] = -beta;
            j_inv[1][3] = alpha;
            j_inv[2][0] = -beta;
            j_inv[2][1] = 1.0 + beta * r / s_eff;
            j_inv[2][3] = (1.0 - r / s_eff) * alpha;
            j_inv[3][2] = 1.0;
            j_inv[3][3] = -1.0;

            // Invert to get J
            double j_full[4][4];
            invert_4x4(j_inv, j_full);

            // Extract 3x3 submatrix (rows 0-2, cols 1-3)
            block[0][0] = j_full[0][1]; block[0][1] = j_full[0][2]; block[0][2] = j_full[0][3];
            block[1][0] = j_full[1][1]; block[1][1] = j_full[1][2]; block[1][2] = j_full[1][3];
            block[2][0] = j_full[2][1]; block[2][1] = j_full[2][2]; block[2][2] = j_full[2][3];

            // No NaN/Inf sanitisation here: matching the CPU implementation,
            // we propagate non-finite values rather than silently falling back
            // to identity. The previous fallback was mathematically wrong —
            // because the dual derivative is D_{K*}(u) = I - D_K(-u), a primal
            // identity block makes the dual derivative I - I = 0, silently
            // zeroing gradients at the cone boundary. With the s_eff floor at
            // 1e-10 and the r/s_eff clamp at 500 above, realistic inputs
            // should not produce NaN; if they do, surfacing the NaN tells the
            // caller something is wrong (CLAUDE.md rule #4). #175
        }

        // Apply Moreau decomposition for dual: D_{K*}(u) = I - D_K(-u)
        double result[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double delta_ij = (i == j) ? 1.0 : 0.0;
                result[i][j] = delta_ij - block[i][j];
            }
        }

        // Store full 3x3 matrix in row-major order (NOT symmetric for exp cone)
        H[h_base + 0] = result[0][0];
        H[h_base + 1] = result[0][1];
        H[h_base + 2] = result[0][2];
        H[h_base + 3] = result[1][0];
        H[h_base + 4] = result[1][1];
        H[h_base + 5] = result[1][2];
        H[h_base + 6] = result[2][0];
        H[h_base + 7] = result[2][1];
        H[h_base + 8] = result[2][2];
    }
}

void compute_exp_derivative(
    double* H,
    const double* u,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numExpCones == 0) return;
    int threads = min((int)numExpCones, 256);
    MOREAU_KERNEL_LAUNCH(compute_exp_derivative_kernel, batchSize, threads, 0, stream,
        H, u, offset, numExpCones, batchSize, m
    );
}

// ============================================================================
// Power Cone Implementation (Projection and Derivative)
// ============================================================================

constexpr double POW_CONE_TOL = 1e-8;
constexpr int POW_MAX_ITERS = 50;

// Check if in power cone interior: x^α * y^(1-α) >= |z| + margin with x,y > margin
__device__ bool in_pow_cone_interior(double x, double y, double abs_z, double alpha, double margin) {
    if (x <= margin || y <= margin) return false;
    return pow(x, alpha) * pow(y, 1.0 - alpha) >= abs_z + margin;
}

// Check if in polar cone interior: x < -margin, y < -margin
__device__ bool in_pow_polar_cone_interior(double x, double y, double abs_z, double alpha, double margin) {
    if (x >= -margin || y >= -margin) return false;
    double lhs = pow(-x / alpha, alpha) * pow(-y / (1.0 - alpha), 1.0 - alpha);
    return lhs >= abs_z + margin;
}

// Helper: compute xi from r in power cone projection
__device__ double pow_calc_xi(double ri, double x, double abs_z, double alpha) {
    double val = x * x + 4.0 * alpha * (abs_z - ri) * ri;
    double xi = 0.5 * (x + sqrt(fmax(val, 0.0)));
    return fmax(xi, POW_CONE_TOL);
}

// Helper: gi function
__device__ double pow_gi(double ri, double xi, double abs_z, double alpha) {
    return 2.0 * pow_calc_xi(ri, xi, abs_z, alpha) - xi;
}

// Helper: f function for Newton
__device__ double pow_calc_f(double ri, double xi, double yi, double alpha) {
    return pow(xi, alpha) * pow(yi, 1.0 - alpha) - ri;
}

// Helper: dxi/dr
__device__ double pow_calc_dxi_dr(double ri, double xi, double x, double abs_z, double alpha) {
    double denom = 2.0 * xi - x;
    if (fabs(denom) < POW_CONE_TOL) return 0.0;
    return alpha * (abs_z - 2.0 * ri) / denom;
}

// Helper: f' for Newton (with safe division)
__device__ double pow_calc_fp(double xi, double yi, double dxidri, double dyidri, double alpha) {
    double alphac = 1.0 - alpha;
    // Safe division: if xi or yi is too small, return 0 for that term
    double term_x = (fabs(xi) > POW_CONE_TOL) ? (alpha * dxidri / xi) : 0.0;
    double term_y = (fabs(yi) > POW_CONE_TOL) ? (alphac * dyidri / yi) : 0.0;
    return pow(xi, alpha) * pow(yi, alphac) * (term_x + term_y) - 1.0;
}

// Project onto primal power cone (without Jacobian computation)
// Power cone K_pow(α) = {(x,y,z) : x^α * y^(1-α) >= |z|, x,y >= 0}
__device__ void project_pow_cone_primal(double x, double y, double z, double alpha, double* out) {
    double abs_z = fabs(z);
    double margin = 1e-6;

    // Interior case: already in cone
    if (in_pow_cone_interior(x, y, abs_z, alpha, margin)) {
        out[0] = x;
        out[1] = y;
        out[2] = z;
        return;
    }

    // Polar interior case: project to zero
    if (in_pow_polar_cone_interior(x, y, abs_z, alpha, margin)) {
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        return;
    }

    // z = 0 case
    if (abs_z <= POW_CONE_TOL) {
        out[0] = fmax(x, 0.0);
        out[1] = fmax(y, 0.0);
        out[2] = 0.0;
        return;
    }

    // General case: Newton iteration to find projection on boundary
    double r = abs_z / 2.0;
    double xi = 0.0, yi = 0.0;

    for (int iter = 0; iter < POW_MAX_ITERS; iter++) {
        xi = pow_calc_xi(r, x, abs_z, alpha);
        yi = pow_calc_xi(r, y, abs_z, 1.0 - alpha);

        double f = pow_calc_f(r, xi, yi, alpha);
        if (fabs(f) < POW_CONE_TOL) break;

        double dxdr = pow_calc_dxi_dr(r, xi, x, abs_z, alpha);
        double dydr = pow_calc_dxi_dr(r, yi, y, abs_z, 1.0 - alpha);
        double fp = pow_calc_fp(xi, yi, dxdr, dydr, alpha);

        if (fabs(fp) < POW_CONE_TOL) break;

        r = r - f / fp;
        r = fmax(r, 0.0);
        r = fmin(r, abs_z);
    }

    out[0] = xi;
    out[1] = yi;
    out[2] = (z >= 0.0) ? r : -r;
}

// Project onto power cone with Jacobian computation
__device__ void project_pow_cone_with_jacobian(
    double x, double y, double z, double alpha,
    double* proj, double jac[3][3]
) {
    double abs_z = fabs(z);
    double margin = 1e-6;

    // Interior case
    if (in_pow_cone_interior(x, y, abs_z, alpha, margin)) {
        proj[0] = x; proj[1] = y; proj[2] = z;
        jac[0][0] = 1.0; jac[0][1] = 0.0; jac[0][2] = 0.0;
        jac[1][0] = 0.0; jac[1][1] = 1.0; jac[1][2] = 0.0;
        jac[2][0] = 0.0; jac[2][1] = 0.0; jac[2][2] = 1.0;
        return;
    }

    // Polar interior case
    if (in_pow_polar_cone_interior(x, y, abs_z, alpha, margin)) {
        proj[0] = 0.0; proj[1] = 0.0; proj[2] = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                jac[i][j] = 0.0;
        return;
    }

    // z = 0 case
    if (abs_z <= POW_CONE_TOL) {
        proj[0] = fmax(x, 0.0);
        proj[1] = fmax(y, 0.0);
        proj[2] = 0.0;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                jac[i][j] = 0.0;

        jac[0][0] = (fabs(x) > POW_CONE_TOL) ? 0.5 * (copysign(1.0, x) + 1.0) : 0.5;
        jac[1][1] = (fabs(y) > POW_CONE_TOL) ? 0.5 * (copysign(1.0, y) + 1.0) : 0.5;

        if (x > 0.0 && y < 0.0) {
            if (alpha > 0.5) jac[2][2] = 1.0;
            else if (alpha < 0.5) jac[2][2] = 0.0;
            else jac[2][2] = (fabs(y) > POW_CONE_TOL) ? x / (2.0 * fabs(y) + x) : 1.0;
        } else if (y > 0.0 && x < 0.0) {
            if (alpha < 0.5) jac[2][2] = 1.0;
            else if (alpha > 0.5) jac[2][2] = 0.0;
            else jac[2][2] = (fabs(x) > POW_CONE_TOL) ? y / (2.0 * fabs(x) + y) : 1.0;
        }
        return;
    }

    // General case: Newton iteration
    double r = abs_z / 2.0;
    double xi = 0.0, yi = 0.0;

    for (int iter = 0; iter < POW_MAX_ITERS; iter++) {
        xi = pow_calc_xi(r, x, abs_z, alpha);
        yi = pow_calc_xi(r, y, abs_z, 1.0 - alpha);

        double f = pow_calc_f(r, xi, yi, alpha);
        if (fabs(f) < POW_CONE_TOL) break;

        double dxdr = pow_calc_dxi_dr(r, xi, x, abs_z, alpha);
        double dydr = pow_calc_dxi_dr(r, yi, y, abs_z, 1.0 - alpha);
        double fp = pow_calc_fp(xi, yi, dxdr, dydr, alpha);

        if (fabs(fp) < POW_CONE_TOL) break;

        r = r - f / fp;
        r = fmax(r, 0.0);
        r = fmin(r, abs_z);
    }

    double x_star = xi;
    double y_star = yi;
    double z_star = (z >= 0.0) ? r : -r;

    proj[0] = x_star;
    proj[1] = y_star;
    proj[2] = z_star;

    // Compute Jacobian using implicit differentiation.
    //
    // The projection satisfies:
    //   x* = 0.5 * (x + sqrt(x^2 + 4*alpha*(|z| - r*)*r*))
    //   y* = 0.5 * (y + sqrt(y^2 + 4*(1-alpha)*(|z| - r*)*r*))
    //   z* = sign(z) * r*
    //   phi(r*) = x*^alpha * y*^(1-alpha) - r* = 0
    //
    // Using implicit differentiation on phi = 0:
    //   dr*/d(...) = -dphi/d(...) / dphi/dr
    //
    // NOTE: Uses implicit differentiation rather than the diffqcp closed-form formula
    // (verified more accurate via JAX autodiff). Matches the CPU Rust implementation.

    double a = alpha;
    double ac = 1.0 - alpha;
    double r_star = r;
    double sign_z = (z != 0.0) ? copysign(1.0, z) : 1.0;

    // Compute sqrt terms: gx = x^2 + 4*a*(|z|-r*)*r*, gy = y^2 + 4*ac*(|z|-r*)*r*
    double gx = x * x + 4.0 * a * (abs_z - r_star) * r_star;
    double gy = y * y + 4.0 * ac * (abs_z - r_star) * r_star;
    double sqrt_gx = sqrt(fmax(gx, POW_CONE_TOL));
    double sqrt_gy = sqrt(fmax(gy, POW_CONE_TOL));

    // Partial derivatives of xi w.r.t. x, |z|, r (holding others constant)
    double dxi_dx = 0.5 + x / (2.0 * sqrt_gx);
    double dxi_dz = a * r_star / sqrt_gx;
    double dxi_dr = a * (abs_z - 2.0 * r_star) / sqrt_gx;

    // Partial derivatives of yi w.r.t. y, |z|, r
    double dyi_dy = 0.5 + y / (2.0 * sqrt_gy);
    double dyi_dz = ac * r_star / sqrt_gy;
    double dyi_dr = ac * (abs_z - 2.0 * r_star) / sqrt_gy;

    // At solution, phi = x*^a * y*^(1-a) - r* = 0, so x*^a * y*^(1-a) = r*
    // Partial derivatives of phi
    double dphi_dx = a * r_star * dxi_dx / x_star;
    double dphi_dy = ac * r_star * dyi_dy / y_star;
    double dphi_dz = a * r_star * dxi_dz / x_star + ac * r_star * dyi_dz / y_star;
    double dphi_dr = a * r_star * dxi_dr / x_star + ac * r_star * dyi_dr / y_star - 1.0;

    // From phi = 0: dr*/d(...) = -dphi/d(...) / dphi/dr
    double dr_dx, dr_dy, dr_dz_abs;
    if (fabs(dphi_dr) > POW_CONE_TOL) {
        dr_dx = -dphi_dx / dphi_dr;
        dr_dy = -dphi_dy / dphi_dr;
        dr_dz_abs = -dphi_dz / dphi_dr;
    } else {
        dr_dx = 0.0;
        dr_dy = 0.0;
        dr_dz_abs = 0.0;
    }

    // Full Jacobian d[x*, y*, z*] / d[x, y, z]
    // dx*/dx = dxi_dx + dxi_dr * dr_dx
    jac[0][0] = dxi_dx + dxi_dr * dr_dx;
    // dx*/dy = dxi_dr * dr_dy (xi doesn't depend on y directly)
    jac[0][1] = dxi_dr * dr_dy;
    // dx*/dz = (dxi_dz + dxi_dr * dr_dz_abs) * sign_z
    jac[0][2] = (dxi_dz + dxi_dr * dr_dz_abs) * sign_z;

    // dy*/dx = dyi_dr * dr_dx (yi doesn't depend on x directly)
    jac[1][0] = dyi_dr * dr_dx;
    // dy*/dy = dyi_dy + dyi_dr * dr_dy
    jac[1][1] = dyi_dy + dyi_dr * dr_dy;
    // dy*/dz = (dyi_dz + dyi_dr * dr_dz_abs) * sign_z
    jac[1][2] = (dyi_dz + dyi_dr * dr_dz_abs) * sign_z;

    // dz*/dx = sign_z * dr_dx
    jac[2][0] = sign_z * dr_dx;
    // dz*/dy = sign_z * dr_dy
    jac[2][1] = sign_z * dr_dy;
    // dz*/dz = dr_dz_abs (for z != 0)
    jac[2][2] = dr_dz_abs;
}

// Compute power cone derivative (Jacobian of projection)
__global__ void compute_power_derivative_kernel(
    double* __restrict__ H,
    const double* __restrict__ u,
    const double* __restrict__ alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numPowerCones; cone += blockDim.x) {
        int64_t u_base = batch * m + offset + cone * 3;
        // Power cone is NOT self-dual, so derivative is NOT symmetric
        // Store full 9 elements per cone (not just upper triangle 6)
        int64_t h_base = batch * numPowerCones * 9 + cone * 9;

        // Input is u = z - s (dual space), we need derivative of Π_{K*}
        // Following diffclarabel: for dual derivative, evaluate at -u
        double xi[3] = {-u[u_base + 0], -u[u_base + 1], -u[u_base + 2]};
        double alpha = alphas[cone];

        // Compute projection and Jacobian at xi (primal derivative)
        double proj[3];
        double block[3][3];
        project_pow_cone_with_jacobian(xi[0], xi[1], xi[2], alpha, proj, block);

        // Apply Moreau decomposition for dual: D_{K*}(u) = I - D_K(-u)
        double result[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double delta_ij = (i == j) ? 1.0 : 0.0;
                result[i][j] = delta_ij - block[i][j];
            }
        }

        // Store full 3x3 matrix in row-major order (NOT symmetric for power cone)
        H[h_base + 0] = result[0][0];
        H[h_base + 1] = result[0][1];
        H[h_base + 2] = result[0][2];
        H[h_base + 3] = result[1][0];
        H[h_base + 4] = result[1][1];
        H[h_base + 5] = result[1][2];
        H[h_base + 6] = result[2][0];
        H[h_base + 7] = result[2][1];
        H[h_base + 8] = result[2][2];
    }
}

void compute_power_derivative(
    double* H,
    const double* u,
    const double* alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
) {
    if (numPowerCones == 0) return;
    int threads = min((int)numPowerCones, 256);
    MOREAU_KERNEL_LAUNCH(compute_power_derivative_kernel, batchSize, threads, 0, stream,
        H, u, alphas, offset, numPowerCones, batchSize, m
    );
}

// ============================================================================
// GenPowerCone Implementation (Projection and Derivative)
// ============================================================================

// Constants matching CPU (Rust) CONE_TOL, POW_MAX_ITERS, and margin values
constexpr double GENPOW_CONE_TOL = 1e-8;
constexpr double GENPOW_MARGIN = 1e-6;
constexpr double GENPOW_NEWTON_TOL = 1e-12;
constexpr double GENPOW_NEWTON_FP_TOL = 1e-14;
constexpr int GENPOW_MAX_ITERS = 200;

// Check if point is in generalized power cone interior (squared form, matches CPU)
// K = {(p,w) : prod(p_i^alpha_i) >= ||w||, p_i >= 0}
// Uses squared comparison: prod(p_i^{2*alpha_i}) >= ||w||^2 + margin
__device__ bool in_genpow_cone_interior(
    const double* v, const double* alphas, int64_t dim1, int64_t dim2, double margin
) {
    // Check p_i > margin
    for (int64_t i = 0; i < dim1; ++i) {
        if (v[i] <= margin) return false;
    }
    // Compute prod(p_i^{2*alpha_i}) and ||w||^2
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; ++i) {
        log_prod += 2.0 * alphas[i] * log(v[i]);
    }
    double norm_w_sq = 0.0;
    for (int64_t i = dim1; i < dim1 + dim2; ++i) {
        norm_w_sq += v[i] * v[i];
    }
    return exp(log_prod) >= norm_w_sq + margin;
}

// Check if point is in generalized power cone polar interior (squared form, matches CPU)
// K° = {(q,r) : prod(-q_i/alpha_i)^alpha_i >= ||r||, q_i <= 0}
// Uses squared comparison: prod((-q_i/alpha_i)^{2*alpha_i}) >= ||r||^2 + margin
__device__ bool in_genpow_polar_interior(
    const double* v, const double* alphas, int64_t dim1, int64_t dim2, double margin
) {
    for (int64_t i = 0; i < dim1; ++i) {
        if (v[i] >= -margin) return false;
    }
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; ++i) {
        log_prod += 2.0 * alphas[i] * log(-v[i] / alphas[i]);
    }
    double norm_w_sq = 0.0;
    for (int64_t i = dim1; i < dim1 + dim2; ++i) {
        norm_w_sq += v[i] * v[i];
    }
    return exp(log_prod) >= norm_w_sq + margin;
}

// Project onto primal generalized power cone via Newton iteration
// Generalizes project_pow_cone_primal to variable dimensions
__device__ void project_genpow_cone_primal(
    const double* v, const double* alphas, int64_t dim1, int64_t dim2,
    double* out
) {
    int64_t dim = dim1 + dim2;

    // Interior: already in cone
    if (in_genpow_cone_interior(v, alphas, dim1, dim2, GENPOW_MARGIN)) {
        for (int64_t i = 0; i < dim; ++i) out[i] = v[i];
        return;
    }

    // Polar interior: project to zero
    if (in_genpow_polar_interior(v, alphas, dim1, dim2, GENPOW_MARGIN)) {
        for (int64_t i = 0; i < dim; ++i) out[i] = 0.0;
        return;
    }

    // ||w|| == 0 case: project p components to max(0, v_i)
    double norm_w_sq = 0.0;
    for (int64_t i = dim1; i < dim; ++i) norm_w_sq += v[i] * v[i];

    if (norm_w_sq < GENPOW_CONE_TOL * GENPOW_CONE_TOL) {
        for (int64_t i = 0; i < dim1; ++i) out[i] = fmax(v[i], 0.0);
        for (int64_t i = dim1; i < dim; ++i) out[i] = 0.0;
        return;
    }

    double abs_w = sqrt(norm_w_sq);

    // Newton iteration: find r such that F(r) = 0
    // where F(r) = prod(xi_i^alpha_i) - r
    // xi_i = 0.5*(v_i + sqrt(v_i^2 + 4*alpha_i*(abs_w - r)*r))
    double r = abs_w / 2.0;

    for (int iter = 0; iter < GENPOW_MAX_ITERS; ++iter) {
        // Compute xi_i for each power dimension
        double log_prod = 0.0;
        double sum_frac = 0.0;  // sum of alpha_i * dxi/dr / xi_i
        bool valid = true;

        for (int64_t i = 0; i < dim1; ++i) {
            double ai = alphas[i];
            double vi = v[i];
            double disc = vi * vi + 4.0 * ai * (abs_w - r) * r;
            if (disc < 0.0) disc = 0.0;
            double xi = 0.5 * (vi + sqrt(disc));
            xi = fmax(xi, GENPOW_CONE_TOL);

            log_prod += ai * log(xi);

            // dxi/dr = ai * (abs_w - 2*r) / (2*xi - vi)
            double denom = 2.0 * xi - vi;
            if (fabs(denom) < GENPOW_CONE_TOL) { valid = false; break; }
            double dxidr = ai * (abs_w - 2.0 * r) / denom;
            sum_frac += ai * dxidr / xi;
        }

        if (!valid) break;

        double prod = exp(log_prod);
        double f = prod - r;
        if (fabs(f) < GENPOW_CONE_TOL) break;

        double fp = prod * sum_frac - 1.0;
        if (fabs(fp) < GENPOW_CONE_TOL) break;

        r = r - f / fp;
        r = fmax(r, 0.0);
        r = fmin(r, abs_w);
    }

    // Recover projection (clamp to CONE_TOL like CPU genpow_calc_pi)
    for (int64_t i = 0; i < dim1; ++i) {
        double ai = alphas[i];
        double vi = v[i];
        double disc = vi * vi + 4.0 * ai * (abs_w - r) * r;
        if (disc < 0.0) disc = 0.0;
        out[i] = fmax(0.5 * (vi + sqrt(disc)), GENPOW_CONE_TOL);
    }
    double scale = (abs_w > GENPOW_CONE_TOL) ? r / abs_w : 0.0;
    for (int64_t i = dim1; i < dim; ++i) {
        out[i] = v[i] * scale;
    }
}

// GenPowerCone dual projection via Moreau: Π_{K*}(z) = z + Π_K(-z)
__global__ void project_genpow_cone_dual_kernel(
    double* __restrict__ pi_u,
    const double* __restrict__ u,
    const double* __restrict__ alphas,
    const int64_t* __restrict__ d_dim1s,
    const int64_t* __restrict__ d_dim2s,
    const int64_t* __restrict__ d_offsets,
    const int64_t* __restrict__ d_alpha_offsets,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t offset,
    int64_t numGenPowerCones,
    int64_t batchSize,
    int64_t m,
    double* __restrict__ work_vec,           // [batchSize][totalGenPowerDim] for neg_v
    int64_t totalGenPowerDim
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numGenPowerCones; cone += blockDim.x) {
        int64_t dim1 = d_dim1s[cone];
        int64_t dim2 = d_dim2s[cone];
        int64_t dim = dim1 + dim2;
        int64_t gp_off = d_offsets[cone];
        int64_t sz_off = d_sz_offsets[cone];
        int64_t alpha_off = d_alpha_offsets[cone];
        int64_t base = batch * m + offset + sz_off;

        // Workspace pointers (neg_v doubles as primal_proj output)
        double* neg_v = work_vec + batch * totalGenPowerDim + gp_off;

        for (int64_t i = 0; i < dim; ++i) {
            neg_v[i] = -u[base + i];
        }

        // project_genpow_cone_primal writes result in-place to pi_u via separate output
        // We use the pi_u buffer directly as primal_proj output
        double* primal_proj = pi_u + base;
        project_genpow_cone_primal(neg_v, &alphas[alpha_off], dim1, dim2, primal_proj);

        for (int64_t i = 0; i < dim; ++i) {
            pi_u[base + i] = u[base + i] + primal_proj[i];
        }
    }
}

void project_genpow_cone_dual(
    double* pi_u,
    const double* u,
    const double* alphas,
    const int64_t* d_dim1s,
    const int64_t* d_dim2s,
    const int64_t* d_offsets,
    const int64_t* d_alpha_offsets,
    int64_t offset,
    int64_t numGenPowerCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream,
    double* work_vec,
    int64_t totalGenPowerDim,
    const int64_t* d_sz_offsets
) {
    if (numGenPowerCones == 0) return;
    int threads = min((int)numGenPowerCones, 256);
    MOREAU_KERNEL_LAUNCH(project_genpow_cone_dual_kernel, batchSize, threads, 0, stream,
        pi_u, u, alphas, d_dim1s, d_dim2s, d_offsets, d_alpha_offsets, d_sz_offsets,
        offset, numGenPowerCones, batchSize, m,
        work_vec, totalGenPowerDim
    );
}

// Compute GenPowerCone derivative in sparse decomposition form
// Outputs: diag + left1*right1^T + left2*right2^T + c3*left3*left3^T
// (Already includes Moreau: D_{K*}(u) = I - D_K(-u))
__global__ void compute_genpow_derivative_sparse_kernel(
    double* __restrict__ sparse_diag,
    double* __restrict__ sparse_left1,
    double* __restrict__ sparse_right1,
    double* __restrict__ sparse_left2,
    double* __restrict__ sparse_right2,
    double* __restrict__ sparse_left3,
    double* __restrict__ sparse_c3,
    const double* __restrict__ u,
    const double* __restrict__ alphas,
    const int64_t* __restrict__ d_dim1s,
    const int64_t* __restrict__ d_dim2s,
    const int64_t* __restrict__ d_offsets,
    const int64_t* __restrict__ d_alpha_offsets,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t offset,
    int64_t numGenPowerCones,
    int64_t totalGenpowDim,
    int64_t batchSize,
    int64_t m,
    double* __restrict__ work_vec,
    double* __restrict__ work_dim1,
    int64_t totalGenPowerAlphas
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numGenPowerCones; cone += blockDim.x) {
        int64_t dim1 = d_dim1s[cone];
        int64_t dim2 = d_dim2s[cone];
        int64_t dim = dim1 + dim2;
        int64_t gp_off = d_offsets[cone];
        int64_t sz_off = d_sz_offsets[cone];
        int64_t alpha_off = d_alpha_offsets[cone];
        int64_t u_base = batch * m + offset + sz_off;
        int64_t out_base = batch * totalGenpowDim + gp_off;

        double margin = GENPOW_MARGIN;
        double tol = GENPOW_CONE_TOL;

        // Output pointers
        double* o_diag = sparse_diag + out_base;
        double* o_l1 = sparse_left1 + out_base;
        double* o_r1 = sparse_right1 + out_base;
        double* o_l2 = sparse_left2 + out_base;
        double* o_r2 = sparse_right2 + out_base;
        double* o_l3 = sparse_left3 + out_base;

        // Workspace: work_dim1 has 7 striped arrays, each of size totalGenPowerAlphas.
        // Within each stripe, this cone's dim1 elements start at alpha_off.
        // Layout: [pi_stars | sqrt_gi | dpi_dpi | dpi_dr | dpi_dnw | dphi_dp | dr_dp]
        //          <-- stride = totalGenPowerAlphas per stripe -->
        double* xi = work_vec + batch * totalGenpowDim + gp_off;
        int64_t d1_base = batch * 7 * totalGenPowerAlphas + alpha_off;
        int64_t d1_stride = totalGenPowerAlphas;
        double* pi_stars = work_dim1 + d1_base;
        double* sqrt_gi  = work_dim1 + d1_base + d1_stride;
        double* dpi_dpi  = work_dim1 + d1_base + 2 * d1_stride;
        double* dpi_dr   = work_dim1 + d1_base + 3 * d1_stride;
        double* dpi_dnw  = work_dim1 + d1_base + 4 * d1_stride;
        double* dphi_dp  = work_dim1 + d1_base + 5 * d1_stride;
        double* dr_dp    = work_dim1 + d1_base + 6 * d1_stride;

        // Helper: zero all output vectors
        auto zero_rank_terms = [&]() {
            for (int64_t i = 0; i < dim; ++i) {
                o_l1[i] = 0.0; o_r1[i] = 0.0;
                o_l2[i] = 0.0; o_r2[i] = 0.0;
                o_l3[i] = 0.0;
            }
            sparse_c3[batch * numGenPowerCones + cone] = 0.0;
        };

        // Evaluate at -u for dual via Moreau
        for (int64_t i = 0; i < dim; ++i) xi[i] = -u[u_base + i];

        const double* cone_alphas = &alphas[alpha_off];
        const double* p = xi;
        const double* w = xi + dim1;

        if (in_genpow_cone_interior(xi, cone_alphas, dim1, dim2, margin)) {
            // Interior: D_K(-u) = I  →  D_{K*}(u) = 0
            for (int64_t i = 0; i < dim; ++i) o_diag[i] = 0.0;
            zero_rank_terms();
            continue;
        }

        if (in_genpow_polar_interior(xi, cone_alphas, dim1, dim2, margin)) {
            // Polar: D_K(-u) = 0  →  D_{K*}(u) = I
            for (int64_t i = 0; i < dim; ++i) o_diag[i] = 1.0;
            zero_rank_terms();
            continue;
        }

        double norm_w_sq = 0.0;
        for (int64_t i = 0; i < dim2; ++i) norm_w_sq += w[i] * w[i];
        double norm_w = sqrt(norm_w_sq);

        if (norm_w <= tol) {
            // w = 0: diagonal only (dual = I - primal_diag)
            for (int64_t i = 0; i < dim1; ++i) {
                double pval = (fabs(p[i]) > tol) ? 0.5 * (copysign(1.0, p[i]) + 1.0) : 0.5;
                o_diag[i] = 1.0 - pval;
            }
            for (int64_t i = 0; i < dim2; ++i) o_diag[dim1 + i] = 1.0;
            zero_rank_terms();
            continue;
        }

        // Newton iteration with tight tolerance for derivative accuracy.
        // Matches CPU (Rust) implementation: copysign regularization on denom
        // instead of hard-breaking, to ensure convergence parity.
        double r_val = norm_w * 0.5;
        for (int iter = 0; iter < GENPOW_MAX_ITERS; ++iter) {
            for (int64_t i = 0; i < dim1; ++i) {
                double ai = cone_alphas[i];
                double vi = p[i];
                double disc = vi * vi + 4.0 * ai * (norm_w - r_val) * r_val;
                if (disc < 0.0) disc = 0.0;
                double psi = 0.5 * (vi + sqrt(disc));
                pi_stars[i] = fmax(psi, tol);
            }

            double log_prod = 0.0;
            for (int64_t i = 0; i < dim1; ++i)
                log_prod += cone_alphas[i] * log(pi_stars[i]);
            double prod = exp(log_prod);
            double f = prod - r_val;
            if (fabs(f) < GENPOW_NEWTON_TOL) break;

            double fp = -1.0;
            for (int64_t i = 0; i < dim1; ++i) {
                double denom = 2.0 * pi_stars[i] - p[i];
                double dpi_dr_i = cone_alphas[i] * (norm_w - 2.0 * r_val) / (denom + copysign(tol, denom));
                if (fabs(pi_stars[i]) > tol)
                    fp += prod * cone_alphas[i] * dpi_dr_i / pi_stars[i];
            }
            if (fabs(fp) < GENPOW_NEWTON_FP_TOL) break;
            r_val -= f / fp;
            r_val = fmax(r_val, 0.0);
            r_val = fmin(r_val, norm_w);
        }

        // Compute intermediates
        for (int64_t i = 0; i < dim1; ++i) {
            double ai = cone_alphas[i];
            double disc = p[i] * p[i] + 4.0 * ai * (norm_w - r_val) * r_val;
            if (disc < 0.0) disc = 0.0;
            pi_stars[i] = fmax(0.5 * (p[i] + sqrt(disc)), tol);
        }

        double log_prod = 0.0;
        for (int64_t i = 0; i < dim1; ++i)
            log_prod += cone_alphas[i] * log(pi_stars[i]);
        double prod = exp(log_prod);

        for (int64_t i = 0; i < dim1; ++i) {
            double gi = p[i] * p[i] + 4.0 * cone_alphas[i] * r_val * (norm_w - r_val);
            sqrt_gi[i] = sqrt(fmax(gi, tol));
        }

        for (int64_t i = 0; i < dim1; ++i)
            dpi_dpi[i] = pi_stars[i] / sqrt_gi[i];

        for (int64_t i = 0; i < dim1; ++i) {
            double denom = 2.0 * pi_stars[i] - p[i];
            dpi_dr[i] = cone_alphas[i] * (norm_w - 2.0 * r_val) / (denom + copysign(tol, denom));
        }

        for (int64_t i = 0; i < dim1; ++i)
            dpi_dnw[i] = cone_alphas[i] * r_val / sqrt_gi[i];

        double dphi_dr_val = -1.0;
        for (int64_t i = 0; i < dim1; ++i) {
            if (fabs(pi_stars[i]) > tol)
                dphi_dr_val += prod * cone_alphas[i] * dpi_dr[i] / pi_stars[i];
        }

        for (int64_t i = 0; i < dim1; ++i)
            dphi_dp[i] = (fabs(pi_stars[i]) > tol) ? prod * cone_alphas[i] * dpi_dpi[i] / pi_stars[i] : 0.0;

        double dphi_dnw_val = 0.0;
        for (int64_t i = 0; i < dim1; ++i) {
            if (fabs(pi_stars[i]) > tol)
                dphi_dnw_val += prod * cone_alphas[i] * dpi_dnw[i] / pi_stars[i];
        }

        double dphi_dr_reg = dphi_dr_val + copysign(tol, dphi_dr_val);
        double dr_dnw = -dphi_dnw_val / dphi_dr_reg;
        for (int64_t i = 0; i < dim1; ++i)
            dr_dp[i] = -dphi_dp[i] / dphi_dr_reg;

        double r_over_nw = r_val / norm_w;

        // Build decomposition vectors (primal: D_K(-u))
        // Then apply Moreau: D_{K*}(u) = I - D_K(-u)
        // For dual: diag = (I - D_primal), negate all rank left vectors

        // D_primal = diag([dpi_dpi; r/||w|| * ones])
        // D_dual = I - D_primal
        for (int64_t i = 0; i < dim1; ++i)
            o_diag[i] = 1.0 - dpi_dpi[i];
        for (int64_t i = 0; i < dim2; ++i)
            o_diag[dim1 + i] = 1.0 - r_over_nw;

        // left1 = -[dpi_dr; ŵ] (negated for dual)
        for (int64_t i = 0; i < dim1; ++i)
            o_l1[i] = -dpi_dr[i];
        for (int64_t i = 0; i < dim2; ++i)
            o_l1[dim1 + i] = -w[i] / norm_w;

        // right1 = [dr_dp; dr_dnw * ŵ]
        for (int64_t i = 0; i < dim1; ++i)
            o_r1[i] = dr_dp[i];
        for (int64_t i = 0; i < dim2; ++i)
            o_r1[dim1 + i] = dr_dnw * w[i] / norm_w;

        // left2 = -[dpi_dnw; zeros] (negated for dual)
        for (int64_t i = 0; i < dim1; ++i)
            o_l2[i] = -dpi_dnw[i];
        for (int64_t i = 0; i < dim2; ++i)
            o_l2[dim1 + i] = 0.0;

        // right2 = [zeros; ŵ]
        for (int64_t i = 0; i < dim1; ++i)
            o_r2[i] = 0.0;
        for (int64_t i = 0; i < dim2; ++i)
            o_r2[dim1 + i] = w[i] / norm_w;

        // left3 = [zeros; w] (same for primal and dual)
        for (int64_t i = 0; i < dim1; ++i)
            o_l3[i] = 0.0;
        for (int64_t i = 0; i < dim2; ++i)
            o_l3[dim1 + i] = w[i];

        if (dim2 == 1) {
            // dim2=1: fold c3*w² = -r/||w|| into diagonal to avoid -r/||w||³ blowup
            // Primal diag was r/||w||, c3 term contributes -r/||w||, net = 0.
            // left1*right1 at ww gives dr_dnw. Total primal ww = dr_dnw.
            // Dual diag = 1 - 0 = 1.
            o_diag[dim1] = 1.0;
            o_l3[dim1] = 0.0;
            sparse_c3[batch * numGenPowerCones + cone] = 0.0;
        } else {
            // c3 = -(-r/||w||³) = r/||w||³ (negated c_ww for dual)
            double c_ww_primal = -r_val / (norm_w * norm_w * norm_w);
            sparse_c3[batch * numGenPowerCones + cone] = -c_ww_primal;
        }
    }
}

void compute_genpow_derivative_sparse(
    double* sparse_diag,
    double* sparse_left1,
    double* sparse_right1,
    double* sparse_left2,
    double* sparse_right2,
    double* sparse_left3,
    double* sparse_c3,
    const double* u,
    const double* alphas,
    const int64_t* d_dim1s,
    const int64_t* d_dim2s,
    const int64_t* d_offsets,
    const int64_t* d_alpha_offsets,
    int64_t offset,
    int64_t numGenPowerCones,
    int64_t totalGenpowDim,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream,
    double* work_vec,
    double* work_dim1,
    int64_t totalGenPowerAlphas,
    const int64_t* d_sz_offsets
) {
    if (numGenPowerCones == 0) return;

    int threads = min((int)numGenPowerCones, 256);
    MOREAU_KERNEL_LAUNCH(compute_genpow_derivative_sparse_kernel, batchSize, threads, 0, stream,
        sparse_diag, sparse_left1, sparse_right1, sparse_left2, sparse_right2,
        sparse_left3, sparse_c3,
        u, alphas, d_dim1s, d_dim2s, d_offsets, d_alpha_offsets, d_sz_offsets,
        offset, numGenPowerCones, totalGenpowDim, batchSize, m,
        work_vec, work_dim1, totalGenPowerAlphas
    );
}

// ============================================================================
// Backward Pass Kernels
// ============================================================================

__global__ void compute_u_from_z_s_kernel(
    double* __restrict__ u,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        int64_t idx = batch * m + i;
        u[idx] = z[idx] - s[idx];
    }
}

void compute_u_from_z_s(
    double* u,
    const double* z,
    const double* s,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (m == 0) return;
    int threads = min((int)m, 256);
    MOREAU_KERNEL_LAUNCH(compute_u_from_z_s_kernel, batchSize, threads, 0, stream,
        u, z, s, m, batchSize
    );
}

__global__ void build_adjoint_rhs_hsde_kernel(
    double* __restrict__ rhs,
    const double* __restrict__ dx_bar,
    const double* __restrict__ dz_bar,
    const double* __restrict__ ds_bar,
    const double* __restrict__ dz_x_bar,         // [batchSize * xn] equilibrated dz_x. May be nullptr or all-zero.
    const double* __restrict__ x,
    const double* __restrict__ z,
    const double* __restrict__ s,
    const double* __restrict__ z_x_eq,           // [batchSize * xn] equilibrated z_x. May be nullptr when xn==0.
    const int64_t* __restrict__ xcone_indices,   // [xn] flat primal indices J across cones (concat in spec order)
    int64_t n,
    int64_t m,
    int64_t xn,                     // direct-x dim (0 for slack-only)
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + 2 * m + xn + 1;
    double* rhs_b = rhs + batch * dim;
    const double* dx_b = dx_bar + batch * n;
    const double* dz_b = dz_bar + batch * m;
    const double* ds_b = ds_bar + batch * m;
    const double* x_b = x + batch * n;
    const double* z_b = z + batch * m;
    const double* s_b = s + batch * m;
    const double* dz_x_b = (dz_x_bar != nullptr) ? dz_x_bar + batch * xn : nullptr;
    const double* z_x_b  = (z_x_eq   != nullptr) ? z_x_eq   + batch * xn : nullptr;

    // rhs[0:n] = dx_bar  (direct-x contribution to x slot is added below)
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        rhs_b[i] = dx_b[i];
    }

    // rhs[n:n+m] = dz_bar + ds_bar (w_bar)
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        rhs_b[n + i] = dz_b[i] + ds_b[i];
    }

    // rhs[n+m:n+2m] = -ds_bar (du_bar)
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        rhs_b[n + m + i] = -ds_b[i];
    }

    // Direct-x slot: `du_x = z_x − x[J]` ⇒ `z_x_internal = du_x + x[J]`.
    // Upstream gradient on z_x flows positively to BOTH du_x slot and the
    // primal-x slot at index J[k]. When dz_x_bar is null, write zeros.
    for (int64_t k = threadIdx.x; k < xn; k += blockDim.x) {
        double dz_x_k = (dz_x_b != nullptr) ? dz_x_b[k] : 0.0;
        rhs_b[n + 2 * m + k] = dz_x_k;
    }

    // Barrier: the regular writes above (rhs_b[i] = dx_b[i] for i < n) and
    // the atomicAdd below target the same primal-x slot at index xj < n.
    // Without this sync, the atomic-add can land in a slot a different
    // warp hasn't yet written, then get clobbered by the plain store —
    // silently dropping the direct-x contribution to dq/db on backward.
    __syncthreads();

    // Atomic-add the dz_x contribution to the primal-x slot so cross-cone
    // collisions are safe (in practice indices are disjoint, but we keep
    // the kernel robust).
    if (dz_x_b != nullptr) {
        for (int64_t k = threadIdx.x; k < xn; k += blockDim.x) {
            int64_t xj = xcone_indices[k];
            atomicAdd(&rhs_b[xj], dz_x_b[k]);
        }
    }

    __syncthreads();

    // rhs[dim-1] = -dx'x - dz'z - ds's - dz_x'z_x_eq  (τ row)
    if (threadIdx.x == 0) {
        double dt_bar = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            dt_bar -= dx_b[i] * x_b[i];
        }
        for (int64_t i = 0; i < m; ++i) {
            dt_bar -= dz_b[i] * z_b[i];
            dt_bar -= ds_b[i] * s_b[i];
        }
        if (dz_x_b != nullptr && z_x_b != nullptr) {
            for (int64_t k = 0; k < xn; ++k) {
                dt_bar -= dz_x_b[k] * z_x_b[k];
            }
        }
        rhs_b[dim - 1] = dt_bar;
    }
}

void build_adjoint_rhs_hsde(
    double* rhs,
    const double* dx_bar,
    const double* dz_bar,
    const double* ds_bar,
    const double* x,
    const double* z,
    const double* s,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(build_adjoint_rhs_hsde_kernel, batchSize, threads, 0, stream,
        rhs, dx_bar, dz_bar, ds_bar,
        /*dz_x_bar=*/nullptr,
        x, z, s,
        /*z_x_eq=*/nullptr,
        /*xcone_indices=*/nullptr,
        n, m, /*xn=*/0, batchSize
    );
}

void build_adjoint_rhs_hsde_with_xcones(
    double* rhs,
    const double* dx_bar,
    const double* dz_bar,
    const double* ds_bar,
    const double* dz_x_bar,
    const double* x,
    const double* z,
    const double* s,
    const double* z_x_eq,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m + xn), 256);
    if (threads < 1) threads = 1;
    MOREAU_KERNEL_LAUNCH(build_adjoint_rhs_hsde_kernel, batchSize, threads, 0, stream,
        rhs, dx_bar, dz_bar, ds_bar, dz_x_bar,
        x, z, s, z_x_eq, xcone_indices,
        n, m, xn, batchSize
    );
}

// Convert IPM-internal direct-x dual to user/original frame:
//   z_x_user[b, k] = z_x_int[b, k] * d[J[k]] / (τ_raw[b] * c_scale[b])
// Mirrors CPU `Variables::unscale` for z_x.
__global__ void unscale_z_x_kernel(
    double* __restrict__ z_x_user,
    const double* __restrict__ z_x_int,
    const double* __restrict__ dinv,
    const double* __restrict__ c_scale,
    const double* __restrict__ tau_raw,
    const int64_t* __restrict__ xcone_indices,
    int64_t n,
    int64_t total_xn,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* dinv_b = dinv + batch * n;
    const double inv_tau_c = 1.0 / (tau_raw[batch] * c_scale[batch]);
    for (int64_t k = threadIdx.x; k < total_xn; k += blockDim.x) {
        int64_t idx = xcone_indices[k];
        double d_j = 1.0 / dinv_b[idx];
        z_x_user[batch * total_xn + k] = z_x_int[batch * total_xn + k] * d_j * inv_tau_c;
    }
}

void unscale_z_x(
    double* z_x_user,
    const double* z_x_int,
    const double* dinv,
    const double* c_scale,
    const double* tau_raw,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t total_xn,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (total_xn <= 0 || batchSize <= 0) return;
    int threads = min((int)total_xn, 256);
    if (threads < 1) threads = 1;
    MOREAU_KERNEL_LAUNCH(unscale_z_x_kernel, batchSize, threads, 0, stream,
        z_x_user, z_x_int, dinv, c_scale, tau_raw, xcone_indices,
        n, total_xn, batchSize
    );
}

// Equilibrate user-frame z_x to the equilibrated τ=1 frame:
//   z_x_eq[b, k] = z_x_user[b, k] * c_scale[b] / d[J[k]]
// (Inverse of `z_x_user = z_x_eq * d[J] / c`.) Mirrors CPU
// solver.rs:694 `z_x_eq = self.variables.z_x[k] * c / d[J[k]]`.
__global__ void equilibrate_z_x_kernel(
    double* __restrict__ z_x_eq,
    const double* __restrict__ z_x_user,
    const double* __restrict__ dinv,
    const double* __restrict__ c_scale,
    const int64_t* __restrict__ xcone_indices,
    int64_t n,
    int64_t xn,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* dinv_b = dinv + batch * n;
    const double c_b = c_scale[batch];
    for (int64_t k = threadIdx.x; k < xn; k += blockDim.x) {
        int64_t idx = xcone_indices[k];
        // d[J] = 1 / dinv[J]; z_x_eq = z_x_user * c / d[J] = z_x_user * c * dinv[J].
        z_x_eq[batch * xn + k] = z_x_user[batch * xn + k] * c_b * dinv_b[idx];
    }
}

void equilibrate_z_x(
    double* z_x_eq,
    const double* z_x_user,
    const double* dinv,
    const double* c_scale,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (xn <= 0 || batchSize <= 0) return;
    int threads = min((int)xn, 256);
    if (threads < 1) threads = 1;
    MOREAU_KERNEL_LAUNCH(equilibrate_z_x_kernel, batchSize, threads, 0, stream,
        z_x_eq, z_x_user, dinv, c_scale, xcone_indices, n, xn, batchSize
    );
}

// Equilibrate user-frame dz_x to the equilibrated frame:
//   dz_x_eq[b, k] = dz_x_bar[b, k] * d[J[k]] / c_scale[b]
// Mirrors `dz_x_eq = dz_x_user * d[J] / c` chain rule on
// `z_x_user = z_x_eq * d[J] / c` from CPU `Variables::unscale`.
__global__ void equilibrate_dz_x_kernel(
    double* __restrict__ dz_x_eq,                  // [batchSize * xn]
    const double* __restrict__ dz_x_bar,           // [batchSize * xn]
    const double* __restrict__ dinv,               // [batchSize * n] (per-batch)
    const double* __restrict__ c_scale,            // [batchSize]
    const int64_t* __restrict__ xcone_indices,     // [xn]
    int64_t n,
    int64_t xn,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* dinv_b = dinv + batch * n;
    const double inv_c = 1.0 / c_scale[batch];

    double* out_b = dz_x_eq + batch * xn;
    const double* in_b = dz_x_bar + batch * xn;

    for (int64_t k = threadIdx.x; k < xn; k += blockDim.x) {
        int64_t idx = xcone_indices[k];
        double d_j = 1.0 / dinv_b[idx];
        out_b[k] = in_b[k] * d_j * inv_c;
    }
}

void equilibrate_dz_x(
    double* dz_x_eq,
    const double* dz_x_bar,
    const double* dinv,
    const double* c_scale,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (xn <= 0 || batchSize <= 0) return;
    int threads = min((int)xn, 256);
    if (threads < 1) threads = 1;
    MOREAU_KERNEL_LAUNCH(equilibrate_dz_x_kernel, batchSize, threads, 0, stream,
        dz_x_eq, dz_x_bar, dinv, c_scale, xcone_indices, n, xn, batchSize
    );
}

__global__ void build_adjoint_rhs_qp_eq_kernel(
    double* __restrict__ rhs,
    const double* __restrict__ dx_bar,
    const double* __restrict__ dz_bar,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + m;
    double* rhs_b = rhs + batch * dim;
    const double* dx_b = dx_bar + batch * n;
    const double* dz_b = dz_bar + batch * m;

    // rhs[0:n] = dx_bar
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        rhs_b[i] = dx_b[i];
    }

    // rhs[n:n+m] = dz_bar
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        rhs_b[n + i] = dz_b[i];
    }
}

void build_adjoint_rhs_qp_eq(
    double* rhs,
    const double* dx_bar,
    const double* dz_bar,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(build_adjoint_rhs_qp_eq_kernel, batchSize, threads, 0, stream,
        rhs, dx_bar, dz_bar, n, m, batchSize
    );
}

__global__ void extract_gradients_hsde_kernel(
    double* __restrict__ dq,
    double* __restrict__ db,
    const double* __restrict__ lambda,
    const double* __restrict__ x,
    const double* __restrict__ z,
    const double* __restrict__ tau,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + 2 * m + xn + 1;
    const double* lam_b = lambda + batch * dim;
    const double* x_b = x + batch * n;
    const double* z_b = z + batch * m;
    double tau_b = tau[batch];

    double* dq_b = dq + batch * n;
    double* db_b = db + batch * m;

    // λ₁ = lam[0:n], λ₂ = lam[n:n+m], λ_τ = lam[n+2m+xn]
    double lam4 = lam_b[dim - 1];

    // dq_bar = -τ * λ₁ + λ_τ * x
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dq_b[i] = -tau_b * lam_b[i] + lam4 * x_b[i];
    }

    // db_bar = τ * λ₂ + λ_τ * z
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        db_b[i] = tau_b * lam_b[n + i] + lam4 * z_b[i];
    }
}

void extract_gradients_hsde(
    double* dq,
    double* db,
    const double* lambda,
    const double* x,
    const double* z,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(extract_gradients_hsde_kernel, batchSize, threads, 0, stream,
        dq, db, lambda, x, z, tau, n, m, /*xn=*/0, batchSize
    );
}

void extract_gradients_hsde_with_xcones(
    double* dq,
    double* db,
    const double* lambda,
    const double* x,
    const double* z,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(extract_gradients_hsde_kernel, batchSize, threads, 0, stream,
        dq, db, lambda, x, z, tau, n, m, xn, batchSize
    );
}

__global__ void extract_gradients_qp_eq_kernel(
    double* __restrict__ dq,
    double* __restrict__ db,
    const double* __restrict__ lambda,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + m;
    const double* lam_b = lambda + batch * dim;
    double* dq_b = dq + batch * n;
    double* db_b = db + batch * m;

    // dq_bar = -λ_x
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dq_b[i] = -lam_b[i];
    }

    // db_bar = λ_z
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        db_b[i] = lam_b[n + i];
    }
}

void extract_gradients_qp_eq(
    double* dq,
    double* db,
    const double* lambda,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(extract_gradients_qp_eq_kernel, batchSize, threads, 0, stream,
        dq, db, lambda, n, m, batchSize
    );
}

__global__ void compute_dP_gradient_hsde_kernel(
    double* __restrict__ dP_values,
    const int64_t* __restrict__ P_row_offsets,
    const int64_t* __restrict__ P_col_indices,
    const double* __restrict__ lambda,
    const double* __restrict__ x,
    const double* __restrict__ tau,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t nnzP,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + 2 * m + xn + 1;
    const double* lam_b = lambda + batch * dim;
    const double* x_b = x + batch * n;
    double tau_b = tau[batch];
    double lam4 = lam_b[dim - 1];

    double* dP_b = dP_values + batch * nnzP;

    // Iterate over nonzeros in P
    // P is a full symmetric matrix (both upper and lower triangle entries stored).
    // Formula from diffclarabel:
    //   dP[i,j] = -0.5 * (τ*λ₁[i]*x[j] + τ*λ₁[j]*x[i]) + λ₄*x[i]*x[j]
    // This gives the same value for both P[i,j] and P[j,i].
    for (int64_t row = 0; row < n; ++row) {
        int64_t row_start = P_row_offsets[row];
        int64_t row_end = P_row_offsets[row + 1];

        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = P_col_indices[k];

            // dP[row,col] = -0.5 * (τ*λ₁[row]*x[col] + τ*λ₁[col]*x[row]) + λ₄*x[row]*x[col]
            double val = -0.5 * tau_b * (lam_b[row] * x_b[col] + lam_b[col] * x_b[row]);
            val += lam4 * x_b[row] * x_b[col];

            dP_b[k] = val;
        }
    }
}

void compute_dP_gradient_hsde(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* lambda,
    const double* x,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzP == 0) return;
    int threads = min((int)nnzP, 256);
    MOREAU_KERNEL_LAUNCH(compute_dP_gradient_hsde_kernel, batchSize, threads, 0, stream,
        dP_values, P_row_offsets, P_col_indices, lambda, x, tau,
        n, m, /*xn=*/0, nnzP, batchSize
    );
}

void compute_dP_gradient_hsde_with_xcones(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* lambda,
    const double* x,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzP == 0) return;
    int threads = min((int)nnzP, 256);
    MOREAU_KERNEL_LAUNCH(compute_dP_gradient_hsde_kernel, batchSize, threads, 0, stream,
        dP_values, P_row_offsets, P_col_indices, lambda, x, tau,
        n, m, xn, nnzP, batchSize
    );
}

__global__ void compute_dP_gradient_qp_eq_kernel(
    double* __restrict__ dP_values,
    const int64_t* __restrict__ P_row_offsets,
    const int64_t* __restrict__ P_col_indices,
    const double* __restrict__ lambda_x,
    const double* __restrict__ x,
    int64_t n,
    int64_t lambda_stride,
    int64_t nnzP,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // lambda_x is stored with stride lambda_stride between batches
    // (e.g., from sol which has n+m elements per batch)
    const double* lam_x_b = lambda_x + batch * lambda_stride;
    const double* x_b = x + batch * n;
    double* dP_b = dP_values + batch * nnzP;

    // P is a full symmetric matrix (both upper and lower triangle entries stored).
    // Formula from diffclarabel:
    //   dP[i,j] = -0.5 * (λ_x[i]*x[j] + λ_x[j]*x[i])
    // This gives the same value for both P[i,j] and P[j,i].
    for (int64_t row = 0; row < n; ++row) {
        int64_t row_start = P_row_offsets[row];
        int64_t row_end = P_row_offsets[row + 1];

        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = P_col_indices[k];

            // dP[row,col] = -0.5 * (λ_x[row]*x[col] + λ_x[col]*x[row])
            double val = -0.5 * (lam_x_b[row] * x_b[col] + lam_x_b[col] * x_b[row]);
            dP_b[k] = val;
        }
    }
}

void compute_dP_gradient_qp_eq(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* lambda_x,
    const double* x,
    int64_t n,
    int64_t lambda_stride,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzP == 0) return;
    int threads = min((int)nnzP, 256);
    MOREAU_KERNEL_LAUNCH(compute_dP_gradient_qp_eq_kernel, batchSize, threads, 0, stream,
        dP_values, P_row_offsets, P_col_indices, lambda_x, x, n, lambda_stride, nnzP, batchSize
    );
}

__global__ void compute_dA_gradient_hsde_kernel(
    double* __restrict__ dA_values,
    const int64_t* __restrict__ A_row_offsets,
    const int64_t* __restrict__ A_col_indices,
    const double* __restrict__ lambda,
    const double* __restrict__ x,
    const double* __restrict__ z,
    const double* /*tau*/,  // unused, kept for API compatibility
    int64_t n,
    int64_t m,
    int64_t lambda_stride,  // stride between batches in lambda
    int64_t z_stride,       // stride between batches in z
    int64_t nnzA,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    const double* lam_b = lambda + batch * lambda_stride;
    const double* lam1_b = lam_b;        // λ₁ at offset 0 (size n)
    const double* lam2_b = lam_b + n;    // λ₂ at offset n (size m)
    const double* x_b = x + batch * n;
    const double* z_b = z + batch * z_stride;

    double* dA_b = dA_values + batch * nnzA;

    // A is m x n
    // dA[row, col] = -λ₁[col] * z[row] - λ₂[row] * x[col]
    // This matches the CPU formula: dA[i,j] = -lam1[j] * z[i] - lam2[i] * x[j]
    for (int64_t row = 0; row < m; ++row) {
        int64_t row_start = A_row_offsets[row];
        int64_t row_end = A_row_offsets[row + 1];

        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = A_col_indices[k];

            double val = -lam1_b[col] * z_b[row] - lam2_b[row] * x_b[col];
            dA_b[k] = val;
        }
    }
}

void compute_dA_gradient_hsde(
    double* dA_values,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* lambda,
    const double* x,
    const double* z,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t lambda_stride,
    int64_t z_stride,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzA == 0) return;
    int threads = min((int)nnzA, 256);
    MOREAU_KERNEL_LAUNCH(compute_dA_gradient_hsde_kernel, batchSize, threads, 0, stream,
        dA_values, A_row_offsets, A_col_indices, lambda, x, z, tau, n, m, lambda_stride, z_stride, nnzA, batchSize
    );
}

__global__ void compute_dA_gradient_qp_eq_kernel(
    double* __restrict__ dA_values,
    const int64_t* __restrict__ A_row_offsets,
    const int64_t* __restrict__ A_col_indices,
    const double* __restrict__ sol,
    const double* __restrict__ x,
    const double* __restrict__ z,
    int64_t n,
    int64_t m,
    int64_t sol_stride,
    int64_t nnzA,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // sol is laid out as [lam_x (n), lam_z (m)] per batch with stride sol_stride
    const double* sol_b = sol + batch * sol_stride;
    const double* lam_x_b = sol_b;           // lam_x at offset 0
    const double* lam_z_b = sol_b + n;       // lam_z at offset n
    const double* x_b = x + batch * n;
    const double* z_b = z + batch * m;
    double* dA_b = dA_values + batch * nnzA;

    for (int64_t row = 0; row < m; ++row) {
        int64_t row_start = A_row_offsets[row];
        int64_t row_end = A_row_offsets[row + 1];

        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = A_col_indices[k];

            // dA[row, col] = -λ_z[row] * x[col] - λ_x[col] * z[row]
            double val = -lam_z_b[row] * x_b[col] - lam_x_b[col] * z_b[row];
            dA_b[k] = val;
        }
    }
}

void compute_dA_gradient_qp_eq(
    double* dA_values,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* sol,
    const double* x,
    const double* z,
    int64_t n,
    int64_t m,
    int64_t sol_stride,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzA == 0) return;
    int threads = min((int)nnzA, 256);
    MOREAU_KERNEL_LAUNCH(compute_dA_gradient_qp_eq_kernel, batchSize, threads, 0, stream,
        dA_values, A_row_offsets, A_col_indices, sol, x, z, n, m, sol_stride, nnzA, batchSize
    );
}

// ============================================================================
// Equilibration Kernels
// ============================================================================

__global__ void transform_upstream_grads_kernel(
    double* __restrict__ dx_eq,
    double* __restrict__ dz_eq,
    double* __restrict__ ds_eq,
    const double* __restrict__ dx,
    const double* __restrict__ dz,
    const double* __restrict__ ds,
    const double* __restrict__ d,
    const double* __restrict__ e,
    const double* __restrict__ einv,
    const double* __restrict__ c,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double c_val = c[batch];

    // dx_eq = D * dx
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dx_eq[batch * n + i] = d[batch * n + i] * dx[batch * n + i];
    }

    // dz_eq = E / c * dz
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        dz_eq[batch * m + i] = e[batch * m + i] / c_val * dz[batch * m + i];
    }

    // ds_eq = E^{-1} * ds
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        ds_eq[batch * m + i] = einv[batch * m + i] * ds[batch * m + i];
    }
}

void transform_upstream_grads_to_equilibrated(
    double* dx_eq,
    double* dz_eq,
    double* ds_eq,
    const double* dx,
    const double* dz,
    const double* ds,
    const double* d,
    const double* e,
    const double* einv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(transform_upstream_grads_kernel, batchSize, threads, 0, stream,
        dx_eq, dz_eq, ds_eq, dx, dz, ds, d, e, einv, c, n, m, batchSize
    );
}

__global__ void transform_output_grads_kernel(
    double* __restrict__ dP_values,
    double* __restrict__ dq,
    double* __restrict__ dA_values,
    double* __restrict__ db,
    const double* __restrict__ dP_eq,
    const double* __restrict__ dq_eq,
    const double* __restrict__ dA_eq,
    const double* __restrict__ db_eq,
    const int64_t* __restrict__ P_row_offsets,
    const int64_t* __restrict__ P_col_indices,
    const int64_t* __restrict__ A_row_offsets,
    const int64_t* __restrict__ A_col_indices,
    const double* __restrict__ d,
    const double* __restrict__ e,
    const double* __restrict__ c,
    int64_t n,
    int64_t m,
    int64_t nnzP,
    int64_t nnzA,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double c_val = c[batch];
    const double* d_b = d + batch * n;
    const double* e_b = e + batch * m;

    // dq += c * D * dq_eq
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dq[batch * n + i] += c_val * d_b[i] * dq_eq[batch * n + i];
    }

    // db += E * db_eq
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        db[batch * m + i] += e_b[i] * db_eq[batch * m + i];
    }

    // dP += c * D * dP_eq * D (at each nonzero position)
    for (int64_t row = 0; row < n; ++row) {
        int64_t row_start = P_row_offsets[row];
        int64_t row_end = P_row_offsets[row + 1];
        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = P_col_indices[k];
            dP_values[batch * nnzP + k] += c_val * d_b[row] * dP_eq[batch * nnzP + k] * d_b[col];
        }
    }

    // dA += E * dA_eq * D (at each nonzero position)
    for (int64_t row = 0; row < m; ++row) {
        int64_t row_start = A_row_offsets[row];
        int64_t row_end = A_row_offsets[row + 1];
        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = A_col_indices[k];
            dA_values[batch * nnzA + k] += e_b[row] * dA_eq[batch * nnzA + k] * d_b[col];
        }
    }
}

void transform_output_grads_to_original(
    double* dP_values,
    double* dq,
    double* dA_values,
    double* db,
    const double* dP_eq,
    const double* dq_eq,
    const double* dA_eq,
    const double* db_eq,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* d,
    const double* e,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t nnzP,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m + nnzP + nnzA), 256);
    MOREAU_KERNEL_LAUNCH(transform_output_grads_kernel, batchSize, threads, 0, stream,
        dP_values, dq, dA_values, db, dP_eq, dq_eq, dA_eq, db_eq,
        P_row_offsets, P_col_indices, A_row_offsets, A_col_indices,
        d, e, c, n, m, nnzP, nnzA, batchSize
    );
}

// ============================================================================
// HSDE Coefficient Kernel
// ============================================================================

__global__ void compute_hsde_coefficients_kernel(
    double* __restrict__ c1,
    double* __restrict__ c2,
    double* __restrict__ c3,
    const double* __restrict__ Px,
    const double* __restrict__ x,
    const double* __restrict__ q,
    const double* __restrict__ b,
    const double* __restrict__ tau,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double tau_b = tau[batch];
    double two_over_tau = 2.0 / tau_b;

    // c1[i] = -(2/τ) * (Px)[i] - q[i]
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        c1[batch * n + i] = -two_over_tau * Px[batch * n + i] - q[batch * n + i];
    }

    // c2[i] = -b[i]
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        c2[batch * m + i] = -b[batch * m + i];
    }

    // c3 = x'Px / τ² (computed by thread 0 for simplicity since n is typically small)
    if (threadIdx.x == 0) {
        double xPx = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            xPx += x[batch * n + i] * Px[batch * n + i];
        }
        c3[batch] = xPx / (tau_b * tau_b);
    }
}

void compute_hsde_coefficients(
    double* c1,
    double* c2,
    double* c3,
    const double* Px,
    const double* x,
    const double* q,
    const double* b,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(compute_hsde_coefficients_kernel, batchSize, threads, 0, stream,
        c1, c2, c3, Px, x, q, b, tau, n, m, batchSize
    );
}

// ============================================================================
// Forward Pass Recovery Kernel
// ============================================================================

__global__ void recover_solution_derivatives_kernel(
    double* __restrict__ dx,
    double* __restrict__ dz,
    double* __restrict__ ds,
    const double* __restrict__ sol,
    const double* __restrict__ x,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + 2 * m + 1;
    const double* sol_b = sol + batch * dim;
    const double* x_b = x + batch * n;
    const double* z_b = z + batch * m;
    const double* s_b = s + batch * m;

    double* dx_b = dx + batch * n;
    double* dz_b = dz + batch * m;
    double* ds_b = ds + batch * m;

    // sol = [dz_x, w, du, dt]
    const double* dz_x = sol_b;           // [0:n]
    const double* w = sol_b + n;          // [n:n+m]
    const double* du = sol_b + n + m;     // [n+m:n+2m]
    double dt = sol_b[dim - 1];           // [n+2m]

    // dx = dz_x - dt * x
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dx_b[i] = dz_x[i] - dt * x_b[i];
    }

    // dz = w - dt * z
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        dz_b[i] = w[i] - dt * z_b[i];
    }

    // ds = w - du - dt * s
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        ds_b[i] = w[i] - du[i] - dt * s_b[i];
    }
}

void recover_solution_derivatives(
    double* dx,
    double* dz,
    double* ds,
    const double* sol,
    const double* x,
    const double* z,
    const double* s,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(recover_solution_derivatives_kernel, batchSize, threads, 0, stream,
        dx, dz, ds, sol, x, z, s, n, m, batchSize
    );
}

// ============================================================================
// QP Equality with Equilibration Kernels
// ============================================================================

__global__ void extract_gradients_qp_eq_with_equilibration_kernel(
    double* __restrict__ dq,
    double* __restrict__ db,
    const double* __restrict__ lambda,
    const double* __restrict__ dinv,
    const double* __restrict__ einv,
    const double* __restrict__ c,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + m;
    const double* lam_b = lambda + batch * dim;
    const double* dinv_b = dinv + batch * n;
    const double* einv_b = einv + batch * m;
    double c_val = c[batch];

    double* dq_b = dq + batch * n;
    double* db_b = db + batch * m;

    // dq = -c * D * lam_x_eq = -c * lam_x / dinv
    // Since D = 1/dinv, we have: dq[i] = -c * lam_x[i] / dinv[i]
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dq_b[i] = -c_val * lam_b[i] / dinv_b[i];
    }

    // db = E * lam_z_eq = lam_z / einv
    // Since E = 1/einv, we have: db[i] = lam_z[i] / einv[i]
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        db_b[i] = lam_b[n + i] / einv_b[i];
    }
}

void extract_gradients_qp_eq_with_equilibration(
    double* dq,
    double* db,
    const double* lambda,
    const double* dinv,
    const double* einv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)(n + m), 256);
    MOREAU_KERNEL_LAUNCH(extract_gradients_qp_eq_with_equilibration_kernel, batchSize, threads, 0, stream,
        dq, db, lambda, dinv, einv, c, n, m, batchSize
    );
}

__global__ void transform_dP_from_equilibrated_kernel(
    double* __restrict__ dP_values,
    const int64_t* __restrict__ P_row_offsets,
    const int64_t* __restrict__ P_col_indices,
    const double* __restrict__ dinv,
    const double* __restrict__ c,
    int64_t n,
    int64_t nnzP,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    const double* dinv_b = dinv + batch * n;
    double c_val = c[batch];
    double* dP_b = dP_values + batch * nnzP;

    // dP[i,j] = c * d[i] * d[j] * dP_eq[i,j] = c / dinv[i] / dinv[j] * dP_eq[i,j]
    for (int64_t row = 0; row < n; ++row) {
        int64_t row_start = P_row_offsets[row];
        int64_t row_end = P_row_offsets[row + 1];

        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = P_col_indices[k];
            dP_b[k] *= c_val / (dinv_b[row] * dinv_b[col]);
        }
    }
}

void transform_dP_from_equilibrated(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* dinv,
    const double* c,
    int64_t n,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzP == 0) return;
    int threads = min((int)nnzP, 256);
    MOREAU_KERNEL_LAUNCH(transform_dP_from_equilibrated_kernel, batchSize, threads, 0, stream,
        dP_values, P_row_offsets, P_col_indices, dinv, c, n, nnzP, batchSize
    );
}

__global__ void transform_dA_from_equilibrated_kernel(
    double* __restrict__ dA_values,
    const int64_t* __restrict__ A_row_offsets,
    const int64_t* __restrict__ A_col_indices,
    const double* __restrict__ dinv,
    const double* __restrict__ einv,
    int64_t n,
    int64_t m,
    int64_t nnzA,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    const double* dinv_b = dinv + batch * n;
    const double* einv_b = einv + batch * m;
    double* dA_b = dA_values + batch * nnzA;

    // dA[i,j] = e[i] * d[j] * dA_eq[i,j] = dA_eq[i,j] / einv[i] / dinv[j]
    for (int64_t row = 0; row < m; ++row) {
        int64_t row_start = A_row_offsets[row];
        int64_t row_end = A_row_offsets[row + 1];

        for (int64_t k = row_start + threadIdx.x; k < row_end; k += blockDim.x) {
            int64_t col = A_col_indices[k];
            dA_b[k] /= (einv_b[row] * dinv_b[col]);
        }
    }
}

void transform_dA_from_equilibrated(
    double* dA_values,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* dinv,
    const double* einv,
    int64_t n,
    int64_t m,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (nnzA == 0) return;
    int threads = min((int)nnzA, 256);
    MOREAU_KERNEL_LAUNCH(transform_dA_from_equilibrated_kernel, batchSize, threads, 0, stream,
        dA_values, A_row_offsets, A_col_indices, dinv, einv, n, m, nnzA, batchSize
    );
}

// ============================================================================
// Forward Pass Kernels
// ============================================================================

__global__ void build_forward_rhs_qp_eq_kernel(
    double* __restrict__ rhs,
    const double* __restrict__ dP_x,      // dP * x [batchSize][n]
    const double* __restrict__ dAt_z,     // dA' * z [batchSize][n]
    const double* __restrict__ dA_x,      // dA * x [batchSize][m]
    const double* __restrict__ dq,        // [batchSize][n]
    const double* __restrict__ db,        // [batchSize][m]
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t dim = n + m;
    double* out = rhs + batch * dim;
    const double* dP_x_b = dP_x + batch * n;
    const double* dAt_z_b = dAt_z + batch * n;
    const double* dA_x_b = dA_x + batch * m;
    const double* dq_b = dq + batch * n;
    const double* db_b = db + batch * m;

    // rhs1[i] = -(dq[i] + dP*x[i] + dA'*z[i]) for i = 0..n-1
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = -(dq_b[i] + dP_x_b[i] + dAt_z_b[i]);
    }

    // rhs2[i] = db[i] - dA*x[i] for i = 0..m-1
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[n + i] = db_b[i] - dA_x_b[i];
    }
}

void build_forward_rhs_qp_eq(
    double* rhs,
    const double* dP_x,
    const double* dAt_z,
    const double* dA_x,
    const double* dq,
    const double* db,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = 256;
    MOREAU_KERNEL_LAUNCH(build_forward_rhs_qp_eq_kernel, batchSize, threads, 0, stream,
        rhs, dP_x, dAt_z, dA_x, dq, db, n, m, batchSize
    );
}

/**
 * Compute x' * dP * x for each batch (scalar reduction).
 * Uses x and dP*x (already computed) to compute the quadratic form.
 */
__global__ void compute_x_dP_x_kernel(
    double* __restrict__ result,          // [batchSize]
    const double* __restrict__ dP_x,      // dP * x [batchSize][n]
    const double* __restrict__ x,         // [batchSize][n]
    int64_t n,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    __shared__ double sdata[256];

    const double* dP_x_b = dP_x + batch * n;
    const double* x_b = x + batch * n;

    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x_b[i] * dP_x_b[i];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[batch] = sdata[0];
    }
}

void compute_x_dP_x(
    double* result,
    const double* dP_x,
    const double* x,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = min((int)n, 256);
    MOREAU_KERNEL_LAUNCH(compute_x_dP_x_kernel, batchSize, threads, 0, stream,
        result, dP_x, x, n, batchSize
    );
}

// ============================================================================
// Forward Pass Kernels for HSDE
// ============================================================================

/**
 * Build forward RHS for HSDE system.
 *
 * The forward HSDE system solves:
 * [P    A'   0   q ] [dz_x]   [r1]
 * [A    I   -I  -b ] [w   ] = [r2]
 * [0    I   -H   0 ] [du  ]   [0 ]
 * [c1   c2   0  c3 ] [dt  ]   [r3]
 *
 * Where:
 * r1 = -(dP*x + dA'*z + dq*τ)
 * r2 = -(dA*x - db*τ)
 * r3 = x'*dP*x/τ + dq'*x + db'*z  (note: uses z, not pi_u, to match backward)
 */
__global__ void build_forward_rhs_hsde_kernel(
    double* __restrict__ rhs,
    const double* __restrict__ dP_x,      // dP * x [batchSize][n]
    const double* __restrict__ dAt_z,     // dA' * z [batchSize][n]
    const double* __restrict__ dA_x,      // dA * x [batchSize][m]
    const double* __restrict__ dq,        // [batchSize][n]
    const double* __restrict__ db,        // [batchSize][m]
    const double* __restrict__ x,         // [batchSize][n]
    const double* __restrict__ z,         // [batchSize][m] - use z instead of pi_u to match backward
    const double* __restrict__ tau,       // [batchSize][1]
    const double* __restrict__ x_dP_x,    // x' * dP * x [batchSize][1]
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int64_t jdim = n + 2 * m + 1;
    double* out = rhs + batch * jdim;
    const double* dP_x_b = dP_x + batch * n;
    const double* dAt_z_b = dAt_z + batch * n;
    const double* dA_x_b = dA_x + batch * m;
    const double* dq_b = dq + batch * n;
    const double* db_b = db + batch * m;
    const double* x_b = x + batch * n;
    const double* z_b = z + batch * m;
    double tau_val = tau[batch];

    // r1[i] = -(dP*x[i] + dA'*z[i] + dq[i]*τ) for i = 0..n-1
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = -(dP_x_b[i] + dAt_z_b[i] + dq_b[i] * tau_val);
    }

    // r2[i] = -(dA*x[i] - db[i]*τ) = -dA*x[i] + db[i]*τ for i = 0..m-1
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[n + i] = -dA_x_b[i] + db_b[i] * tau_val;
    }

    // r3_block = 0 (rows n+m to n+2m-1)
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[n + m + i] = 0.0;
    }

    // r3 = x'*dP*x/τ + dq'*x + db'*z (last element)
    // Use shared memory for reductions
    __shared__ double dqx_shared[256];
    __shared__ double dbz_shared[256];

    double dqx_local = 0.0;
    double dbz_local = 0.0;

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dqx_local += dq_b[i] * x_b[i];
    }
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        dbz_local += db_b[i] * z_b[i];
    }

    dqx_shared[threadIdx.x] = dqx_local;
    dbz_shared[threadIdx.x] = dbz_local;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            dqx_shared[threadIdx.x] += dqx_shared[threadIdx.x + stride];
            dbz_shared[threadIdx.x] += dbz_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double r3 = x_dP_x[batch] / tau_val + dqx_shared[0] + dbz_shared[0];
        out[jdim - 1] = r3;
    }
}

void build_forward_rhs_hsde(
    double* rhs,
    const double* dP_x,
    const double* dAt_z,
    const double* dA_x,
    const double* dq,
    const double* db,
    const double* x,
    const double* pi_u,  // Note: parameter is named pi_u but we use z (passed as pi_u) to match backward
    const double* tau,
    const double* x_dP_x,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threads = 256;
    MOREAU_KERNEL_LAUNCH(build_forward_rhs_hsde_kernel, batchSize, threads, 0, stream,
        rhs, dP_x, dAt_z, dA_x, dq, db, x, pi_u, tau, x_dP_x, n, m, batchSize
    );
}

// ============================================================================
// Fused Projection + Derivative Kernels (Item 8)
// ============================================================================
// These kernels compute u = z - s, projection, and derivative in one pass
// per cone type, avoiding redundant memory reads and (for exp/power cones)
// redundant Newton iterations.

// Fused zero cone: u = z - s, pi = u (identity projection, no derivative)
__global__ void fused_zero_proj_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t numZeroCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        int64_t idx = batch * m + i;
        double u_val = z[idx] - s[idx];
        u[idx] = u_val;
        pi_u[idx] = u_val;  // Identity projection for zero cone dual
    }
}

// Fused nonneg: u = z - s, pi = max(u, 0), H = (u >= 0 ? 1 : 0)
__global__ void fused_nonneg_proj_deriv_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    double* __restrict__ H_diag,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t idx = batch * m + offset + i;
        double u_val = z[idx] - s[idx];
        u[idx] = u_val;
        pi_u[idx] = fmax(u_val, 0.0);
        H_diag[batch * numNonnegCones + i] = (u_val >= 0.0) ? 1.0 : 0.0;
    }
}

// Fused SOC: u = z - s, SOC projection, SOC derivative (variable dim)
// Dense cones (dim<=4): stores upper-tri in H with per-cone offsets
// Sparse cones (dim>4): stores diag + rank-2 decomposition
__global__ void fused_soc_proj_deriv_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    // Dense derivative output (dim<=4)
    double* __restrict__ H,
    const int64_t* __restrict__ d_soc_Hs_offsets,
    int64_t totalDenseSocHsEntries,
    // Sparse derivative output (dim>4)
    double* __restrict__ sparse_diag,
    double* __restrict__ sparse_v1,
    double* __restrict__ sparse_v2,
    double* __restrict__ sparse_c1,
    double* __restrict__ sparse_c2,
    const int64_t* __restrict__ d_soc_sparse_indices,
    const int64_t* __restrict__ d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    // Common params
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_sz_offsets,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numSocCones; cone += blockDim.x) {
        int64_t dim = d_soc_dims[cone];
        int64_t sz_off = d_soc_sz_offsets ? d_soc_sz_offsets[cone] : d_soc_offsets[cone];
        int64_t base = batch * m + offset + sz_off;

        // Compute u = z - s and ||x[1:]||
        double t = z[base + 0] - s[base + 0];
        u[base + 0] = t;
        double norm_x_sq = 0.0;
        for (int64_t i = 1; i < dim; ++i) {
            double ui = z[base + i] - s[base + i];
            u[base + i] = ui;
            norm_x_sq += ui * ui;
        }
        double norm_x = safe_sqrt(norm_x_sq);

        // Determine if this is a sparse or dense cone
        bool is_sparse = (d_soc_sparse_indices != nullptr && d_soc_sparse_indices[cone] >= 0);

        if (t >= norm_x) {
            // Interior: projection = identity, derivative = I
            pi_u[base + 0] = u[base + 0];
            for (int64_t i = 1; i < dim; ++i) {
                pi_u[base + i] = u[base + i];
            }
            if (is_sparse) {
                int64_t si = d_soc_sparse_indices[cone];
                int64_t sp_off = d_soc_sparse_offsets[cone];
                int64_t s_base = batch * totalSparseSocDim + sp_off;
                for (int64_t i = 0; i < dim; ++i) {
                    sparse_diag[s_base + i] = 1.0;
                    sparse_v1[s_base + i] = 0.0;
                    sparse_v2[s_base + i] = 0.0;
                }
                sparse_c1[batch * numSparseSoc + si] = 0.0;
                sparse_c2[batch * numSparseSoc + si] = 0.0;
            } else {
                int64_t hs_off = d_soc_Hs_offsets[cone];
                int64_t h_base = batch * totalDenseSocHsEntries + hs_off;
                int64_t idx = 0;
                for (int64_t r = 0; r < dim; ++r) {
                    for (int64_t c = r; c < dim; ++c) {
                        H[h_base + idx] = (r == c) ? 1.0 : 0.0;
                        ++idx;
                    }
                }
            }
        } else if (t <= -norm_x) {
            // Negative cone: projection = 0, derivative = 0
            for (int64_t i = 0; i < dim; ++i) {
                pi_u[base + i] = 0.0;
            }
            if (is_sparse) {
                int64_t si = d_soc_sparse_indices[cone];
                int64_t sp_off = d_soc_sparse_offsets[cone];
                int64_t s_base = batch * totalSparseSocDim + sp_off;
                for (int64_t i = 0; i < dim; ++i) {
                    sparse_diag[s_base + i] = 0.0;
                    sparse_v1[s_base + i] = 0.0;
                    sparse_v2[s_base + i] = 0.0;
                }
                sparse_c1[batch * numSparseSoc + si] = 0.0;
                sparse_c2[batch * numSparseSoc + si] = 0.0;
            } else {
                int64_t hs_off = d_soc_Hs_offsets[cone];
                int64_t h_base = batch * totalDenseSocHsEntries + hs_off;
                int64_t hs_count = dim * (dim + 1) / 2;
                for (int64_t i = 0; i < hs_count; ++i) {
                    H[h_base + i] = 0.0;
                }
            }
        } else {
            // Boundary case
            double scale = 0.5 * (t + norm_x);
            double inv_norm = safe_div(1.0, norm_x, 0.0);
            pi_u[base + 0] = scale;
            for (int64_t i = 1; i < dim; ++i) {
                pi_u[base + i] = scale * u[base + i] * inv_norm;
            }

            double t_over_norm = t * inv_norm;

            if (is_sparse) {
                // Sparse: H = diag(d) + c1*v1*v1^T + c2*v2*v2^T
                // d = (0, α, ..., α), α = (t + ||x||) / (2*||x||)
                // v1 = (1, û), c1 = 0.5; v2 = (0, û), c2 = -α
                int64_t si = d_soc_sparse_indices[cone];
                int64_t sp_off = d_soc_sparse_offsets[cone];
                int64_t s_base = batch * totalSparseSocDim + sp_off;
                double alpha = 0.5 * (t + norm_x) * inv_norm;

                sparse_diag[s_base] = 0.0;
                sparse_v1[s_base] = 1.0;
                sparse_v2[s_base] = 0.0;
                for (int64_t i = 1; i < dim; ++i) {
                    double x_n = u[base + i] * inv_norm;
                    sparse_diag[s_base + i] = alpha;
                    sparse_v1[s_base + i] = x_n;
                    sparse_v2[s_base + i] = x_n;
                }
                sparse_c1[batch * numSparseSoc + si] = 0.5;
                sparse_c2[batch * numSparseSoc + si] = -alpha;
            } else {
                // Dense: upper triangle
                int64_t hs_off = d_soc_Hs_offsets[cone];
                int64_t h_base = batch * totalDenseSocHsEntries + hs_off;
                int64_t idx = 0;
                for (int64_t r = 0; r < dim; ++r) {
                    for (int64_t c = r; c < dim; ++c) {
                        double val;
                        if (r == 0 && c == 0) {
                            val = 0.5;
                        } else if (r == 0) {
                            val = 0.5 * u[base + c] * inv_norm;
                        } else {
                            double x_r = u[base + r] * inv_norm;
                            double x_c = u[base + c] * inv_norm;
                            double delta = (r == c) ? 1.0 : 0.0;
                            val = 0.5 * (delta + t_over_norm * (delta - x_r * x_c));
                        }
                        H[h_base + idx] = val;
                        ++idx;
                    }
                }
            }
        }
    }
}

// Fused exp: u = z - s, dual projection, derivative (sharing Newton iteration)
__global__ void fused_exp_proj_deriv_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    double* __restrict__ H,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numExpCones; cone += blockDim.x) {
        int64_t base = batch * m + offset + cone * 3;
        int64_t h_base = batch * numExpCones * 9 + cone * 9;

        // Compute u = z - s
        double v0 = z[base + 0] - s[base + 0];
        double v1 = z[base + 1] - s[base + 1];
        double v2 = z[base + 2] - s[base + 2];
        u[base + 0] = v0;
        u[base + 1] = v1;
        u[base + 2] = v2;

        // For dual projection via Moreau: Π_{K*}(v) = v + Π_K(-v)
        double neg_v[3] = {-v0, -v1, -v2};
        double primal_proj[3];
        project_exp_cone_primal(neg_v, primal_proj);

        pi_u[base + 0] = v0 + primal_proj[0];
        pi_u[base + 1] = v1 + primal_proj[1];
        pi_u[base + 2] = v2 + primal_proj[2];

        // Derivative: evaluate at xi = -u (same as neg_v)
        double xi[3] = {neg_v[0], neg_v[1], neg_v[2]};
        double margin = 1e-6;
        double block[3][3];

        if (in_exp_interior(xi[0], xi[1], xi[2], margin)) {
            block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
            block[1][0] = 0.0; block[1][1] = 1.0; block[1][2] = 0.0;
            block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 1.0;
        } else if (in_exp_dual_interior(-xi[0], -xi[1], -xi[2], margin)) {
            block[0][0] = 0.0; block[0][1] = 0.0; block[0][2] = 0.0;
            block[1][0] = 0.0; block[1][1] = 0.0; block[1][2] = 0.0;
            block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 0.0;
        } else if (xi[0] < -margin && xi[1] < -margin) {
            block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
            block[1][0] = 0.0; block[1][1] = 0.0; block[1][2] = 0.0;
            block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = (xi[2] >= 0.0) ? 1.0 : 0.0;
        } else {
            // Boundary case: REUSE the primal projection result (primal_proj = Π_K(-u))
            // instead of calling project_exp_cone_primal again
            double r = primal_proj[0], ss = primal_proj[1], t = primal_proj[2];
            double s_min = 1e-6;
            double s_eff = fmax(ss, s_min);
            double l = t - xi[2];
            double r_over_s = fmin(r / s_eff, 500.0);
            double alpha = exp(r_over_s);
            double beta = l * r / (s_eff * s_eff) * alpha;

            double j_inv[4][4] = {{0}};
            j_inv[0][0] = alpha;
            j_inv[0][1] = (-r + s_eff) / s_eff * alpha;
            j_inv[0][2] = -1.0;
            j_inv[1][0] = 1.0 + l / s_eff * alpha;
            j_inv[1][1] = -beta;
            j_inv[1][3] = alpha;
            j_inv[2][0] = -beta;
            j_inv[2][1] = 1.0 + beta * r / s_eff;
            j_inv[2][3] = (1.0 - r / s_eff) * alpha;
            j_inv[3][2] = 1.0;
            j_inv[3][3] = -1.0;

            double j_full[4][4];
            invert_4x4(j_inv, j_full);

            block[0][0] = j_full[0][1]; block[0][1] = j_full[0][2]; block[0][2] = j_full[0][3];
            block[1][0] = j_full[1][1]; block[1][1] = j_full[1][2]; block[1][2] = j_full[1][3];
            block[2][0] = j_full[2][1]; block[2][1] = j_full[2][2]; block[2][2] = j_full[2][3];

            bool bad = false;
            for (int i = 0; i < 3 && !bad; i++)
                for (int j = 0; j < 3 && !bad; j++)
                    if (isnan(block[i][j]) || isinf(block[i][j]))
                        bad = true;
            if (bad) {
                block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
                block[1][0] = 0.0; block[1][1] = 1.0; block[1][2] = 0.0;
                block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 1.0;
            }
        }

        // Moreau: D_{K*}(u) = I - D_K(-u)
        H[h_base + 0] = 1.0 - block[0][0];
        H[h_base + 1] =     - block[0][1];
        H[h_base + 2] =     - block[0][2];
        H[h_base + 3] =     - block[1][0];
        H[h_base + 4] = 1.0 - block[1][1];
        H[h_base + 5] =     - block[1][2];
        H[h_base + 6] =     - block[2][0];
        H[h_base + 7] =     - block[2][1];
        H[h_base + 8] = 1.0 - block[2][2];
    }
}

// Fused power: u = z - s, dual projection, derivative (sharing Newton via project_pow_cone_with_jacobian)
__global__ void fused_power_proj_deriv_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    double* __restrict__ H,
    const double* __restrict__ z,
    const double* __restrict__ s,
    const double* __restrict__ alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numPowerCones; cone += blockDim.x) {
        int64_t base = batch * m + offset + cone * 3;
        int64_t h_base = batch * numPowerCones * 9 + cone * 9;
        double alpha = alphas[cone];

        // Compute u = z - s
        double ux = z[base + 0] - s[base + 0];
        double uy = z[base + 1] - s[base + 1];
        double uz = z[base + 2] - s[base + 2];
        u[base + 0] = ux;
        u[base + 1] = uy;
        u[base + 2] = uz;

        // For dual projection: Π_{K*}(u) = u + Π_K(-u)
        // For derivative: evaluate at -u, get both projection and Jacobian
        double proj[3];
        double block[3][3];
        project_pow_cone_with_jacobian(-ux, -uy, -uz, alpha, proj, block);

        // Dual projection via Moreau
        pi_u[base + 0] = ux + proj[0];
        pi_u[base + 1] = uy + proj[1];
        pi_u[base + 2] = uz + proj[2];

        // Moreau: D_{K*}(u) = I - D_K(-u)
        H[h_base + 0] = 1.0 - block[0][0];
        H[h_base + 1] =     - block[0][1];
        H[h_base + 2] =     - block[0][2];
        H[h_base + 3] =     - block[1][0];
        H[h_base + 4] = 1.0 - block[1][1];
        H[h_base + 5] =     - block[1][2];
        H[h_base + 6] =     - block[2][0];
        H[h_base + 7] =     - block[2][1];
        H[h_base + 8] = 1.0 - block[2][2];
    }
}

// ============================================================================
// Dispatch functions for fused projection + derivative
// ============================================================================

void fused_cone_projection_and_derivative(
    double* u,
    double* pi_u,
    double* nonneg_H,
    double* soc_H,
    const int64_t* d_soc_Hs_offsets,
    int64_t totalDenseSocHsEntries,
    double* soc_sparse_diag,
    double* soc_sparse_v1,
    double* soc_sparse_v2,
    double* soc_sparse_c1,
    double* soc_sparse_c2,
    const int64_t* d_soc_sparse_indices,
    const int64_t* d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    double* exp_H,
    double* power_H,
    const double* z,
    const double* s,
    const double* powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int64_t offset = 0;

    if (numZeroCones > 0) {
        int threads = min((int)numZeroCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_zero_proj_kernel, batchSize, threads, 0, stream,
            u, pi_u, z, s, numZeroCones, batchSize, m
        );
        offset += numZeroCones;
    }

    if (numNonnegCones > 0) {
        int threads = min((int)numNonnegCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_nonneg_proj_deriv_kernel, batchSize, threads, 0, stream,
            u, pi_u, nonneg_H, z, s, offset, numNonnegCones, batchSize, m
        );
        offset += numNonnegCones;
    }

    if (numSocCones > 0) {
        int threads = min((int)numSocCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_soc_proj_deriv_kernel, batchSize, threads, 0, stream,
            u, pi_u,
            soc_H, d_soc_Hs_offsets, totalDenseSocHsEntries,
            soc_sparse_diag, soc_sparse_v1, soc_sparse_v2,
            soc_sparse_c1, soc_sparse_c2,
            d_soc_sparse_indices, d_soc_sparse_offsets,
            totalSparseSocDim, numSparseSoc,
            z, s, offset, numSocCones, d_soc_dims, d_soc_offsets,
            d_soc_sz_offsets, batchSize, m
        );
        offset += totalSocDim;
    }

    if (numExpCones > 0) {
        int threads = min((int)numExpCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_exp_proj_deriv_kernel, batchSize, threads, 0, stream,
            u, pi_u, exp_H, z, s, offset, numExpCones, batchSize, m
        );
        offset += numExpCones * 3;
    }

    if (numPowerCones > 0) {
        int threads = min((int)numPowerCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_power_proj_deriv_kernel, batchSize, threads, 0, stream,
            u, pi_u, power_H, z, s, powerAlphas, offset, numPowerCones, batchSize, m
        );
    }
}

// ============================================================================
// Asymmetric direct-x cone Jacobian (IFT-direct backward)
// ============================================================================
//
// Computes dense H_x = DΠ_K*(u_x) for asymmetric direct-x cones (Exp,
// Power), where u_x_k = z_x_k − x[J_xc[k]]. The asymmetric primal/dual
// swap matters only for the FORWARD scaling; for backward, the dual cone
// projection Jacobian is the same function as for the slack form. See CPU
// `get_cone_derivative_sparse_xcones` (cones.rs) which dispatches Exp/Pow
// direct-x through the slack-form `derivative_cone_sparse(u, slack, dual=true)`.
//
// One block per batch; threads stride over numXCones. Each cone reads its
// (variable-length) slice via xcone_indices/xcone_numel_offsets, computes
// the per-kind Jacobian, and writes into the kind-specific dense buffer at
// the per-cone xcone_h_off slot.
__global__ void compute_xcone_asymm_H_kernel(
    double* __restrict__ xcone_exp_H,                 // [batchSize][totalXExpKkt] dense 3*3 per Exp x-cone
    double* __restrict__ xcone_pow_H,                 // [batchSize][totalXPowKkt] dense 3*3 per Power x-cone
    const double* __restrict__ x_eq,                  // [batchSize][n]
    const double* __restrict__ z_x_eq,                // [batchSize][totalXConeNumel]
    const int64_t* __restrict__ xcone_kinds,          // [numXCones]  3=Exp, 4=Power, others skipped
    const int64_t* __restrict__ xcone_indices,        // [totalXConeNumel] flat J indices
    const int64_t* __restrict__ xcone_numel_offsets,  // [numXCones+1] prefix sum of dims
    const int64_t* __restrict__ xcone_h_off,          // [numXCones]   per-cone offset into kind-specific H
    const int64_t* __restrict__ d_xcone_pow_idx,      // [numXCones]   per-Power index, -1 if not Power
    const double* __restrict__ d_xcone_pow_alpha,    // [numXPower]   α per Power x-cone
    int64_t numXCones,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXExpKkt,
    int64_t totalXPowKkt,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* x_b  = x_eq   + batch * n;
    const double* zx_b = z_x_eq + batch * totalXConeNumel;

    for (int64_t xc = threadIdx.x; xc < numXCones; xc += blockDim.x) {
        int64_t kind = xcone_kinds[xc];
        if (kind != 3 && kind != 4) continue;

        int64_t off = xcone_h_off[xc];
        int64_t nu  = xcone_numel_offsets[xc];

        // u_x_k = z_x_k − x[J_xc[k]] (Exact mode dual-side input).
        double v0 = zx_b[nu + 0] - x_b[xcone_indices[nu + 0]];
        double v1 = zx_b[nu + 1] - x_b[xcone_indices[nu + 1]];
        double v2 = zx_b[nu + 2] - x_b[xcone_indices[nu + 2]];

        double block[3][3];

        if (kind == 3) {
            // Exp cone — mirrors compute_exp_derivative_kernel body.
            // Evaluate at xi = -u_x for the dual-derivative-via-Moreau path.
            double xi[3] = {-v0, -v1, -v2};
            double margin = 1e-6;

            if (in_exp_interior(xi[0], xi[1], xi[2], margin)) {
                block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
                block[1][0] = 0.0; block[1][1] = 1.0; block[1][2] = 0.0;
                block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 1.0;
            } else if (in_exp_dual_interior(-xi[0], -xi[1], -xi[2], margin)) {
                block[0][0] = 0.0; block[0][1] = 0.0; block[0][2] = 0.0;
                block[1][0] = 0.0; block[1][1] = 0.0; block[1][2] = 0.0;
                block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 0.0;
            } else if (xi[0] < -margin && xi[1] < -margin) {
                block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
                block[1][0] = 0.0; block[1][1] = 0.0; block[1][2] = 0.0;
                block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = (xi[2] >= 0.0) ? 1.0 : 0.0;
            } else {
                double rs[3];
                project_exp_cone_primal(xi, rs);
                double r = rs[0], s = rs[1], t = rs[2];
                double s_min = 1e-6;
                double s_eff = fmax(s, s_min);
                double l = t - xi[2];
                double r_over_s = fmin(r / s_eff, 500.0);
                double alpha_e = exp(r_over_s);
                double beta = l * r / (s_eff * s_eff) * alpha_e;
                double j_inv[4][4] = {{0}};
                j_inv[0][0] = alpha_e;
                j_inv[0][1] = (-r + s_eff) / s_eff * alpha_e;
                j_inv[0][2] = -1.0;
                j_inv[1][0] = 1.0 + l / s_eff * alpha_e;
                j_inv[1][1] = -beta;
                j_inv[1][3] = alpha_e;
                j_inv[2][0] = -beta;
                j_inv[2][1] = 1.0 + beta * r / s_eff;
                j_inv[2][3] = (1.0 - r / s_eff) * alpha_e;
                j_inv[3][2] = 1.0;
                j_inv[3][3] = -1.0;
                double j_full[4][4];
                invert_4x4(j_inv, j_full);
                block[0][0] = j_full[0][1]; block[0][1] = j_full[0][2]; block[0][2] = j_full[0][3];
                block[1][0] = j_full[1][1]; block[1][1] = j_full[1][2]; block[1][2] = j_full[1][3];
                block[2][0] = j_full[2][1]; block[2][1] = j_full[2][2]; block[2][2] = j_full[2][3];
                bool bad = false;
                for (int i = 0; i < 3 && !bad; i++)
                    for (int j = 0; j < 3 && !bad; j++)
                        if (isnan(block[i][j]) || isinf(block[i][j])) bad = true;
                if (bad) {
                    block[0][0] = 1.0; block[0][1] = 0.0; block[0][2] = 0.0;
                    block[1][0] = 0.0; block[1][1] = 1.0; block[1][2] = 0.0;
                    block[2][0] = 0.0; block[2][1] = 0.0; block[2][2] = 1.0;
                }
            }
            double* H = xcone_exp_H + batch * totalXExpKkt + off;
            // Moreau: D_{K*}(u) = I − D_K(−u). Row-major 3*3.
            H[0] = 1.0 - block[0][0]; H[1] =     - block[0][1]; H[2] =     - block[0][2];
            H[3] =     - block[1][0]; H[4] = 1.0 - block[1][1]; H[5] =     - block[1][2];
            H[6] =     - block[2][0]; H[7] =     - block[2][1]; H[8] = 1.0 - block[2][2];
        } else {
            // Power cone — Newton solve via project_pow_cone_with_jacobian
            // at -u, then Moreau decomposition.
            double alpha = d_xcone_pow_alpha[d_xcone_pow_idx[xc]];
            double proj[3];
            project_pow_cone_with_jacobian(-v0, -v1, -v2, alpha, proj, block);
            double* H = xcone_pow_H + batch * totalXPowKkt + off;
            H[0] = 1.0 - block[0][0]; H[1] =     - block[0][1]; H[2] =     - block[0][2];
            H[3] =     - block[1][0]; H[4] = 1.0 - block[1][1]; H[5] =     - block[1][2];
            H[6] =     - block[2][0]; H[7] =     - block[2][1]; H[8] = 1.0 - block[2][2];
        }
    }
}

void compute_xcone_asymm_H(
    double* xcone_exp_H,
    double* xcone_pow_H,
    const double* x_eq,
    const double* z_x_eq,
    const int64_t* xcone_kinds,
    const int64_t* xcone_indices,
    const int64_t* xcone_numel_offsets,
    const int64_t* xcone_h_off,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t numXCones,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXExpKkt,
    int64_t totalXPowKkt,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (numXCones == 0 || (totalXExpKkt == 0 && totalXPowKkt == 0)) return;
    int threads = min((int)numXCones, 256);
    MOREAU_KERNEL_LAUNCH(compute_xcone_asymm_H_kernel, batchSize, threads, 0, stream,
        xcone_exp_H, xcone_pow_H,
        x_eq, z_x_eq,
        xcone_kinds, xcone_indices, xcone_numel_offsets, xcone_h_off,
        d_xcone_pow_idx, d_xcone_pow_alpha,
        numXCones, n, totalXConeNumel,
        totalXExpKkt, totalXPowKkt, batchSize
    );
}

// ============================================================================
// Direct-x SOC Jacobian (IFT-direct backward, rank-2 sparse expansion)
// ============================================================================
//
// For direct-x SOC cones with dim > 4, computes the rank-2 decomposition
// of DΠ_K*(u_x) where u_x = z_x − x[J]:
//   H = diag(d) + c1 * v1 v1' + c2 * v2 v2'
// using the slack-side Moreau formulas (`compute_soc_derivative_sparse_kernel`)
// adapted to direct-x indexing. Cones with dim ≤ 4 stay on the dense
// `xcone_soc_H` path (handled by `compute_xcone_H_kernel`).
__global__ void compute_xcone_soc_rank2_kernel(
    double* __restrict__ sparse_diag,                  // [batch][totalSparseXSocDim]
    double* __restrict__ sparse_v1,
    double* __restrict__ sparse_v2,
    double* __restrict__ sparse_c1,                    // [batch][numSparseXSoc]
    double* __restrict__ sparse_c2,
    const double* __restrict__ x_eq,                   // [batch][n]
    const double* __restrict__ z_x_eq,                 // [batch][totalXConeNumel]
    const int64_t* __restrict__ xcone_indices,
    const int64_t* __restrict__ xcone_numel_offsets,
    const int64_t* __restrict__ d_xsoc_sparse_dim_offsets,  // [numSparseXSoc+1]
    const int64_t* __restrict__ d_xsoc_sparse_to_xc,        // [numSparseXSoc] -> xcone idx
    const int64_t* __restrict__ d_xsoc_sparse_dims,         // [numSparseXSoc]
    int64_t numSparseXSoc,
    int64_t totalSparseXSocDim,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* x_b  = x_eq   + batch * n;
    const double* zx_b = z_x_eq + batch * totalXConeNumel;

    for (int64_t s = threadIdx.x; s < numSparseXSoc; s += blockDim.x) {
        int64_t xc_idx     = d_xsoc_sparse_to_xc[s];
        int64_t dim        = d_xsoc_sparse_dims[s];
        int64_t sparse_off = d_xsoc_sparse_dim_offsets[s];
        int64_t nu         = xcone_numel_offsets[xc_idx];
        int64_t s_base     = batch * totalSparseXSocDim + sparse_off;

        // u_x[i] = z_x[i] - x[J[i]]
        double t = zx_b[nu + 0] - x_b[xcone_indices[nu + 0]];
        double norm_x_sq = 0.0;
        for (int64_t i = 1; i < dim; ++i) {
            double u_i = zx_b[nu + i] - x_b[xcone_indices[nu + i]];
            norm_x_sq += u_i * u_i;
        }
        double norm_x = sqrt(norm_x_sq);

        if (t >= norm_x) {
            // Interior: H = I → diag = 1, ranks = 0.
            for (int64_t i = 0; i < dim; ++i) {
                sparse_diag[s_base + i] = 1.0;
                sparse_v1[s_base + i]   = 0.0;
                sparse_v2[s_base + i]   = 0.0;
            }
            sparse_c1[batch * numSparseXSoc + s] = 0.0;
            sparse_c2[batch * numSparseXSoc + s] = 0.0;
        } else if (t <= -norm_x) {
            // Polar: H = 0.
            for (int64_t i = 0; i < dim; ++i) {
                sparse_diag[s_base + i] = 0.0;
                sparse_v1[s_base + i]   = 0.0;
                sparse_v2[s_base + i]   = 0.0;
            }
            sparse_c1[batch * numSparseXSoc + s] = 0.0;
            sparse_c2[batch * numSparseXSoc + s] = 0.0;
        } else {
            // Boundary: H = diag(0, α, ..., α) + 0.5*v1*v1' + (-α)*v2*v2'
            double inv_norm = (norm_x > 1e-300) ? (1.0 / norm_x) : 0.0;
            double alpha = 0.5 * (t + norm_x) * inv_norm;
            sparse_diag[s_base] = 0.0;
            for (int64_t i = 1; i < dim; ++i) sparse_diag[s_base + i] = alpha;
            sparse_v1[s_base] = 1.0;
            for (int64_t i = 1; i < dim; ++i) {
                double u_i = zx_b[nu + i] - x_b[xcone_indices[nu + i]];
                sparse_v1[s_base + i] = u_i * inv_norm;
            }
            sparse_v2[s_base] = 0.0;
            for (int64_t i = 1; i < dim; ++i) {
                double u_i = zx_b[nu + i] - x_b[xcone_indices[nu + i]];
                sparse_v2[s_base + i] = u_i * inv_norm;
            }
            sparse_c1[batch * numSparseXSoc + s] = 0.5;
            sparse_c2[batch * numSparseXSoc + s] = -alpha;
        }
    }
}

void compute_xcone_soc_rank2(
    double* sparse_diag,
    double* sparse_v1,
    double* sparse_v2,
    double* sparse_c1,
    double* sparse_c2,
    const double* x_eq,
    const double* z_x_eq,
    const int64_t* xcone_indices,
    const int64_t* xcone_numel_offsets,
    const int64_t* d_xsoc_sparse_dim_offsets,
    const int64_t* d_xsoc_sparse_to_xc,
    const int64_t* d_xsoc_sparse_dims,
    int64_t numSparseXSoc,
    int64_t totalSparseXSocDim,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (numSparseXSoc == 0) return;
    int threads = min((int)numSparseXSoc, 256);
    MOREAU_KERNEL_LAUNCH(compute_xcone_soc_rank2_kernel, batchSize, threads, 0, stream,
        sparse_diag, sparse_v1, sparse_v2, sparse_c1, sparse_c2,
        x_eq, z_x_eq, xcone_indices, xcone_numel_offsets,
        d_xsoc_sparse_dim_offsets, d_xsoc_sparse_to_xc, d_xsoc_sparse_dims,
        numSparseXSoc, totalSparseXSocDim, n, totalXConeNumel, batchSize
    );
}

// ============================================================================
// Direct-x GenPowerCone Jacobian (IFT-direct backward, dense expansion)
// ============================================================================
//
// Mirrors the slack-side `compute_genpow_derivative_sparse_kernel` body
// (Newton on r, boundary detection, rank-3 decomposition under Moreau)
// but reads `u_x = z_x − x[J]` from direct-x storage and writes the
// fully expanded dense `dim*dim` block into `xcone_genpow_H`. The CPU
// reference uses the same dense expansion in
// `build_hsde_augmented_system_sparse_full` (kkt.rs) — the comment
// there notes that direct-x sparse expansion "is a future
// optimization." We adopt the same dense form here.
//
// One block per batch; threads stride over numXCones and early-out for
// kinds other than GenPower. Workspace pointers (`work_vec`,
// `work_dim1`, the six rank-3 stripes) are sized to the direct-x
// totals (`totalXGenPowerDim` / `totalXGenPowerAlphas`) and indexed
// per-cone by the cone's `gp_off` / `alpha_off`.
__global__ void compute_xcone_genpow_H_kernel(
    double* __restrict__ xcone_genpow_H,              // [batch][totalXGenPowKkt]
    double* __restrict__ rank3_diag,                  // [batch][totalXGenPowerDim]
    double* __restrict__ rank3_left1,
    double* __restrict__ rank3_right1,
    double* __restrict__ rank3_left2,
    double* __restrict__ rank3_right2,
    double* __restrict__ rank3_left3,
    double* __restrict__ rank3_c3,                    // [batch][numXGenPowerCones]
    const double* __restrict__ x_eq,                  // [batch][n]
    const double* __restrict__ z_x_eq,                // [batch][totalXConeNumel]
    const int64_t* __restrict__ xcone_kinds,          // [numXCones]
    const int64_t* __restrict__ xcone_indices,        // [totalXConeNumel] flat J indices
    const int64_t* __restrict__ xcone_numel_offsets,  // [numXCones+1]
    const int64_t* __restrict__ xcone_h_off,          // [numXCones] per-cone offset into H
    const int64_t* __restrict__ d_xcone_genpow_idx,   // [numXCones] per-GenPow index, -1 if not
    const int64_t* __restrict__ d_xcone_genpow_dim1s, // [numXGenPow]
    const int64_t* __restrict__ d_xcone_genpow_dim2s, // [numXGenPow]
    const int64_t* __restrict__ d_xcone_genpow_alpha_offsets,  // [numXGenPow+1]
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,    // [numXGenPow+1]
    const double* __restrict__ d_xcone_genpow_alphas,         // flat alphas
    double* __restrict__ work_vec,                    // [batch][totalXGenPowerDim] xi scratch
    double* __restrict__ work_dim1,                   // [batch][7 * totalXGenPowerAlphas]
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXGenPowKkt,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    const double* x_b  = x_eq   + batch * n;
    const double* zx_b = z_x_eq + batch * totalXConeNumel;

    for (int64_t xc = threadIdx.x; xc < numXCones; xc += blockDim.x) {
        if (xcone_kinds[xc] != 5) continue;  // 5 = GenPower
        int64_t gp = d_xcone_genpow_idx[xc];
        int64_t dim1 = d_xcone_genpow_dim1s[gp];
        int64_t dim2 = d_xcone_genpow_dim2s[gp];
        int64_t dim  = dim1 + dim2;
        int64_t gp_off    = d_xcone_genpow_dim_offsets[gp];
        int64_t alpha_off = d_xcone_genpow_alpha_offsets[gp];
        int64_t nu        = xcone_numel_offsets[xc];
        int64_t h_off     = xcone_h_off[xc];

        double* H_dense = xcone_genpow_H + batch * totalXGenPowKkt + h_off;

        double margin = GENPOW_MARGIN;
        double tol    = GENPOW_CONE_TOL;

        // Per-cone scratch (slices into the global workspace).
        double* xi    = work_vec + batch * totalXGenPowerDim + gp_off;
        int64_t d1_base   = batch * 7 * totalXGenPowerAlphas + alpha_off;
        int64_t d1_stride = totalXGenPowerAlphas;
        double* pi_stars = work_dim1 + d1_base;
        double* sqrt_gi  = work_dim1 + d1_base + d1_stride;
        double* dpi_dpi  = work_dim1 + d1_base + 2 * d1_stride;
        double* dpi_dr   = work_dim1 + d1_base + 3 * d1_stride;
        double* dpi_dnw  = work_dim1 + d1_base + 4 * d1_stride;
        double* dphi_dp  = work_dim1 + d1_base + 5 * d1_stride;
        double* dr_dp    = work_dim1 + d1_base + 6 * d1_stride;

        // Rank-3 outputs (slices into the global rank-3 stripes).
        int64_t out_base = batch * totalXGenPowerDim + gp_off;
        double* o_diag = rank3_diag   + out_base;
        double* o_l1   = rank3_left1  + out_base;
        double* o_r1   = rank3_right1 + out_base;
        double* o_l2   = rank3_left2  + out_base;
        double* o_r2   = rank3_right2 + out_base;
        double* o_l3   = rank3_left3  + out_base;
        double* o_c3   = rank3_c3 + batch * numXGenPowerCones + gp;

        auto zero_rank_terms = [&]() {
            for (int64_t i = 0; i < dim; ++i) {
                o_l1[i] = 0.0; o_r1[i] = 0.0;
                o_l2[i] = 0.0; o_r2[i] = 0.0;
                o_l3[i] = 0.0;
            }
            *o_c3 = 0.0;
        };

        // Gather u_x = z_x − x[J]; evaluate at xi = -u_x for the
        // dual-via-Moreau path.
        for (int64_t i = 0; i < dim; ++i) {
            double u_i = zx_b[nu + i] - x_b[xcone_indices[nu + i]];
            xi[i] = -u_i;
        }

        const double* cone_alphas = d_xcone_genpow_alphas + alpha_off;
        const double* p = xi;
        const double* w = xi + dim1;

        bool emitted_diag = false;
        if (in_genpow_cone_interior(xi, cone_alphas, dim1, dim2, margin)) {
            for (int64_t i = 0; i < dim; ++i) o_diag[i] = 0.0;
            zero_rank_terms();
            emitted_diag = true;
        } else if (in_genpow_polar_interior(xi, cone_alphas, dim1, dim2, margin)) {
            for (int64_t i = 0; i < dim; ++i) o_diag[i] = 1.0;
            zero_rank_terms();
            emitted_diag = true;
        }

        double norm_w = 0.0;
        if (!emitted_diag) {
            double norm_w_sq = 0.0;
            for (int64_t i = 0; i < dim2; ++i) norm_w_sq += w[i] * w[i];
            norm_w = sqrt(norm_w_sq);
            if (norm_w <= tol) {
                for (int64_t i = 0; i < dim1; ++i) {
                    double pval = (fabs(p[i]) > tol) ? 0.5 * (copysign(1.0, p[i]) + 1.0) : 0.5;
                    o_diag[i] = 1.0 - pval;
                }
                for (int64_t i = 0; i < dim2; ++i) o_diag[dim1 + i] = 1.0;
                zero_rank_terms();
                emitted_diag = true;
            }
        }

        if (!emitted_diag) {
            // Newton iteration on r (closed-form pi_stars given r).
            double r_val = norm_w * 0.5;
            for (int iter = 0; iter < GENPOW_MAX_ITERS; ++iter) {
                for (int64_t i = 0; i < dim1; ++i) {
                    double ai = cone_alphas[i];
                    double vi = p[i];
                    double disc = vi * vi + 4.0 * ai * (norm_w - r_val) * r_val;
                    if (disc < 0.0) disc = 0.0;
                    double psi = 0.5 * (vi + sqrt(disc));
                    pi_stars[i] = fmax(psi, tol);
                }
                double log_prod = 0.0;
                for (int64_t i = 0; i < dim1; ++i)
                    log_prod += cone_alphas[i] * log(pi_stars[i]);
                double prod = exp(log_prod);
                double f = prod - r_val;
                if (fabs(f) < GENPOW_NEWTON_TOL) break;

                double fp = -1.0;
                for (int64_t i = 0; i < dim1; ++i) {
                    double denom = 2.0 * pi_stars[i] - p[i];
                    double dpi_dr_i = cone_alphas[i] * (norm_w - 2.0 * r_val) /
                                      (denom + copysign(tol, denom));
                    if (fabs(pi_stars[i]) > tol)
                        fp += prod * cone_alphas[i] * dpi_dr_i / pi_stars[i];
                }
                if (fabs(fp) < GENPOW_NEWTON_FP_TOL) break;
                r_val -= f / fp;
                r_val = fmax(r_val, 0.0);
                r_val = fmin(r_val, norm_w);
            }

            // Recompute final intermediates at converged r_val.
            for (int64_t i = 0; i < dim1; ++i) {
                double ai = cone_alphas[i];
                double disc = p[i] * p[i] + 4.0 * ai * (norm_w - r_val) * r_val;
                if (disc < 0.0) disc = 0.0;
                pi_stars[i] = fmax(0.5 * (p[i] + sqrt(disc)), tol);
            }
            double log_prod = 0.0;
            for (int64_t i = 0; i < dim1; ++i)
                log_prod += cone_alphas[i] * log(pi_stars[i]);
            double prod = exp(log_prod);

            for (int64_t i = 0; i < dim1; ++i) {
                double gi = p[i] * p[i] + 4.0 * cone_alphas[i] * r_val * (norm_w - r_val);
                sqrt_gi[i] = sqrt(fmax(gi, tol));
            }
            for (int64_t i = 0; i < dim1; ++i) dpi_dpi[i] = pi_stars[i] / sqrt_gi[i];
            for (int64_t i = 0; i < dim1; ++i) {
                double denom = 2.0 * pi_stars[i] - p[i];
                dpi_dr[i] = cone_alphas[i] * (norm_w - 2.0 * r_val) /
                            (denom + copysign(tol, denom));
            }
            for (int64_t i = 0; i < dim1; ++i)
                dpi_dnw[i] = cone_alphas[i] * r_val / sqrt_gi[i];

            double dphi_dr_val = -1.0;
            for (int64_t i = 0; i < dim1; ++i) {
                if (fabs(pi_stars[i]) > tol)
                    dphi_dr_val += prod * cone_alphas[i] * dpi_dr[i] / pi_stars[i];
            }
            for (int64_t i = 0; i < dim1; ++i)
                dphi_dp[i] = (fabs(pi_stars[i]) > tol)
                             ? prod * cone_alphas[i] * dpi_dpi[i] / pi_stars[i] : 0.0;
            double dphi_dnw_val = 0.0;
            for (int64_t i = 0; i < dim1; ++i) {
                if (fabs(pi_stars[i]) > tol)
                    dphi_dnw_val += prod * cone_alphas[i] * dpi_dnw[i] / pi_stars[i];
            }

            double dphi_dr_reg = dphi_dr_val + copysign(tol, dphi_dr_val);
            double dr_dnw = -dphi_dnw_val / dphi_dr_reg;
            for (int64_t i = 0; i < dim1; ++i)
                dr_dp[i] = -dphi_dp[i] / dphi_dr_reg;

            double r_over_nw = r_val / norm_w;

            for (int64_t i = 0; i < dim1; ++i) o_diag[i] = 1.0 - dpi_dpi[i];
            for (int64_t i = 0; i < dim2; ++i) o_diag[dim1 + i] = 1.0 - r_over_nw;

            for (int64_t i = 0; i < dim1; ++i) o_l1[i] = -dpi_dr[i];
            for (int64_t i = 0; i < dim2; ++i) o_l1[dim1 + i] = -w[i] / norm_w;

            for (int64_t i = 0; i < dim1; ++i) o_r1[i] = dr_dp[i];
            for (int64_t i = 0; i < dim2; ++i) o_r1[dim1 + i] = dr_dnw * w[i] / norm_w;

            for (int64_t i = 0; i < dim1; ++i) o_l2[i] = -dpi_dnw[i];
            for (int64_t i = 0; i < dim2; ++i) o_l2[dim1 + i] = 0.0;

            for (int64_t i = 0; i < dim1; ++i) o_r2[i] = 0.0;
            for (int64_t i = 0; i < dim2; ++i) o_r2[dim1 + i] = w[i] / norm_w;

            for (int64_t i = 0; i < dim1; ++i) o_l3[i] = 0.0;
            for (int64_t i = 0; i < dim2; ++i) o_l3[dim1 + i] = w[i];

            if (dim2 == 1) {
                // Fold c3*w² into diagonal — see slack kernel for derivation.
                o_diag[dim1] = 1.0;
                o_l3[dim1] = 0.0;
                *o_c3 = 0.0;
            } else {
                double c_ww_primal = -r_val / (norm_w * norm_w * norm_w);
                *o_c3 = -c_ww_primal;
            }
        }

        // Expand rank-3 form to dense H_x[k][l] (row-major).
        // H = diag(o_diag) + o_l1 ⊗ o_r1 + o_l2 ⊗ o_r2 + c3 * o_l3 ⊗ o_l3.
        double c3_val = *o_c3;
        for (int64_t k = 0; k < dim; ++k) {
            double dk = o_diag[k];
            double l1k = o_l1[k];
            double l2k = o_l2[k];
            double l3k = o_l3[k];
            for (int64_t l = 0; l < dim; ++l) {
                double v = (k == l) ? dk : 0.0;
                v += l1k * o_r1[l];
                v += l2k * o_r2[l];
                v += c3_val * l3k * o_l3[l];
                H_dense[k * dim + l] = v;
            }
        }
    }
}

void compute_xcone_genpow_H(
    double* xcone_genpow_H,
    double* rank3_diag,
    double* rank3_left1,
    double* rank3_right1,
    double* rank3_left2,
    double* rank3_right2,
    double* rank3_left3,
    double* rank3_c3,
    const double* x_eq,
    const double* z_x_eq,
    const int64_t* xcone_kinds,
    const int64_t* xcone_indices,
    const int64_t* xcone_numel_offsets,
    const int64_t* xcone_h_off,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const double*  d_xcone_genpow_alphas,
    double* work_vec,
    double* work_dim1,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXGenPowKkt,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (numXGenPowerCones == 0 || totalXGenPowKkt == 0) return;
    int threads = min((int)numXCones, 256);
    MOREAU_KERNEL_LAUNCH(compute_xcone_genpow_H_kernel, batchSize, threads, 0, stream,
        xcone_genpow_H,
        rank3_diag, rank3_left1, rank3_right1,
        rank3_left2, rank3_right2, rank3_left3, rank3_c3,
        x_eq, z_x_eq,
        xcone_kinds, xcone_indices, xcone_numel_offsets, xcone_h_off,
        d_xcone_genpow_idx, d_xcone_genpow_dim1s, d_xcone_genpow_dim2s,
        d_xcone_genpow_alpha_offsets, d_xcone_genpow_dim_offsets,
        d_xcone_genpow_alphas,
        work_vec, work_dim1,
        numXCones, numXGenPowerCones,
        totalXGenPowerDim, totalXGenPowerAlphas,
        n, totalXConeNumel, totalXGenPowKkt, batchSize
    );
}

// Forward path: fuse u = z - s into projection (no derivative)
__global__ void fused_u_and_proj_nonneg_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        int64_t idx = batch * m + offset + i;
        double u_val = z[idx] - s[idx];
        u[idx] = u_val;
        pi_u[idx] = fmax(u_val, 0.0);
    }
}

__global__ void fused_u_and_proj_soc_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_sz_offsets,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numSocCones; cone += blockDim.x) {
        int64_t dim = d_soc_dims[cone];
        int64_t sz_off = d_soc_sz_offsets ? d_soc_sz_offsets[cone] : d_soc_offsets[cone];
        int64_t base = batch * m + offset + sz_off;

        double t = z[base + 0] - s[base + 0];
        u[base + 0] = t;
        double norm_x_sq = 0.0;
        for (int64_t i = 1; i < dim; ++i) {
            double ui = z[base + i] - s[base + i];
            u[base + i] = ui;
            norm_x_sq += ui * ui;
        }
        double norm_x = safe_sqrt(norm_x_sq);

        if (t >= norm_x) {
            pi_u[base + 0] = t;
            for (int64_t i = 1; i < dim; ++i) {
                pi_u[base + i] = u[base + i];
            }
        } else if (t <= -norm_x) {
            for (int64_t i = 0; i < dim; ++i) {
                pi_u[base + i] = 0.0;
            }
        } else {
            double scale = 0.5 * (t + norm_x);
            double inv_norm = safe_div(1.0, norm_x, 0.0);
            pi_u[base + 0] = scale;
            for (int64_t i = 1; i < dim; ++i) {
                pi_u[base + i] = scale * u[base + i] * inv_norm;
            }
        }
    }
}

__global__ void fused_u_and_proj_exp_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    const double* __restrict__ z,
    const double* __restrict__ s,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numExpCones; cone += blockDim.x) {
        int64_t base = batch * m + offset + cone * 3;
        double v0 = z[base + 0] - s[base + 0];
        double v1 = z[base + 1] - s[base + 1];
        double v2 = z[base + 2] - s[base + 2];
        u[base + 0] = v0;
        u[base + 1] = v1;
        u[base + 2] = v2;

        double neg_v[3] = {-v0, -v1, -v2};
        double primal_proj[3];
        project_exp_cone_primal(neg_v, primal_proj);

        pi_u[base + 0] = v0 + primal_proj[0];
        pi_u[base + 1] = v1 + primal_proj[1];
        pi_u[base + 2] = v2 + primal_proj[2];
    }
}

__global__ void fused_u_and_proj_power_kernel(
    double* __restrict__ u,
    double* __restrict__ pi_u,
    const double* __restrict__ z,
    const double* __restrict__ s,
    const double* __restrict__ alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    for (int64_t cone = threadIdx.x; cone < numPowerCones; cone += blockDim.x) {
        int64_t base = batch * m + offset + cone * 3;
        double alpha = alphas[cone];

        double ux = z[base + 0] - s[base + 0];
        double uy = z[base + 1] - s[base + 1];
        double uz = z[base + 2] - s[base + 2];
        u[base + 0] = ux;
        u[base + 1] = uy;
        u[base + 2] = uz;

        double primal_proj[3];
        project_pow_cone_primal(-ux, -uy, -uz, alpha, primal_proj);

        pi_u[base + 0] = ux + primal_proj[0];
        pi_u[base + 1] = uy + primal_proj[1];
        pi_u[base + 2] = uz + primal_proj[2];
    }
}

void fused_u_and_cone_projection(
    double* u,
    double* pi_u,
    const double* z,
    const double* s,
    const double* powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int64_t offset = 0;

    if (numZeroCones > 0) {
        int threads = min((int)numZeroCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_zero_proj_kernel, batchSize, threads, 0, stream,
            u, pi_u, z, s, numZeroCones, batchSize, m
        );
        offset += numZeroCones;
    }

    if (numNonnegCones > 0) {
        int threads = min((int)numNonnegCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_u_and_proj_nonneg_kernel, batchSize, threads, 0, stream,
            u, pi_u, z, s, offset, numNonnegCones, batchSize, m
        );
        offset += numNonnegCones;
    }

    if (numSocCones > 0) {
        int threads = min((int)numSocCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_u_and_proj_soc_kernel, batchSize, threads, 0, stream,
            u, pi_u, z, s, offset, numSocCones, d_soc_dims, d_soc_offsets,
            d_soc_sz_offsets, batchSize, m
        );
        offset += totalSocDim;
    }

    if (numExpCones > 0) {
        int threads = min((int)numExpCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_u_and_proj_exp_kernel, batchSize, threads, 0, stream,
            u, pi_u, z, s, offset, numExpCones, batchSize, m
        );
        offset += numExpCones * 3;
    }

    if (numPowerCones > 0) {
        int threads = min((int)numPowerCones, 256);
        MOREAU_KERNEL_LAUNCH(fused_u_and_proj_power_kernel, batchSize, threads, 0, stream,
            u, pi_u, z, s, powerAlphas, offset, numPowerCones, batchSize, m
        );
    }
}

// ============================================================================
// PSD Cone Projection and Derivative
// ============================================================================
// Implementation moved to diff_psd.cu (separate compilation unit) to avoid
// nvcc/gcc15 cwchar issues with <vector> includes needed by cones.hpp/diff.hpp.

} // namespace moreau
