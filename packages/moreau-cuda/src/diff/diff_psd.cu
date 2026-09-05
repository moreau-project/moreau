/**
 * @file diff_psd.cu
 * @brief PSD cone projection and derivative for backward pass
 *
 * Separated from kernels.cu to allow including cones.hpp and diff.hpp
 * (which pull in <vector>) without triggering nvcc/gcc15 cwchar issues.
 */

#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/diff/diff_kernels.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>

namespace moreau {

// ============================================================================
// Device kernels for PSD backward pass
// ============================================================================

static constexpr double PSD_SQRT2 = 1.4142135623730951;
static constexpr double PSD_INV_SQRT2 = 0.7071067811865476;

// Materialise the k-th svec basis vector directly into the symmetric
// matrix it would produce after svec_to_mat — saves the
// `cudaMemsetAsync(svec, 0) + set_element_kernel + diff_svec_to_mat_kernel`
// trio per column in the PSD Jacobian loops, cutting 3 launches per k.
// The decoded (i, j) for svec position k satisfies k = j(j+1)/2 + i
// with i ≤ j. Zeroes out the whole n×n matrix and writes the basis
// pattern (1 on diagonal i==j, PSD_INV_SQRT2 on the off-diagonal mirror
// pair). Single launch per k.
__global__ void build_E_k_matrix_kernel(
    double* __restrict__ mat, int64_t k, int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n2 = n * n;
    if (tid >= n2) return;
    mat[tid] = 0.0;

    // Decode (i, j) from k just like diff_svec_to_mat_kernel.
    int64_t j = 0, acc = 0;
    while (acc + j + 1 <= k) { acc += j + 1; j++; }
    int64_t i = k - acc;

    if (tid == j * n + i) {
        mat[tid] = (i == j) ? 1.0 : PSD_INV_SQRT2;
    } else if (i != j && tid == i * n + j) {
        mat[tid] = PSD_INV_SQRT2;
    }
}

// svec → dense symmetric matrix (column-major)
__global__ void diff_svec_to_mat_kernel(
    double* __restrict__ mat, const double* __restrict__ svec, int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t svec_dim = n * (n + 1) / 2;
    if (tid >= svec_dim) return;

    // Decode: svec index k = j*(j+1)/2 + i, with i <= j
    int64_t j = 0, acc = 0;
    while (acc + j + 1 <= tid) { acc += j + 1; j++; }
    int64_t i = tid - acc;

    double val = svec[tid];
    if (i == j) {
        mat[j * n + i] = val;
    } else {
        double unscaled = val * PSD_INV_SQRT2;
        mat[j * n + i] = unscaled;
        mat[i * n + j] = unscaled;
    }
}

// dense symmetric matrix → svec
__global__ void diff_mat_to_svec_kernel(
    double* __restrict__ svec, const double* __restrict__ mat, int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t svec_dim = n * (n + 1) / 2;
    if (tid >= svec_dim) return;

    int64_t j = 0, acc = 0;
    while (acc + j + 1 <= tid) { acc += j + 1; j++; }
    int64_t i = tid - acc;

    if (i == j) {
        svec[tid] = mat[j * n + i];
    } else {
        // mat is symmetric, so average is just the value itself
        svec[tid] = mat[j * n + i] * PSD_SQRT2;
    }
}

// clamp eigenvalues: out[i] = max(in[i], 0)
__global__ void clamp_eigenvalues_kernel(
    double* __restrict__ out, const double* __restrict__ in, int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmax(in[i], 0.0);
}

// scale columns: out[i + col*n] = Q[i + col*n] * scale[col]
__global__ void scale_columns_kernel(
    double* __restrict__ out, const double* __restrict__ Q, const double* __restrict__ scale, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t col = idx / n;
    out[idx] = Q[idx] * scale[col];
}

// Ω[i,j] for PSD cone projection derivative using spectral continuous extension.
// When both eigenvalues have the same sign, use exact closed-form to avoid
// catastrophic cancellation in (max(λ_i,0) - max(λ_j,0)) / (λ_i - λ_j).
// Reference: Nobel, Candes, Boyd, "Tractable Evaluation of Stein's Unbiased
//   Risk Estimate for Convex Regularizers", §C
__global__ void build_omega_kernel(
    double* __restrict__ omega, const double* __restrict__ eigvals, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t i = idx % n;
    int64_t j = idx / n;

    double li = eigvals[i], lj = eigvals[j];

    if (i == j) {
        // Diagonal: derivative of max(λ,0) at λ
        omega[idx] = (li >= 0.0) ? 1.0 : 0.0;
    } else if (li >= 0.0 && lj >= 0.0) {
        // Both non-negative: Ω = 1 exactly (continuous extension limit)
        omega[idx] = 1.0;
    } else if (li <= 0.0 && lj <= 0.0) {
        // Both non-positive: Ω = 0 exactly
        omega[idx] = 0.0;
    } else {
        // Mixed signs: |λ_i - λ_j| is bounded away from zero, formula is stable
        omega[idx] = (fmax(li, 0.0) - fmax(lj, 0.0)) / (li - lj);
    }
}

// Hadamard: A[i] *= B[i]
__global__ void hadamard_kernel(double* __restrict__ A, const double* __restrict__ B, int64_t n2) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) A[idx] *= B[idx];
}

// 1×1 PSD projection: pi_u = max(u, 0)
__global__ void project_psd_1x1_kernel(
    double* __restrict__ pi_u, const double* __restrict__ u, int64_t offset,
    int64_t numPsdCones, const int64_t* __restrict__ d_dims, const int64_t* __restrict__ d_sz_offsets,
    int64_t batchSize, int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    for (int64_t cone = threadIdx.x; cone < numPsdCones; cone += blockDim.x) {
        if (d_dims[cone] != 1) continue;
        int64_t base = batch * m + offset + d_sz_offsets[cone];
        pi_u[base] = fmax(u[base], 0.0);
    }
}

// 1×1 PSD derivative: H = (u >= 0) ? 1 : 0
__global__ void psd_derivative_1x1_kernel(
    double* __restrict__ psd_H, const double* __restrict__ u, int64_t offset,
    int64_t numPsdCones, const int64_t* __restrict__ d_dims, const int64_t* __restrict__ d_sz_offsets,
    const int64_t* __restrict__ d_Hs_offsets, int64_t totalPsdHsEntries,
    int64_t batchSize, int64_t m
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    for (int64_t cone = threadIdx.x; cone < numPsdCones; cone += blockDim.x) {
        if (d_dims[cone] != 1) continue;
        int64_t base = batch * m + offset + d_sz_offsets[cone];
        psd_H[batch * totalPsdHsEntries + d_Hs_offsets[cone]] =
            (u[base] >= 0.0) ? 1.0 : 0.0;
    }
}

// Scatter Jacobian column k into upper-triangle H storage
__global__ void scatter_jacobian_col_kernel(
    double* __restrict__ H, const double* __restrict__ col, int64_t k
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > k) return;
    H[k * (k + 1) / 2 + i] = col[i];
}

// Set single element of vector to a value (avoids H2D transfer for basis vectors)
__global__ void set_element_kernel(double* __restrict__ vec, int64_t idx, double val) {
    if (threadIdx.x == 0) vec[idx] = val;
}

// ============================================================================
// Host-side sz_offset computation
// ============================================================================

static std::vector<int64_t> compute_psd_sz_offsets_host(const Cones& cones) {
    if (cones.numPsdCones == 0) return {};
    std::vector<int64_t> orig_offsets(cones.psdConeDimsOriginal.size() + 1, 0);
    for (size_t i = 0; i < cones.psdConeDimsOriginal.size(); ++i) {
        int64_t n = cones.psdConeDimsOriginal[i];
        orig_offsets[i + 1] = orig_offsets[i] + n * (n + 1) / 2;
    }
    std::vector<int64_t> sz(cones.numPsdCones);
    for (int64_t i = 0; i < cones.numPsdCones; ++i)
        sz[i] = orig_offsets[cones.psdSortPerm[i]];
    return sz;
}

// ============================================================================
// PSD Projection (eigendecomp + clamp + reconstruct)
// ============================================================================

void project_psd_cone_dual(
    double* pi_u, const double* u, int64_t offset,
    const Cones& cones, ConeDerivatives& derivs,
    int64_t batchSize, int64_t m, cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    // 1×1 cones: simple kernel
    int threads = std::min((int)cones.numPsdCones, 256);
    MOREAU_KERNEL_LAUNCH(project_psd_1x1_kernel, batchSize, threads, 0, stream,
        pi_u, u, offset, cones.numPsdCones,
        cones.d_psd_dims, cones.d_psd_sz_offsets, batchSize, m);

    // Check for dim > 1
    bool has_general = false;
    for (int64_t i = 0; i < cones.numPsdCones; i++)
        if (cones.psdConeDims[i] > 1) { has_general = true; break; }
    if (!has_general) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    auto sz_offsets = compute_psd_sz_offsets_host(cones);
    int64_t matsq_off = 0, mat_off = 0;

    for (int64_t cone = 0; cone < cones.numPsdCones; cone++) {
        int64_t n = cones.psdConeDims[cone];
        int64_t n2 = n * n;
        int64_t svec_dim = n * (n + 1) / 2;
        int64_t sz_off = sz_offsets[cone];

        if (n <= 1) {
            if (n == 1) {
                for (int64_t b = 0; b < batchSize; b++) {
                    cudaMemcpyAsync(
                        derivs.psd_eigvals.data() + b * derivs.totalPsdMatDim + mat_off,
                        u + b * m + offset + sz_off,
                        sizeof(double), cudaMemcpyDeviceToDevice, stream);
                    double one = 1.0;
                    cudaMemcpyAsync(
                        derivs.psd_eigvecs.data() + b * derivs.totalPsdMatSqDim + matsq_off,
                        &one, sizeof(double), cudaMemcpyHostToDevice, stream);
                }
            }
            matsq_off += n2; mat_off += n;
            continue;
        }

        int blk_svec = (svec_dim + 255) / 256;
        int blk_n2 = (n2 + 255) / 256;
        int blk_n = (n + 255) / 256;

        for (int64_t b = 0; b < batchSize; b++) {
            double* eigvecs = derivs.psd_eigvecs.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            double* eigvals = derivs.psd_eigvals.data() + b * derivs.totalPsdMatDim + mat_off;
            double* work = derivs.psd_work_mat.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            double* work2 = derivs.psd_work_mat2.data() + b * derivs.totalPsdMatSqDim + matsq_off;

            // svec → matrix (into eigvecs, will be overwritten by syevd)
            MOREAU_KERNEL_LAUNCH(diff_svec_to_mat_kernel, blk_svec, 256, 0, stream,
                eigvecs, u + b * m + offset + sz_off, n);

            // Eigendecompose in-place
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                            n, eigvecs, n, eigvals,
                            derivs.d_psd_cusolver_work, derivs.d_psd_cusolver_work_size,
                            derivs.d_psd_info);

            // Clamp eigenvalues (use omega workspace as temp)
            double* clamped = derivs.psd_omega.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            MOREAU_KERNEL_LAUNCH(clamp_eigenvalues_kernel, blk_n, 256, 0, stream, clamped, eigvals, n);

            // Q_scaled = Q * diag(clamped)
            MOREAU_KERNEL_LAUNCH(scale_columns_kernel, blk_n2, 256, 0, stream, work, eigvecs, clamped, n);

            // result = Q_scaled * Q^T
            double one = 1.0, zero = 0.0;
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                          n, n, n, &one, work, n, eigvecs, n, &zero, work2, n);

            // matrix → svec → pi_u
            MOREAU_KERNEL_LAUNCH(diff_mat_to_svec_kernel, blk_svec, 256, 0, stream,
                pi_u + b * m + offset + sz_off, work2, n);
        }

        matsq_off += n2; mat_off += n;
    }
}

// ============================================================================
// PSD Derivative (Ω-matrix)
// ============================================================================

void compute_psd_derivative(
    double* psd_H, const double* u, int64_t offset,
    const Cones& cones, ConeDerivatives& derivs,
    int64_t batchSize, int64_t m, cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    // 1×1 cones
    int threads = std::min((int)cones.numPsdCones, 256);
    MOREAU_KERNEL_LAUNCH(psd_derivative_1x1_kernel, batchSize, threads, 0, stream,
        psd_H, u, offset, cones.numPsdCones,
        cones.d_psd_dims, cones.d_psd_sz_offsets, cones.d_psd_Hs_offsets,
        derivs.totalPsdHsEntries, batchSize, m);

    bool has_general = false;
    for (int64_t i = 0; i < cones.numPsdCones; i++)
        if (cones.psdConeDims[i] > 1) { has_general = true; break; }
    if (!has_general) return;

    cublasHandle_t cublas = cones.cublasH_;
    cublasSetStream_v2(cublas, stream);

    int64_t matsq_off = 0, mat_off = 0, svec_off = 0, hs_off = 0;

    for (int64_t cone = 0; cone < cones.numPsdCones; cone++) {
        int64_t n = cones.psdConeDims[cone];
        int64_t n2 = n * n;
        int64_t svec_dim = n * (n + 1) / 2;
        int64_t hs_size = svec_dim * (svec_dim + 1) / 2;

        if (n <= 1) {
            matsq_off += n2; mat_off += n;
            svec_off += svec_dim; hs_off += hs_size;
            continue;
        }

        int blk_n2 = (n2 + 255) / 256;
        int blk_svec = (svec_dim + 255) / 256;

        for (int64_t b = 0; b < batchSize; b++) {
            double* Q = derivs.psd_eigvecs.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            double* lam = derivs.psd_eigvals.data() + b * derivs.totalPsdMatDim + mat_off;
            double* omega = derivs.psd_omega.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            double* W1 = derivs.psd_work_mat.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            double* W2 = derivs.psd_work_mat2.data() + b * derivs.totalPsdMatSqDim + matsq_off;
            double* sv = derivs.psd_work_svec.data() + b * derivs.totalPsdSvecDim + svec_off;
            double* H = psd_H + b * derivs.totalPsdHsEntries + hs_off;

            // Build Ω from cached eigenvalues
            MOREAU_KERNEL_LAUNCH(build_omega_kernel, blk_n2, 256, 0, stream, omega, lam, n);

            double one = 1.0, zero = 0.0;

            for (int64_t k = 0; k < svec_dim; k++) {
                // E_k directly into matrix form (fused
                // memset+set_element+svec_to_mat — 3 launches → 1).
                MOREAU_KERNEL_LAUNCH(build_E_k_matrix_kernel, blk_n2, 256, 0, stream, W1, k, n);

                // Q^T * E_k * Q
                cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, Q, n, W1, n, &zero, W2, n);
                cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, W2, n, Q, n, &zero, W1, n);

                // Hadamard with Ω
                MOREAU_KERNEL_LAUNCH(hadamard_kernel, blk_n2, 256, 0, stream, W1, omega, n2);

                // Q * (...) * Q^T
                cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, Q, n, W1, n, &zero, W2, n);
                cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, W2, n, Q, n, &zero, W1, n);

                // mat → svec (writes full result to sv)
                MOREAU_KERNEL_LAUNCH(diff_mat_to_svec_kernel, blk_svec, 256, 0, stream, sv, W1, n);

                // Scatter column k into H upper triangle
                int blk_k = (k + 1 + 255) / 256;
                MOREAU_KERNEL_LAUNCH(scatter_jacobian_col_kernel, blk_k, 256, 0, stream, H, sv, k);
            }
        }

        matsq_off += n2; mat_off += n;
        svec_off += svec_dim; hs_off += hs_size;
    }
}

// ============================================================================
// Direct-x PSD Derivative
// ============================================================================
//
// Mirrors `compute_psd_derivative` but for IFT-direct backward: u_x = z_x −
// x[J] (gathered svec) is built on the fly, eigendecomp is computed fresh
// (not cached from forward), and the Jacobian is written as a full dense
// `svec_dim × svec_dim` row-major matrix per direct-x PSD cone (the
// populate kernel reads `H[k][l]` directly without unpacking).

// Gather u_x_svec[k] = z_x_eq[xc_off + k] − x_eq[xcone_indices[xc_off + k]]
__global__ void gather_xcone_psd_u_kernel(
    double* __restrict__ u_x_svec,                  // [svec_dim] scratch
    const double* __restrict__ x_eq_b,              // [n]
    const double* __restrict__ z_x_eq_b,            // [totalXConeNumel]
    const int64_t* __restrict__ xcone_indices_b,    // [totalXConeNumel]
    int64_t xc_off,
    int64_t svec_dim
) {
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= svec_dim) return;
    int64_t Jxk = xcone_indices_b[xc_off + k];
    u_x_svec[k] = z_x_eq_b[xc_off + k] - x_eq_b[Jxk];
}

// Scatter Jacobian column k into row-major dense H storage:
// H[i, k] = col_svec[i] for i = 0..svec_dim-1.
__global__ void scatter_jacobian_col_dense_kernel(
    double* __restrict__ H, const double* __restrict__ col_svec, int64_t k, int64_t svec_dim
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= svec_dim) return;
    H[i * svec_dim + k] = col_svec[i];
}

void compute_xcone_psd_derivative(
    double* xcone_psd_H,
    const double* x_eq,
    const double* z_x_eq,
    const Cones& cones,
    ConeDerivatives& derivs,
    int64_t totalXConeNumel,
    int64_t totalXPsdKkt,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream
) {
    // Find direct-x PSD cones; nothing to do if none.
    bool has_psd = false;
    int64_t max_n_dim = 0, max_svec_dim = 0;
    for (const auto& xc : cones.dir_cones) {
        if (xc.kind == XConeKind::PSD) {
            has_psd = true;
            max_n_dim = std::max(max_n_dim, xc.psd_k);
            int64_t sv = static_cast<int64_t>(xc.indices.size());
            max_svec_dim = std::max(max_svec_dim, sv);
        }
    }
    if (!has_psd) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    // Allocate cuSOLVER work buffer sized for the largest direct-x PSD cone
    // if the existing workspace is insufficient. Reuse `derivs.d_psd_cusolver_work`
    // when its size already covers `max_n_dim`.
    int needed_work_size = 0;
    cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_LOWER,
                                 (int)max_n_dim, nullptr, (int)max_n_dim,
                                 nullptr, &needed_work_size);
    double* cusolver_work = derivs.d_psd_cusolver_work;
    int cusolver_work_size = derivs.d_psd_cusolver_work_size;
    bool work_owned = false;
    if (cusolver_work == nullptr || cusolver_work_size < needed_work_size) {
        cudaMalloc(&cusolver_work, sizeof(double) * needed_work_size);
        cusolver_work_size = needed_work_size;
        work_owned = true;
    }
    int* d_info = derivs.d_psd_info;
    bool info_owned = false;
    if (d_info == nullptr) {
        cudaMalloc(&d_info, sizeof(int));
        info_owned = true;
    }

    // Allocate per-call scratch sized for the largest direct-x PSD cone:
    // mat (n²), eigvecs (n²), eigvals (n), omega (n²), W1, W2 (n² each),
    // u_x_svec, col_svec (svec_dim each), and a flat indices buffer.
    int64_t max_n2 = max_n_dim * max_n_dim;
    double* d_eigvecs;  cudaMalloc(&d_eigvecs, sizeof(double) * max_n2);
    double* d_eigvals;  cudaMalloc(&d_eigvals, sizeof(double) * max_n_dim);
    double* d_omega;    cudaMalloc(&d_omega,   sizeof(double) * max_n2);
    double* d_W1;       cudaMalloc(&d_W1,      sizeof(double) * max_n2);
    double* d_W2;       cudaMalloc(&d_W2,      sizeof(double) * max_n2);
    double* d_u_svec;   cudaMalloc(&d_u_svec,  sizeof(double) * max_svec_dim);
    double* d_col_svec; cudaMalloc(&d_col_svec,sizeof(double) * max_svec_dim);

    // Upload flattened xcone indices (per-cone, rebuilt each call — small).
    std::vector<int64_t> indices_flat;
    indices_flat.reserve(totalXConeNumel);
    for (const auto& xc : cones.dir_cones) {
        for (int64_t v : xc.indices) indices_flat.push_back(v);
    }
    int64_t* d_indices_flat = nullptr;
    cudaMalloc(&d_indices_flat, sizeof(int64_t) * totalXConeNumel);
    cudaMemcpyAsync(d_indices_flat, indices_flat.data(),
                    sizeof(int64_t) * totalXConeNumel,
                    cudaMemcpyHostToDevice, stream);

    // Iterate cones. Track running offsets:
    //   xc_off:   into z_x_eq / xcone_indices (sum of dims so far)
    //   psd_off:  into xcone_psd_H (sum of svec_dim² so far for PSD cones)
    int64_t xc_off = 0;
    int64_t psd_off = 0;
    for (const auto& xc : cones.dir_cones) {
        int64_t cone_dim = static_cast<int64_t>(xc.indices.size());
        if (xc.kind != XConeKind::PSD) {
            xc_off += cone_dim;
            continue;
        }

        int64_t n_mat = xc.psd_k;
        int64_t n2 = n_mat * n_mat;
        int64_t svec_dim = cone_dim;  // == n_mat * (n_mat + 1) / 2

        int blk_n2  = (int)((n2 + 255) / 256);
        int blk_n   = (int)((n_mat + 255) / 256);
        int blk_sv  = (int)((svec_dim + 255) / 256);

        for (int64_t b = 0; b < batchSize; b++) {
            const double* x_b  = x_eq + b * n;
            const double* zx_b = z_x_eq + b * totalXConeNumel;
            double* H_b = xcone_psd_H + b * totalXPsdKkt + psd_off;

            // Gather u_x_svec = z_x_eq[xc_off..xc_off+sv] - x_eq[J]
            MOREAU_KERNEL_LAUNCH(gather_xcone_psd_u_kernel, blk_sv, 256, 0, stream,
                d_u_svec, x_b, zx_b, d_indices_flat, xc_off, svec_dim);

            // svec → matrix into eigvecs (will be overwritten by syevd).
            MOREAU_KERNEL_LAUNCH(diff_svec_to_mat_kernel, blk_sv, 256, 0, stream,
                d_eigvecs, d_u_svec, n_mat);

            // Eigendecompose in-place: eigvecs ← Q, eigvals ← λ.
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER,
                             (int)n_mat, d_eigvecs, (int)n_mat, d_eigvals,
                             cusolver_work, cusolver_work_size, d_info);

            // Build Ω from eigenvalues.
            MOREAU_KERNEL_LAUNCH(build_omega_kernel, blk_n2, 256, 0, stream, d_omega, d_eigvals, n_mat);

            double one = 1.0, zero = 0.0;
            for (int64_t k = 0; k < svec_dim; k++) {
                // E_k directly into matrix form (fused
                // memset+set_element+svec_to_mat — 3 launches → 1).
                MOREAU_KERNEL_LAUNCH(build_E_k_matrix_kernel, blk_n2, 256, 0, stream, d_W1, k, n_mat);

                // Q^T E_k Q.
                cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                               (int)n_mat, (int)n_mat, (int)n_mat,
                               &one, d_eigvecs, (int)n_mat,
                               d_W1, (int)n_mat, &zero, d_W2, (int)n_mat);
                cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               (int)n_mat, (int)n_mat, (int)n_mat,
                               &one, d_W2, (int)n_mat,
                               d_eigvecs, (int)n_mat, &zero, d_W1, (int)n_mat);

                // Hadamard with Ω.
                MOREAU_KERNEL_LAUNCH(hadamard_kernel, blk_n2, 256, 0, stream, d_W1, d_omega, n2);

                // Q (...) Q^T.
                cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               (int)n_mat, (int)n_mat, (int)n_mat,
                               &one, d_eigvecs, (int)n_mat,
                               d_W1, (int)n_mat, &zero, d_W2, (int)n_mat);
                cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                               (int)n_mat, (int)n_mat, (int)n_mat,
                               &one, d_W2, (int)n_mat,
                               d_eigvecs, (int)n_mat, &zero, d_W1, (int)n_mat);

                // matrix → svec → col k of Jacobian.
                MOREAU_KERNEL_LAUNCH(diff_mat_to_svec_kernel, blk_sv, 256, 0, stream, d_col_svec, d_W1, n_mat);

                // Scatter into row-major dense H_b: H[i, k] = col[i].
                MOREAU_KERNEL_LAUNCH(scatter_jacobian_col_dense_kernel, blk_sv, 256, 0, stream,
                    H_b, d_col_svec, k, svec_dim);
            }
        }

        xc_off  += cone_dim;
        psd_off += svec_dim * svec_dim;
    }

    cudaStreamSynchronize(stream);
    cudaFree(d_eigvecs);
    cudaFree(d_eigvals);
    cudaFree(d_omega);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_u_svec);
    cudaFree(d_col_svec);
    cudaFree(d_indices_flat);
    if (work_owned) cudaFree(cusolver_work);
    if (info_owned) cudaFree(d_info);
}

} // namespace moreau
