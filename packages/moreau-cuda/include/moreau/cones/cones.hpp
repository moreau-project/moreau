/**
 * @file cones.hpp
 * @brief Cone projection and operations
 */

#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "moreau/vector/vector.hpp"

namespace moreau {

// Forward declaration
enum class ScalingStrategy;

namespace cones {
// SOC/GenPow cones with total dim > PARALLEL_THRESHOLD are processed by a
// block-per-cone parallel kernel (intra-cone parallelism via shared-memory
// reduction). Cones at or below the threshold stay in the composite
// thread-per-cone kernel, which is more efficient for dims where a single
// thread's serial work is small. Algorithm follows the ClarabelGPU paper
// (§4.5, "Dynamic parallelism for second-order cones"), but uses a
// block-per-cone pattern rather than CUDA dynamic parallelism.
// Dim > THRESHOLD goes through update_large_soc_scaling_kernel (one
// block per (batch, cone), `SOC_PARALLEL_BLOCK_SIZE` threads doing
// shared-memory tail reductions). Dim <= THRESHOLD goes through the
// composite kernel's single-thread-per-cone path inline. Threshold=4
// puts EVERY rank-2 sparse SOC (dim > 4) on the parallel path — the
// large-SOC kernel already asserts `dim > 4` (always-sparse invariant),
// so this is the maximally-parallel setting consistent with its
// assumptions. Only the tiny dense cones (dim ∈ {2, 3, 4}, dense
// 6/10/15-entry Hs with no tail loop) stay inline, which is the right
// shape — block-per-cone overhead would dominate at that size. The
// old threshold of 32 left dim ∈ (4, 32] on the slow inline path.
constexpr int64_t SOC_PARALLEL_THRESHOLD = 4;
constexpr int SOC_PARALLEL_BLOCK_SIZE = 256;
constexpr int64_t GENPOW_PARALLEL_THRESHOLD = 32;
constexpr int GENPOW_PARALLEL_BLOCK_SIZE = 256;
} // namespace cones

/**
 * @brief Per-batch margin computation results
 */
struct MarginResults {
    double min_margin;
    double pos_margin;
};


/**
 * @brief Kind tag for direct-x cone entries.
 *
 * Mirrors the `SupportedXConeT` variants on the Rust/CPU side. Direct-x
 * cones constrain `x[J] ∈ K_J` on subvectors of the primal x rather
 * than on slack variables.
 */
enum class XConeKind : int {
    Nonneg = 0,
    SOC = 1,
    PSD = 2,
    Exp = 3,
    Power = 4,
    GenPower = 5,
};


/**
 * @brief User-facing descriptor for a direct-x cone constraint.
 *
 * @field kind     Cone type (Nonneg, SOC, or PSD).
 * @field indices  Indices into x, length >= 1. Must be disjoint across
 *                 all direct-x cones in a problem (enforced at the
 *                 Python/FFI boundary). For PSD: length must equal
 *                 psd_k * (psd_k + 1) / 2 (column-major svec).
 * @field psd_k    Matrix side length for PSD cones; ignored for other
 *                 kinds. Must satisfy indices.size() == psd_k(psd_k+1)/2.
 */
struct SupportedXConeT {
    XConeKind kind = XConeKind::Nonneg;
    std::vector<int64_t> indices;
    int64_t psd_k = 0;

    // Power cone exponent α ∈ (0, 1). Only used when kind == Power.
    double power_alpha = 0.0;

    // GenPower cone parameters. Only used when kind == GenPower.
    // gen_power_alphas has dim1 entries (positive, sum to 1.0).
    // gen_power_dim2 is the tail dimension >= 1.
    // indices.size() must equal gen_power_alphas.size() + gen_power_dim2.
    std::vector<double> gen_power_alphas;
    int64_t gen_power_dim2 = 0;

    int64_t numel() const { return static_cast<int64_t>(indices.size()); }
    int64_t degree() const {
        switch (kind) {
            case XConeKind::Nonneg:   return numel();
            case XConeKind::SOC:      return int64_t{1};
            case XConeKind::PSD:      return psd_k;
            // Asymmetric cones — barrier degree (ν).
            case XConeKind::Exp:      return int64_t{3};
            case XConeKind::Power:    return int64_t{3};
            case XConeKind::GenPower: return numel();  // = dim1 + dim2
        }
        return int64_t{0};
    }
};


/**
 * @brief Cone structure information for conic optimization problems
 *
 * Describes the Cartesian product of convex cones K = K₁ × K₂ × ... × Kₖ
 * where the slack variable s ∈ K.
 *
 * Usage pattern:
 * 1. User creates ConeStructure with default constructor
 * 2. User sets cone counts and powerAlphas
 * 3. User passes to Solver constructor
 * 4. Solver calls initialize() to allocate device memory
 */
struct Cones {
    // Cone counts (set by user before passing to solver)
    int64_t numZeroCones;
    int64_t numNonnegCones;
    int64_t numExpCones;
    int64_t numSocCones;
    int64_t numPowerCones;
    int64_t numPsdCones;
    int64_t numGenPowerCones;

    // Direct-x cones. Supported end-to-end on CUDA (Nonneg, SOC, Exp, Power,
    // GenPower, PSD). initialize() validates each entry's kind/dimensions
    // (e.g. SOC needs >= 2 indices, PSD needs indices.size() == k(k+1)/2) and
    // rejects only genuinely malformed cones.
    std::vector<SupportedXConeT> dir_cones;
    int64_t numXCones;

    // SOC cone dimensions (set by user, host vector)
    // After initialize(), sorted by ascending dimension for warp coherence.
    std::vector<int64_t> socConeDims;  // Size = numSocCones, each >= 2 (sorted)
    std::vector<int64_t> socConeDimsOriginal;  // Original user-specified order (for KKT construction)
    std::vector<int64_t> socSortPerm;  // perm[sorted_idx] = original_idx

    // Power cone parameters (set by user, host vector)
    std::vector<double> powerAlphas;  // Size = numPowerCones

    // PSD cone dimensions (set by user, host vector)
    // Each entry is the matrix dimension n (constraint uses n(n+1)/2 svec elements).
    // After initialize(), sorted by ascending dimension for warp coherence.
    std::vector<int64_t> psdConeDims;          // Size = numPsdCones (sorted)
    std::vector<int64_t> psdConeDimsOriginal;  // Original user-specified order (for KKT)
    std::vector<int64_t> psdSortPerm;          // perm[sorted_idx] = original_idx

    // Derived PSD quantities (computed in initialize())
    int64_t totalPsdSvecDim;       // sum of n_i(n_i+1)/2 (total svec elements in s/z)
    int64_t totalPsdMatDim;        // sum of n_i (total matrix dimensions = degree contribution)
    int64_t totalPsdMatSqDim;      // sum of n_i² (total dense matrix elements)
    int64_t totalPsdHsEntries;     // sum of svec_dim_i*(svec_dim_i+1)/2 (Hessian upper triangle entries)

    // Device arrays for PSD dimension/offset info
    int64_t* d_psd_dims;           // [numPsdCones] - matrix dimension of each PSD cone
    int64_t* d_psd_svec_offsets;   // [numPsdCones+1] - prefix sum of svec dims (for internal arrays)
    int64_t* d_psd_sz_offsets;     // [numPsdCones] - original offset into s/z for each sorted cone
    int64_t* d_psd_Hs_offsets;     // [numPsdCones+1] - prefix sum of Hs entries
    int64_t* d_psd_mat_offsets;    // [numPsdCones+1] - prefix sum of n_i (for lambda array)
    int64_t* d_psd_matsq_offsets;  // [numPsdCones+1] - prefix sum of n_i² (for matrix workspace)
    // Generalized power cone parameters (set by user, host vectors)
    // Each GenPowerCone has dim1 alphas (summing to 1) and dim2 >= 1
    // Total cone dimension = dim1 + dim2
    // After initialize(), sorted by ascending total dim for warp coherence.
    std::vector<double> genPowerAlphas;   // Flattened alpha values (sorted order)
    std::vector<int64_t> genPowerDim1s;   // dim1 per cone (sorted order)
    std::vector<int64_t> genPowerDim2s;   // dim2 per cone (sorted order)
    std::vector<double> genPowerAlphasOriginal;   // Original user order (for KKT)
    std::vector<int64_t> genPowerDim1sOriginal;   // Original user order (for KKT)
    std::vector<int64_t> genPowerDim2sOriginal;   // Original user order (for KKT)
    std::vector<int64_t> genPowerSortPerm;  // perm[sorted_idx] = original_idx

    // Derived SOC quantities (computed in initialize())
    int64_t totalSocDim;           // sum of all SOC dims
    int64_t totalSocHsEntries;     // sum of Hs entries per cone (dense: tri(dim), sparse: dim)
    int64_t totalSocHsDenseEntries; // sum of dim*(dim+1)/2 per cone (always dense, for diff)
    int64_t numSparseSoc;          // count of SOC cones with dim > 4
    // Count of SOC cones large enough to use the block-per-cone parallel kernel
    // (dim > SOC_PARALLEL_THRESHOLD). Because socConeDims is sorted ascending in
    // initialize(), these cones occupy indices [numSocCones - numLargeSoc, numSocCones).
    int64_t numLargeSoc;

    // Derived direct-x quantities (computed in initialize(); zero when numXCones == 0)
    int64_t totalXConeNumel;       // sum of indices.size() across dir_cones
    int64_t totalXConeHsEntries;   // sum of Hs entries per x-cone (nonneg: dim, SOC dense: tri, SOC sparse: dim, PSD: svec_dim*(svec_dim+1)/2)
    int64_t totalSparseXSocDim;    // sum of dim over x-cone SOC with dim > 4
    int64_t numSparseXSoc;         // count of x-cone SOC with dim > 4

    // PSD direct-x derived quantities (parallel to slack PSD totals).
    int64_t numXPsdCones;          // count of x-cones with kind == PSD
    int64_t totalXPsdSvecDim;      // sum of indices.size() over x-PSD = sum k(k+1)/2
    int64_t totalXPsdMatDim;       // sum of psd_k over x-PSD (eigenvalue array length)
    int64_t totalXPsdMatSqDim;     // sum of psd_k^2 over x-PSD (matrix workspace length)
    int64_t totalXPsdHsEntries;    // sum of svec_dim*(svec_dim+1)/2 over x-PSD = direct-x PSD Hs upper-tri entries

    // Asymmetric direct-x derived quantities. Each Exp/Power cone is 3-D
    // dense; GenPower is variable-dim. Hs storage uses the standard packed
    // upper-tri (Exp/Power: 6 entries each) and is allocated inside the
    // shared `xcone_Hs` buffer, picked up by `refresh_xcone_hs`.
    int64_t numXExpCones;          // count of x-cones with kind == Exp
    int64_t numXPowerCones;        // count of x-cones with kind == Power
    int64_t numXGenPowerCones;     // count of x-cones with kind == GenPower

    // Derived GenPowerCone quantities (computed in initialize())
    int64_t totalGenPowerDim;        // sum of (dim1+dim2) across all GenPowerCones
    int64_t totalGenPowerAlphas;     // sum of dim1 across all GenPowerCones (= genPowerAlphas.size())
    int64_t totalGenPowerGradEntries; // sum of (dim1+dim2) per cone = totalGenPowerDim
    int64_t totalGenPowerHsEntries;  // sum of Hs entries per cone (dense: tri(dim), sparse: dim)
    int64_t totalGenPowerHsDenseEntries; // sum of dim*(dim+1)/2 per cone (always dense, for diff)
    int64_t totalGenPowerDimSq;      // sum of (dim1+dim2)^2 across all GenPowerCones
    int64_t totalGenPowerDim2;       // sum of dim2 across all GenPowerCones
    int64_t numSparseGenPow;         // count of cones with total dim > 4
    // Count of GenPowerCones large enough to use the block-per-cone kernel
    // (dim > GENPOW_PARALLEL_THRESHOLD). Sorted-ascending layout puts them
    // at indices [numGenPowerCones - numLargeGenPow, numGenPowerCones).
    int64_t numLargeGenPow;

    // Device arrays for SOC dimension/offset info
    int64_t* d_soc_dims;           // [numSocCones] - dimension of each SOC cone
    int64_t* d_soc_offsets;        // [numSocCones+1] - prefix sum of sorted dims (for internal arrays)
    int64_t* d_soc_sz_offsets;     // [numSocCones] - original offset into s/z for each sorted cone
    int64_t* d_soc_Hs_offsets;     // [numSocCones+1] - prefix sum of Hs entries
    int64_t* d_soc_Hs_dense_offsets; // [numSocCones+1] - prefix sum of dense upper-tri entries (for diff)
    int64_t* d_soc_sparse_offsets;   // [numSocCones+1] - prefix sum of dim for sparse-only cones (0 for dense)
    int64_t* d_soc_sparse_indices;   // [numSocCones] - sparse cone index for each SOC cone (-1 if dense)

    // Device arrays for direct-x cone dimension/offset info
    // (allocated in initialize() only when numXCones > 0; nullptr otherwise).
    //
    // d_xcone_kinds[i]      - XConeKind value for cone i (0=Nonneg, 1=SOC, 2=PSD)
    // d_xcone_dims[i]       - number of primal indices for cone i
    // d_xcone_numel_offsets - prefix sum of dims, length numXCones+1
    // d_xcone_hs_offsets    - prefix sum of Hs entries, length numXCones+1
    // d_xcone_indices       - flattened indices into x, length totalXConeNumel
    // d_xcone_sorted_indices - sorted copy of indices per cone (CSC column validity)
    // d_xcone_cone_pos_for_sorted - permutation sorted→cone-local position
    // d_xcone_sparse_indices - [numXCones] sparse index (>=0) if SOC+dim>4, -1 otherwise
    // d_xcone_sparse_offsets - [numXCones+1] prefix sum of dim for sparse x-SOC cones only
    // d_xcone_kind_per_entry - [totalXConeNumel] kind of the cone each flat entry
    //                           belongs to. Lets flat-iteration kernels (nonneg
    //                           step-math) skip SOC entries with O(1) check.
    // d_xcone_psd_idx        - [numXCones] index into PSD-only arrays (-1 if not PSD)
    // d_xcone_psd_k          - [numXPsdCones] matrix dim k for each x-PSD cone
    // d_xcone_psd_svec_offsets - [numXPsdCones+1] prefix sum of svec_dim for x-PSD only
    // d_xcone_psd_mat_offsets  - [numXPsdCones+1] prefix sum of k (lambda offsets)
    // d_xcone_psd_matsq_offsets - [numXPsdCones+1] prefix sum of k^2 (matrix workspace offsets)
    // d_xcone_psd_hs_offsets    - [numXPsdCones+1] prefix sum of svec_dim*(svec_dim+1)/2 (Hs upper-tri)
    // d_xcone_psd_in_full_offsets - [numXPsdCones] start offset of each PSD x-cone within
    //                          the all-xcones flat layout (= d_xcone_numel_offsets[c]).
    //                          Doubles as offset into d_xcone_indices[] and z_x[].
    int64_t* d_xcone_kinds;
    int64_t* d_xcone_dims;
    int64_t* d_xcone_numel_offsets;
    int64_t* d_xcone_hs_offsets;
    int64_t* d_xcone_indices;
    int64_t* d_xcone_sorted_indices;
    int64_t* d_xcone_cone_pos_for_sorted;
    int64_t* d_xcone_sparse_indices;
    int64_t* d_xcone_sparse_offsets;
    int64_t* d_xcone_kind_per_entry;
    int64_t* d_xcone_psd_idx;
    int64_t* d_xcone_psd_k;
    // Host mirror of d_xcone_psd_k, populated once in initialize(). The
    // PSD scaling/step kernels need per-cone k on the host to drive cuBLAS
    // sizes; pulling it from the device per IPM iteration costs a D→H copy
    // + stream sync (violates the "no host↔device transfers in the loop"
    // invariant). Cache it once at construction.
    std::vector<int64_t> xconePsdDimsHost;
    int64_t* d_xcone_psd_svec_offsets;
    int64_t* d_xcone_psd_mat_offsets;
    int64_t* d_xcone_psd_matsq_offsets;
    int64_t* d_xcone_psd_hs_offsets;
    int64_t* d_xcone_psd_in_full_offsets;

    // Asymmetric direct-x device tables.
    // d_xcone_exp_idx[c]    - index into per-Exp arrays (-1 if not Exp)
    // d_xcone_pow_idx[c]    - index into per-Power arrays (-1 if not Power)
    // d_xcone_pow_alpha[i]  - α for the i-th Power x-cone (i = d_xcone_pow_idx[c])
    // d_xcone_genpow_idx[c] - index into per-GenPower arrays (-1 if not GenPower)
    // d_xcone_genpow_dim1s[i]         - dim1 for the i-th GenPower x-cone
    // d_xcone_genpow_dim2s[i]         - dim2 for the i-th GenPower x-cone
    // d_xcone_genpow_alpha_offsets[i] - prefix sum of dim1 (length numXGenPowerCones+1)
    // d_xcone_genpow_dim_offsets[i]   - prefix sum of (dim1+dim2) (length numXGenPowerCones+1)
    // d_xcone_genpow_sparse_idx[i]    - sparse cone index (-1 if dense)
    // d_xcone_genpow_sparse_offsets[i]- prefix sum of dim over sparse-only (length numXGenPowerCones+1)
    // d_xcone_genpow_sparse_alpha_offsets[sc] - alpha offset (a_off) for the sc-th sparse GenPow cone
    //                                          (global, into xcone_genpow_q; length numSparseXGenPow)
    // d_xcone_genpow_sparse_q_offsets[sc]     - sparse-only alpha prefix (for H_xcone_genpow_q_idx_)
    //                                          (starts from 0 for sparse-only; length numSparseXGenPow)
    // d_xcone_genpow_sparse_r_offsets[sc]     - sparse-only dim2 prefix (for H_xcone_genpow_r_idx_)
    //                                          (starts from 0 for sparse-only; length numSparseXGenPow)
    // d_xcone_genpow_sparse_to_gidx[sc] - genpow cone index for the sc-th sparse cone
    //                                     (length numSparseXGenPow; reverse lookup for dim1/dim2)
    // d_xcone_genpow_alphas[off+j]    - alpha values (flattened, indexed by alpha_offsets)
    int64_t* d_xcone_exp_idx;
    int64_t* d_xcone_pow_idx;
    double*  d_xcone_pow_alpha;
    int64_t* d_xcone_genpow_idx;
    int64_t* d_xcone_genpow_dim1s;
    int64_t* d_xcone_genpow_dim2s;
    int64_t* d_xcone_genpow_alpha_offsets;
    int64_t* d_xcone_genpow_dim_offsets;
    int64_t* d_xcone_genpow_sparse_idx;
    int64_t* d_xcone_genpow_sparse_offsets;
    int64_t* d_xcone_genpow_sparse_alpha_offsets;
    int64_t* d_xcone_genpow_sparse_q_offsets;
    int64_t* d_xcone_genpow_sparse_r_offsets;
    int64_t* d_xcone_genpow_sparse_to_gidx;
    double*  d_xcone_genpow_alphas;

    // Derived direct-x GenPower quantities (computed in initialize())
    int64_t totalXGenPowerDim;        // sum of (dim1+dim2) over x-GenPow cones
    int64_t totalXGenPowerAlphas;     // sum of dim1 over x-GenPow cones
    int64_t totalXGenPowerDim2;       // sum of dim2 over x-GenPow cones
    int64_t numSparseXGenPow;         // count of x-GenPow with dim > 4
    int64_t totalSparseXGenPowDim;    // sum of dim over sparse x-GenPow cones

    // Device arrays for GenPowerCone dimension/offset info
    double*  d_genPowerAlphas;     // [totalGenPowerAlphas] - flattened alpha values (sorted order)
    int64_t* d_genPowerDim1s;      // [numGenPowerCones] - dim1 of each cone (sorted order)
    int64_t* d_genPowerDim2s;      // [numGenPowerCones] - dim2 of each cone (sorted order)
    int64_t* d_genPowerOffsets;    // [numGenPowerCones+1] - prefix sum of (dim1+dim2) (sorted)
    int64_t* d_genPowerAlphaOffsets; // [numGenPowerCones+1] - prefix sum of dim1 (sorted)
    int64_t* d_genPowerHsOffsets;  // [numGenPowerCones+1] - prefix sum of Hs entries (dense: tri, sparse: diag)
    int64_t* d_genPowerHsDenseOffsets; // [numGenPowerCones+1] - prefix sum of dense upper-tri entries (for diff)
    int64_t* d_genPowerDimSqOffsets; // [numGenPowerCones+1] - prefix sum of (dim1+dim2)^2
    int64_t* d_genPowerSzOffsets;  // [numGenPowerCones] - original position in s/z for each sorted cone
    int64_t* d_genPowerDims;       // [numGenPowerCones] - total dim (dim1+dim2) per sorted cone
    int64_t* d_genPowerSparseOffsets; // [numGenPowerCones+1] - prefix sum of dim for sparse-only cones
    int64_t* d_genPowerSparseIndices; // [numGenPowerCones] - sparse cone index (-1 if dense)

    // Internal state (initialized by solver)
    int64_t batchSize;
    double* d_powerAlphas;  // Device copy of powerAlphas

    // Device memory for margin computation (allocated in initialize())
    MarginResults* d_batch_margin_results;  // Pre-allocated buffer for per-batch margin results

    // Device memory for barrier computation (allocated in initialize())
    double* d_barrier_work;  // Pre-allocated per-batch buffer for barrier accumulation
    double* h_barrier_pinned;  // Host-visible pinned pointer for async barrier readback

    // Mapped pinned memory for scaling success detection (avoids CPU/GPU sync).
    // Per-batch scaling success flags.  GPU kernel writes via d_scaling_success;
    // host reads via h_scaling_success_pinned.  Array of batchSize int32_t.
    int32_t* d_scaling_success;           // Device-mapped pointer (for kernel writes)
    int32_t* h_scaling_success_pinned;    // Host-visible pinned pointer (for CPU reads)

    // Nonnegative cone working data
    // Stores w and λ for each nonnegative cone constraint
    BatchedVector nonneg_w;  // [batchSize][numNonnegCones]
    BatchedVector nonneg_lambda;  // [batchSize][numNonnegCones]

    // Second-order cone working data (variable dimension)
    BatchedVector soc_u;  // [batchSize][totalSocDim] - rank-2 update u (sparse cones only)
    BatchedVector soc_v;  // [batchSize][totalSocDim] - rank-2 update v (sparse cones only)
    BatchedVector soc_d;  // [batchSize][numSocCones] - rank-2 scalar d (sparse cones only)
    BatchedVector soc_Hs; // [batchSize][totalSocHsEntries] - dense: upper tri, sparse: diag only
    BatchedVector soc_w;  // [batchSize][totalSocDim] - W scaling vector for each cone
    BatchedVector soc_lambda;  // [batchSize][totalSocDim] - λ = sqrt(s .* z) for each cone
    BatchedVector soc_eta;  // [batchSize][numSocCones] - η scaling factor for each cone

    // Exponential cone working data
    // Each exp cone (3 constraints) stores:
    // - H_dual: Hessian of dual barrier (3x3 symmetric = 6 upper triangular elements)
    // - Hs: scaling matrix μH(z) (3x3 symmetric = 6 elements)
    // - grad: gradient of dual barrier (3 elements)
    // - z: copy of z at scaling point (3 elements)
    BatchedVector exp_H_dual;  // [batchSize][numExpCones * 6]
    BatchedVector exp_Hs;      // [batchSize][numExpCones * 6]
    BatchedVector exp_grad;    // [batchSize][numExpCones * 3]
    BatchedVector exp_z;       // [batchSize][numExpCones * 3]

    // Power cone working data
    // Each power cone (3 constraints) stores:
    // - H_dual: Hessian of dual barrier (3x3 symmetric = 6 upper triangular elements)
    // - Hs: scaling matrix μH(z) (3x3 symmetric = 6 elements)
    // - grad: gradient of dual barrier (3 elements)
    // - z: copy of z at scaling point (3 elements)
    BatchedVector power_H_dual;  // [batchSize][numPowerCones * 6]
    BatchedVector power_Hs;      // [batchSize][numPowerCones * 6]
    BatchedVector power_grad;    // [batchSize][numPowerCones * 3]
    BatchedVector power_z;       // [batchSize][numPowerCones * 3]

    // PSD cone working data
    // Each PSD cone with matrix dim n uses svec_dim = n(n+1)/2 constraints.
    // Scaling requires Cholesky, SVD, and symmetric Kronecker product.
    BatchedVector psd_lambda;       // [batchSize][totalPsdMatDim] - eigenvalues λ
    BatchedVector psd_Lambdaisqrt;  // [batchSize][totalPsdMatDim] - Λ^{-1/2} = 1/√λ
    BatchedVector psd_R;            // [batchSize][totalPsdMatSqDim] - R scaling matrix (n×n per cone)
    BatchedVector psd_Rinv;         // [batchSize][totalPsdMatSqDim] - R^{-1} (n×n per cone)
    BatchedVector psd_Hs;           // [batchSize][totalPsdHsEntries] - Hessian upper triangle
    BatchedVector psd_work_mat1;    // [batchSize][totalPsdMatSqDim] - workspace matrix
    BatchedVector psd_work_mat2;    // [batchSize][totalPsdMatSqDim] - workspace matrix
    BatchedVector psd_work_mat3;    // [batchSize][totalPsdMatSqDim] - workspace matrix
    BatchedVector psd_work_svec;    // [batchSize][totalPsdSvecDim] - workspace svec

    // cuSOLVER/cuBLAS handles for PSD operations (Cholesky, SVD, eigendecomp)
    cusolverDnHandle_t cusolverH_ = nullptr;
    cublasHandle_t cublasH_ = nullptr;

    // Pre-allocated cuSOLVER info and workspace for PSD scaling
    int* d_psd_info_ = nullptr;        // cuSOLVER info output (device, pre-allocated)
    double* d_psd_work_ = nullptr;     // cuSOLVER dpotrf workspace (device, pre-allocated)
    int psd_work_size_ = 0;            // Size of dpotrf workspace
    double* d_psd_gesvd_work_ = nullptr;  // cuSOLVER dgesvd workspace
    int psd_gesvd_work_size_ = 0;
    double* d_psd_syevd_work_ = nullptr;  // cuSOLVER dsyevd workspace (for step length)
    int psd_syevd_work_size_ = 0;
    // Generalized power cone working data (variable dimension)
    // Uses rank-3 representation: Hs = μ*(D + pp' - qq' - rr')
    // Instead of storing full Hs, store diagonal D and vectors p, q, r
    BatchedVector genpow_grad;   // [batchSize][totalGenPowerDim] - dual barrier gradient
    BatchedVector genpow_z;      // [batchSize][totalGenPowerDim] - copy of z at scaling point
    BatchedVector genpow_Hs;     // [batchSize][totalGenPowerHsEntries] - packed upper-tri of Hs
    BatchedVector genpow_p;      // [batchSize][totalGenPowerDim] - rank-3 update vector p
    BatchedVector genpow_q;      // [batchSize][totalGenPowerAlphas] - rank-3 update vector q (dim1 per cone)
    BatchedVector genpow_r;      // [batchSize][totalGenPowerDim - totalGenPowerAlphas] - rank-3 update vector r (dim2 per cone)
    BatchedVector genpow_d1;     // [batchSize][totalGenPowerAlphas] - diagonal vector d1 (dim1 per cone)
    BatchedVector genpow_d2;     // [batchSize][numGenPowerCones] - diagonal scalar d2 (one per cone)
    BatchedVector genpow_mu;     // [batchSize][1] - mu per batch (stored during scaling for mul_Hs)

    // Direct-x cone working data. Analogous to the slack cone buffers above but
    // indexed over x[J] subvectors instead of slack rows. Allocated when
    // numXCones > 0.
    //
    // Nonneg x-cones: w, lambda live in xcone_w / xcone_lambda (shared with SOC
    // since all entries are per-primal-index); eta is unused (always 0).
    // SOC x-cones dense (dim<=4): xcone_Hs stores upper-tri packed.
    // SOC x-cones sparse (dim>4): xcone_Hs stores diagonal only and the rank-2
    //   expansion columns live in xcone_u / xcone_v / xcone_d (same semantics
    //   as soc_u / soc_v / soc_d for slack SOCs).
    BatchedVector xcone_w;       // [batchSize][totalXConeNumel]
    BatchedVector xcone_lambda;  // [batchSize][totalXConeNumel]
    BatchedVector xcone_eta;     // [batchSize][numXCones]   (SOC only; 0 for nonneg entries)
    BatchedVector xcone_Hs;      // [batchSize][totalXConeHsEntries]
    BatchedVector xcone_u;       // [batchSize][totalSparseXSocDim]
    BatchedVector xcone_v;       // [batchSize][totalSparseXSocDim]
    BatchedVector xcone_d;       // [batchSize][numSparseXSoc]
    BatchedVector xcone_z;       // [batchSize][totalXConeNumel]  mirror of Variables.z_x for kernels
    // Asymmetric direct-x: ∇F_primal(x) at the scaling point, used by
    // `add_combined_ds_shift_*` (shift = σμ·∇F_primal(x_pt)). Shares the
    // numel layout with xcone_w/xcone_lambda. Zero-filled for symmetric
    // (nonneg/SOC/PSD) entries.
    BatchedVector xcone_grad_primal;  // [batchSize][totalXConeNumel]

    // PSD direct-x working data. Mirrors the slack-PSD buffers but
    // indexed over x-PSD cones only. Allocated when numXPsdCones > 0.
    BatchedVector xcone_psd_lambda;        // [batchSize][totalXPsdMatDim]    eigenvalues λ
    BatchedVector xcone_psd_Lambdaisqrt;   // [batchSize][totalXPsdMatDim]    Λ^{-1/2}
    BatchedVector xcone_psd_R;             // [batchSize][totalXPsdMatSqDim]  R scaling matrix (k×k per cone)
    BatchedVector xcone_psd_Rinv;          // [batchSize][totalXPsdMatSqDim]  R^{-1}
    BatchedVector xcone_psd_Hs;            // [batchSize][totalXPsdHsEntries] svec×svec upper-tri Hessian
    BatchedVector xcone_psd_work_mat1;     // [batchSize][totalXPsdMatSqDim]  workspace
    BatchedVector xcone_psd_work_mat2;     // [batchSize][totalXPsdMatSqDim]
    BatchedVector xcone_psd_work_mat3;     // [batchSize][totalXPsdMatSqDim]
    BatchedVector xcone_psd_combined_scratch; // [batchSize][totalXConeNumel] svec scratch for `recover_dz_x_combined_psd`
    // Eigenvalue scratch for compute_xcone_psd_alpha when slack PSD is empty.
    // Sized [batchSize][max_xcone_psd_k] so each batch slot can hold the
    // eigvals of the largest xcone-PSD cone. Avoids per-iter cudaMalloc.
    // When slack PSD is present, psd_work_svec is reused instead.
    BatchedVector xcone_psd_eigvals_scratch;  // [batchSize][max_xcone_psd_k]

    // Direct-x GenPower working storage.  Mirrors the slack genpow_* buffers
    // but uses primal-barrier formulas (update_primal_grad_H).
    // Allocated when numXGenPowerCones > 0.
    BatchedVector xcone_genpow_p;   // [batchSize][totalXGenPowerDim]
    BatchedVector xcone_genpow_q;   // [batchSize][totalXGenPowerAlphas]
    BatchedVector xcone_genpow_r;   // [batchSize][totalXGenPowerDim2]
    BatchedVector xcone_genpow_d1;  // [batchSize][totalXGenPowerAlphas]
    BatchedVector xcone_genpow_d2;  // [batchSize][numXGenPowerCones]

    // Mosek-Tunçel rank-6 PD scaling for direct-x GenPow (mirrors CPU
    // GenPowerConeData::pd_axes/pd_coefs/pd_signs/pd_active on the primal
    // side). Allocated when numXGenPowerCones > 0. When pd_active[c] = 0.0,
    // the cone runs rank-3 only (D + pp' - qq' - rr'); when 1.0, the rank-6
    // axis correction is added: Hs += Σ_k sign_k · coef_k · a_k a_k'.
    BatchedVector xcone_genpow_pd_axes;     // (B, 6 * totalXGenPowerDim)
    BatchedVector xcone_genpow_pd_coefs;    // (B, 6 * numXGenPowerCones)
    BatchedVector xcone_genpow_pd_signs;    // (B, 6 * numXGenPowerCones), ±1.0 as double
    BatchedVector xcone_genpow_pd_active;   // (B, numXGenPowerCones), 0.0 or 1.0
    BatchedVector xcone_genpow_pd_workspace;// (B, 8 * totalXGenPowerDim)

    // Smoothing workspace (for Newton iteration in smoothing_all_cones kernel)
    // Replaces per-thread stack arrays, enabling arbitrary GenPowerCone dimensions.
    BatchedVector genpow_smooth_zlocal; // [batchSize][totalGenPowerDim]
    BatchedVector genpow_smooth_wlocal; // [batchSize][totalGenPowerDim]
    BatchedVector genpow_smooth_res;    // [batchSize][totalGenPowerDim]
    BatchedVector genpow_smooth_delta;  // [batchSize][totalGenPowerDim]
    BatchedVector genpow_smooth_hmat;   // [batchSize][totalGenPowerDimSq]
    BatchedVector genpow_smooth_lmat;   // [batchSize][totalGenPowerDimSq]

    // 3rd-order Mehrotra η correction workspace (used in combined_ds_shift).
    // Mirrors the CPU `correction_u` / `correction_gpsi` scratch in
    // GenPowerConeData. `genpow_correction_u` holds the rank-3-SMW solve
    // `H_dual⁻¹·ds` (length `totalGenPowerDim`); `genpow_correction_gpsi`
    // holds `g_ψ` (length `totalGenPowerDim`). Both are dim-sized per cone
    // and packed by cone offset.
    BatchedVector genpow_correction_u;    // [batchSize][totalGenPowerDim]
    BatchedVector genpow_correction_gpsi; // [batchSize][totalGenPowerDim]

    // Mosek-Tunçel rank-6 sparse PD scaling (mirrors CPU
    // GenPowerConeData::pd_axes/pd_coefs/pd_signs/pd_active). When
    // pd_active[i] is true, the cone's Hs is
    //   Hs = μ·H_dual + Σ_{k=0..6} pd_signs[k] · pd_coefs[k] · pd_axes[k] · pd_axes[k]'
    // Otherwise these are zero and Hs = μ·H_dual (dual fallback).
    //
    // Layout:
    //   pd_axes  : (B, 6 · totalGenPowerDim)   — 6 axis vectors, each
    //                                            dim-sized per cone, packed
    //                                            by axis-then-cone within batch
    //   pd_coefs : (B, 6 · numGenPowerCones)   — 6 coefs per cone (≥ 0)
    //   pd_signs : (B, 6 · numGenPowerCones)   — int8 signs (±1)
    //   pd_active: (B, numGenPowerCones)       — int8 boolean per cone
    //
    // Workspace for the QR-based projector + 4×4 Jacobi eigendecomp in
    // try_compute_pd_axes_kernel:
    //   pd_workspace : (B, 8 · totalGenPowerDim) — packs gs, δs, δz,
    //                                              e_z, e_dzp, q3, q4, h_z
    BatchedVector genpow_pd_axes;     // (B, 6 * totalGenPowerDim) — flat
    BatchedVector genpow_pd_coefs;    // (B, 6 * numGenPowerCones)
    BatchedVector genpow_pd_signs;    // (B, 6 * numGenPowerCones), ±1.0 stored as double
    BatchedVector genpow_pd_active;   // (B, numGenPowerCones), 0.0 or 1.0 stored as double
    BatchedVector genpow_pd_workspace;// (B, 8 * totalGenPowerDim)

    /**
     * @brief Default constructor for user setup
     */
    Cones()
        : numZeroCones(0),
          numNonnegCones(0),
          numExpCones(0),
          numSocCones(0),
          numPowerCones(0),
          numPsdCones(0),
          totalPsdSvecDim(0),
          totalPsdMatDim(0),
          totalPsdMatSqDim(0),
          totalPsdHsEntries(0),
          d_psd_dims(nullptr),
          d_psd_svec_offsets(nullptr),
          d_psd_sz_offsets(nullptr),
          d_psd_Hs_offsets(nullptr),
          d_psd_mat_offsets(nullptr),
          d_psd_matsq_offsets(nullptr),
          numGenPowerCones(0),
          numXCones(0),
          totalXConeNumel(0),
          totalXConeHsEntries(0),
          totalSparseXSocDim(0),
          numSparseXSoc(0),
          numXPsdCones(0),
          totalXPsdSvecDim(0),
          totalXPsdMatDim(0),
          totalXPsdMatSqDim(0),
          totalXPsdHsEntries(0),
          d_xcone_kinds(nullptr),
          d_xcone_dims(nullptr),
          d_xcone_numel_offsets(nullptr),
          d_xcone_hs_offsets(nullptr),
          d_xcone_indices(nullptr),
          d_xcone_sorted_indices(nullptr),
          d_xcone_cone_pos_for_sorted(nullptr),
          d_xcone_sparse_indices(nullptr),
          d_xcone_sparse_offsets(nullptr),
          d_xcone_kind_per_entry(nullptr),
          d_xcone_psd_idx(nullptr),
          d_xcone_psd_k(nullptr),
          d_xcone_psd_svec_offsets(nullptr),
          d_xcone_psd_mat_offsets(nullptr),
          d_xcone_psd_matsq_offsets(nullptr),
          d_xcone_psd_hs_offsets(nullptr),
          d_xcone_psd_in_full_offsets(nullptr),
          numXExpCones(0),
          numXPowerCones(0),
          numXGenPowerCones(0),
          d_xcone_exp_idx(nullptr),
          d_xcone_pow_idx(nullptr),
          d_xcone_pow_alpha(nullptr),
          d_xcone_genpow_idx(nullptr),
          d_xcone_genpow_dim1s(nullptr),
          d_xcone_genpow_dim2s(nullptr),
          d_xcone_genpow_alpha_offsets(nullptr),
          d_xcone_genpow_dim_offsets(nullptr),
          d_xcone_genpow_sparse_idx(nullptr),
          d_xcone_genpow_sparse_offsets(nullptr),
          d_xcone_genpow_sparse_alpha_offsets(nullptr),
          d_xcone_genpow_sparse_q_offsets(nullptr),
          d_xcone_genpow_sparse_r_offsets(nullptr),
          d_xcone_genpow_sparse_to_gidx(nullptr),
          d_xcone_genpow_alphas(nullptr),
          totalXGenPowerDim(0),
          totalXGenPowerAlphas(0),
          totalXGenPowerDim2(0),
          numSparseXGenPow(0),
          totalSparseXGenPowDim(0),
          totalGenPowerDim(0),
          totalGenPowerAlphas(0),
          totalGenPowerGradEntries(0),
          totalGenPowerHsEntries(0),
          totalGenPowerHsDenseEntries(0),
          totalGenPowerDimSq(0),
          totalGenPowerDim2(0),
          numSparseGenPow(0),
          numLargeGenPow(0),
          totalSocDim(0),
          totalSocHsEntries(0),
          totalSocHsDenseEntries(0),
          numSparseSoc(0),
          numLargeSoc(0),
          d_soc_dims(nullptr),
          d_soc_offsets(nullptr),
          d_soc_sz_offsets(nullptr),
          d_soc_Hs_offsets(nullptr),
          d_soc_Hs_dense_offsets(nullptr),
          d_soc_sparse_offsets(nullptr),
          d_soc_sparse_indices(nullptr),
          d_genPowerAlphas(nullptr),
          d_genPowerDim1s(nullptr),
          d_genPowerDim2s(nullptr),
          d_genPowerOffsets(nullptr),
          d_genPowerAlphaOffsets(nullptr),
          d_genPowerHsOffsets(nullptr),
          d_genPowerHsDenseOffsets(nullptr),
          d_genPowerDimSqOffsets(nullptr),
          d_genPowerSzOffsets(nullptr),
          d_genPowerDims(nullptr),
          d_genPowerSparseOffsets(nullptr),
          d_genPowerSparseIndices(nullptr),
          batchSize(0),
          d_powerAlphas(nullptr),
          d_batch_margin_results(nullptr),
          d_barrier_work(nullptr),
          h_barrier_pinned(nullptr),
          d_scaling_success(nullptr),
          h_scaling_success_pinned(nullptr),
          nonneg_w(1, 1),
          nonneg_lambda(1, 1),
          soc_u(1, 1),
          soc_v(1, 1),
          soc_d(1, 1),
          soc_Hs(1, 1),
          soc_w(1, 1),
          soc_lambda(1, 1),
          soc_eta(1, 1),
          exp_H_dual(1, 1),
          exp_Hs(1, 1),
          exp_grad(1, 1),
          exp_z(1, 1),
          power_H_dual(1, 1),
          power_Hs(1, 1),
          power_grad(1, 1),
          power_z(1, 1),
          psd_lambda(1, 1),
          psd_Lambdaisqrt(1, 1),
          psd_R(1, 1),
          psd_Rinv(1, 1),
          psd_Hs(1, 1),
          psd_work_mat1(1, 1),
          psd_work_mat2(1, 1),
          psd_work_mat3(1, 1),
          psd_work_svec(1, 1),
          genpow_grad(1, 1),
          genpow_z(1, 1),
          genpow_Hs(1, 1),
          genpow_p(1, 1),
          genpow_q(1, 1),
          genpow_r(1, 1),
          genpow_d1(1, 1),
          genpow_d2(1, 1),
          genpow_mu(1, 1),
          genpow_smooth_zlocal(1, 1),
          genpow_smooth_wlocal(1, 1),
          genpow_smooth_res(1, 1),
          genpow_smooth_delta(1, 1),
          genpow_smooth_hmat(1, 1),
          genpow_smooth_lmat(1, 1),
          genpow_correction_u(1, 1),
          genpow_correction_gpsi(1, 1),
          genpow_pd_axes(1, 1),
          genpow_pd_coefs(1, 1),
          genpow_pd_signs(1, 1),
          genpow_pd_active(1, 1),
          genpow_pd_workspace(1, 1),
          xcone_w(1, 1),
          xcone_lambda(1, 1),
          xcone_eta(1, 1),
          xcone_Hs(1, 1),
          xcone_u(1, 1),
          xcone_v(1, 1),
          xcone_d(1, 1),
          xcone_z(1, 1),
          xcone_grad_primal(1, 1),
          xcone_psd_lambda(1, 1),
          xcone_psd_Lambdaisqrt(1, 1),
          xcone_psd_R(1, 1),
          xcone_psd_Rinv(1, 1),
          xcone_psd_Hs(1, 1),
          xcone_psd_work_mat1(1, 1),
          xcone_psd_work_mat2(1, 1),
          xcone_psd_work_mat3(1, 1),
          xcone_psd_combined_scratch(1, 1),
          xcone_psd_eigvals_scratch(1, 1),
          xcone_genpow_p(1, 1),
          xcone_genpow_q(1, 1),
          xcone_genpow_r(1, 1),
          xcone_genpow_d1(1, 1),
          xcone_genpow_d2(1, 1),
          xcone_genpow_pd_axes(1, 1),
          xcone_genpow_pd_coefs(1, 1),
          xcone_genpow_pd_signs(1, 1),
          xcone_genpow_pd_active(1, 1),
          xcone_genpow_pd_workspace(1, 1)
    {}

    /**
     * @brief Copy constructor (copies metadata, not device memory)
     */
    Cones(const Cones& other)
        : numZeroCones(other.numZeroCones),
          numNonnegCones(other.numNonnegCones),
          numExpCones(other.numExpCones),
          numSocCones(other.numSocCones),
          numPowerCones(other.numPowerCones),
          numGenPowerCones(other.numGenPowerCones),
          numPsdCones(other.numPsdCones),
          dir_cones(other.dir_cones),
          numXCones(other.numXCones),
          totalXConeNumel(0),
          totalXConeHsEntries(0),
          totalSparseXSocDim(0),
          numSparseXSoc(0),
          numXPsdCones(0),
          totalXPsdSvecDim(0),
          totalXPsdMatDim(0),
          totalXPsdMatSqDim(0),
          totalXPsdHsEntries(0),
          d_xcone_kinds(nullptr),
          d_xcone_dims(nullptr),
          d_xcone_numel_offsets(nullptr),
          d_xcone_hs_offsets(nullptr),
          d_xcone_indices(nullptr),
          d_xcone_sorted_indices(nullptr),
          d_xcone_cone_pos_for_sorted(nullptr),
          d_xcone_sparse_indices(nullptr),
          d_xcone_sparse_offsets(nullptr),
          d_xcone_kind_per_entry(nullptr),
          d_xcone_psd_idx(nullptr),
          d_xcone_psd_k(nullptr),
          d_xcone_psd_svec_offsets(nullptr),
          d_xcone_psd_mat_offsets(nullptr),
          d_xcone_psd_matsq_offsets(nullptr),
          d_xcone_psd_hs_offsets(nullptr),
          d_xcone_psd_in_full_offsets(nullptr),
          numXExpCones(0),
          numXPowerCones(0),
          numXGenPowerCones(0),
          d_xcone_exp_idx(nullptr),
          d_xcone_pow_idx(nullptr),
          d_xcone_pow_alpha(nullptr),
          d_xcone_genpow_idx(nullptr),
          d_xcone_genpow_dim1s(nullptr),
          d_xcone_genpow_dim2s(nullptr),
          d_xcone_genpow_alpha_offsets(nullptr),
          d_xcone_genpow_dim_offsets(nullptr),
          d_xcone_genpow_sparse_idx(nullptr),
          d_xcone_genpow_sparse_offsets(nullptr),
          d_xcone_genpow_sparse_alpha_offsets(nullptr),
          d_xcone_genpow_sparse_q_offsets(nullptr),
          d_xcone_genpow_sparse_r_offsets(nullptr),
          d_xcone_genpow_sparse_to_gidx(nullptr),
          d_xcone_genpow_alphas(nullptr),
          totalXGenPowerDim(0),
          totalXGenPowerAlphas(0),
          totalXGenPowerDim2(0),
          numSparseXGenPow(0),
          totalSparseXGenPowDim(0),
          socConeDims(other.socConeDims),
          powerAlphas(other.powerAlphas),
          psdConeDims(other.psdConeDims),
          totalPsdSvecDim(0),
          totalPsdMatDim(0),
          totalPsdMatSqDim(0),
          totalPsdHsEntries(0),
          d_psd_dims(nullptr),
          d_psd_svec_offsets(nullptr),
          d_psd_sz_offsets(nullptr),
          d_psd_Hs_offsets(nullptr),
          d_psd_mat_offsets(nullptr),
          d_psd_matsq_offsets(nullptr),
          genPowerAlphas(other.genPowerAlphas),
          genPowerDim1s(other.genPowerDim1s),
          genPowerDim2s(other.genPowerDim2s),
          totalGenPowerDim(0),
          totalGenPowerAlphas(0),
          totalGenPowerGradEntries(0),
          totalGenPowerHsEntries(0),
          totalGenPowerHsDenseEntries(0),
          totalGenPowerDimSq(0),
          totalGenPowerDim2(0),
          numSparseGenPow(0),
          numLargeGenPow(0),
          totalSocDim(0),
          totalSocHsEntries(0),
          totalSocHsDenseEntries(0),
          numSparseSoc(0),
          numLargeSoc(0),
          d_soc_dims(nullptr),
          d_soc_offsets(nullptr),
          d_soc_sz_offsets(nullptr),
          d_soc_Hs_offsets(nullptr),
          d_soc_Hs_dense_offsets(nullptr),
          d_soc_sparse_offsets(nullptr),
          d_soc_sparse_indices(nullptr),
          d_genPowerAlphas(nullptr),
          d_genPowerDim1s(nullptr),
          d_genPowerDim2s(nullptr),
          d_genPowerOffsets(nullptr),
          d_genPowerAlphaOffsets(nullptr),
          d_genPowerHsOffsets(nullptr),
          d_genPowerHsDenseOffsets(nullptr),
          d_genPowerDimSqOffsets(nullptr),
          d_genPowerSzOffsets(nullptr),
          d_genPowerDims(nullptr),
          d_genPowerSparseOffsets(nullptr),
          d_genPowerSparseIndices(nullptr),
          batchSize(0),
          d_powerAlphas(nullptr),
          d_batch_margin_results(nullptr),
          d_barrier_work(nullptr),
          h_barrier_pinned(nullptr),
          d_scaling_success(nullptr),
          h_scaling_success_pinned(nullptr),
          nonneg_w(1, 1),
          nonneg_lambda(1, 1),
          soc_u(1, 1),
          soc_v(1, 1),
          soc_d(1, 1),
          soc_Hs(1, 1),
          soc_w(1, 1),
          soc_lambda(1, 1),
          soc_eta(1, 1),
          exp_H_dual(1, 1),
          exp_Hs(1, 1),
          exp_grad(1, 1),
          exp_z(1, 1),
          power_H_dual(1, 1),
          power_Hs(1, 1),
          power_grad(1, 1),
          power_z(1, 1),
          psd_lambda(1, 1),
          psd_Lambdaisqrt(1, 1),
          psd_R(1, 1),
          psd_Rinv(1, 1),
          psd_Hs(1, 1),
          psd_work_mat1(1, 1),
          psd_work_mat2(1, 1),
          psd_work_mat3(1, 1),
          psd_work_svec(1, 1),
          genpow_grad(1, 1),
          genpow_z(1, 1),
          genpow_Hs(1, 1),
          genpow_p(1, 1),
          genpow_q(1, 1),
          genpow_r(1, 1),
          genpow_d1(1, 1),
          genpow_d2(1, 1),
          genpow_mu(1, 1),
          genpow_smooth_zlocal(1, 1),
          genpow_smooth_wlocal(1, 1),
          genpow_smooth_res(1, 1),
          genpow_smooth_delta(1, 1),
          genpow_smooth_hmat(1, 1),
          genpow_smooth_lmat(1, 1),
          genpow_correction_u(1, 1),
          genpow_correction_gpsi(1, 1),
          genpow_pd_axes(1, 1),
          genpow_pd_coefs(1, 1),
          genpow_pd_signs(1, 1),
          genpow_pd_active(1, 1),
          genpow_pd_workspace(1, 1),
          xcone_w(1, 1),
          xcone_lambda(1, 1),
          xcone_eta(1, 1),
          xcone_Hs(1, 1),
          xcone_u(1, 1),
          xcone_v(1, 1),
          xcone_d(1, 1),
          xcone_z(1, 1),
          xcone_grad_primal(1, 1),
          xcone_psd_lambda(1, 1),
          xcone_psd_Lambdaisqrt(1, 1),
          xcone_psd_R(1, 1),
          xcone_psd_Rinv(1, 1),
          xcone_psd_Hs(1, 1),
          xcone_psd_work_mat1(1, 1),
          xcone_psd_work_mat2(1, 1),
          xcone_psd_work_mat3(1, 1),
          xcone_psd_combined_scratch(1, 1),
          xcone_psd_eigvals_scratch(1, 1),
          xcone_genpow_p(1, 1),
          xcone_genpow_q(1, 1),
          xcone_genpow_r(1, 1),
          xcone_genpow_d1(1, 1),
          xcone_genpow_d2(1, 1),
          xcone_genpow_pd_axes(1, 1),
          xcone_genpow_pd_coefs(1, 1),
          xcone_genpow_pd_signs(1, 1),
          xcone_genpow_pd_active(1, 1),
          xcone_genpow_pd_workspace(1, 1)
    {}

    /**
     * @brief Initialize device memory (called by solver after construction)
     *
     * Validates cone parameters and allocates GPU memory for cone working data.
     * @throws std::invalid_argument if cone parameters are invalid (e.g., power alpha not in (0,1))
     */
    void initialize(int64_t batchSize_, cudaStream_t stream = 0) {
        if (batchSize > 0) {
            return;  // Already initialized
        }

        // Sync numXCones from dir_cones (dir_cones is authoritative).
        // Direct-x nonneg and SOC both supported end-to-end on CUDA (any
        // dim ≥ 2). SOC with dim ≤ 4 uses the dense Hs form; dim > 4
        // uses the rank-2 u/v/d sparse expansion. Both paths are
        // streaming (no stack-array dim cap), matching CPU.
        numXCones = static_cast<int64_t>(dir_cones.size());
        if (numXCones > 0) {
            for (const auto& xc : dir_cones) {
                if (xc.kind == XConeKind::Nonneg) continue;
                if (xc.kind == XConeKind::SOC) {
                    // Match Python `DirectConeSpec.validate_kind_size` line 110
                    // (`SOC x-cone requires >= 2 indices`). Direct construction
                    // from FFI (JAX / torch) bypasses the Python validator, so
                    // CUDA needs the same floor: a 1-index SOC has empty `v`,
                    // which leaves dense Hs degenerate (rank-1) and triggers
                    // the symmetric-cone PD scaling on a singular block.
                    if (xc.indices.size() < 2) {
                        throw std::invalid_argument(
                            "Cones: direct-x SOC requires >= 2 indices, got " +
                            std::to_string(xc.indices.size()));
                    }
                    continue;
                }
                if (xc.kind == XConeKind::Exp) continue;
                if (xc.kind == XConeKind::Power) continue;
                if (xc.kind == XConeKind::GenPower) continue;
                if (xc.kind == XConeKind::PSD) {
                    const int64_t k = xc.psd_k;
                    if (k < 1) {
                        throw std::invalid_argument(
                            "Cones: direct-x PSD cone psd_k must be >= 1, got " +
                            std::to_string(k));
                    }
                    const int64_t expected = k * (k + 1) / 2;
                    if (static_cast<int64_t>(xc.indices.size()) != expected) {
                        throw std::invalid_argument(
                            "Cones: direct-x PSD cone indices.size() = " +
                            std::to_string(xc.indices.size()) + " but psd_k = " +
                            std::to_string(k) + " requires " +
                            std::to_string(expected));
                    }
                    continue;
                }
                throw std::runtime_error(
                    "Cones: unknown direct-x cone kind on CUDA.");
            }
        }

        // Sync numPowerCones from powerAlphas (powerAlphas is authoritative)
        if (!powerAlphas.empty()) {
            numPowerCones = static_cast<int64_t>(powerAlphas.size());
        }

        // Validate power cone alphas: each must be in (0, 1) exclusive
        if (numPowerCones > 0) {
            if (static_cast<int64_t>(powerAlphas.size()) != numPowerCones) {
                throw std::invalid_argument(
                    "Cones: powerAlphas size (" + std::to_string(powerAlphas.size()) +
                    ") must match numPowerCones (" + std::to_string(numPowerCones) + ")");
            }
            for (int64_t i = 0; i < numPowerCones; ++i) {
                double alpha = powerAlphas[i];
                if (alpha <= 0.0 || alpha >= 1.0) {
                    throw std::invalid_argument(
                        "Cones: power cone alpha[" + std::to_string(i) + "] = " +
                        std::to_string(alpha) + " must be in (0, 1) exclusive");
                }
            }
        }

        batchSize = batchSize_;

        // Allocate margin computation buffers
        cudaMalloc(&d_batch_margin_results, sizeof(MarginResults) * batchSize);

        // Allocate barrier computation buffer (one per batch element)
        cudaMalloc(&d_barrier_work, sizeof(double) * batchSize);
        // Pinned host memory for async barrier readback (avoids cudaStreamSynchronize)
        cudaHostAlloc(&h_barrier_pinned, sizeof(double) * batchSize, cudaHostAllocDefault);

        // Allocate mapped pinned memory for scaling success flag.
        // Per-batch scaling success: GPU writes via d_scaling_success; host reads
        // via h_scaling_success_pinned.  Avoids explicit D2H copy + sync.
        cudaHostAlloc(&h_scaling_success_pinned, sizeof(int32_t) * batchSize, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_scaling_success, h_scaling_success_pinned, 0);

        // Allocate and initialize nonnegative cone working data
        if (numNonnegCones > 0) {
            nonneg_w = BatchedVector(numNonnegCones, batchSize);
            nonneg_lambda = BatchedVector(numNonnegCones, batchSize);
            nonneg_w.setToConstant(0.0, stream);
            nonneg_lambda.setToConstant(0.0, stream);
        }

        // Sync numSocCones from socConeDims (socConeDims is authoritative)
        if (!socConeDims.empty()) {
            numSocCones = static_cast<int64_t>(socConeDims.size());
        } else if (numSocCones > 0) {
            // Backward compat: numSocCones set but socConeDims empty → assume dim=3
            socConeDims.assign(numSocCones, 3);
        }

        // Allocate and initialize SOC working data (variable dimensions)
        if (numSocCones > 0) {

            // Validate dimensions before sorting
            for (int64_t i = 0; i < numSocCones; ++i) {
                if (socConeDims[i] < 2) {
                    throw std::invalid_argument(
                        "Cones: socConeDims[" + std::to_string(i) + "] = " +
                        std::to_string(socConeDims[i]) + " must be >= 2");
                }
            }

            // Compute original prefix sums BEFORE sorting (for s/z offsets)
            std::vector<int64_t> orig_offsets(numSocCones + 1);
            orig_offsets[0] = 0;
            for (int64_t i = 0; i < numSocCones; ++i) {
                orig_offsets[i + 1] = orig_offsets[i] + socConeDims[i];
            }

            // Sort cones by dimension for warp coherence
            std::vector<int64_t> perm(numSocCones);
            std::iota(perm.begin(), perm.end(), 0);
            std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return socConeDims[a] < socConeDims[b];
            });

            // Build sz_offsets: original position in s/z for each sorted cone
            std::vector<int64_t> sz_offsets_host(numSocCones);
            for (int64_t i = 0; i < numSocCones; ++i) {
                sz_offsets_host[i] = orig_offsets[perm[i]];
            }

            // Save original dims and sort permutation for KKT construction
            socConeDimsOriginal = socConeDims;
            socSortPerm = perm;
            std::vector<int64_t> sorted_dims(numSocCones);
            for (int64_t i = 0; i < numSocCones; ++i) {
                sorted_dims[i] = socConeDims[perm[i]];
            }
            socConeDims = sorted_dims;

            // Compute prefix sums and totals from SORTED dims
            totalSocDim = 0;
            totalSocHsEntries = 0;
            totalSocHsDenseEntries = 0;
            numSparseSoc = 0;
            numLargeSoc = 0;
            std::vector<int64_t> soc_offsets_host(numSocCones + 1);
            std::vector<int64_t> soc_Hs_offsets_host(numSocCones + 1);
            std::vector<int64_t> soc_Hs_dense_offsets_host(numSocCones + 1);
            soc_offsets_host[0] = 0;
            soc_Hs_offsets_host[0] = 0;
            soc_Hs_dense_offsets_host[0] = 0;
            for (int64_t i = 0; i < numSocCones; ++i) {
                int64_t dim = socConeDims[i];
                totalSocDim += dim;
                soc_offsets_host[i + 1] = totalSocDim;
                // Dense (dim<=4): full upper triangle; Sparse (dim>4): diagonal only
                int64_t hs_entries = (dim <= 4) ? dim * (dim + 1) / 2 : dim;
                totalSocHsEntries += hs_entries;
                soc_Hs_offsets_host[i + 1] = totalSocHsEntries;
                // Always-dense upper triangle entries (for diff path)
                int64_t dense_entries = dim * (dim + 1) / 2;
                totalSocHsDenseEntries += dense_entries;
                soc_Hs_dense_offsets_host[i + 1] = totalSocHsDenseEntries;
                if (dim > 4) numSparseSoc++;
                if (dim > cones::SOC_PARALLEL_THRESHOLD) numLargeSoc++;
            }

            // Compute sparse SOC prefix sums (for O(1) offset lookup in KKT kernel)
            std::vector<int64_t> soc_sparse_offsets_host(numSocCones + 1);
            std::vector<int64_t> soc_sparse_indices_host(numSocCones);
            soc_sparse_offsets_host[0] = 0;
            {
                int64_t sparse_dim_acc = 0;
                int64_t sparse_idx = 0;
                for (int64_t i = 0; i < numSocCones; ++i) {
                    int64_t dim = socConeDims[i];
                    if (dim > 4) {
                        soc_sparse_indices_host[i] = sparse_idx++;
                        sparse_dim_acc += dim;
                    } else {
                        soc_sparse_indices_host[i] = -1;
                    }
                    soc_sparse_offsets_host[i + 1] = sparse_dim_acc;
                }
            }

            // Upload dimension/offset arrays to device
            cudaMalloc(&d_soc_dims, sizeof(int64_t) * numSocCones);
            cudaMalloc(&d_soc_offsets, sizeof(int64_t) * (numSocCones + 1));
            cudaMalloc(&d_soc_sz_offsets, sizeof(int64_t) * numSocCones);
            cudaMalloc(&d_soc_Hs_offsets, sizeof(int64_t) * (numSocCones + 1));
            cudaMalloc(&d_soc_Hs_dense_offsets, sizeof(int64_t) * (numSocCones + 1));
            cudaMalloc(&d_soc_sparse_offsets, sizeof(int64_t) * (numSocCones + 1));
            cudaMalloc(&d_soc_sparse_indices, sizeof(int64_t) * numSocCones);
            cudaMemcpyAsync(d_soc_dims, socConeDims.data(),
                           sizeof(int64_t) * numSocCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_soc_offsets, soc_offsets_host.data(),
                           sizeof(int64_t) * (numSocCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_soc_sz_offsets, sz_offsets_host.data(),
                           sizeof(int64_t) * numSocCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_soc_Hs_offsets, soc_Hs_offsets_host.data(),
                           sizeof(int64_t) * (numSocCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_soc_Hs_dense_offsets, soc_Hs_dense_offsets_host.data(),
                           sizeof(int64_t) * (numSocCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_soc_sparse_offsets, soc_sparse_offsets_host.data(),
                           sizeof(int64_t) * (numSocCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_soc_sparse_indices, soc_sparse_indices_host.data(),
                           sizeof(int64_t) * numSocCones,
                           cudaMemcpyHostToDevice, stream);

            // Allocate working vectors
            soc_u = BatchedVector(totalSocDim, batchSize);
            soc_v = BatchedVector(totalSocDim, batchSize);
            soc_d = BatchedVector(numSocCones, batchSize);
            soc_eta = BatchedVector(numSocCones, batchSize);
            soc_Hs = BatchedVector(totalSocHsEntries, batchSize);
            soc_w = BatchedVector(totalSocDim, batchSize);
            soc_lambda = BatchedVector(totalSocDim, batchSize);
            soc_u.setToConstant(0.0, stream);
            soc_v.setToConstant(0.0, stream);
            soc_d.setToConstant(0.0, stream);
            soc_eta.setToConstant(0.0, stream);
            soc_Hs.setToConstant(0.0, stream);
            soc_w.setToConstant(0.0, stream);
            soc_lambda.setToConstant(0.0, stream);
        }

        // Allocate and initialize exponential cone working data
        if (numExpCones > 0) {
            exp_H_dual = BatchedVector(numExpCones * 6, batchSize);  // 6 = upper tri of 3x3
            exp_Hs = BatchedVector(numExpCones * 6, batchSize);
            exp_grad = BatchedVector(numExpCones * 3, batchSize);
            exp_z = BatchedVector(numExpCones * 3, batchSize);
            // Initialize to identity-like matrices and zero vectors
            exp_H_dual.setToConstant(0.0, stream);
            exp_Hs.setToConstant(0.0, stream);
            exp_grad.setToConstant(0.0, stream);
            exp_z.setToConstant(0.0, stream);
        }

        // Allocate and initialize power cone working data
        if (numPowerCones > 0) {
            // Validate powerAlphas size matches numPowerCones
            if (powerAlphas.size() != static_cast<size_t>(numPowerCones)) {
                throw std::invalid_argument(
                    "powerAlphas.size() (" + std::to_string(powerAlphas.size()) +
                    ") must equal numPowerCones (" + std::to_string(numPowerCones) + ")"
                );
            }

            power_H_dual = BatchedVector(numPowerCones * 6, batchSize);
            power_Hs = BatchedVector(numPowerCones * 6, batchSize);
            power_grad = BatchedVector(numPowerCones * 3, batchSize);
            power_z = BatchedVector(numPowerCones * 3, batchSize);
            // Initialize to identity-like matrices and zero vectors
            power_H_dual.setToConstant(0.0, stream);
            power_Hs.setToConstant(0.0, stream);
            power_grad.setToConstant(0.0, stream);
            power_z.setToConstant(0.0, stream);

            // Copy powerAlphas to device
            cudaMalloc(&d_powerAlphas, sizeof(double) * numPowerCones);
            cudaMemcpy(d_powerAlphas, powerAlphas.data(),
                      sizeof(double) * numPowerCones,
                      cudaMemcpyHostToDevice);
        }

        // Derive numPsdCones from psdConeDims (psdConeDims is authoritative)
        numPsdCones = static_cast<int64_t>(psdConeDims.size());

        // Allocate and initialize PSD cone working data
        if (numPsdCones > 0) {
            // Validate dimensions
            for (int64_t i = 0; i < numPsdCones; ++i) {
                if (psdConeDims[i] < 1) {
                    throw std::invalid_argument(
                        "Cones: psdConeDims[" + std::to_string(i) + "] = " +
                        std::to_string(psdConeDims[i]) + " must be >= 1");
                }
            }
        }

        // Validate generalized power cone dimensions before sorting/allocation
        if (!genPowerDim1s.empty()) {
            numGenPowerCones = static_cast<int64_t>(genPowerDim1s.size());
        }
        if (numGenPowerCones > 0) {
            if (static_cast<int64_t>(genPowerDim1s.size()) != numGenPowerCones ||
                static_cast<int64_t>(genPowerDim2s.size()) != numGenPowerCones) {
                throw std::invalid_argument(
                    "Cones: genPowerDim1s/Dim2s size must match numGenPowerCones");
            }
            // Validate dimensions and alpha sizes before sorting
            {
                int64_t expected_alphas = 0;
                for (int64_t i = 0; i < numGenPowerCones; ++i) {
                    if (genPowerDim1s[i] < 1) {
                        throw std::invalid_argument(
                            "Cones: genPowerDim1s[" + std::to_string(i) + "] must be >= 1");
                    }
                    if (genPowerDim2s[i] < 1) {
                        throw std::invalid_argument(
                            "Cones: genPowerDim2s[" + std::to_string(i) + "] must be >= 1");
                    }
                    expected_alphas += genPowerDim1s[i];
                }
                if (static_cast<int64_t>(genPowerAlphas.size()) != expected_alphas) {
                    throw std::invalid_argument(
                        "Cones: genPowerAlphas size (" + std::to_string(genPowerAlphas.size()) +
                        ") must equal sum of dim1s (" + std::to_string(expected_alphas) + ")");
                }
            }
        }

        // Allocate and initialize PSD cone sorting and device data
        if (numPsdCones > 0) {
            // Compute original prefix sums BEFORE sorting (for s/z offsets)
            std::vector<int64_t> orig_svec_offsets(numPsdCones + 1);
            orig_svec_offsets[0] = 0;
            for (int64_t i = 0; i < numPsdCones; ++i) {
                int64_t n = psdConeDims[i];
                orig_svec_offsets[i + 1] = orig_svec_offsets[i] + n * (n + 1) / 2;
            }

            // Sort cones by dimension for warp coherence
            std::vector<int64_t> perm(numPsdCones);
            std::iota(perm.begin(), perm.end(), 0);
            std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return psdConeDims[a] < psdConeDims[b];
            });

            // Build sz_offsets: original position in s/z for each sorted cone
            std::vector<int64_t> sz_offsets_host(numPsdCones);
            for (int64_t i = 0; i < numPsdCones; ++i) {
                sz_offsets_host[i] = orig_svec_offsets[perm[i]];
            }

            // Save original dims and sort permutation for KKT construction
            psdConeDimsOriginal = psdConeDims;
            psdSortPerm = perm;
            std::vector<int64_t> sorted_dims(numPsdCones);
            for (int64_t i = 0; i < numPsdCones; ++i) {
                sorted_dims[i] = psdConeDims[perm[i]];
            }
            psdConeDims = sorted_dims;

            // Compute prefix sums and totals from SORTED dims
            totalPsdSvecDim = 0;
            totalPsdMatDim = 0;
            totalPsdMatSqDim = 0;
            totalPsdHsEntries = 0;
            std::vector<int64_t> psd_svec_offsets_host(numPsdCones + 1);
            std::vector<int64_t> psd_Hs_offsets_host(numPsdCones + 1);
            std::vector<int64_t> psd_mat_offsets_host(numPsdCones + 1);
            std::vector<int64_t> psd_matsq_offsets_host(numPsdCones + 1);
            psd_svec_offsets_host[0] = 0;
            psd_Hs_offsets_host[0] = 0;
            psd_mat_offsets_host[0] = 0;
            psd_matsq_offsets_host[0] = 0;
            for (int64_t i = 0; i < numPsdCones; ++i) {
                int64_t n = psdConeDims[i];
                int64_t svec_dim = n * (n + 1) / 2;
                totalPsdSvecDim += svec_dim;
                totalPsdMatDim += n;
                totalPsdMatSqDim += n * n;
                totalPsdHsEntries += svec_dim * (svec_dim + 1) / 2;
                psd_svec_offsets_host[i + 1] = totalPsdSvecDim;
                psd_Hs_offsets_host[i + 1] = totalPsdHsEntries;
                psd_mat_offsets_host[i + 1] = totalPsdMatDim;
                psd_matsq_offsets_host[i + 1] = totalPsdMatSqDim;
            }

            // Upload dimension/offset arrays to device
            cudaMalloc(&d_psd_dims, sizeof(int64_t) * numPsdCones);
            cudaMalloc(&d_psd_svec_offsets, sizeof(int64_t) * (numPsdCones + 1));
            cudaMalloc(&d_psd_sz_offsets, sizeof(int64_t) * numPsdCones);
            cudaMalloc(&d_psd_Hs_offsets, sizeof(int64_t) * (numPsdCones + 1));
            cudaMalloc(&d_psd_mat_offsets, sizeof(int64_t) * (numPsdCones + 1));
            cudaMalloc(&d_psd_matsq_offsets, sizeof(int64_t) * (numPsdCones + 1));
            cudaMemcpyAsync(d_psd_dims, psdConeDims.data(),
                           sizeof(int64_t) * numPsdCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_psd_svec_offsets, psd_svec_offsets_host.data(),
                           sizeof(int64_t) * (numPsdCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_psd_sz_offsets, sz_offsets_host.data(),
                           sizeof(int64_t) * numPsdCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_psd_Hs_offsets, psd_Hs_offsets_host.data(),
                           sizeof(int64_t) * (numPsdCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_psd_mat_offsets, psd_mat_offsets_host.data(),
                           sizeof(int64_t) * (numPsdCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_psd_matsq_offsets, psd_matsq_offsets_host.data(),
                           sizeof(int64_t) * (numPsdCones + 1),
                           cudaMemcpyHostToDevice, stream);

            // Allocate working vectors
            psd_lambda = BatchedVector(totalPsdMatDim, batchSize);
            psd_Lambdaisqrt = BatchedVector(totalPsdMatDim, batchSize);
            psd_R = BatchedVector(totalPsdMatSqDim, batchSize);
            psd_Rinv = BatchedVector(totalPsdMatSqDim, batchSize);
            psd_Hs = BatchedVector(totalPsdHsEntries, batchSize);
            psd_work_mat1 = BatchedVector(totalPsdMatSqDim, batchSize);
            psd_work_mat2 = BatchedVector(totalPsdMatSqDim, batchSize);
            psd_work_mat3 = BatchedVector(totalPsdMatSqDim, batchSize);
            psd_work_svec = BatchedVector(totalPsdSvecDim, batchSize);
            psd_lambda.setToConstant(0.0, stream);
            psd_Lambdaisqrt.setToConstant(0.0, stream);
            psd_R.setToConstant(0.0, stream);
            psd_Rinv.setToConstant(0.0, stream);
            psd_Hs.setToConstant(0.0, stream);
            psd_work_mat1.setToConstant(0.0, stream);
            psd_work_mat2.setToConstant(0.0, stream);
            psd_work_mat3.setToConstant(0.0, stream);
            psd_work_svec.setToConstant(0.0, stream);

        }
        // cuSOLVER / cuBLAS handles + workspaces are shared between slack PSD
        // and direct-x PSD; size them to the largest k across both.
        // (Direct-x PSD dir_cones[].psd_k is captured below in numXPsdCones; we
        // peek into dir_cones here since the direct-x init runs after this block.)
        int64_t max_dim_for_psd = 0;
        if (numPsdCones > 0) {
            max_dim_for_psd = std::max(max_dim_for_psd,
                *std::max_element(psdConeDims.begin(), psdConeDims.end()));
        }
        for (const auto& xc : dir_cones) {
            if (xc.kind == XConeKind::PSD && xc.psd_k > max_dim_for_psd) {
                max_dim_for_psd = xc.psd_k;
            }
        }
        if (max_dim_for_psd > 0) {
            // Create cuSOLVER and cuBLAS handles
            cusolverDnCreate(&cusolverH_);
            cusolverDnSetStream(cusolverH_, stream);
            cublasCreate(&cublasH_);
            cublasSetStream(cublasH_, stream);

            cudaMalloc(&d_psd_info_, sizeof(int));
            cusolverDnDpotrf_bufferSize(cusolverH_, CUBLAS_FILL_MODE_LOWER,
                                        max_dim_for_psd, nullptr, max_dim_for_psd, &psd_work_size_);
            cudaMalloc(&d_psd_work_, sizeof(double) * psd_work_size_);
            cusolverDnDgesvd_bufferSize(cusolverH_, max_dim_for_psd, max_dim_for_psd, &psd_gesvd_work_size_);
            cudaMalloc(&d_psd_gesvd_work_, sizeof(double) * psd_gesvd_work_size_);
            cusolverDnDsyevd_bufferSize(cusolverH_, CUSOLVER_EIG_MODE_VECTOR,
                                         CUBLAS_FILL_MODE_LOWER,
                                         max_dim_for_psd, nullptr, max_dim_for_psd,
                                         nullptr, &psd_syevd_work_size_);
            cudaMalloc(&d_psd_syevd_work_, sizeof(double) * psd_syevd_work_size_);
        }

        // Allocate and initialize generalized power cone sorting and device data
        if (numGenPowerCones > 0) {
            std::vector<int64_t> orig_offsets(numGenPowerCones + 1);
            std::vector<int64_t> orig_alpha_offsets(numGenPowerCones + 1);
            orig_offsets[0] = 0;
            orig_alpha_offsets[0] = 0;
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                orig_offsets[i + 1] = orig_offsets[i] + genPowerDim1s[i] + genPowerDim2s[i];
                orig_alpha_offsets[i + 1] = orig_alpha_offsets[i] + genPowerDim1s[i];
            }

            // Sort cones by total dimension for warp coherence
            std::vector<int64_t> perm(numGenPowerCones);
            std::iota(perm.begin(), perm.end(), 0);
            std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return (genPowerDim1s[a] + genPowerDim2s[a]) < (genPowerDim1s[b] + genPowerDim2s[b]);
            });

            // Build sz_offsets: original position in s/z for each sorted cone
            std::vector<int64_t> sz_offsets_host(numGenPowerCones);
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                sz_offsets_host[i] = orig_offsets[perm[i]];
            }

            // Save originals and sort permutation for KKT construction
            genPowerDim1sOriginal = genPowerDim1s;
            genPowerDim2sOriginal = genPowerDim2s;
            genPowerAlphasOriginal = genPowerAlphas;
            genPowerSortPerm = perm;

            // Apply permutation to dim1s, dim2s
            std::vector<int64_t> sorted_dim1s(numGenPowerCones);
            std::vector<int64_t> sorted_dim2s(numGenPowerCones);
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                sorted_dim1s[i] = genPowerDim1s[perm[i]];
                sorted_dim2s[i] = genPowerDim2s[perm[i]];
            }
            genPowerDim1s = sorted_dim1s;
            genPowerDim2s = sorted_dim2s;

            // Reorder alphas according to permutation
            std::vector<double> sorted_alphas;
            sorted_alphas.reserve(genPowerAlphas.size());
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                int64_t orig_idx = perm[i];
                int64_t alpha_start = orig_alpha_offsets[orig_idx];
                int64_t dim1 = genPowerDim1s[i];
                for (int64_t j = 0; j < dim1; ++j) {
                    sorted_alphas.push_back(genPowerAlphas[alpha_start + j]);
                }
            }
            genPowerAlphas = sorted_alphas;

            // Validate alpha values (positivity and sum-to-one)
            totalGenPowerAlphas = 0;
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                totalGenPowerAlphas += genPowerDim1s[i];
            }
            int64_t alpha_off = 0;
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                double asum = 0.0;
                for (int64_t j = 0; j < genPowerDim1s[i]; ++j) {
                    double a = genPowerAlphas[alpha_off + j];
                    if (a <= 0.0) {
                        throw std::invalid_argument(
                            "Cones: genPowerAlphas values must be > 0");
                    }
                    asum += a;
                }
                if (std::abs(asum - 1.0) > 1e-8 * genPowerDim1s[i]) {
                    throw std::invalid_argument(
                        "Cones: genPowerAlphas for cone " + std::to_string(i) +
                        " must sum to 1, got " + std::to_string(asum));
                }
                alpha_off += genPowerDim1s[i];
            }

            // Compute prefix sums and totals from SORTED dims
            totalGenPowerDim = 0;
            totalGenPowerHsEntries = 0;
            totalGenPowerHsDenseEntries = 0;
            totalGenPowerDimSq = 0;
            totalGenPowerDim2 = 0;
            numSparseGenPow = 0;
            numLargeGenPow = 0;
            std::vector<int64_t> offsets_host(numGenPowerCones + 1);
            std::vector<int64_t> alpha_offsets_host(numGenPowerCones + 1);
            std::vector<int64_t> hs_offsets_host(numGenPowerCones + 1);
            std::vector<int64_t> hs_dense_offsets_host(numGenPowerCones + 1);
            std::vector<int64_t> dimsq_offsets_host(numGenPowerCones + 1);
            std::vector<int64_t> dims_host(numGenPowerCones);
            offsets_host[0] = 0;
            alpha_offsets_host[0] = 0;
            hs_offsets_host[0] = 0;
            hs_dense_offsets_host[0] = 0;
            dimsq_offsets_host[0] = 0;

            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                int64_t dim1 = genPowerDim1s[i];
                int64_t dim2 = genPowerDim2s[i];
                int64_t dim = dim1 + dim2;
                dims_host[i] = dim;
                totalGenPowerDim += dim;
                totalGenPowerDim2 += dim2;
                totalGenPowerDimSq += dim * dim;
                offsets_host[i + 1] = totalGenPowerDim;
                alpha_offsets_host[i + 1] = alpha_offsets_host[i] + dim1;
                // Dense (dim<=4): full upper triangle; Sparse (dim>4): diagonal only
                int64_t hs_entries = (dim <= 4) ? dim * (dim + 1) / 2 : dim;
                totalGenPowerHsEntries += hs_entries;
                hs_offsets_host[i + 1] = totalGenPowerHsEntries;
                // Always-dense upper triangle entries (for diff path)
                int64_t dense_entries = dim * (dim + 1) / 2;
                totalGenPowerHsDenseEntries += dense_entries;
                hs_dense_offsets_host[i + 1] = totalGenPowerHsDenseEntries;
                dimsq_offsets_host[i + 1] = totalGenPowerDimSq;
                if (dim > 4) numSparseGenPow++;
                if (dim > cones::GENPOW_PARALLEL_THRESHOLD) numLargeGenPow++;
            }
            totalGenPowerGradEntries = totalGenPowerDim;

            // Compute sparse GenPow prefix sums (for O(1) offset lookup in KKT kernel)
            std::vector<int64_t> sparse_offsets_host(numGenPowerCones + 1);
            std::vector<int64_t> sparse_indices_host(numGenPowerCones);
            sparse_offsets_host[0] = 0;
            {
                int64_t sparse_dim_acc = 0;
                int64_t sparse_idx = 0;
                for (int64_t i = 0; i < numGenPowerCones; ++i) {
                    int64_t dim = dims_host[i];
                    if (dim > 4) {
                        sparse_indices_host[i] = sparse_idx++;
                        sparse_dim_acc += dim;
                    } else {
                        sparse_indices_host[i] = -1;
                    }
                    sparse_offsets_host[i + 1] = sparse_dim_acc;
                }
            }

            // Upload arrays to device
            cudaMalloc(&d_genPowerAlphas, sizeof(double) * totalGenPowerAlphas);
            cudaMalloc(&d_genPowerDim1s, sizeof(int64_t) * numGenPowerCones);
            cudaMalloc(&d_genPowerDim2s, sizeof(int64_t) * numGenPowerCones);
            cudaMalloc(&d_genPowerOffsets, sizeof(int64_t) * (numGenPowerCones + 1));
            cudaMalloc(&d_genPowerAlphaOffsets, sizeof(int64_t) * (numGenPowerCones + 1));
            cudaMalloc(&d_genPowerHsOffsets, sizeof(int64_t) * (numGenPowerCones + 1));
            cudaMalloc(&d_genPowerHsDenseOffsets, sizeof(int64_t) * (numGenPowerCones + 1));
            cudaMalloc(&d_genPowerDimSqOffsets, sizeof(int64_t) * (numGenPowerCones + 1));
            cudaMalloc(&d_genPowerSzOffsets, sizeof(int64_t) * numGenPowerCones);
            cudaMalloc(&d_genPowerDims, sizeof(int64_t) * numGenPowerCones);
            cudaMalloc(&d_genPowerSparseOffsets, sizeof(int64_t) * (numGenPowerCones + 1));
            cudaMalloc(&d_genPowerSparseIndices, sizeof(int64_t) * numGenPowerCones);
            cudaMemcpyAsync(d_genPowerAlphas, genPowerAlphas.data(),
                           sizeof(double) * totalGenPowerAlphas,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerDim1s, genPowerDim1s.data(),
                           sizeof(int64_t) * numGenPowerCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerDim2s, genPowerDim2s.data(),
                           sizeof(int64_t) * numGenPowerCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerOffsets, offsets_host.data(),
                           sizeof(int64_t) * (numGenPowerCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerAlphaOffsets, alpha_offsets_host.data(),
                           sizeof(int64_t) * (numGenPowerCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerHsOffsets, hs_offsets_host.data(),
                           sizeof(int64_t) * (numGenPowerCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerHsDenseOffsets, hs_dense_offsets_host.data(),
                           sizeof(int64_t) * (numGenPowerCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerDimSqOffsets, dimsq_offsets_host.data(),
                           sizeof(int64_t) * (numGenPowerCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerSzOffsets, sz_offsets_host.data(),
                           sizeof(int64_t) * numGenPowerCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerDims, dims_host.data(),
                           sizeof(int64_t) * numGenPowerCones,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerSparseOffsets, sparse_offsets_host.data(),
                           sizeof(int64_t) * (numGenPowerCones + 1),
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_genPowerSparseIndices, sparse_indices_host.data(),
                           sizeof(int64_t) * numGenPowerCones,
                           cudaMemcpyHostToDevice, stream);

            // Allocate working vectors
            genpow_grad = BatchedVector(totalGenPowerDim, batchSize);
            genpow_z = BatchedVector(totalGenPowerDim, batchSize);
            genpow_Hs = BatchedVector(totalGenPowerHsEntries, batchSize);
            genpow_p = BatchedVector(totalGenPowerDim, batchSize);
            genpow_q = BatchedVector(totalGenPowerAlphas, batchSize);
            genpow_r = BatchedVector(totalGenPowerDim2, batchSize);
            genpow_d1 = BatchedVector(totalGenPowerAlphas, batchSize);
            genpow_d2 = BatchedVector(numGenPowerCones, batchSize);
            genpow_mu = BatchedVector(1, batchSize);
            genpow_grad.setToConstant(0.0, stream);
            genpow_z.setToConstant(0.0, stream);
            genpow_Hs.setToConstant(0.0, stream);
            genpow_p.setToConstant(0.0, stream);
            genpow_q.setToConstant(0.0, stream);
            genpow_r.setToConstant(0.0, stream);
            genpow_d1.setToConstant(0.0, stream);
            genpow_d2.setToConstant(0.0, stream);

            // Smoothing workspace (replaces per-thread stack arrays)
            genpow_smooth_zlocal = BatchedVector(totalGenPowerDim, batchSize);
            genpow_smooth_wlocal = BatchedVector(totalGenPowerDim, batchSize);
            genpow_smooth_res = BatchedVector(totalGenPowerDim, batchSize);
            genpow_smooth_delta = BatchedVector(totalGenPowerDim, batchSize);
            genpow_smooth_hmat = BatchedVector(totalGenPowerDimSq, batchSize);
            genpow_smooth_lmat = BatchedVector(totalGenPowerDimSq, batchSize);
            genpow_smooth_zlocal.setToConstant(0.0, stream);
            genpow_smooth_wlocal.setToConstant(0.0, stream);
            genpow_smooth_res.setToConstant(0.0, stream);
            genpow_smooth_delta.setToConstant(0.0, stream);
            genpow_smooth_hmat.setToConstant(0.0, stream);
            genpow_smooth_lmat.setToConstant(0.0, stream);

            // Higher-correction workspace (rank-3 SMW solve + g_ψ).
            genpow_correction_u    = BatchedVector(totalGenPowerDim, batchSize);
            genpow_correction_gpsi = BatchedVector(totalGenPowerDim, batchSize);
            genpow_correction_u.setToConstant(0.0, stream);
            genpow_correction_gpsi.setToConstant(0.0, stream);

            // PD-scaling rank-6 sparse expansion state (mirrors CPU).
            genpow_pd_axes      = BatchedVector(6 * totalGenPowerDim, batchSize);
            genpow_pd_coefs     = BatchedVector(6 * numGenPowerCones, batchSize);
            genpow_pd_signs     = BatchedVector(6 * numGenPowerCones, batchSize);
            genpow_pd_active    = BatchedVector(numGenPowerCones, batchSize);
            genpow_pd_workspace = BatchedVector(8 * totalGenPowerDim, batchSize);
            genpow_pd_axes.setToConstant(0.0, stream);
            genpow_pd_coefs.setToConstant(0.0, stream);
            genpow_pd_signs.setToConstant(1.0, stream);
            genpow_pd_active.setToConstant(0.0, stream);
            genpow_pd_workspace.setToConstant(0.0, stream);
        }

        // Direct-x cones: compute totals, upload metadata arrays, and
        // allocate batched working storage. PSD direct-x is plumbed
        // through the metadata layer here; the runtime kernels (scaling,
        // step math, KKT) gate further down with NotImplementedError
        // until M2-M4 land.
        if (numXCones > 0) {
            totalXConeNumel = 0;
            totalXConeHsEntries = 0;
            totalSparseXSocDim = 0;
            numSparseXSoc = 0;
            numXPsdCones = 0;
            totalXPsdSvecDim = 0;
            totalXPsdMatDim = 0;
            totalXPsdMatSqDim = 0;
            totalXPsdHsEntries = 0;
            numXExpCones = 0;
            numXPowerCones = 0;
            numXGenPowerCones = 0;
            totalXGenPowerDim = 0;
            totalXGenPowerAlphas = 0;
            totalXGenPowerDim2 = 0;
            numSparseXGenPow = 0;
            totalSparseXGenPowDim = 0;

            std::vector<int64_t> kinds_host(numXCones);
            std::vector<int64_t> dims_host(numXCones);
            std::vector<int64_t> numel_offsets_host(numXCones + 1, 0);
            std::vector<int64_t> hs_offsets_host(numXCones + 1, 0);
            std::vector<int64_t> sparse_indices_host(numXCones, -1);
            std::vector<int64_t> sparse_offsets_host(numXCones + 1, 0);
            std::vector<int64_t> psd_idx_host(numXCones, -1);
            std::vector<int64_t> indices_host;
            std::vector<int64_t> sorted_indices_host;
            std::vector<int64_t> cone_pos_for_sorted_host;
            std::vector<int64_t> kind_per_entry_host;
            // PSD-only auxiliary metadata
            std::vector<int64_t> psd_k_host;
            std::vector<int64_t> psd_svec_offsets_host{0};
            std::vector<int64_t> psd_mat_offsets_host{0};
            std::vector<int64_t> psd_matsq_offsets_host{0};
            std::vector<int64_t> psd_hs_offsets_host{0};
            std::vector<int64_t> psd_in_full_offsets_host;
            // Asymmetric direct-x auxiliary metadata.
            std::vector<int64_t> exp_idx_host(numXCones, -1);
            std::vector<int64_t> pow_idx_host(numXCones, -1);
            std::vector<double>  pow_alpha_host;
            // GenPower direct-x metadata.
            std::vector<int64_t> genpow_idx_host(numXCones, -1);
            std::vector<int64_t> genpow_dim1s_host;
            std::vector<int64_t> genpow_dim2s_host;
            std::vector<int64_t> genpow_alpha_offsets_host{0};
            std::vector<int64_t> genpow_dim_offsets_host{0};
            std::vector<int64_t> genpow_sparse_idx_host;
            std::vector<int64_t> genpow_sparse_offsets_host{0};
            std::vector<int64_t> genpow_sparse_alpha_offsets_host;  // global a_off per sparse cone
            std::vector<int64_t> genpow_sparse_q_offsets_host{0};   // sparse-only dim1 prefix
            std::vector<int64_t> genpow_sparse_r_offsets_host{0};   // sparse-only dim2 prefix
            std::vector<int64_t> genpow_sparse_to_gidx_host;        // gidx per sparse cone
            std::vector<double>  genpow_alphas_host;

            for (int64_t c = 0; c < numXCones; ++c) {
                const auto& xc = dir_cones[(size_t)c];
                kinds_host[(size_t)c] = static_cast<int64_t>(xc.kind);
                const int64_t dim = static_cast<int64_t>(xc.indices.size());
                dims_host[(size_t)c] = dim;
                numel_offsets_host[(size_t)c + 1] = numel_offsets_host[(size_t)c] + dim;

                // Hs entry count per cone: nonneg -> dim; SOC dense -> dim*(dim+1)/2;
                // SOC sparse -> dim (off-diagonal coupling lives in u/v cols);
                // PSD -> dim*(dim+1)/2 (full upper-tri of the svec×svec Hs).
                int64_t hs_entries = dim;
                if (xc.kind == XConeKind::SOC && dim <= 4) {
                    hs_entries = dim * (dim + 1) / 2;
                } else if (xc.kind == XConeKind::PSD) {
                    hs_entries = dim * (dim + 1) / 2;
                } else if (xc.kind == XConeKind::Exp ||
                           xc.kind == XConeKind::Power) {
                    // Asymmetric Exp/Power are 3D-dense: 6 packed upper-tri entries.
                    hs_entries = dim * (dim + 1) / 2;
                } else if (xc.kind == XConeKind::GenPower) {
                    // GenPower: dense (dim<=4) -> full upper-tri; sparse (dim>4) -> diagonal only.
                    hs_entries = (dim <= 4) ? dim * (dim + 1) / 2 : dim;
                }
                hs_offsets_host[(size_t)c + 1] = hs_offsets_host[(size_t)c] + hs_entries;

                const bool is_sparse_soc =
                    (xc.kind == XConeKind::SOC) && (dim > 4);
                if (is_sparse_soc) {
                    sparse_indices_host[(size_t)c] = numSparseXSoc;
                    sparse_offsets_host[(size_t)c + 1] =
                        sparse_offsets_host[(size_t)c] + dim;
                    totalSparseXSocDim += dim;
                    ++numSparseXSoc;
                } else {
                    sparse_offsets_host[(size_t)c + 1] =
                        sparse_offsets_host[(size_t)c];
                }

                if (xc.kind == XConeKind::PSD) {
                    const int64_t k = xc.psd_k;
                    psd_idx_host[(size_t)c] = numXPsdCones;
                    psd_k_host.push_back(k);
                    psd_in_full_offsets_host.push_back(numel_offsets_host[(size_t)c]);
                    totalXPsdSvecDim += dim;
                    totalXPsdMatDim += k;
                    totalXPsdMatSqDim += k * k;
                    totalXPsdHsEntries += hs_entries;
                    psd_svec_offsets_host.push_back(totalXPsdSvecDim);
                    psd_mat_offsets_host.push_back(totalXPsdMatDim);
                    psd_matsq_offsets_host.push_back(totalXPsdMatSqDim);
                    psd_hs_offsets_host.push_back(totalXPsdHsEntries);
                    ++numXPsdCones;
                } else if (xc.kind == XConeKind::Exp) {
                    if (dim != 3) {
                        throw std::runtime_error(
                            "ExponentialXCone must have exactly 3 indices");
                    }
                    exp_idx_host[(size_t)c] = numXExpCones;
                    ++numXExpCones;
                } else if (xc.kind == XConeKind::Power) {
                    if (dim != 3) {
                        throw std::runtime_error(
                            "PowerXCone must have exactly 3 indices");
                    }
                    if (xc.power_alpha <= 0.0 || xc.power_alpha >= 1.0) {
                        throw std::runtime_error(
                            "PowerXCone alpha must lie in (0, 1)");
                    }
                    pow_idx_host[(size_t)c] = numXPowerCones;
                    pow_alpha_host.push_back(xc.power_alpha);
                    ++numXPowerCones;
                } else if (xc.kind == XConeKind::GenPower) {
                    const int64_t dim1 = static_cast<int64_t>(xc.gen_power_alphas.size());
                    const int64_t dim2 = xc.gen_power_dim2;
                    if (dim1 < 1 || dim2 < 1) {
                        throw std::runtime_error(
                            "GenPowerXCone requires dim1 >= 1 and dim2 >= 1");
                    }
                    if (dim1 + dim2 != dim) {
                        throw std::runtime_error(
                            "GenPowerXCone: indices.size() must equal dim1 + dim2");
                    }
                    double asum = 0.0;
                    for (double a : xc.gen_power_alphas) {
                        if (a <= 0.0) throw std::runtime_error("GenPowerXCone: alphas must be > 0");
                        asum += a;
                    }
                    // Match Python `DirectConeSpec._types.py:178` (relative
                    // tolerance scaled by dim1) and slack genpow at
                    // line 1438. Absolute 1e-8 was tighter than Python
                    // for dim1 > 1, so a problem that passed the
                    // Python validator could fail here at large dim1.
                    if (std::abs(asum - 1.0) > 1e-8 * static_cast<double>(dim1)) {
                        throw std::runtime_error(
                            "GenPowerXCone: alphas must sum to 1, got " +
                            std::to_string(asum));
                    }
                    genpow_idx_host[(size_t)c] = numXGenPowerCones;
                    genpow_dim1s_host.push_back(dim1);
                    genpow_dim2s_host.push_back(dim2);
                    totalXGenPowerDim += dim;
                    totalXGenPowerAlphas += dim1;
                    totalXGenPowerDim2 += dim2;
                    genpow_alpha_offsets_host.push_back(totalXGenPowerAlphas);
                    genpow_dim_offsets_host.push_back(totalXGenPowerDim);
                    for (double a : xc.gen_power_alphas) genpow_alphas_host.push_back(a);
                    if (dim > 4) {
                        genpow_sparse_idx_host.push_back(numSparseXGenPow);
                        genpow_sparse_offsets_host.push_back(
                            genpow_sparse_offsets_host.back() + dim);
                        // Global alpha offset (into xcone_genpow_q data array)
                        genpow_sparse_alpha_offsets_host.push_back(
                            totalXGenPowerAlphas - dim1);
                        // Sparse-only dim1/dim2 prefix (into H_xcone_genpow_q/r_idx_ arrays)
                        genpow_sparse_q_offsets_host.push_back(
                            genpow_sparse_q_offsets_host.back() + dim1);
                        genpow_sparse_r_offsets_host.push_back(
                            genpow_sparse_r_offsets_host.back() + dim2);
                        // Reverse genpow index mapping
                        genpow_sparse_to_gidx_host.push_back(numXGenPowerCones);
                        totalSparseXGenPowDim += dim;
                        ++numSparseXGenPow;
                    } else {
                        genpow_sparse_idx_host.push_back(-1);
                        genpow_sparse_offsets_host.push_back(
                            genpow_sparse_offsets_host.back());
                    }
                    ++numXGenPowerCones;
                }

                // Flat index arrays: indices in user order + sorted copy
                // with cone-local permutation (for rank-2 column writes).
                for (int64_t v : xc.indices) indices_host.push_back(v);
                // Per-entry kind tag so flat-iteration nonneg kernels can
                // skip SOC entries (and vice versa for SOC flat passes).
                for (int64_t p = 0; p < dim; ++p)
                    kind_per_entry_host.push_back(static_cast<int64_t>(xc.kind));
                std::vector<std::pair<int64_t, int64_t>> pairs;
                pairs.reserve((size_t)dim);
                for (int64_t p = 0; p < dim; ++p)
                    pairs.emplace_back(xc.indices[(size_t)p], p);
                std::sort(pairs.begin(), pairs.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                for (const auto& kv : pairs) {
                    sorted_indices_host.push_back(kv.first);
                    cone_pos_for_sorted_host.push_back(kv.second);
                }

                totalXConeNumel += dim;
                totalXConeHsEntries += hs_entries;
            }

            auto upload_i64 = [&](const std::vector<int64_t>& host,
                                   int64_t** dptr) {
                if (host.empty()) { *dptr = nullptr; return; }
                size_t bytes = sizeof(int64_t) * host.size();
                auto e = cudaMalloc(dptr, bytes);
                if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
                cudaMemcpyAsync(*dptr, host.data(), bytes,
                                cudaMemcpyHostToDevice, stream);
            };
            upload_i64(kinds_host,              &d_xcone_kinds);
            upload_i64(dims_host,               &d_xcone_dims);
            upload_i64(numel_offsets_host,      &d_xcone_numel_offsets);
            upload_i64(hs_offsets_host,         &d_xcone_hs_offsets);
            upload_i64(indices_host,            &d_xcone_indices);
            upload_i64(sorted_indices_host,     &d_xcone_sorted_indices);
            upload_i64(cone_pos_for_sorted_host,&d_xcone_cone_pos_for_sorted);
            upload_i64(sparse_indices_host,     &d_xcone_sparse_indices);
            upload_i64(sparse_offsets_host,     &d_xcone_sparse_offsets);
            upload_i64(kind_per_entry_host,     &d_xcone_kind_per_entry);
            upload_i64(psd_idx_host,            &d_xcone_psd_idx);
            if (numXExpCones > 0) {
                upload_i64(exp_idx_host, &d_xcone_exp_idx);
            }
            if (numXPowerCones > 0) {
                upload_i64(pow_idx_host, &d_xcone_pow_idx);
                size_t bytes = sizeof(double) * pow_alpha_host.size();
                auto e = cudaMalloc(&d_xcone_pow_alpha, bytes);
                if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
                cudaMemcpyAsync(d_xcone_pow_alpha, pow_alpha_host.data(),
                                bytes, cudaMemcpyHostToDevice, stream);
            }
            if (numXPsdCones > 0) {
                upload_i64(psd_k_host,                &d_xcone_psd_k);
                xconePsdDimsHost = psd_k_host;  // cache for kernels that drive cuBLAS sizes
                upload_i64(psd_svec_offsets_host,     &d_xcone_psd_svec_offsets);
                upload_i64(psd_mat_offsets_host,      &d_xcone_psd_mat_offsets);
                upload_i64(psd_matsq_offsets_host,    &d_xcone_psd_matsq_offsets);
                upload_i64(psd_hs_offsets_host,       &d_xcone_psd_hs_offsets);
                upload_i64(psd_in_full_offsets_host,  &d_xcone_psd_in_full_offsets);
            }
            if (numXGenPowerCones > 0) {
                upload_i64(genpow_idx_host,            &d_xcone_genpow_idx);
                upload_i64(genpow_dim1s_host,          &d_xcone_genpow_dim1s);
                upload_i64(genpow_dim2s_host,          &d_xcone_genpow_dim2s);
                upload_i64(genpow_alpha_offsets_host,  &d_xcone_genpow_alpha_offsets);
                upload_i64(genpow_dim_offsets_host,    &d_xcone_genpow_dim_offsets);
                upload_i64(genpow_sparse_idx_host,           &d_xcone_genpow_sparse_idx);
                upload_i64(genpow_sparse_offsets_host,       &d_xcone_genpow_sparse_offsets);
                upload_i64(genpow_sparse_alpha_offsets_host, &d_xcone_genpow_sparse_alpha_offsets);
                upload_i64(genpow_sparse_q_offsets_host,     &d_xcone_genpow_sparse_q_offsets);
                upload_i64(genpow_sparse_r_offsets_host,     &d_xcone_genpow_sparse_r_offsets);
                upload_i64(genpow_sparse_to_gidx_host,       &d_xcone_genpow_sparse_to_gidx);
                {
                    size_t bytes = sizeof(double) * genpow_alphas_host.size();
                    auto e = cudaMalloc(&d_xcone_genpow_alphas, bytes);
                    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
                    cudaMemcpyAsync(d_xcone_genpow_alphas, genpow_alphas_host.data(),
                                    bytes, cudaMemcpyHostToDevice, stream);
                }
                // Working storage for p/q/r/d1/d2.
                xcone_genpow_p  = BatchedVector(totalXGenPowerDim,    batchSize);
                xcone_genpow_q  = BatchedVector(totalXGenPowerAlphas, batchSize);
                xcone_genpow_r  = BatchedVector(totalXGenPowerDim2,   batchSize);
                xcone_genpow_d1 = BatchedVector(totalXGenPowerAlphas, batchSize);
                xcone_genpow_d2 = BatchedVector(numXGenPowerCones,    batchSize);
                xcone_genpow_p.setToConstant(0.0,  stream);
                xcone_genpow_q.setToConstant(0.0,  stream);
                xcone_genpow_r.setToConstant(0.0,  stream);
                xcone_genpow_d1.setToConstant(0.0, stream);
                xcone_genpow_d2.setToConstant(0.0, stream);
                // Mosek-Tunçel rank-6 PD scaling state for direct-x. Layout
                // mirrors slack: 6 axes × dim per cone, 6 coefs/signs per cone,
                // 1 active flag per cone, 8×dim workspace per cone (used by the
                // pd-axis compute kernel).
                xcone_genpow_pd_axes      = BatchedVector(6 * totalXGenPowerDim, batchSize);
                xcone_genpow_pd_coefs     = BatchedVector(6 * numXGenPowerCones, batchSize);
                xcone_genpow_pd_signs     = BatchedVector(6 * numXGenPowerCones, batchSize);
                xcone_genpow_pd_active    = BatchedVector(numXGenPowerCones,     batchSize);
                xcone_genpow_pd_workspace = BatchedVector(8 * totalXGenPowerDim, batchSize);
                xcone_genpow_pd_axes.setToConstant(0.0, stream);
                xcone_genpow_pd_coefs.setToConstant(0.0, stream);
                xcone_genpow_pd_signs.setToConstant(1.0, stream);
                xcone_genpow_pd_active.setToConstant(0.0, stream);
                xcone_genpow_pd_workspace.setToConstant(0.0, stream);
            }

            // Batched working storage. Allocate w/lambda/Hs/z sized to
            // the authoritative totals; the SOC-only buffers stay at
            // placeholder (1,1) for nonneg-only runs.
            xcone_w      = BatchedVector(totalXConeNumel, batchSize);
            xcone_lambda = BatchedVector(totalXConeNumel, batchSize);
            xcone_Hs     = BatchedVector(totalXConeHsEntries, batchSize);
            xcone_z      = BatchedVector(totalXConeNumel, batchSize);
            xcone_w.setToConstant(0.0, stream);
            xcone_lambda.setToConstant(0.0, stream);
            xcone_Hs.setToConstant(0.0, stream);
            xcone_z.setToConstant(0.0, stream);
            // Asymmetric direct-x: ∇F_primal(x) at scaling point. Allocated
            // alongside xcone_w with the same numel layout. Used by
            // `add_combined_ds_shift_*` for σμ·∇F shift.
            xcone_grad_primal = BatchedVector(totalXConeNumel, batchSize);
            xcone_grad_primal.setToConstant(0.0, stream);

            // xcone_eta is written by the SOC scaling kernel for every
            // SOC direct-x cone (dense and sparse). Allocate whenever
            // there's at least one x-cone so the numXCones-sized flat
            // write stays in bounds. Nonneg entries leave their slot at 0.
            xcone_eta = BatchedVector(numXCones, batchSize);
            xcone_eta.setToConstant(0.0, stream);

            // u/v/d only needed for rank-2 sparse SOC (dim > 4).
            if (numSparseXSoc > 0) {
                xcone_u   = BatchedVector(totalSparseXSocDim, batchSize);
                xcone_v   = BatchedVector(totalSparseXSocDim, batchSize);
                xcone_d   = BatchedVector(numSparseXSoc, batchSize);
                xcone_u.setToConstant(0.0, stream);
                xcone_v.setToConstant(0.0, stream);
                xcone_d.setToConstant(0.0, stream);
            }

            // PSD direct-x workspaces (parallel to slack PSD).
            if (numXPsdCones > 0) {
                xcone_psd_lambda      = BatchedVector(totalXPsdMatDim,    batchSize);
                xcone_psd_Lambdaisqrt = BatchedVector(totalXPsdMatDim,    batchSize);
                xcone_psd_R           = BatchedVector(totalXPsdMatSqDim,  batchSize);
                xcone_psd_Rinv        = BatchedVector(totalXPsdMatSqDim,  batchSize);
                xcone_psd_Hs          = BatchedVector(totalXPsdHsEntries, batchSize);
                xcone_psd_work_mat1   = BatchedVector(totalXPsdMatSqDim,  batchSize);
                xcone_psd_work_mat2   = BatchedVector(totalXPsdMatSqDim,  batchSize);
                xcone_psd_work_mat3   = BatchedVector(totalXPsdMatSqDim,  batchSize);
                xcone_psd_combined_scratch = BatchedVector(totalXConeNumel, batchSize);
                // Eigvals scratch sized to batchSize × max(xcone_psd_k).
                int64_t max_xpsd_k = 0;
                for (int64_t k : psd_k_host) max_xpsd_k = std::max(max_xpsd_k, k);
                xcone_psd_eigvals_scratch = BatchedVector(max_xpsd_k > 0 ? max_xpsd_k : 1, batchSize);
                xcone_psd_lambda.setToConstant(0.0, stream);
                xcone_psd_Lambdaisqrt.setToConstant(0.0, stream);
                xcone_psd_R.setToConstant(0.0, stream);
                xcone_psd_Rinv.setToConstant(0.0, stream);
                xcone_psd_Hs.setToConstant(0.0, stream);
                xcone_psd_work_mat1.setToConstant(0.0, stream);
                xcone_psd_work_mat2.setToConstant(0.0, stream);
                xcone_psd_work_mat3.setToConstant(0.0, stream);
                xcone_psd_combined_scratch.setToConstant(0.0, stream);
                xcone_psd_eigvals_scratch.setToConstant(0.0, stream);
            }
        }

        cudaStreamSynchronize(stream);
    }

    // No copy assignment (owns device memory)
    Cones& operator=(const Cones&) = delete;

    /**
     * @brief Move constructor - transfers ownership and nulls source pointers
     *
     * Custom implementation required because default move would copy raw pointers
     * causing double-free in destructor.
     */
    Cones(Cones&& other) noexcept
        : numZeroCones(other.numZeroCones),
          numNonnegCones(other.numNonnegCones),
          numExpCones(other.numExpCones),
          numSocCones(other.numSocCones),
          numPowerCones(other.numPowerCones),
          numPsdCones(other.numPsdCones),
          numGenPowerCones(other.numGenPowerCones),
          socConeDims(std::move(other.socConeDims)),
          socConeDimsOriginal(std::move(other.socConeDimsOriginal)),
          socSortPerm(std::move(other.socSortPerm)),
          powerAlphas(std::move(other.powerAlphas)),
          psdConeDims(std::move(other.psdConeDims)),
          psdConeDimsOriginal(std::move(other.psdConeDimsOriginal)),
          psdSortPerm(std::move(other.psdSortPerm)),
          totalPsdSvecDim(other.totalPsdSvecDim),
          totalPsdMatDim(other.totalPsdMatDim),
          totalPsdMatSqDim(other.totalPsdMatSqDim),
          totalPsdHsEntries(other.totalPsdHsEntries),
          d_psd_dims(other.d_psd_dims),
          d_psd_svec_offsets(other.d_psd_svec_offsets),
          d_psd_sz_offsets(other.d_psd_sz_offsets),
          d_psd_Hs_offsets(other.d_psd_Hs_offsets),
          d_psd_mat_offsets(other.d_psd_mat_offsets),
          d_psd_matsq_offsets(other.d_psd_matsq_offsets),
          genPowerAlphas(std::move(other.genPowerAlphas)),
          genPowerDim1s(std::move(other.genPowerDim1s)),
          genPowerDim2s(std::move(other.genPowerDim2s)),
          genPowerAlphasOriginal(std::move(other.genPowerAlphasOriginal)),
          genPowerDim1sOriginal(std::move(other.genPowerDim1sOriginal)),
          genPowerDim2sOriginal(std::move(other.genPowerDim2sOriginal)),
          genPowerSortPerm(std::move(other.genPowerSortPerm)),
          totalGenPowerDim(other.totalGenPowerDim),
          totalGenPowerAlphas(other.totalGenPowerAlphas),
          totalGenPowerGradEntries(other.totalGenPowerGradEntries),
          totalGenPowerHsEntries(other.totalGenPowerHsEntries),
          totalGenPowerHsDenseEntries(other.totalGenPowerHsDenseEntries),
          totalGenPowerDimSq(other.totalGenPowerDimSq),
          totalGenPowerDim2(other.totalGenPowerDim2),
          numSparseGenPow(other.numSparseGenPow),
          totalSocDim(other.totalSocDim),
          totalSocHsEntries(other.totalSocHsEntries),
          totalSocHsDenseEntries(other.totalSocHsDenseEntries),
          numSparseSoc(other.numSparseSoc),
          d_soc_dims(other.d_soc_dims),
          d_soc_offsets(other.d_soc_offsets),
          d_soc_sz_offsets(other.d_soc_sz_offsets),
          d_soc_Hs_offsets(other.d_soc_Hs_offsets),
          d_soc_Hs_dense_offsets(other.d_soc_Hs_dense_offsets),
          d_soc_sparse_offsets(other.d_soc_sparse_offsets),
          d_soc_sparse_indices(other.d_soc_sparse_indices),
          d_genPowerAlphas(other.d_genPowerAlphas),
          d_genPowerDim1s(other.d_genPowerDim1s),
          d_genPowerDim2s(other.d_genPowerDim2s),
          d_genPowerOffsets(other.d_genPowerOffsets),
          d_genPowerAlphaOffsets(other.d_genPowerAlphaOffsets),
          d_genPowerHsOffsets(other.d_genPowerHsOffsets),
          d_genPowerHsDenseOffsets(other.d_genPowerHsDenseOffsets),
          d_genPowerDimSqOffsets(other.d_genPowerDimSqOffsets),
          d_genPowerSzOffsets(other.d_genPowerSzOffsets),
          d_genPowerDims(other.d_genPowerDims),
          d_genPowerSparseOffsets(other.d_genPowerSparseOffsets),
          d_genPowerSparseIndices(other.d_genPowerSparseIndices),
          batchSize(other.batchSize),
          d_powerAlphas(other.d_powerAlphas),
          d_batch_margin_results(other.d_batch_margin_results),
          d_barrier_work(other.d_barrier_work),
          h_barrier_pinned(other.h_barrier_pinned),
          d_scaling_success(other.d_scaling_success),
          h_scaling_success_pinned(other.h_scaling_success_pinned),
          nonneg_w(std::move(other.nonneg_w)),
          nonneg_lambda(std::move(other.nonneg_lambda)),
          soc_u(std::move(other.soc_u)),
          soc_v(std::move(other.soc_v)),
          soc_d(std::move(other.soc_d)),
          soc_Hs(std::move(other.soc_Hs)),
          soc_w(std::move(other.soc_w)),
          soc_lambda(std::move(other.soc_lambda)),
          soc_eta(std::move(other.soc_eta)),
          exp_H_dual(std::move(other.exp_H_dual)),
          exp_Hs(std::move(other.exp_Hs)),
          exp_grad(std::move(other.exp_grad)),
          exp_z(std::move(other.exp_z)),
          power_H_dual(std::move(other.power_H_dual)),
          power_Hs(std::move(other.power_Hs)),
          power_grad(std::move(other.power_grad)),
          power_z(std::move(other.power_z)),
          psd_lambda(std::move(other.psd_lambda)),
          psd_Lambdaisqrt(std::move(other.psd_Lambdaisqrt)),
          psd_R(std::move(other.psd_R)),
          psd_Rinv(std::move(other.psd_Rinv)),
          psd_Hs(std::move(other.psd_Hs)),
          psd_work_mat1(std::move(other.psd_work_mat1)),
          psd_work_mat2(std::move(other.psd_work_mat2)),
          psd_work_mat3(std::move(other.psd_work_mat3)),
          psd_work_svec(std::move(other.psd_work_svec)),
          cusolverH_(other.cusolverH_),
          cublasH_(other.cublasH_),
          d_psd_info_(other.d_psd_info_),
          d_psd_work_(other.d_psd_work_),
          psd_work_size_(other.psd_work_size_),
          d_psd_gesvd_work_(other.d_psd_gesvd_work_),
          psd_gesvd_work_size_(other.psd_gesvd_work_size_),
          d_psd_syevd_work_(other.d_psd_syevd_work_),
          psd_syevd_work_size_(other.psd_syevd_work_size_),
          genpow_grad(std::move(other.genpow_grad)),
          genpow_z(std::move(other.genpow_z)),
          genpow_Hs(std::move(other.genpow_Hs)),
          genpow_p(std::move(other.genpow_p)),
          genpow_q(std::move(other.genpow_q)),
          genpow_r(std::move(other.genpow_r)),
          genpow_d1(std::move(other.genpow_d1)),
          genpow_d2(std::move(other.genpow_d2)),
          genpow_mu(std::move(other.genpow_mu)),
          genpow_smooth_zlocal(std::move(other.genpow_smooth_zlocal)),
          genpow_smooth_wlocal(std::move(other.genpow_smooth_wlocal)),
          genpow_smooth_res(std::move(other.genpow_smooth_res)),
          genpow_smooth_delta(std::move(other.genpow_smooth_delta)),
          genpow_smooth_hmat(std::move(other.genpow_smooth_hmat)),
          genpow_smooth_lmat(std::move(other.genpow_smooth_lmat)),
          genpow_correction_u(std::move(other.genpow_correction_u)),
          genpow_correction_gpsi(std::move(other.genpow_correction_gpsi)),
          genpow_pd_axes(std::move(other.genpow_pd_axes)),
          genpow_pd_coefs(std::move(other.genpow_pd_coefs)),
          genpow_pd_signs(std::move(other.genpow_pd_signs)),
          genpow_pd_active(std::move(other.genpow_pd_active)),
          genpow_pd_workspace(std::move(other.genpow_pd_workspace)),
          dir_cones(std::move(other.dir_cones)),
          numXCones(other.numXCones),
          totalXConeNumel(other.totalXConeNumel),
          totalXConeHsEntries(other.totalXConeHsEntries),
          totalSparseXSocDim(other.totalSparseXSocDim),
          numSparseXSoc(other.numSparseXSoc),
          numXPsdCones(other.numXPsdCones),
          totalXPsdSvecDim(other.totalXPsdSvecDim),
          totalXPsdMatDim(other.totalXPsdMatDim),
          totalXPsdMatSqDim(other.totalXPsdMatSqDim),
          totalXPsdHsEntries(other.totalXPsdHsEntries),
          d_xcone_kinds(other.d_xcone_kinds),
          d_xcone_dims(other.d_xcone_dims),
          d_xcone_numel_offsets(other.d_xcone_numel_offsets),
          d_xcone_hs_offsets(other.d_xcone_hs_offsets),
          d_xcone_indices(other.d_xcone_indices),
          d_xcone_sorted_indices(other.d_xcone_sorted_indices),
          d_xcone_cone_pos_for_sorted(other.d_xcone_cone_pos_for_sorted),
          d_xcone_sparse_indices(other.d_xcone_sparse_indices),
          d_xcone_sparse_offsets(other.d_xcone_sparse_offsets),
          d_xcone_kind_per_entry(other.d_xcone_kind_per_entry),
          d_xcone_psd_idx(other.d_xcone_psd_idx),
          d_xcone_psd_k(other.d_xcone_psd_k),
          d_xcone_psd_svec_offsets(other.d_xcone_psd_svec_offsets),
          d_xcone_psd_mat_offsets(other.d_xcone_psd_mat_offsets),
          d_xcone_psd_matsq_offsets(other.d_xcone_psd_matsq_offsets),
          d_xcone_psd_hs_offsets(other.d_xcone_psd_hs_offsets),
          d_xcone_psd_in_full_offsets(other.d_xcone_psd_in_full_offsets),
          numXExpCones(other.numXExpCones),
          numXPowerCones(other.numXPowerCones),
          numXGenPowerCones(other.numXGenPowerCones),
          d_xcone_exp_idx(other.d_xcone_exp_idx),
          d_xcone_pow_idx(other.d_xcone_pow_idx),
          d_xcone_pow_alpha(other.d_xcone_pow_alpha),
          d_xcone_genpow_idx(other.d_xcone_genpow_idx),
          d_xcone_genpow_dim1s(other.d_xcone_genpow_dim1s),
          d_xcone_genpow_dim2s(other.d_xcone_genpow_dim2s),
          d_xcone_genpow_alpha_offsets(other.d_xcone_genpow_alpha_offsets),
          d_xcone_genpow_dim_offsets(other.d_xcone_genpow_dim_offsets),
          d_xcone_genpow_sparse_idx(other.d_xcone_genpow_sparse_idx),
          d_xcone_genpow_sparse_offsets(other.d_xcone_genpow_sparse_offsets),
          d_xcone_genpow_alphas(other.d_xcone_genpow_alphas),
          totalXGenPowerDim(other.totalXGenPowerDim),
          totalXGenPowerAlphas(other.totalXGenPowerAlphas),
          totalXGenPowerDim2(other.totalXGenPowerDim2),
          numSparseXGenPow(other.numSparseXGenPow),
          totalSparseXGenPowDim(other.totalSparseXGenPowDim),
          xcone_w(std::move(other.xcone_w)),
          xcone_lambda(std::move(other.xcone_lambda)),
          xcone_eta(std::move(other.xcone_eta)),
          xcone_Hs(std::move(other.xcone_Hs)),
          xcone_u(std::move(other.xcone_u)),
          xcone_v(std::move(other.xcone_v)),
          xcone_d(std::move(other.xcone_d)),
          xcone_z(std::move(other.xcone_z)),
          xcone_grad_primal(std::move(other.xcone_grad_primal)),
          xcone_psd_lambda(std::move(other.xcone_psd_lambda)),
          xcone_psd_Lambdaisqrt(std::move(other.xcone_psd_Lambdaisqrt)),
          xcone_psd_R(std::move(other.xcone_psd_R)),
          xcone_psd_Rinv(std::move(other.xcone_psd_Rinv)),
          xcone_psd_Hs(std::move(other.xcone_psd_Hs)),
          xcone_psd_work_mat1(std::move(other.xcone_psd_work_mat1)),
          xcone_psd_work_mat2(std::move(other.xcone_psd_work_mat2)),
          xcone_psd_work_mat3(std::move(other.xcone_psd_work_mat3)),
          xcone_psd_combined_scratch(std::move(other.xcone_psd_combined_scratch)),
          xcone_psd_eigvals_scratch(std::move(other.xcone_psd_eigvals_scratch)),
          xcone_genpow_p(std::move(other.xcone_genpow_p)),
          xcone_genpow_q(std::move(other.xcone_genpow_q)),
          xcone_genpow_r(std::move(other.xcone_genpow_r)),
          xcone_genpow_d1(std::move(other.xcone_genpow_d1)),
          xcone_genpow_d2(std::move(other.xcone_genpow_d2)),
          xcone_genpow_pd_axes(std::move(other.xcone_genpow_pd_axes)),
          xcone_genpow_pd_coefs(std::move(other.xcone_genpow_pd_coefs)),
          xcone_genpow_pd_signs(std::move(other.xcone_genpow_pd_signs)),
          xcone_genpow_pd_active(std::move(other.xcone_genpow_pd_active)),
          xcone_genpow_pd_workspace(std::move(other.xcone_genpow_pd_workspace))
    {
        // Null out source pointers to prevent double-free
        other.d_soc_dims = nullptr;
        other.d_soc_offsets = nullptr;
        other.d_soc_sz_offsets = nullptr;
        other.d_soc_Hs_offsets = nullptr;
        other.d_soc_Hs_dense_offsets = nullptr;
        other.d_soc_sparse_offsets = nullptr;
        other.d_soc_sparse_indices = nullptr;
        other.d_psd_dims = nullptr;
        other.d_psd_svec_offsets = nullptr;
        other.d_psd_sz_offsets = nullptr;
        other.d_psd_Hs_offsets = nullptr;
        other.d_psd_mat_offsets = nullptr;
        other.d_psd_matsq_offsets = nullptr;
        other.d_genPowerAlphas = nullptr;
        other.d_genPowerDim1s = nullptr;
        other.d_genPowerDim2s = nullptr;
        other.d_genPowerOffsets = nullptr;
        other.d_genPowerAlphaOffsets = nullptr;
        other.d_genPowerHsOffsets = nullptr;
        other.d_genPowerHsDenseOffsets = nullptr;
        other.d_genPowerDimSqOffsets = nullptr;
        other.d_genPowerSzOffsets = nullptr;
        other.d_genPowerDims = nullptr;
        other.d_genPowerSparseOffsets = nullptr;
        other.d_genPowerSparseIndices = nullptr;
        other.d_powerAlphas = nullptr;
        other.d_batch_margin_results = nullptr;
        other.d_barrier_work = nullptr;
        other.h_barrier_pinned = nullptr;
        other.d_scaling_success = nullptr;
        other.h_scaling_success_pinned = nullptr;
        other.cusolverH_ = nullptr;
        other.cublasH_ = nullptr;
        other.d_psd_info_ = nullptr;
        other.d_psd_work_ = nullptr;
        other.d_psd_gesvd_work_ = nullptr;
        other.d_psd_syevd_work_ = nullptr;
        other.batchSize = 0;
        other.d_xcone_kinds = nullptr;
        other.d_xcone_dims = nullptr;
        other.d_xcone_numel_offsets = nullptr;
        other.d_xcone_hs_offsets = nullptr;
        other.d_xcone_indices = nullptr;
        other.d_xcone_sorted_indices = nullptr;
        other.d_xcone_cone_pos_for_sorted = nullptr;
        other.d_xcone_sparse_indices = nullptr;
        other.d_xcone_sparse_offsets = nullptr;
        other.d_xcone_kind_per_entry = nullptr;
        other.d_xcone_psd_idx = nullptr;
        other.d_xcone_psd_k = nullptr;
        other.d_xcone_psd_svec_offsets = nullptr;
        other.d_xcone_psd_mat_offsets = nullptr;
        other.d_xcone_psd_matsq_offsets = nullptr;
        other.d_xcone_psd_hs_offsets = nullptr;
        other.d_xcone_psd_in_full_offsets = nullptr;
        other.d_xcone_exp_idx = nullptr;
        other.d_xcone_pow_idx = nullptr;
        other.d_xcone_pow_alpha = nullptr;
        other.d_xcone_genpow_idx = nullptr;
        other.d_xcone_genpow_dim1s = nullptr;
        other.d_xcone_genpow_dim2s = nullptr;
        other.d_xcone_genpow_alpha_offsets = nullptr;
        other.d_xcone_genpow_dim_offsets = nullptr;
        other.d_xcone_genpow_sparse_idx = nullptr;
        other.d_xcone_genpow_sparse_offsets = nullptr;
        other.d_xcone_genpow_sparse_alpha_offsets = nullptr;
        other.d_xcone_genpow_sparse_q_offsets = nullptr;
        other.d_xcone_genpow_sparse_r_offsets = nullptr;
        other.d_xcone_genpow_sparse_to_gidx = nullptr;
        other.d_xcone_genpow_alphas = nullptr;
        other.numXExpCones = 0;
        other.numXPowerCones = 0;
        other.numXGenPowerCones = 0;
        other.totalXGenPowerDim = 0;
        other.totalXGenPowerAlphas = 0;
        other.totalXGenPowerDim2 = 0;
        other.numSparseXGenPow = 0;
        other.totalSparseXGenPowDim = 0;
        other.numXCones = 0;
        other.totalXConeNumel = 0;
        other.totalXConeHsEntries = 0;
        other.totalSparseXSocDim = 0;
        other.numSparseXSoc = 0;
        other.numXPsdCones = 0;
        other.totalXPsdSvecDim = 0;
        other.totalXPsdMatDim = 0;
        other.totalXPsdMatSqDim = 0;
        other.totalXPsdHsEntries = 0;
    }

    /**
     * @brief Move assignment - transfers ownership and nulls source pointers
     */
    Cones& operator=(Cones&& other) noexcept {
        if (this != &other) {
            // Free existing resources
            if (d_soc_dims) cudaFree(d_soc_dims);
            if (d_soc_offsets) cudaFree(d_soc_offsets);
            if (d_soc_sz_offsets) cudaFree(d_soc_sz_offsets);
            if (d_soc_Hs_offsets) cudaFree(d_soc_Hs_offsets);
            if (d_soc_Hs_dense_offsets) cudaFree(d_soc_Hs_dense_offsets);
            if (d_soc_sparse_offsets) cudaFree(d_soc_sparse_offsets);
            if (d_soc_sparse_indices) cudaFree(d_soc_sparse_indices);
            if (d_psd_dims) cudaFree(d_psd_dims);
            if (d_psd_svec_offsets) cudaFree(d_psd_svec_offsets);
            if (d_psd_sz_offsets) cudaFree(d_psd_sz_offsets);
            if (d_psd_Hs_offsets) cudaFree(d_psd_Hs_offsets);
            if (d_psd_mat_offsets) cudaFree(d_psd_mat_offsets);
            if (d_psd_matsq_offsets) cudaFree(d_psd_matsq_offsets);
            if (d_genPowerAlphas) cudaFree(d_genPowerAlphas);
            if (d_genPowerDim1s) cudaFree(d_genPowerDim1s);
            if (d_genPowerDim2s) cudaFree(d_genPowerDim2s);
            if (d_genPowerOffsets) cudaFree(d_genPowerOffsets);
            if (d_genPowerAlphaOffsets) cudaFree(d_genPowerAlphaOffsets);
            if (d_genPowerHsOffsets) cudaFree(d_genPowerHsOffsets);
            if (d_genPowerHsDenseOffsets) cudaFree(d_genPowerHsDenseOffsets);
            if (d_genPowerDimSqOffsets) cudaFree(d_genPowerDimSqOffsets);
            if (d_genPowerSzOffsets) cudaFree(d_genPowerSzOffsets);
            if (d_genPowerDims) cudaFree(d_genPowerDims);
            if (d_genPowerSparseOffsets) cudaFree(d_genPowerSparseOffsets);
            if (d_genPowerSparseIndices) cudaFree(d_genPowerSparseIndices);
            if (d_powerAlphas) cudaFree(d_powerAlphas);
            if (d_batch_margin_results) cudaFree(d_batch_margin_results);
            if (d_barrier_work) cudaFree(d_barrier_work);
            if (h_barrier_pinned) cudaFreeHost(h_barrier_pinned);
            if (h_scaling_success_pinned) cudaFreeHost(h_scaling_success_pinned);
            if (d_xcone_kinds) cudaFree(d_xcone_kinds);
            if (d_xcone_dims) cudaFree(d_xcone_dims);
            if (d_xcone_numel_offsets) cudaFree(d_xcone_numel_offsets);
            if (d_xcone_hs_offsets) cudaFree(d_xcone_hs_offsets);
            if (d_xcone_indices) cudaFree(d_xcone_indices);
            if (d_xcone_sorted_indices) cudaFree(d_xcone_sorted_indices);
            if (d_xcone_cone_pos_for_sorted) cudaFree(d_xcone_cone_pos_for_sorted);
            if (d_xcone_sparse_indices) cudaFree(d_xcone_sparse_indices);
            if (d_xcone_sparse_offsets) cudaFree(d_xcone_sparse_offsets);
            if (d_xcone_kind_per_entry) cudaFree(d_xcone_kind_per_entry);
            // Direct-x PSD device arrays — the destructor frees these but
            // the move-assign "free existing resources" block was missing
            // them, so move-assigning a Cones that already owned PSD x-cone
            // metadata leaked every one of these pointers.
            if (d_xcone_psd_idx) cudaFree(d_xcone_psd_idx);
            if (d_xcone_psd_k) cudaFree(d_xcone_psd_k);
            if (d_xcone_psd_svec_offsets) cudaFree(d_xcone_psd_svec_offsets);
            if (d_xcone_psd_mat_offsets) cudaFree(d_xcone_psd_mat_offsets);
            if (d_xcone_psd_matsq_offsets) cudaFree(d_xcone_psd_matsq_offsets);
            if (d_xcone_psd_hs_offsets) cudaFree(d_xcone_psd_hs_offsets);
            if (d_xcone_psd_in_full_offsets) cudaFree(d_xcone_psd_in_full_offsets);
            if (d_xcone_exp_idx) cudaFree(d_xcone_exp_idx);
            if (d_xcone_pow_idx) cudaFree(d_xcone_pow_idx);
            if (d_xcone_pow_alpha) cudaFree(d_xcone_pow_alpha);
            if (d_xcone_genpow_idx) cudaFree(d_xcone_genpow_idx);
            if (d_xcone_genpow_dim1s) cudaFree(d_xcone_genpow_dim1s);
            if (d_xcone_genpow_dim2s) cudaFree(d_xcone_genpow_dim2s);
            if (d_xcone_genpow_alpha_offsets) cudaFree(d_xcone_genpow_alpha_offsets);
            if (d_xcone_genpow_dim_offsets) cudaFree(d_xcone_genpow_dim_offsets);
            if (d_xcone_genpow_sparse_idx) cudaFree(d_xcone_genpow_sparse_idx);
            if (d_xcone_genpow_sparse_offsets) cudaFree(d_xcone_genpow_sparse_offsets);
            if (d_xcone_genpow_sparse_alpha_offsets) cudaFree(d_xcone_genpow_sparse_alpha_offsets);
            if (d_xcone_genpow_sparse_q_offsets) cudaFree(d_xcone_genpow_sparse_q_offsets);
            if (d_xcone_genpow_sparse_r_offsets) cudaFree(d_xcone_genpow_sparse_r_offsets);
            if (d_xcone_genpow_sparse_to_gidx) cudaFree(d_xcone_genpow_sparse_to_gidx);
            if (d_xcone_genpow_alphas) cudaFree(d_xcone_genpow_alphas);
            if (d_psd_info_) cudaFree(d_psd_info_);
            if (d_psd_work_) cudaFree(d_psd_work_);
            if (d_psd_gesvd_work_) cudaFree(d_psd_gesvd_work_);
            if (d_psd_syevd_work_) cudaFree(d_psd_syevd_work_);
            if (cusolverH_) cusolverDnDestroy(cusolverH_);
            if (cublasH_) cublasDestroy(cublasH_);

            // Copy scalar values
            numZeroCones = other.numZeroCones;
            numNonnegCones = other.numNonnegCones;
            numExpCones = other.numExpCones;
            numSocCones = other.numSocCones;
            numPowerCones = other.numPowerCones;
            numPsdCones = other.numPsdCones;
            numGenPowerCones = other.numGenPowerCones;
            totalSocDim = other.totalSocDim;
            totalSocHsEntries = other.totalSocHsEntries;
            totalSocHsDenseEntries = other.totalSocHsDenseEntries;
            numSparseSoc = other.numSparseSoc;
            totalPsdSvecDim = other.totalPsdSvecDim;
            totalPsdMatDim = other.totalPsdMatDim;
            totalPsdMatSqDim = other.totalPsdMatSqDim;
            totalPsdHsEntries = other.totalPsdHsEntries;
            totalGenPowerDim = other.totalGenPowerDim;
            totalGenPowerAlphas = other.totalGenPowerAlphas;
            totalGenPowerGradEntries = other.totalGenPowerGradEntries;
            totalGenPowerHsEntries = other.totalGenPowerHsEntries;
            totalGenPowerHsDenseEntries = other.totalGenPowerHsDenseEntries;
            totalGenPowerDimSq = other.totalGenPowerDimSq;
            totalGenPowerDim2 = other.totalGenPowerDim2;
            numSparseGenPow = other.numSparseGenPow;
            batchSize = other.batchSize;
            numXCones = other.numXCones;
            totalXConeNumel = other.totalXConeNumel;
            totalXConeHsEntries = other.totalXConeHsEntries;
            totalSparseXSocDim = other.totalSparseXSocDim;
            numSparseXSoc = other.numSparseXSoc;

            // Move vectors
            dir_cones = std::move(other.dir_cones);
            socConeDims = std::move(other.socConeDims);
            socConeDimsOriginal = std::move(other.socConeDimsOriginal);
            socSortPerm = std::move(other.socSortPerm);
            powerAlphas = std::move(other.powerAlphas);
            psdConeDims = std::move(other.psdConeDims);
            psdConeDimsOriginal = std::move(other.psdConeDimsOriginal);
            psdSortPerm = std::move(other.psdSortPerm);
            genPowerAlphas = std::move(other.genPowerAlphas);
            genPowerDim1s = std::move(other.genPowerDim1s);
            genPowerDim2s = std::move(other.genPowerDim2s);
            genPowerAlphasOriginal = std::move(other.genPowerAlphasOriginal);
            genPowerDim1sOriginal = std::move(other.genPowerDim1sOriginal);
            genPowerDim2sOriginal = std::move(other.genPowerDim2sOriginal);
            genPowerSortPerm = std::move(other.genPowerSortPerm);

            // Transfer pointer ownership
            d_soc_dims = other.d_soc_dims;
            d_soc_offsets = other.d_soc_offsets;
            d_soc_sz_offsets = other.d_soc_sz_offsets;
            d_soc_Hs_offsets = other.d_soc_Hs_offsets;
            d_soc_Hs_dense_offsets = other.d_soc_Hs_dense_offsets;
            d_soc_sparse_offsets = other.d_soc_sparse_offsets;
            d_soc_sparse_indices = other.d_soc_sparse_indices;
            d_psd_dims = other.d_psd_dims;
            d_psd_svec_offsets = other.d_psd_svec_offsets;
            d_psd_sz_offsets = other.d_psd_sz_offsets;
            d_psd_Hs_offsets = other.d_psd_Hs_offsets;
            d_psd_mat_offsets = other.d_psd_mat_offsets;
            d_psd_matsq_offsets = other.d_psd_matsq_offsets;
            d_genPowerAlphas = other.d_genPowerAlphas;
            d_genPowerDim1s = other.d_genPowerDim1s;
            d_genPowerDim2s = other.d_genPowerDim2s;
            d_genPowerOffsets = other.d_genPowerOffsets;
            d_genPowerAlphaOffsets = other.d_genPowerAlphaOffsets;
            d_genPowerHsOffsets = other.d_genPowerHsOffsets;
            d_genPowerHsDenseOffsets = other.d_genPowerHsDenseOffsets;
            d_genPowerDimSqOffsets = other.d_genPowerDimSqOffsets;
            d_genPowerSzOffsets = other.d_genPowerSzOffsets;
            d_genPowerDims = other.d_genPowerDims;
            d_genPowerSparseOffsets = other.d_genPowerSparseOffsets;
            d_genPowerSparseIndices = other.d_genPowerSparseIndices;
            d_powerAlphas = other.d_powerAlphas;
            d_batch_margin_results = other.d_batch_margin_results;
            d_barrier_work = other.d_barrier_work;
            h_barrier_pinned = other.h_barrier_pinned;
            d_scaling_success = other.d_scaling_success;
            h_scaling_success_pinned = other.h_scaling_success_pinned;
            d_xcone_kinds = other.d_xcone_kinds;
            d_xcone_kind_per_entry = other.d_xcone_kind_per_entry;
            d_xcone_dims = other.d_xcone_dims;
            d_xcone_numel_offsets = other.d_xcone_numel_offsets;
            d_xcone_hs_offsets = other.d_xcone_hs_offsets;
            d_xcone_indices = other.d_xcone_indices;
            d_xcone_sorted_indices = other.d_xcone_sorted_indices;
            d_xcone_cone_pos_for_sorted = other.d_xcone_cone_pos_for_sorted;
            d_xcone_sparse_indices = other.d_xcone_sparse_indices;
            d_xcone_sparse_offsets = other.d_xcone_sparse_offsets;
            d_xcone_psd_idx = other.d_xcone_psd_idx;
            d_xcone_psd_k = other.d_xcone_psd_k;
            d_xcone_psd_svec_offsets = other.d_xcone_psd_svec_offsets;
            d_xcone_psd_mat_offsets = other.d_xcone_psd_mat_offsets;
            d_xcone_psd_matsq_offsets = other.d_xcone_psd_matsq_offsets;
            d_xcone_psd_hs_offsets = other.d_xcone_psd_hs_offsets;
            d_xcone_psd_in_full_offsets = other.d_xcone_psd_in_full_offsets;
            d_xcone_exp_idx = other.d_xcone_exp_idx;
            d_xcone_pow_idx = other.d_xcone_pow_idx;
            d_xcone_pow_alpha = other.d_xcone_pow_alpha;
            d_xcone_genpow_idx = other.d_xcone_genpow_idx;
            d_xcone_genpow_dim1s = other.d_xcone_genpow_dim1s;
            d_xcone_genpow_dim2s = other.d_xcone_genpow_dim2s;
            d_xcone_genpow_alpha_offsets = other.d_xcone_genpow_alpha_offsets;
            d_xcone_genpow_dim_offsets = other.d_xcone_genpow_dim_offsets;
            d_xcone_genpow_sparse_idx = other.d_xcone_genpow_sparse_idx;
            d_xcone_genpow_sparse_offsets = other.d_xcone_genpow_sparse_offsets;
            d_xcone_genpow_sparse_alpha_offsets = other.d_xcone_genpow_sparse_alpha_offsets;
            d_xcone_genpow_sparse_q_offsets = other.d_xcone_genpow_sparse_q_offsets;
            d_xcone_genpow_sparse_r_offsets = other.d_xcone_genpow_sparse_r_offsets;
            d_xcone_genpow_sparse_to_gidx = other.d_xcone_genpow_sparse_to_gidx;
            d_xcone_genpow_alphas = other.d_xcone_genpow_alphas;
            numXExpCones = other.numXExpCones;
            numXPowerCones = other.numXPowerCones;
            numXGenPowerCones = other.numXGenPowerCones;
            totalXGenPowerDim = other.totalXGenPowerDim;
            totalXGenPowerAlphas = other.totalXGenPowerAlphas;
            totalXGenPowerDim2 = other.totalXGenPowerDim2;
            numSparseXGenPow = other.numSparseXGenPow;
            totalSparseXGenPowDim = other.totalSparseXGenPowDim;
            numXPsdCones = other.numXPsdCones;
            totalXPsdSvecDim = other.totalXPsdSvecDim;
            totalXPsdMatDim = other.totalXPsdMatDim;
            totalXPsdMatSqDim = other.totalXPsdMatSqDim;
            totalXPsdHsEntries = other.totalXPsdHsEntries;
            cusolverH_ = other.cusolverH_;
            cublasH_ = other.cublasH_;
            d_psd_info_ = other.d_psd_info_;
            d_psd_work_ = other.d_psd_work_;
            psd_work_size_ = other.psd_work_size_;
            d_psd_gesvd_work_ = other.d_psd_gesvd_work_;
            psd_gesvd_work_size_ = other.psd_gesvd_work_size_;
            d_psd_syevd_work_ = other.d_psd_syevd_work_;
            psd_syevd_work_size_ = other.psd_syevd_work_size_;

            // Move BatchedVectors
            nonneg_w = std::move(other.nonneg_w);
            nonneg_lambda = std::move(other.nonneg_lambda);
            soc_u = std::move(other.soc_u);
            soc_v = std::move(other.soc_v);
            soc_d = std::move(other.soc_d);
            soc_Hs = std::move(other.soc_Hs);
            soc_w = std::move(other.soc_w);
            soc_lambda = std::move(other.soc_lambda);
            soc_eta = std::move(other.soc_eta);
            exp_H_dual = std::move(other.exp_H_dual);
            exp_Hs = std::move(other.exp_Hs);
            exp_grad = std::move(other.exp_grad);
            exp_z = std::move(other.exp_z);
            power_H_dual = std::move(other.power_H_dual);
            power_Hs = std::move(other.power_Hs);
            power_grad = std::move(other.power_grad);
            power_z = std::move(other.power_z);
            psd_lambda = std::move(other.psd_lambda);
            psd_Lambdaisqrt = std::move(other.psd_Lambdaisqrt);
            psd_R = std::move(other.psd_R);
            psd_Rinv = std::move(other.psd_Rinv);
            psd_Hs = std::move(other.psd_Hs);
            psd_work_mat1 = std::move(other.psd_work_mat1);
            psd_work_mat2 = std::move(other.psd_work_mat2);
            psd_work_mat3 = std::move(other.psd_work_mat3);
            psd_work_svec = std::move(other.psd_work_svec);
            genpow_grad = std::move(other.genpow_grad);
            genpow_z = std::move(other.genpow_z);
            genpow_Hs = std::move(other.genpow_Hs);
            genpow_p = std::move(other.genpow_p);
            genpow_q = std::move(other.genpow_q);
            genpow_r = std::move(other.genpow_r);
            genpow_d1 = std::move(other.genpow_d1);
            genpow_d2 = std::move(other.genpow_d2);
            genpow_mu = std::move(other.genpow_mu);
            genpow_smooth_zlocal = std::move(other.genpow_smooth_zlocal);
            genpow_smooth_wlocal = std::move(other.genpow_smooth_wlocal);
            genpow_smooth_res = std::move(other.genpow_smooth_res);
            genpow_smooth_delta = std::move(other.genpow_smooth_delta);
            genpow_smooth_hmat = std::move(other.genpow_smooth_hmat);
            genpow_smooth_lmat = std::move(other.genpow_smooth_lmat);
            genpow_correction_u = std::move(other.genpow_correction_u);
            genpow_correction_gpsi = std::move(other.genpow_correction_gpsi);
            genpow_pd_axes = std::move(other.genpow_pd_axes);
            genpow_pd_coefs = std::move(other.genpow_pd_coefs);
            genpow_pd_signs = std::move(other.genpow_pd_signs);
            genpow_pd_active = std::move(other.genpow_pd_active);
            genpow_pd_workspace = std::move(other.genpow_pd_workspace);
            xcone_w = std::move(other.xcone_w);
            xcone_lambda = std::move(other.xcone_lambda);
            xcone_eta = std::move(other.xcone_eta);
            xcone_Hs = std::move(other.xcone_Hs);
            xcone_u = std::move(other.xcone_u);
            xcone_v = std::move(other.xcone_v);
            xcone_d = std::move(other.xcone_d);
            xcone_z = std::move(other.xcone_z);
            xcone_grad_primal = std::move(other.xcone_grad_primal);
            xcone_psd_lambda = std::move(other.xcone_psd_lambda);
            xcone_psd_Lambdaisqrt = std::move(other.xcone_psd_Lambdaisqrt);
            xcone_psd_Hs = std::move(other.xcone_psd_Hs);
            xcone_psd_R = std::move(other.xcone_psd_R);
            xcone_psd_Rinv = std::move(other.xcone_psd_Rinv);
            xcone_psd_work_mat1 = std::move(other.xcone_psd_work_mat1);
            xcone_psd_work_mat2 = std::move(other.xcone_psd_work_mat2);
            xcone_psd_work_mat3 = std::move(other.xcone_psd_work_mat3);
            xcone_psd_combined_scratch = std::move(other.xcone_psd_combined_scratch);
            xcone_psd_eigvals_scratch = std::move(other.xcone_psd_eigvals_scratch);
            xcone_genpow_p = std::move(other.xcone_genpow_p);
            xcone_genpow_q = std::move(other.xcone_genpow_q);
            xcone_genpow_r = std::move(other.xcone_genpow_r);
            xcone_genpow_d1 = std::move(other.xcone_genpow_d1);
            xcone_genpow_d2 = std::move(other.xcone_genpow_d2);
            xcone_genpow_pd_axes = std::move(other.xcone_genpow_pd_axes);
            xcone_genpow_pd_coefs = std::move(other.xcone_genpow_pd_coefs);
            xcone_genpow_pd_signs = std::move(other.xcone_genpow_pd_signs);
            xcone_genpow_pd_active = std::move(other.xcone_genpow_pd_active);
            xcone_genpow_pd_workspace = std::move(other.xcone_genpow_pd_workspace);

            // Null out source pointers
            other.d_soc_dims = nullptr;
            other.d_soc_offsets = nullptr;
            other.d_soc_sz_offsets = nullptr;
            other.d_soc_Hs_offsets = nullptr;
            other.d_soc_Hs_dense_offsets = nullptr;
            other.d_soc_sparse_offsets = nullptr;
            other.d_soc_sparse_indices = nullptr;
            other.d_psd_dims = nullptr;
            other.d_psd_svec_offsets = nullptr;
            other.d_psd_sz_offsets = nullptr;
            other.d_psd_Hs_offsets = nullptr;
            other.d_psd_mat_offsets = nullptr;
            other.d_psd_matsq_offsets = nullptr;
            other.d_genPowerAlphas = nullptr;
            other.d_genPowerDim1s = nullptr;
            other.d_genPowerDim2s = nullptr;
            other.d_genPowerOffsets = nullptr;
            other.d_genPowerAlphaOffsets = nullptr;
            other.d_genPowerHsOffsets = nullptr;
            other.d_genPowerHsDenseOffsets = nullptr;
            other.d_genPowerDimSqOffsets = nullptr;
            other.d_genPowerSzOffsets = nullptr;
            other.d_genPowerDims = nullptr;
            other.d_genPowerSparseOffsets = nullptr;
            other.d_genPowerSparseIndices = nullptr;
            other.d_powerAlphas = nullptr;
            other.d_batch_margin_results = nullptr;
            other.d_barrier_work = nullptr;
            other.h_barrier_pinned = nullptr;
            other.d_scaling_success = nullptr;
            other.h_scaling_success_pinned = nullptr;
            other.cusolverH_ = nullptr;
            other.cublasH_ = nullptr;
            other.d_psd_info_ = nullptr;
            other.d_psd_work_ = nullptr;
            other.d_psd_gesvd_work_ = nullptr;
            other.d_psd_syevd_work_ = nullptr;
            other.batchSize = 0;
            other.d_xcone_kinds = nullptr;
            other.d_xcone_dims = nullptr;
            other.d_xcone_numel_offsets = nullptr;
            other.d_xcone_hs_offsets = nullptr;
            other.d_xcone_indices = nullptr;
            other.d_xcone_sorted_indices = nullptr;
            other.d_xcone_cone_pos_for_sorted = nullptr;
            other.d_xcone_sparse_indices = nullptr;
            other.d_xcone_sparse_offsets = nullptr;
            other.d_xcone_kind_per_entry = nullptr;
            other.d_xcone_psd_idx = nullptr;
            other.d_xcone_psd_k = nullptr;
            other.d_xcone_psd_svec_offsets = nullptr;
            other.d_xcone_psd_mat_offsets = nullptr;
            other.d_xcone_psd_matsq_offsets = nullptr;
            other.d_xcone_psd_hs_offsets = nullptr;
            other.d_xcone_psd_in_full_offsets = nullptr;
            other.d_xcone_exp_idx = nullptr;
            other.d_xcone_pow_idx = nullptr;
            other.d_xcone_pow_alpha = nullptr;
            other.d_xcone_genpow_idx = nullptr;
            other.d_xcone_genpow_dim1s = nullptr;
            other.d_xcone_genpow_dim2s = nullptr;
            other.d_xcone_genpow_alpha_offsets = nullptr;
            other.d_xcone_genpow_dim_offsets = nullptr;
            other.d_xcone_genpow_sparse_idx = nullptr;
            other.d_xcone_genpow_sparse_offsets = nullptr;
            other.d_xcone_genpow_sparse_alpha_offsets = nullptr;
            other.d_xcone_genpow_sparse_q_offsets = nullptr;
            other.d_xcone_genpow_sparse_r_offsets = nullptr;
            other.d_xcone_genpow_sparse_to_gidx = nullptr;
            other.d_xcone_genpow_alphas = nullptr;
            other.numXExpCones = 0;
            other.numXPowerCones = 0;
            other.numXGenPowerCones = 0;
            other.totalXGenPowerDim = 0;
            other.totalXGenPowerAlphas = 0;
            other.totalXGenPowerDim2 = 0;
            other.numSparseXGenPow = 0;
            other.totalSparseXGenPowDim = 0;
            other.numXCones = 0;
            other.totalXConeNumel = 0;
            other.totalXConeHsEntries = 0;
            other.totalSparseXSocDim = 0;
            other.numSparseXSoc = 0;
            other.numXPsdCones = 0;
            other.totalXPsdSvecDim = 0;
            other.totalXPsdMatDim = 0;
            other.totalXPsdMatSqDim = 0;
            other.totalXPsdHsEntries = 0;
        }
        return *this;
    }

    ~Cones() {
        if (d_soc_dims) { cudaFree(d_soc_dims); d_soc_dims = nullptr; }
        if (d_soc_offsets) { cudaFree(d_soc_offsets); d_soc_offsets = nullptr; }
        if (d_soc_sz_offsets) { cudaFree(d_soc_sz_offsets); d_soc_sz_offsets = nullptr; }
        if (d_soc_Hs_offsets) { cudaFree(d_soc_Hs_offsets); d_soc_Hs_offsets = nullptr; }
        if (d_soc_Hs_dense_offsets) { cudaFree(d_soc_Hs_dense_offsets); d_soc_Hs_dense_offsets = nullptr; }
        if (d_soc_sparse_offsets) { cudaFree(d_soc_sparse_offsets); d_soc_sparse_offsets = nullptr; }
        if (d_soc_sparse_indices) { cudaFree(d_soc_sparse_indices); d_soc_sparse_indices = nullptr; }
        if (d_psd_dims) { cudaFree(d_psd_dims); d_psd_dims = nullptr; }
        if (d_psd_svec_offsets) { cudaFree(d_psd_svec_offsets); d_psd_svec_offsets = nullptr; }
        if (d_psd_sz_offsets) { cudaFree(d_psd_sz_offsets); d_psd_sz_offsets = nullptr; }
        if (d_psd_Hs_offsets) { cudaFree(d_psd_Hs_offsets); d_psd_Hs_offsets = nullptr; }
        if (d_psd_mat_offsets) { cudaFree(d_psd_mat_offsets); d_psd_mat_offsets = nullptr; }
        if (d_psd_matsq_offsets) { cudaFree(d_psd_matsq_offsets); d_psd_matsq_offsets = nullptr; }
        if (d_genPowerAlphas) { cudaFree(d_genPowerAlphas); d_genPowerAlphas = nullptr; }
        if (d_genPowerDim1s) { cudaFree(d_genPowerDim1s); d_genPowerDim1s = nullptr; }
        if (d_genPowerDim2s) { cudaFree(d_genPowerDim2s); d_genPowerDim2s = nullptr; }
        if (d_genPowerOffsets) { cudaFree(d_genPowerOffsets); d_genPowerOffsets = nullptr; }
        if (d_genPowerAlphaOffsets) { cudaFree(d_genPowerAlphaOffsets); d_genPowerAlphaOffsets = nullptr; }
        if (d_genPowerHsOffsets) { cudaFree(d_genPowerHsOffsets); d_genPowerHsOffsets = nullptr; }
        if (d_genPowerHsDenseOffsets) { cudaFree(d_genPowerHsDenseOffsets); d_genPowerHsDenseOffsets = nullptr; }
        if (d_genPowerDimSqOffsets) { cudaFree(d_genPowerDimSqOffsets); d_genPowerDimSqOffsets = nullptr; }
        if (d_genPowerSzOffsets) { cudaFree(d_genPowerSzOffsets); d_genPowerSzOffsets = nullptr; }
        if (d_genPowerDims) { cudaFree(d_genPowerDims); d_genPowerDims = nullptr; }
        if (d_genPowerSparseOffsets) { cudaFree(d_genPowerSparseOffsets); d_genPowerSparseOffsets = nullptr; }
        if (d_genPowerSparseIndices) { cudaFree(d_genPowerSparseIndices); d_genPowerSparseIndices = nullptr; }
        if (d_powerAlphas) { cudaFree(d_powerAlphas); d_powerAlphas = nullptr; }
        if (d_psd_info_) { cudaFree(d_psd_info_); d_psd_info_ = nullptr; }
        if (d_psd_work_) { cudaFree(d_psd_work_); d_psd_work_ = nullptr; }
        if (d_psd_gesvd_work_) { cudaFree(d_psd_gesvd_work_); d_psd_gesvd_work_ = nullptr; }
        if (d_psd_syevd_work_) { cudaFree(d_psd_syevd_work_); d_psd_syevd_work_ = nullptr; }
        if (d_batch_margin_results) { cudaFree(d_batch_margin_results); d_batch_margin_results = nullptr; }
        if (d_barrier_work) { cudaFree(d_barrier_work); d_barrier_work = nullptr; }
        if (h_barrier_pinned) { cudaFreeHost(h_barrier_pinned); h_barrier_pinned = nullptr; }
        // d_scaling_success is a device-mapped pointer from h_scaling_success_pinned — don't cudaFree it
        if (h_scaling_success_pinned) { cudaFreeHost(h_scaling_success_pinned); h_scaling_success_pinned = nullptr; d_scaling_success = nullptr; }

        // Direct-x cone device arrays (nullptr when no x-cones; guarded
        // checks are cheap and keep this path uniform).
        if (d_xcone_kinds) { cudaFree(d_xcone_kinds); d_xcone_kinds = nullptr; }
        if (d_xcone_dims) { cudaFree(d_xcone_dims); d_xcone_dims = nullptr; }
        if (d_xcone_numel_offsets) { cudaFree(d_xcone_numel_offsets); d_xcone_numel_offsets = nullptr; }
        if (d_xcone_hs_offsets) { cudaFree(d_xcone_hs_offsets); d_xcone_hs_offsets = nullptr; }
        if (d_xcone_indices) { cudaFree(d_xcone_indices); d_xcone_indices = nullptr; }
        if (d_xcone_sorted_indices) { cudaFree(d_xcone_sorted_indices); d_xcone_sorted_indices = nullptr; }
        if (d_xcone_cone_pos_for_sorted) { cudaFree(d_xcone_cone_pos_for_sorted); d_xcone_cone_pos_for_sorted = nullptr; }
        if (d_xcone_sparse_indices) { cudaFree(d_xcone_sparse_indices); d_xcone_sparse_indices = nullptr; }
        if (d_xcone_sparse_offsets) { cudaFree(d_xcone_sparse_offsets); d_xcone_sparse_offsets = nullptr; }
        if (d_xcone_kind_per_entry) { cudaFree(d_xcone_kind_per_entry); d_xcone_kind_per_entry = nullptr; }
        if (d_xcone_psd_idx) { cudaFree(d_xcone_psd_idx); d_xcone_psd_idx = nullptr; }
        if (d_xcone_psd_k) { cudaFree(d_xcone_psd_k); d_xcone_psd_k = nullptr; }
        if (d_xcone_psd_svec_offsets) { cudaFree(d_xcone_psd_svec_offsets); d_xcone_psd_svec_offsets = nullptr; }
        if (d_xcone_psd_mat_offsets) { cudaFree(d_xcone_psd_mat_offsets); d_xcone_psd_mat_offsets = nullptr; }
        if (d_xcone_psd_matsq_offsets) { cudaFree(d_xcone_psd_matsq_offsets); d_xcone_psd_matsq_offsets = nullptr; }
        if (d_xcone_psd_hs_offsets) { cudaFree(d_xcone_psd_hs_offsets); d_xcone_psd_hs_offsets = nullptr; }
        if (d_xcone_psd_in_full_offsets) { cudaFree(d_xcone_psd_in_full_offsets); d_xcone_psd_in_full_offsets = nullptr; }
        if (d_xcone_exp_idx) { cudaFree(d_xcone_exp_idx); d_xcone_exp_idx = nullptr; }
        if (d_xcone_pow_idx) { cudaFree(d_xcone_pow_idx); d_xcone_pow_idx = nullptr; }
        if (d_xcone_pow_alpha) { cudaFree(d_xcone_pow_alpha); d_xcone_pow_alpha = nullptr; }
        if (d_xcone_genpow_idx) { cudaFree(d_xcone_genpow_idx); d_xcone_genpow_idx = nullptr; }
        if (d_xcone_genpow_dim1s) { cudaFree(d_xcone_genpow_dim1s); d_xcone_genpow_dim1s = nullptr; }
        if (d_xcone_genpow_dim2s) { cudaFree(d_xcone_genpow_dim2s); d_xcone_genpow_dim2s = nullptr; }
        if (d_xcone_genpow_alpha_offsets) { cudaFree(d_xcone_genpow_alpha_offsets); d_xcone_genpow_alpha_offsets = nullptr; }
        if (d_xcone_genpow_dim_offsets) { cudaFree(d_xcone_genpow_dim_offsets); d_xcone_genpow_dim_offsets = nullptr; }
        if (d_xcone_genpow_sparse_idx) { cudaFree(d_xcone_genpow_sparse_idx); d_xcone_genpow_sparse_idx = nullptr; }
        if (d_xcone_genpow_sparse_offsets) { cudaFree(d_xcone_genpow_sparse_offsets); d_xcone_genpow_sparse_offsets = nullptr; }
        if (d_xcone_genpow_sparse_alpha_offsets) { cudaFree(d_xcone_genpow_sparse_alpha_offsets); d_xcone_genpow_sparse_alpha_offsets = nullptr; }
        if (d_xcone_genpow_sparse_q_offsets) { cudaFree(d_xcone_genpow_sparse_q_offsets); d_xcone_genpow_sparse_q_offsets = nullptr; }
        if (d_xcone_genpow_sparse_r_offsets) { cudaFree(d_xcone_genpow_sparse_r_offsets); d_xcone_genpow_sparse_r_offsets = nullptr; }
        if (d_xcone_genpow_sparse_to_gidx) { cudaFree(d_xcone_genpow_sparse_to_gidx); d_xcone_genpow_sparse_to_gidx = nullptr; }
        if (d_xcone_genpow_alphas) { cudaFree(d_xcone_genpow_alphas); d_xcone_genpow_alphas = nullptr; }

        if (cusolverH_) { cusolverDnDestroy(cusolverH_); cusolverH_ = nullptr; }
        if (cublasH_) { cublasDestroy(cublasH_); cublasH_ = nullptr; }
    }

    bool isSymmetric() const noexcept {
        return numExpCones == 0 && numPowerCones == 0 && numGenPowerCones == 0 &&
               numXExpCones == 0 && numXPowerCones == 0 && numXGenPowerCones == 0;
    }

    /**
     * @brief Check if all cones allow primal-dual scaling
     *
     * All currently implemented cones (zero, nonneg, SOC, exp, power) support
     * primal-dual scaling. The generalized power cone returns false
     * (nonsymmetric cone, dual-only scaling).
     *
     * @return true if all cone types support primal-dual scaling, false otherwise
     */
    bool allows_primal_dual_scaling() const noexcept {
        // GenPow PD scaling uses the rank-9 augmented KKT (3 dual rank-3
        // axes for q/r/p + 6 Mosek-Tunçel PD axes). KKT structure is
        // allocated by `KKTData` with `p = 9 * numSparseGenPow`, axis
        // values are computed by `compute_genpow_pd_axes` after each
        // `update_scaling` call, and `update_kkt_H_block_kernel` writes
        // them into the expansion columns each iteration.
        return true;
    }

    /**
     * @brief Set identity scaling for symmetric cones
     *
     * For LP initialization, we need H = I (identity) so the KKT system is:
     * [0   A']   [x]   [rhs_x]
     * [A  -I ] * [z] = [rhs_z]
     *
     * For nonnegative cones: w = λ = 1, so Hs = w² = 1 (identity)
     * For SOC cones: η = 1, w = (1, 0, 0, ...), so Hs = η²(2ww' - J) = I
     */
    void set_identity_scaling(cudaStream_t stream = 0) {
        // For nonnegative cones, identity scaling means w = λ = 1
        if (numNonnegCones > 0) {
            nonneg_w.setToConstant(1.0, stream);
            nonneg_lambda.setToConstant(1.0, stream);
        }

        // For SOC cones, identity scaling means η = 1, w = (1, 0, ..., 0)
        // Dense: Hs = η²(2ww' - J) = I
        // Sparse: Hs = η²·diag(d, 1, ..., 1) with d=0.5, u[0]=1/√2, v=0
        if (numSocCones > 0) {
            soc_eta.setToConstant(1.0, stream);

            // Set sparse expansion data for identity scaling
            // d = 0.5, u = (1/√2, 0, ..., 0), v = 0
            std::vector<double> u_init(totalSocDim * batchSize, 0.0);
            std::vector<double> d_init(numSocCones * batchSize, 0.0);
            {
                int64_t off = 0;
                for (int64_t cone = 0; cone < numSocCones; cone++) {
                    int64_t dim = socConeDims[cone];
                    if (dim > 4) {
                        for (int64_t batch = 0; batch < batchSize; batch++) {
                            u_init[batch * totalSocDim + off] = 0.7071067811865476;  // 1/√2
                            d_init[batch * numSocCones + cone] = 0.5;
                        }
                    }
                    off += dim;
                }
            }
            cudaMemcpyAsync(soc_u.data(), u_init.data(),
                           sizeof(double) * totalSocDim * batchSize,
                           cudaMemcpyHostToDevice, stream);
            soc_v.setToConstant(0.0, stream);
            cudaMemcpyAsync(soc_d.data(), d_init.data(),
                           sizeof(double) * numSocCones * batchSize,
                           cudaMemcpyHostToDevice, stream);

            // Set w = (1, 0, ..., 0) for each SOC cone (variable dim)
            std::vector<double> w_init(totalSocDim * batchSize, 0.0);
            int64_t off = 0;
            for (int64_t cone = 0; cone < numSocCones; cone++) {
                int64_t dim = socConeDims[cone];
                for (int64_t batch = 0; batch < batchSize; batch++) {
                    w_init[batch * totalSocDim + off] = 1.0;
                }
                off += dim;
            }
            cudaMemcpyAsync(soc_w.data(), w_init.data(),
                           sizeof(double) * totalSocDim * batchSize,
                           cudaMemcpyHostToDevice, stream);

            // Set soc_Hs to identity for each cone
            // Dense (dim<=4): upper triangular format; Sparse (dim>4): diagonal only
            std::vector<double> Hs_identity(totalSocHsEntries * batchSize, 0.0);
            int64_t hs_off = 0;
            for (int64_t cone = 0; cone < numSocCones; cone++) {
                int64_t dim = socConeDims[cone];
                bool is_sparse = (dim > 4);
                for (int64_t batch = 0; batch < batchSize; batch++) {
                    int64_t idx = batch * totalSocHsEntries + hs_off;
                    if (is_sparse) {
                        // Diagonal-only: η²·diag(d, 1, ..., 1), with η=1, d=0.5
                        // First entry is d=0.5; rest are 1.0
                        // The expansion columns add η²*(uu' - vv') to reconstruct full Hs = I
                        Hs_identity[idx] = 0.5;
                        for (int64_t i = 1; i < dim; i++) {
                            Hs_identity[idx + i] = 1.0;
                        }
                    } else {
                        // Set diagonal entries of packed upper triangle to 1
                        // Row-major upper triangle: row i, col j >= i
                        int64_t k = 0;
                        for (int64_t row = 0; row < dim; row++) {
                            for (int64_t col = row; col < dim; col++) {
                                if (row == col) Hs_identity[idx + k] = 1.0;
                                k++;
                            }
                        }
                    }
                }
                hs_off += is_sparse ? dim : dim * (dim + 1) / 2;
            }
            cudaMemcpyAsync(soc_Hs.data(), Hs_identity.data(),
                           sizeof(double) * totalSocHsEntries * batchSize,
                           cudaMemcpyHostToDevice, stream);
        }

        // For PSD cones, identity scaling: Hs = I (svec_dim × svec_dim identity, upper triangle)
        // Also set R = I, Rinv = I, lambda = 1 for each cone
        if (numPsdCones > 0) {
            // Set psd_Hs to identity in upper-triangle format
            std::vector<double> psd_Hs_identity(totalPsdHsEntries * batchSize, 0.0);
            int64_t hs_off = 0;
            for (int64_t cone = 0; cone < numPsdCones; cone++) {
                int64_t dim = psdConeDims[cone];
                int64_t svec_dim = dim * (dim + 1) / 2;
                // Set diagonal entries of the svec_dim × svec_dim identity
                for (int64_t k = 0; k < svec_dim; k++) {
                    int64_t diag_idx = k * (k + 1) / 2 + k;
                    for (int64_t batch = 0; batch < batchSize; batch++) {
                        psd_Hs_identity[batch * totalPsdHsEntries + hs_off + diag_idx] = 1.0;
                    }
                }
                hs_off += svec_dim * (svec_dim + 1) / 2;
            }
            cudaMemcpyAsync(psd_Hs.data(), psd_Hs_identity.data(),
                           sizeof(double) * totalPsdHsEntries * batchSize,
                           cudaMemcpyHostToDevice, stream);

            // Set lambda = 1 for all PSD cones
            psd_lambda.setToConstant(1.0, stream);
            psd_Lambdaisqrt.setToConstant(1.0, stream);

            // Set R = I, Rinv = I for each cone (column-major n×n identity)
            std::vector<double> eye_data(totalPsdMatSqDim * batchSize, 0.0);
            int64_t matsq_off = 0;
            for (int64_t cone = 0; cone < numPsdCones; cone++) {
                int64_t dim = psdConeDims[cone];
                for (int64_t k = 0; k < dim; k++) {
                    for (int64_t batch = 0; batch < batchSize; batch++) {
                        eye_data[batch * totalPsdMatSqDim + matsq_off + k * dim + k] = 1.0;
                    }
                }
                matsq_off += dim * dim;
            }
            cudaMemcpyAsync(psd_R.data(), eye_data.data(),
                           sizeof(double) * totalPsdMatSqDim * batchSize,
                           cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(psd_Rinv.data(), eye_data.data(),
                           sizeof(double) * totalPsdMatSqDim * batchSize,
                           cudaMemcpyHostToDevice, stream);
        }

        cudaStreamSynchronize(stream);
    }

    /**
     * @brief Unit initialization for variables s and z
     *
     * Initializes the slack (s) and dual (z) variables to standard starting points
     * for each cone type. This is called during solver initialization.
     *
     * @param s Slack variables [batchSize][m]
     * @param z Dual variables [batchSize][m]
     * @param stream CUDA stream for async operations
     */
    void unit_initialization(BatchedVector& s, BatchedVector& z, cudaStream_t stream = 0);

    /**
     * @brief Compute margins for cone variables (per-batch)
     *
     * Returns per-batch minimum and positive margins.
     * This is the correct version for batched solves that need isolation.
     *
     * @param z Variables to check (s for primal, z for dual)
     * @param min_margin_out Output: minimum margin per batch [batchSize]
     * @param pos_margin_out Output: positive margin sum per batch [batchSize]
     * @param is_primal_cone true for primal cone (s), false for dual cone (z)
     * @param stream CUDA stream for async operations
     */
    void margins_batched(const BatchedVector& z, BatchedVector& min_margin_out, BatchedVector& pos_margin_out,
                         bool is_primal_cone, cudaStream_t stream = 0);

    /**
     * @brief Shift variables into cone interior (per-batch alpha)
     *
     * Shifts each batch's variables by its own alpha value.
     * This is the correct version for batched solves that need isolation.
     *
     * @param z Variables to shift (s for primal, z for dual)
     * @param alpha Per-batch shift amounts [batchSize]
     * @param is_primal_cone true for primal cone (s), false for dual cone (z)
     * @param stream CUDA stream for async operations
     */
    void scaled_unit_shift_batched(BatchedVector& z, const BatchedVector& alpha, bool is_primal_cone, cudaStream_t stream = 0);

    /**
     * @brief Fused: compute margins + alpha + apply shift in one kernel
     */
    void fused_margins_and_shift(BatchedVector& z, bool is_primal_cone, double total_degree, cudaStream_t stream = 0);

    /**
     * @brief Shift PSD cones into interior using eigendecomp-based margins
     * Matches CPU _shift_to_cone_interior for PSD cones.
     */
    void psd_shift_to_interior(BatchedVector& z, double total_degree, cudaStream_t stream = 0);

    /**
     * @brief Get total number of constraints
     */
    [[nodiscard]] int64_t totalConstraints() const noexcept {
        // Use totalSocDim if initialized, otherwise compute from socConeDims
        int64_t socDim = totalSocDim;
        if (socDim == 0 && numSocCones > 0) {
            if (!socConeDims.empty()) {
                for (auto d : socConeDims) socDim += d;
            } else {
                socDim = numSocCones * 3;  // fallback before initialize()
            }
        }
        // Use totalPsdSvecDim if initialized, otherwise compute from psdConeDims
        int64_t psdDim = totalPsdSvecDim;
        if (psdDim == 0 && numPsdCones > 0) {
            if (!psdConeDims.empty()) {
                for (auto d : psdConeDims) psdDim += d * (d + 1) / 2;
            }
        }
        // Use totalGenPowerDim if initialized, otherwise compute from genPowerDim1s/Dim2s
        int64_t gpDim = totalGenPowerDim;
        if (gpDim == 0 && numGenPowerCones > 0) {
            for (int64_t i = 0; i < numGenPowerCones; ++i) {
                gpDim += genPowerDim1s[i] + genPowerDim2s[i];
            }
        }
        return numZeroCones + numNonnegCones + socDim + psdDim + numExpCones * 3 + numPowerCones * 3 + gpDim;
    }

    // Public slack-vector layout, matching Clarabel/CVXPY and the CPU backend:
    // Zero, Nonnegative, SOC, PSD, Exponential, Power, Generalized Power.
    [[nodiscard]] int64_t psdOffset() const noexcept {
        return numZeroCones + numNonnegCones + totalSocDim;
    }

    [[nodiscard]] int64_t expOffset() const noexcept {
        return psdOffset() + totalPsdSvecDim;
    }

    [[nodiscard]] int64_t powerOffset() const noexcept {
        return expOffset() + numExpCones * 3;
    }

    [[nodiscard]] int64_t genPowerOffset() const noexcept {
        return powerOffset() + numPowerCones * 3;
    }

    /**
     * @brief Get total cone degree for complementarity measure
     *
     * Cone degrees:
     * - Zero cone: 0
     * - Nonneg cone: dim (number of variables)
     * - SOC cone: 1 per cone
     * - Exp cone: 3 per cone
     * - Power cone: 3 per cone
     */
    [[nodiscard]] int64_t degree() const noexcept {
        // PSD cone degree = matrix dim n per cone
        int64_t psdDegree = totalPsdMatDim;
        if (psdDegree == 0 && numPsdCones > 0) {
            if (!psdConeDims.empty()) {
                for (auto d : psdConeDims) psdDegree += d;
            }
        }
        // GenPowerCone degree = dim1 + 1 per cone
        int64_t gpDeg = 0;
        for (int64_t i = 0; i < numGenPowerCones; ++i) {
            gpDeg += genPowerDim1s[i] + 1;
        }
        // Direct-x cone degree: nonneg contributes numel per cone, SOC
        // contributes 1 per cone, PSD contributes psd_k per cone (matches
        // the CPU CompositeXCone layout and slack PSD).
        int64_t xDeg = 0;
        for (const auto& xc : dir_cones) {
            if (xc.kind == XConeKind::Nonneg) {
                xDeg += static_cast<int64_t>(xc.indices.size());
            } else if (xc.kind == XConeKind::SOC) {
                xDeg += 1;
            } else if (xc.kind == XConeKind::Exp || xc.kind == XConeKind::Power) {
                xDeg += 3;
            } else if (xc.kind == XConeKind::PSD) {
                xDeg += xc.psd_k;
            } else if (xc.kind == XConeKind::GenPower) {
                // GenPower direct-x: degree = dim1 + 1 per cone
                // (same formula as slack GenPowerCone: genPowerDim1s[i] + 1)
                xDeg += static_cast<int64_t>(xc.gen_power_alphas.size()) + 1;
            }
        }
        return 0 * numZeroCones +        // Zero cones contribute 0
               1 * numNonnegCones +      // Nonneg cones contribute dim (1 per variable)
               1 * numSocCones +         // SOC cones contribute 1 per cone
               3 * numExpCones +         // Exp cones contribute 3 per cone
               3 * numPowerCones +       // Power cones contribute 3 per cone
               psdDegree +               // PSD cones contribute n per cone
               gpDeg +                   // GenPowerCones contribute dim1+1 per cone
               xDeg;                     // Direct-x cones (nonneg: numel; SOC: 1)
    }

    /**
     * @brief Update cone scaling based on current s, z, and μ
     *
     * Updates the scaling matrices for each cone type based on the
     * current primal slack s and dual z variables.
     *
     * @param s Slack variables [batchSize][m]
     * @param z Dual variables [batchSize][m]
     * @param μ Complementarity measure
     * @param scaling Scaling strategy (PrimalDual or Dual)
     * @param stream CUDA stream for async operations
     * @return true if scaling succeeded, false if scaling failed
     */
    bool update_scaling(const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ, ScalingStrategy scaling, cudaStream_t stream = 0, const int8_t* pd_enabled_per_batch = nullptr);

    /**
     * @brief Compute affine step ds for all cones
     *
     * Computes the affine step direction for the slack variable s.
     * Different cone types have different formulas:
     * - Zero cone: ds = 0
     * - Nonnegative: ds = λ²
     * - SOC: ds = λ ∘ λ (circle product)
     * - Exp/Power: ds = s (copy)
     *
     * @param ds Output: affine step direction [batchSize][m]
     * @param s Current slack variables [batchSize][m]
     * @param stream CUDA stream for async operations
     */
    void affine_ds(BatchedVector& ds, const BatchedVector& s, cudaStream_t stream = 0);

    /**
     * @brief Compute Δs from Δz offset
     *
     * For combined step direction, computes the offset term for Δs based on Δz.
     * This is used in the formula: HₛΔz + Δs = -c
     *
     * @param out Output vector [batchSize][m]
     * @param ds Input ds vector [batchSize][m]
     * @param work Work vector [batchSize][m] (unused in most cones)
     * @param z Dual variables [batchSize][m]
     * @param stream CUDA stream for async operations
     */
    void Δs_from_Δz_offset(BatchedVector& out, const BatchedVector& ds, BatchedVector& work, const BatchedVector& z, cudaStream_t stream = 0);

    /**
     * @brief Multiply by Hs matrix (cone scaling Hessian)
     *
     * Computes y = Hs * x where Hs is the scaled Hessian:
     * - Zero cone: y = 0
     * - Nonnegative: y = w² * x (element-wise)
     * - SOC: y = η²(2ww^T - J)x where J = diag(1,-I)
     * - Exp/Power: y = Hs * x (stored 3x3 matrix multiply)
     *
     * @param y Output vector [batchSize][m]
     * @param x Input vector [batchSize][m]
     * @param work Work vector [batchSize][m]
     * @param stream CUDA stream for async operations
     */
    void mul_Hs(BatchedVector& y, const BatchedVector& x, BatchedVector& work, cudaStream_t stream = 0);

    /**
     * @brief Calculate maximum step length to stay within cones
     *
     * Computes the maximum α such that (s + α*ds, z + α*dz) remains in the cone.
     * Returns a pair (αz, αs) for dual and primal step lengths.
     *
     * @param dz Step direction for dual variable [batchSize][m]
     * @param ds Step direction for slack variable [batchSize][m]
     * @param z Current dual variable [batchSize][m]
     * @param s Current slack variable [batchSize][m]
     * @param alpha_max Maximum allowed step length [batchSize][1]
     * @param alpha_z Output: maximum step length for z [batchSize][1]
     * @param alpha_s Output: maximum step length for s [batchSize][1]
     * @param stream CUDA stream for async operations
     */
    void step_length(
        const BatchedVector& dz,
        const BatchedVector& ds,
        const BatchedVector& z,
        const BatchedVector& s,
        const BatchedVector& alpha_max,
        BatchedVector& alpha_z,
        BatchedVector& alpha_s,
        double backtrack_step = 0.8,
        double min_step_length = 1e-4,
        cudaStream_t stream = 0);

    /**
     * @brief Compute combined step DS shift for Mehrotra correction
     *
     * For symmetric cones, computes: shift = W^-1(step_s) ∘ W(step_z) - σμe
     * Where ∘ is the circle product (Jordan algebra) and e is the unit element.
     *
     * This is used in the combined step calculation for predictor-corrector methods.
     * The step_z and step_s vectors are also modified in place (scaled by W and W^-1).
     *
     * @param shift Output: combined shift vector [batchSize][m]
     * @param step_z Input/Output: affine step for z, modified to W(step_z) [batchSize][m]
     * @param step_s Input/Output: affine step for s, modified to W^-1(step_s) [batchSize][m]
     * @param sigma_mu Product of centering parameter and complementarity μ [batchSize]
     * @param stream CUDA stream for async operations
     */
    void combined_ds_shift(
        BatchedVector& shift,
        BatchedVector& step_z,
        BatchedVector& step_s,
        const BatchedVector& sigma_mu,
        cudaStream_t stream = 0);

    /**
     * @brief Compute barrier function for cones at variables + α * step
     *
     * Sum of barrier functions over all cone types:
     * - Zero cone: 0
     * - Nonnegative: -sum(log(s_i * z_i))
     * - SOC: -0.5 * log(residual_s * residual_z)
     * - Exponential: barrier_primal(s) + barrier_dual(z)
     * - Power: barrier_primal(s) + barrier_dual(z)
     *
     * @param z Dual variables [batchSize][m]
     * @param s Slack variables [batchSize][m]
     * @param dz Dual step direction [batchSize][m]
     * @param ds Slack step direction [batchSize][m]
     * @param alpha Step length
     * @param stream CUDA stream for async operations
     * @return Sum of barrier values across all batch elements
     */
    double computeBarrier(
        const BatchedVector& z,
        const BatchedVector& s,
        const BatchedVector& dz,
        const BatchedVector& ds,
        double alpha,
        cudaStream_t stream = 0);

    /**
     * @brief Apply cone smoothing for warm start projection
     *
     * Projects z onto the mu-central path using per-cone smoothing formulas.
     * After calling this, recover s = z - work.
     *
     * @param z Dual variables to smooth (modified in-place) [batchSize][m]
     * @param work Preserved quantity work = z_orig - s_orig [batchSize][m]
     * @param mu Per-batch warmness parameter [batchSize]
     * @param stream CUDA stream
     */
    void smoothing(BatchedVector& z, const BatchedVector& work, const BatchedVector& mu, cudaStream_t stream = 0);

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept {
        size_t total = 0;

        if (numNonnegCones > 0) {
            total += nonneg_w.memoryUsage() + nonneg_lambda.memoryUsage();
        }

        if (numSocCones > 0) {
            total += soc_u.memoryUsage() + soc_v.memoryUsage() + soc_d.memoryUsage() +
                     soc_Hs.memoryUsage() + soc_w.memoryUsage() + soc_lambda.memoryUsage() +
                     soc_eta.memoryUsage();
        }

        if (numExpCones > 0) {
            total += exp_H_dual.memoryUsage() + exp_Hs.memoryUsage() +
                     exp_grad.memoryUsage() + exp_z.memoryUsage();
        }

        if (numPowerCones > 0) {
            total += power_H_dual.memoryUsage() + power_Hs.memoryUsage() +
                     power_grad.memoryUsage() + power_z.memoryUsage();
            if (d_powerAlphas) {
                total += sizeof(double) * numPowerCones;
            }
        }

        if (numGenPowerCones > 0) {
            total += genpow_grad.memoryUsage() + genpow_z.memoryUsage() +
                     genpow_Hs.memoryUsage() + genpow_p.memoryUsage() +
                     genpow_q.memoryUsage() + genpow_r.memoryUsage() +
                     genpow_d1.memoryUsage() + genpow_d2.memoryUsage();
            if (d_genPowerAlphas) {
                total += sizeof(double) * totalGenPowerAlphas;
                total += sizeof(int64_t) * numGenPowerCones * 2;  // dim1s, dim2s
                total += sizeof(int64_t) * (numGenPowerCones + 1) * 3;  // offsets
            }
        }

        return total;
    }
};

} // namespace moreau
