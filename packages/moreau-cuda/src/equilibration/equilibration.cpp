// equilibration.cpp
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "moreau/equilibration/equilibration.hpp"
#include "moreau/equilibration/equilibration_kernels.cuh"
#include "moreau/cuda/utils.hpp"

namespace moreau {

void equilibration(
    EquilibrationData& equil,
    CSR& P,
    int64_t* matrixPRowOf,
    CSR& A,
    int64_t* matrixARowOf,
    BatchedVector& q,
    BatchedVector& b,
    const EquilibrationSettings& settings,
    const Cones& cones,
    cudaStream_t stream)
{
    equil.reset(stream);

    int n = P.rows();
    int m = A.rows();
    int nnzP = P.nnz();
    int batchSize = P.batchSize();

    BatchedVector dMeanColNormP(1, batchSize);
    BatchedVector dInfNormQ(1, batchSize);

    for (int iter = 0; iter < settings.max_iter; iter++) {
        // For LP (empty P), skip P row norms computation
        if (nnzP > 0) {
            compute_batch_row_norms_P(P, matrixPRowOf, equil.dwork, stream);
        }

        compute_batch_col_norms_A_noreset(A, equil.dwork, stream);

        compute_batch_row_norms_A(A, matrixARowOf, equil.ework, stream);

        process_equilibration(
            equil.dwork, equil.d, equil.ework, equil.e,
            settings.scale_min, settings.scale_max, n, m, batchSize, stream
        );

        scale_data(P, matrixPRowOf, A, matrixARowOf, q, b, &equil.dwork, &equil.ework, stream);

        // Compute row norms of P and conditional cost scaling
        // For LP (empty P), skip this step
        if (nnzP > 0) {
            compute_batch_row_norms_P(P, matrixPRowOf, equil.dwork, stream);

            compute_mean(equil.dwork, dMeanColNormP, stream);
            compute_inf_norm(q, dInfNormQ, stream);

            conditional_scale_cost(
                P, q, equil.c, dMeanColNormP, dInfNormQ,
                settings.scale_min, settings.scale_max, stream
            );
        }
    }

    // Rectify cone equilibration
    bool any_changed = false;

    // Zero and nonneg cones: set ework = 1.0 (no change)
    if (cones.numZeroCones + cones.numNonnegCones > 0) {
        set_array(equil.ework, 1.0, 0, cones.numZeroCones + cones.numNonnegCones, equil.ework.batchSize(), stream);
    }

    int64_t cone_offset = cones.numZeroCones + cones.numNonnegCones;

    // SOC cones: ework[i] = mean(e_cone) / e[i]
    if (cones.numSocCones > 0 && cones.totalSocDim > 0) {
        rectify_soc_cone_equilibration(
            equil.ework, equil.e,
            cone_offset,
            cones.d_soc_offsets,
            cones.d_soc_dims,
            cones.numSocCones,
            stream,
            cones.d_soc_sz_offsets
        );
        cone_offset += cones.totalSocDim;
        any_changed = true;
    }

    // PSD cones (variable-size groups -- need PSD-specific rectification)
    if (cones.numPsdCones > 0) {
        rectify_psd_cone_equilibration(
            equil.ework, equil.e,
            cone_offset,
            cones.d_psd_dims,
            cones.numPsdCones,
            stream,
            cones.d_psd_sz_offsets
        );
        cone_offset += cones.totalPsdSvecDim;
        any_changed = true;
    }

    // Exponential and power cones are all three-dimensional.
    int64_t cone_3d_size = cones.numExpCones * 3 + cones.numPowerCones * 3;
    if (cone_3d_size > 0) {
        rectify_cone_equilibration(
            equil.ework, equil.e,
            cone_offset,
            cone_3d_size,
            stream
        );
        cone_offset += cone_3d_size;
        any_changed = true;
    }

    // GenPowerCone: variable-dim rectification (like SOC)
    if (cones.numGenPowerCones > 0 && cones.totalGenPowerDim > 0) {
        rectify_genpow_cone_equilibration(
            equil.ework, equil.e,
            cone_offset,
            cones.d_genPowerOffsets,
            cones.d_genPowerDim1s,
            cones.d_genPowerDim2s,
            cones.numGenPowerCones,
            stream,
            cones.d_genPowerSzOffsets
        );
        any_changed = true;
    }

    // Only rescale if cones were rectified
    if (any_changed) {
        // Scale only A and b (d=nullptr means skip P and q)
        scale_data(P, matrixPRowOf, A, matrixARowOf, q, b, nullptr, &equil.ework, stream);

        // Update e: e *= ework
        hadamard(equil.e.data(), equil.ework.data(), equil.e.n(), equil.ework.batchSize(), stream);
    }

    // Direct-x equilibration rectification (x-space, mirror of CPU
    // problemdata.rs:309-342). For cones with
    // `requires_uniform_x_scaling()==true` we collapse `d[J]` to a single
    // geometric mean so `x[J] ∈ K_J` stays valid under `x̃ = D⁻¹·x`. Cones
    // requiring this: SOC, PSD, Exp, Power, GenPower (all are invariant
    // under uniform positive scaling but NOT per-coordinate). Nonneg is
    // invariant under per-entry scaling, so it's left at the Ruiz value.
    // Without this, x[J] leaves the original cone after unscaling.
    if (cones.numXCones > 0) {
        bool any_xcone_rectified = false;
        // Initialize dwork to 1 (identity rectification) for all n entries;
        // kernel only overwrites direct-x cone indices that need uniform.
        set_array(equil.dwork, 1.0, 0, n, equil.dwork.batchSize(), stream);

        rectify_xcone_soc_d_equilibration(
            equil.dwork, equil.d,
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets, cones.d_xcone_indices,
            cones.numXCones, n, stream
        );

        // Trigger rescale if any cone requiring uniform scaling is present.
        for (const auto& xc : cones.dir_cones) {
            if (xc.kind == XConeKind::SOC ||
                xc.kind == XConeKind::PSD ||
                xc.kind == XConeKind::Exp ||
                xc.kind == XConeKind::Power ||
                xc.kind == XConeKind::GenPower) {
                any_xcone_rectified = true;
                break;
            }
        }
        if (any_xcone_rectified) {
            // Apply the d-rectification to P, q, A only; b is a
            // constraint-space vector so it's not touched here (e=nullptr).
            scale_data(P, matrixPRowOf, A, matrixARowOf, q, b, &equil.dwork, nullptr, stream);
            // Update stored d: d *= dwork, so subsequent dinv reflects it.
            hadamard(equil.d.data(), equil.dwork.data(), equil.d.n(), equil.dwork.batchSize(), stream);
        }
    }

    compute_reciprocal(equil.d, equil.dinv, stream);
    compute_reciprocal(equil.e, equil.einv, stream);

    // Cache final mean P row norm for equilibrate_vectors_only
    // This allows subsequent solves with new q/b to compute cost scaling
    if (nnzP > 0) {
        compute_batch_row_norms_P(P, matrixPRowOf, equil.dwork, stream);
        compute_mean(equil.dwork, equil.mean_P_row_norm, stream);
    }

    // No sync needed here - caller is responsible for synchronization
    // if they need host-side access to equilibration results
}

void equilibrate_vectors_only(
    EquilibrationData& equil,
    CSR& P,
    int64_t* matrixPRowOf,
    BatchedVector& q,
    BatchedVector& b,
    const EquilibrationSettings& settings,
    cudaStream_t stream)
{
    int n = q.n();
    int m = b.n();
    int nnzP = P.nnz();
    int batchSize = q.batchSize();

    // Apply stored d scaling to new q: q = d * q_new
    hadamard(q.data(), equil.d.data(), n, batchSize, stream);

    // Apply stored e scaling to new b: b = e * b_new
    hadamard(b.data(), equil.e.data(), m, batchSize, stream);

    // Compute cost scaling adjustment for new q
    // This recomputes c based on cached P row norms and new q inf norm
    if (nnzP > 0) {
        BatchedVector dInfNormQ(1, batchSize);
        compute_inf_norm(q, dInfNormQ, stream);

        // conditional_scale_cost computes a relative c adjustment and applies it to P, q, c
        // It uses the cached mean_P_row_norm and newly computed inf_norm_q
        conditional_scale_cost(
            P, q, equil.c, equil.mean_P_row_norm, dInfNormQ,
            settings.scale_min, settings.scale_max, stream
        );

        // Update cached mean_P_row_norm since P was potentially rescaled
        compute_batch_row_norms_P(P, matrixPRowOf, equil.dwork, stream);
        compute_mean(equil.dwork, equil.mean_P_row_norm, stream);
    }
}

std::vector<int64_t> computeRowOfFromCSR(const int64_t* rowOffsets, int64_t n, int64_t nnz) {
    std::vector<int64_t> row_of(nnz);

    for (int64_t i = 0; i < n; ++i) {
        for (int64_t p = rowOffsets[i]; p < rowOffsets[i + 1]; ++p) {
            row_of[p] = i;
        }
    }

    return row_of;
}

int64_t* allocateAndCopyRowOfToGPU(const std::vector<int64_t>& h_row_of, cudaStream_t stream) {
    if (h_row_of.empty()) {
        return nullptr;
    }

    int64_t* d_row_of = nullptr;
    size_t bytes = h_row_of.size() * sizeof(int64_t);

    CUDA_THROW(cudaMalloc(&d_row_of, bytes));
    CUDA_THROW(cudaMemcpyAsync(d_row_of, h_row_of.data(), bytes,
                               cudaMemcpyHostToDevice, stream));

    return d_row_of;
}

} // namespace moreau
