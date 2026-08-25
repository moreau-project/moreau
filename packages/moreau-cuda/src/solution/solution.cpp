/**
 * @file solution.cpp
 * @brief Implementation of solution post-processing
 */

#include "moreau/solution/solution.hpp"
#include "moreau/solver/data.hpp"
#include "moreau/solver/solver_kernels.cuh"
#include "moreau/variables/variables.hpp"
#include "moreau/solver/info.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/vector/vector_kernels.cuh"
#include <cuda_runtime.h>

namespace moreau {

void Solution::post_process(
    int64_t n,
    int64_t m,
    int64_t batchSize,
    const EquilibrationData& equilibration,
    Variables& variables,
    const Info& info,
    bool equilibration_enabled,
    cudaStream_t stream
) {
    // Copy status from info
    cudaMemcpyAsync(status.get(), info.status_device,
                    sizeof(int32_t) * batchSize, cudaMemcpyDeviceToDevice, stream);

    // Update iteration count
    this->iterations = info.iterations;

    // Use saved cost values (captured at convergence) to avoid corruption
    // from update_info_kernel continuing to overwrite for later-converging batches
    cudaMemcpyAsync(obj_val.data(), cost_primal_raw.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(obj_val_dual.data(), cost_dual_raw.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(r_prim.data(), info.res_primal.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(r_dual.data(), info.res_dual.data(),
                    sizeof(double) * batchSize, cudaMemcpyDeviceToDevice, stream);

    // Set objective values to NaN for infeasible batches (matching CPU solver)
    set_infeasible_obj_nan(obj_val, obj_val_dual, info.status_device, batchSize, stream);

    // Normalize by τ (optimal) or κ (infeasible) per batch, matching CPU solver.
    // For infeasible problems, dividing by κ produces a proper infeasibility certificate.
    // Use saved raw variables to avoid NaN values if solver continued past convergence.
    BatchedVector scale(1, batchSize);
    build_normalization_scale(scale, τ_raw, κ_raw, info.status_device, batchSize, stream);

    BatchedVector x_normalized(n, batchSize);
    BatchedVector s_normalized(m, batchSize);
    BatchedVector z_normalized(m, batchSize);

    div_per_batch(x_normalized, x_raw, scale, stream);
    div_per_batch(s_normalized, s_raw, scale, stream);
    div_per_batch(z_normalized, z_raw, scale, stream);

    // Then unscale the variables to get a solution to the original (user-provided) problem
    // This reverses the equilibration scaling applied during problem setup

    if (equilibration_enabled) {
        // Unscale x: x = (x/τ) .* d = (x/τ) / dinv
        elementwise_div(this->x, x_normalized, equilibration.dinv, stream);

        // Unscale z: z = (z/τ) .* e / c = (z/τ) / einv / c
        // First: z_temp = z_normalized / einv = z_normalized * e
        BatchedVector z_temp(m, batchSize);
        elementwise_div(z_temp, z_normalized, equilibration.einv, stream);

        // Then: z = z_temp / c (c is a scalar per batch [1, batchSize])
        div_per_batch(this->z, z_temp, equilibration.c, stream);

        // Unscale s: s = (s/τ) .* einv = (s/τ) / e
        elementwise_div(this->s, s_normalized, equilibration.e, stream);
    } else {
        // No equilibration: just use normalized variables
        cudaMemcpyAsync(this->x.data(), x_normalized.data(),
                        sizeof(double) * n * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(this->z.data(), z_normalized.data(),
                        sizeof(double) * m * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(this->s.data(), s_normalized.data(),
                        sizeof(double) * m * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

void Solution::finalize(const Info& info) {
    solve_time = info.solve_time;
}

} // namespace moreau
