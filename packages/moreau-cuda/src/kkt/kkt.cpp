/**
 * @file kkt.cpp
 * @brief KKT system update implementation
 */

#include "moreau/kkt/kkt.hpp"
#include "moreau/vector/vector_kernels.cuh"
#include "moreau/kkt/kkt_kernels.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace moreau {

bool KKTData::update(
    Cones& cones,
    const BatchedVector& s,
    const BatchedVector& z,
    const BatchedVector& mu,
    ScalingStrategy scaling,
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    const BatchedVector& q,
    const BatchedVector& b,
    BatchedVector& workx,
    BatchedVector& const_rhs,
    BatchedVector& const_sol,
    BatchedVector& x2,
    BatchedVector& z2,
    cudaStream_t stream
) {
    // Precondition: populate() must be called before update()
    // Use runtime check instead of assert to catch this in release builds
    if (!populated_) {
        throw std::logic_error("KKTData::update() called before populate()");
    }

    // Update cone scaling (this also populates soc_Hs internally)
    bool success = cones.update_scaling(s, z, mu, scaling, stream);
    if (!success) return false;

    // Update KKT H block with new cone scaling values
    update_H(cones, mu.data(), stream);

    // Regularize and refactorize the KKT system
    success = regularize_and_refactor(
        static_regularization_enable,
        static_regularization_constant,
        static_regularization_proportional,
        stream
    );

    if (!success) return false;

    // Fused: negate q and pack with b into const RHS  (2→1 kernel)
    negate_and_pack_const_rhs_kernel(
        q.data(), b.data(), const_rhs.data(), n, m, p, batchSize, stream);

    // Solve the KKT system
    solve(const_rhs.data(), const_sol.data(), stream);

    // Extract solution from interleaved format
    // De-interleave into x2/z2 with one kernel launch (expansion entries discarded)
    unpack_const_sol(
        const_sol.data(), x2.data(), z2.data(), n, m, p, batchSize, stream);

    // No sync needed - x2/z2 only used in subsequent GPU operations on same stream
    return true;
}

// Factor + solve constant RHS [-q; b] → (x2, z2)
bool KKTData::updateFactorOnly(
    Cones& cones,
    const BatchedVector& s,
    const BatchedVector& z,
    const BatchedVector& mu,
    ScalingStrategy scaling,
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    const BatchedVector& q,
    const BatchedVector& b,
    BatchedVector& workx,
    BatchedVector& const_rhs,
    BatchedVector& const_sol,
    BatchedVector& x2,
    BatchedVector& z2,
    cudaStream_t stream
) {
    if (!populated_) {
        throw std::logic_error("KKTData::updateFactorOnly() called before populate()");
    }

    update_H(cones, mu.data(), stream);

    bool success = regularize_and_refactor(
        static_regularization_enable,
        static_regularization_constant,
        static_regularization_proportional,
        stream
    );

    if (!success) return false;

    // Fused: negate q and pack with b into const RHS  (2→1 kernel)
    negate_and_pack_const_rhs_kernel(
        q.data(), b.data(), const_rhs.data(), n, m, p, batchSize, stream);

    solve(const_rhs.data(), const_sol.data(), stream);

    // Unpack solution into x2, z2
    unpack_const_sol(
        const_sol.data(), x2.data(), z2.data(), n, m, p, batchSize, stream);

    return true;
}

} // namespace moreau
