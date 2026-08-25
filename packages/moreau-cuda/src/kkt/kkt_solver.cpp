/**
 * @file kkt_solver.cpp
 * @brief Factory function for KKT solver backend selection
 */

#include "moreau/kkt/kkt_solver.hpp"
#include "moreau/kkt/kkt.hpp"
#include "moreau/kkt/riccati.hpp"
#include "moreau/kkt/kkt_woodbury.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace moreau {

std::unique_ptr<KKTSolver> make_kkt_solver(
    int64_t n, int64_t m, int64_t batchSize,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones, const Settings& settings,
    cudaStream_t stream)
{
    auto requested = settings.ipm.kktSolverType;

    // --- Try Riccati ---
    bool try_riccati = (requested == KKTSolverType::Auto ||
                       requested == KKTSolverType::Riccati);

    if (try_riccati) {
        // Allow a bandwidth-reducing (RCM) reorder to recover block-tridiagonal
        // structure from a scrambled variable order. The backward mirrors the
        // same reorder (DiffKKT::initialize_riccati passes the matching perm to
        // DiffRiccatiData), so this is safe with gradients on as well.
        std::vector<int32_t> perm;
        auto blocks = detect_block_tridiagonal(
            n, m, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA, cones,
            /*allow_permute=*/true, &perm);

        if (!blocks.empty()) {
            int32_t max_blk = *std::max_element(blocks.begin(), blocks.end());
            bool fits_smem = (max_blk <= riccati_smem_max_block());

            if (fits_smem) {
                return std::make_unique<RiccatiKKTData>(
                    n, m, batchSize, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA,
                    cones, blocks, perm, stream);
            }

            if (requested == KKTSolverType::Riccati) {
                throw std::runtime_error(
                    "Riccati solver requested but max block size (" +
                    std::to_string(max_blk) + ") exceeds device shared memory limit (" +
                    std::to_string(riccati_smem_max_block()) +
                    "). Use device='auto' or kktSolverType='cudss' for this problem.");
            }
        } else if (requested == KKTSolverType::Riccati) {
            throw std::runtime_error(
                "Riccati solver requested but problem is not block-tridiagonal");
        }
    }

    // --- Try Woodbury ---
    // Woodbury is opt-in only (direct_solve_method='woodbury'). It can be
    // slower than cuDSS for many problem sizes. When direct-x cones are
    // present with enable_grad, the *forward* Woodbury solve is still used;
    // the backward routes through the general DiffKKT/cuDSS adjoint instead
    // (DiffWoodbury has no x-cone path) — see the `use_woodbury_backward`
    // gate in diff.cu. No rejection needed: the combination composes.
    bool try_woodbury = (requested == KKTSolverType::Woodbury);

    if (try_woodbury) {
        if (WoodburyKKTData::isCompatible(n, m, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA, cones)) {
            return std::make_unique<WoodburyKKTData>(
                n, m, batchSize, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA,
                cones, stream);
        }

        if (requested == KKTSolverType::Woodbury) {
            throw std::runtime_error(
                "Woodbury solver requested but problem structure is not compatible "
                "(requires diagonal P, zero + nonneg cones only, k_total < n)");
        }
    }

    // --- Fallback to cuDSS ---
    const double dyn_reg_eps = settings.ipm.dynamicRegularizationEnable
                                   ? settings.ipm.dynamicRegularizationEps
                                   : 0.0;
    return std::make_unique<KKTData>(
        n, m, batchSize, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA, cones,
        settings.ipm.cudssIrSteps, settings.ipm.cudssPivotEnable, stream,
        /*maxLuNnz=*/-1, dyn_reg_eps);
}

} // namespace moreau
