/**
 * @file kkt_solver.hpp
 * @brief Abstract KKT solver interface and factory function
 *
 * KKTSolver is the abstract base class for three KKT solver backends:
 * - KKTData (CuDSS): General-purpose sparse LDL
 * - RiccatiKKTData: Block-tridiagonal Cholesky for MPC/MHE problems
 * - WoodburyKKTData: Diagonal P + low-rank/sparse A (portfolio, factor models)
 *
 * Auto-detection priority: Riccati > Woodbury > CuDSS.
 */

#pragma once

#include <memory>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "moreau/settings/settings.hpp"

namespace moreau {

// Forward declarations
class CSR;
struct Cones;
class BatchedVector;
enum class ScalingStrategy;

/**
 * @brief Abstract KKT solver base class
 *
 * Concrete backends (KKTData, RiccatiKKTData, WoodburyKKTData) inherit from
 * this and implement the pure virtual methods. CompiledSolver stores a
 * unique_ptr<KKTSolver> created by make_kkt_solver().
 */
class KKTSolver {
public:
    virtual ~KKTSolver() = default;
    KKTSolver() = default;
    KKTSolver(const KKTSolver&) = delete;
    KKTSolver& operator=(const KKTSolver&) = delete;

    // --- Pure virtual: common interface ---

    virtual void populate(CSR& P, CSR& A, cudaStream_t stream = 0) = 0;
    virtual void update_H(const Cones& cones, const double* mu_data = nullptr, cudaStream_t stream = 0) = 0;

    virtual bool update(
        Cones& cones,
        const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q, const BatchedVector& b,
        BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
        BatchedVector& x2, BatchedVector& z2,
        cudaStream_t stream = 0) = 0;

    virtual bool update(
        Cones& cones,
        const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0) = 0;

    virtual bool updateFactorOnly(
        Cones& cones,
        const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q, const BatchedVector& b,
        BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
        BatchedVector& x2, BatchedVector& z2,
        cudaStream_t stream = 0) = 0;

    virtual void solve(const double* rhs, double* sol, cudaStream_t stream = 0) = 0;
    virtual void solve2(const double* rhs, double* sol, cudaStream_t stream = 0) = 0;

    virtual bool regularize_and_refactor(
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0) = 0;

    [[nodiscard]] virtual size_t memoryUsage() const noexcept = 0;
    [[nodiscard]] virtual KKTSolverType solverType() const noexcept = 0;

    // --- Virtual with defaults ---

    virtual void solve_combined(
        const double* affine_rhs, double* affine_sol,
        double* const_x, double* const_z,
        cudaStream_t stream = 0)
    {
        // Default: just solve affine (constant already solved in updateFactorOnly)
        solve(affine_rhs, affine_sol, stream);
    }

    virtual void setCublasHandle(cublasHandle_t) {}

    // --- Direct-x cone hooks ---
    // Default no-ops so backends that don't support direct-x cones (Riccati,
    // Woodbury) simply reject them up-front via Cones::initialize() and
    // never get called with non-empty x_cones. KKTData overrides these.
    virtual void init_xcone_px_baseline(const Cones& /*cones*/,
                                        cudaStream_t /*stream*/ = 0) {}
    virtual void refresh_xcone_hs(const Cones& /*cones*/,
                                  cudaStream_t /*stream*/ = 0) {}

    /// Raw device pointers needed by the fused scale+scatter pass in
    /// `Variables::scale_cones`. Zero-initialised default tuple means the
    /// caller must fall back to the classic `refresh_xcone_hs` path. KKTData
    /// overrides this with real pointers into its (1,1)-block layout.
    struct XConeScatterTargets {
        double* kkt_values = nullptr;
        const double* xcone_px_baseline = nullptr;
        const int64_t* H_xcone_hs_idx = nullptr;
        const int64_t* H_xcone_u_idx = nullptr;
        const int64_t* H_xcone_v_idx = nullptr;
        const int64_t* H_xcone_exp_diag_idx = nullptr;
        // Direct-x sparse GenPow expansion indices (rank-9 PD).
        // H_xcone_exp_diag_idx holds SOC entries first, then GenPow entries.
        // These pointers index into the GenPow-specific portions.
        const int64_t* H_xcone_genpow_q_idx = nullptr;
        const int64_t* H_xcone_genpow_r_idx = nullptr;
        const int64_t* H_xcone_genpow_p_idx = nullptr;
        // 6 PD-axis off-diag column index arrays (one slot per cone-block row).
        const int64_t* H_xcone_genpow_pd_axis_idx[6] = {nullptr, nullptr, nullptr,
                                                        nullptr, nullptr, nullptr};
        // Offset into H_xcone_exp_diag_idx where GenPow entries start
        int64_t xcone_genpow_exp_diag_offset = 0;
        int64_t nnzKKT = 0;
    };
    virtual XConeScatterTargets xcone_scatter_targets() { return {}; }

    // --- Non-virtual no-ops ---
    void setupL2Persistence(cudaStream_t = 0) {}
    void resetL2Persistence(cudaStream_t = 0) {}

    // --- Convenience (non-virtual) ---
    [[nodiscard]] KKTSolverType actualSolverType() const noexcept { return solverType(); }
    [[nodiscard]] bool isRiccati() const noexcept { return solverType() == KKTSolverType::Riccati; }
    [[nodiscard]] bool isWoodbury() const noexcept { return solverType() == KKTSolverType::Woodbury; }
};

/**
 * @brief Factory: auto-detects structure and returns the right KKT solver backend
 *
 * Detection priority: Riccati > Woodbury > CuDSS.
 */
std::unique_ptr<KKTSolver> make_kkt_solver(
    int64_t n, int64_t m, int64_t batchSize,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones, const Settings& settings,
    cudaStream_t stream = 0);

} // namespace moreau
