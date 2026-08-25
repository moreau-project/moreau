#pragma once

/**
 * @file chordal_info.hpp
 * @brief Host-side chordal decomposition for PSD cones
 *
 * Preprocesses sparse PSD cone problems by decomposing large sparse PSD cones
 * into smaller dense cliques. The solver operates on the decomposed problem
 * transparently. This is a C++ port of the Rust chordal decomposition in
 * packages/moreau-cpu/src/solver/chordal/.
 *
 * Flow:
 * 1. analyze() — detect sparsity, build clique tree, merge, compute augmented structure
 * 2. augment_*() — fill per-problem numerical values into precomputed structure
 * 3. (solver runs on decomposed problem)
 * 4. reverse_*() + complete_z() — map solution back to original dimensions
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include <cmath>

#include "moreau/chordal/supernode_tree.hpp"

namespace moreau {

// Forward declare (defined in symbolic_ldl.hpp)
struct SymbolicLDLResult;

// ============================================================================
// SparsityPattern — per-cone decomposition metadata
// ============================================================================

struct SparsityPattern {
    SuperNodeTree sntree;
    std::vector<int64_t> ordering;  // vertex permutation: orig_idx = ordering[permuted_idx]
    int64_t orig_index;             // which cone in original problem this came from
};

// ============================================================================
// ConeMapEntry — maps each decomposed cone to its origin
// ============================================================================

struct ConeMapEntry {
    int64_t orig_index;
    int64_t pattern_index;  // -1 if not decomposed
    int64_t clique_index;   // -1 if not decomposed

    bool is_decomposed() const { return pattern_index >= 0; }
};

// ============================================================================
// ChordalInfo — main coordinator
// ============================================================================

struct ChordalInfo {
    // Original problem dimensions
    int64_t n_orig = 0;
    int64_t m_orig = 0;

    // Original cone structure
    int64_t num_zero = 0;
    int64_t num_nonneg = 0;
    int64_t total_soc = 0;
    int64_t num_exp = 0;
    int64_t num_power = 0;
    std::vector<int64_t> orig_psd_dims;

    // Decomposition data
    std::vector<SparsityPattern> spatterns;
    std::vector<ConeMapEntry> cone_maps;

    // Augmented problem structure (CSC on host)
    std::vector<int64_t> P_aug_colptr, P_aug_rowind;
    std::vector<int64_t> A_aug_colptr, A_aug_rowind;
    std::vector<double> A_aug_values;  // structural values (+1/-1 for overlaps)
    // Mapping: augmented_values[orig_to_aug[k]] corresponds to original entry k
    // (in user CSR order after composition). Length = {P,A}_nnz_orig.
    std::vector<int64_t> P_orig_to_aug;
    std::vector<int64_t> A_orig_to_aug;
    int64_t n_aug = 0, m_aug = 0;
    int64_t P_nnz_orig = 0;
    int64_t A_nnz_orig = 0;

    // Row permutation: b_aug[aug_row] came from b_orig[b_row_map[aug_row]]
    // INT64_MAX for overlap rows (no original row)
    std::vector<int64_t> b_row_map;

    // Decomposed PSD dims (after chordal, these replace orig_psd_dims)
    std::vector<int64_t> decomposed_psd_dims;

    // ========================================================================
    // Construction
    // ========================================================================

    /**
     * @brief Analyze sparsity and build chordal decomposition
     *
     * @param A_colptr CSC column pointers of constraint matrix A
     * @param A_rowind CSC row indices of A
     * @param A_ncol Number of columns in A (= n)
     * @param A_nrow Number of rows in A (= m)
     * @param psd_dims Matrix dimensions of PSD cones
     * @param num_psd Number of PSD cones
     * @param num_zero_ Number of zero cones
     * @param num_nonneg_ Number of nonneg cones
     * @param total_soc_ Total SOC dimension
     * @param num_exp_ Number of exp cones
     * @param num_power_ Number of power cones
     */
    static ChordalInfo analyze(
        const int64_t* A_colptr, const int64_t* A_rowind,
        int64_t A_ncol, int64_t A_nrow,
        const int64_t* psd_dims, int64_t num_psd,
        int64_t num_zero_, int64_t num_nonneg_,
        int64_t total_soc_, int64_t num_exp_, int64_t num_power_,
        const bool* b_sparsity_pattern = nullptr
    );

    bool is_decomposed() const { return !spatterns.empty(); }

    // ========================================================================
    // Value-phase augmentation (per-problem)
    // ========================================================================

    /// Fill P values: original values into first P_nnz_orig slots
    void augment_P_values(const double* P_orig, double* P_aug, int64_t P_nnz_aug) const;

    /// Fill A values: original values into first A_nnz_orig slots, overlaps pre-baked
    void augment_A_values(const double* A_orig, double* A_aug, int64_t A_nnz_aug) const;

    /// Zero-pad q: q_aug[0:n_orig] = q_orig, rest = 0
    void augment_q(const double* q_orig, double* q_aug) const;

    /// Permute b: b_aug[aug_row] = b_orig[b_row_map[aug_row]] (0 for overlap rows)
    void augment_b(const double* b_orig, double* b_aug) const;

    // ========================================================================
    // Solution reverse mapping
    // ========================================================================

    /// reverse_s: ACCUMULATE clique values into original (+=)
    void reverse_s(const double* s_aug, double* s_orig) const;

    /// reverse_z: OVERWRITE original with last clique's values (=)
    void reverse_z(const double* z_aug, double* z_orig) const;

    // ========================================================================
    // PSD completion (fill structural zeros in dual z)
    // ========================================================================

    /// Complete z to be truly PSD at structural zeros
    /// Operates on z in original dimensions, after reverse mapping
    void complete_z(double* z, int64_t psd_offset) const;

    // ========================================================================
    // Backward pass adjoints
    // ========================================================================

    void adjoint_reverse_s(const double* ds_orig, double* ds_aug) const;
    void adjoint_reverse_z(const double* dz_orig, double* dz_aug) const;
    void adjoint_complete_z(double* dz, int64_t psd_offset) const;
    void adjoint_augment_dP(const double* dP_aug, double* dP_orig) const;
    void adjoint_augment_dA(const double* dA_aug, double* dA_orig) const;
    void adjoint_augment_dq(const double* dq_aug, double* dq_orig) const;
    void adjoint_augment_db(const double* db_aug, double* db_orig) const;

private:
    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Find aggregate sparsity mask from A and optional b sparsity pattern
    static std::vector<bool> find_aggregate_sparsity_mask(
        const int64_t* A_colptr, const int64_t* A_rowind,
        int64_t A_ncol, int64_t A_nrow,
        const bool* b_sparsity_pattern = nullptr);

    /// Analyze one PSD cone's sparsity pattern
    void analyse_psd_sparsity(
        std::vector<bool>& nz_mask, int64_t conedim, int64_t coneidx);

    /// Build compact augmented problem structure
    void build_compact_augmentation(
        const int64_t* P_colptr, const int64_t* P_rowind, int64_t P_ncol,
        const int64_t* A_colptr, const int64_t* A_rowind, int64_t A_ncol, int64_t A_nrow);

    /// Helper: svec index utilities
    static int64_t triangular_number(int64_t n) { return n * (n + 1) / 2; }
    static int64_t triangular_index(int64_t k) { return k * (k + 3) / 2; }
    static std::pair<int64_t, int64_t> upper_triangular_index_to_coord(int64_t idx) {
        int64_t col = 0;
        while ((col + 1) * (col + 2) / 2 <= idx) col++;
        int64_t row = idx - col * (col + 1) / 2;
        return {row, col};
    }
    static int64_t coord_to_upper_triangular_index(int64_t row, int64_t col) {
        return col * (col + 1) / 2 + row;
    }
};

// ============================================================================
// Inline implementations of simple methods
// ============================================================================

inline void ChordalInfo::augment_P_values(const double* P_orig, double* P_aug, int64_t P_nnz_aug) const {
    // Zero everything first, then scatter original values to mapped positions
    std::memset(P_aug, 0, sizeof(double) * P_nnz_aug);
    if (P_nnz_orig > 0 && P_orig) {
        for (int64_t k = 0; k < P_nnz_orig; ++k) {
            P_aug[P_orig_to_aug[k]] = P_orig[k];
        }
    }
}

inline void ChordalInfo::augment_A_values(const double* A_orig, double* A_aug, int64_t A_nnz_aug) const {
    // Copy structural template (includes +1/-1 for overlaps)
    std::memcpy(A_aug, A_aug_values.data(), sizeof(double) * A_nnz_aug);
    // Place original values at their mapped positions in the augmented CSC
    if (A_nnz_orig > 0 && A_orig) {
        for (int64_t k = 0; k < A_nnz_orig; ++k) {
            A_aug[A_orig_to_aug[k]] = A_orig[k];
        }
    }
}

inline void ChordalInfo::augment_q(const double* q_orig, double* q_aug) const {
    std::memset(q_aug, 0, sizeof(double) * n_aug);
    if (n_orig > 0 && q_orig) {
        std::memcpy(q_aug, q_orig, sizeof(double) * n_orig);
    }
}

inline void ChordalInfo::augment_b(const double* b_orig, double* b_aug) const {
    std::memset(b_aug, 0, sizeof(double) * m_aug);
    if (b_orig) {
        for (int64_t i = 0; i < m_aug; i++) {
            if (b_row_map[i] != INT64_MAX) {
                b_aug[i] = b_orig[b_row_map[i]];
            }
        }
    }
}

inline void ChordalInfo::adjoint_augment_dP(const double* dP_aug, double* dP_orig) const {
    // Gather: adjoint of P value scatter
    if (P_nnz_orig > 0) {
        for (int64_t k = 0; k < P_nnz_orig; ++k) {
            dP_orig[k] = dP_aug[P_orig_to_aug[k]];
        }
    }
}

inline void ChordalInfo::adjoint_augment_dq(const double* dq_aug, double* dq_orig) const {
    // Slice: only first n_orig entries
    std::memcpy(dq_orig, dq_aug, sizeof(double) * n_orig);
}

inline void ChordalInfo::adjoint_augment_db(const double* db_aug, double* db_orig) const {
    // Gather: adjoint of b permutation
    std::memset(db_orig, 0, sizeof(double) * m_orig);
    for (int64_t i = 0; i < m_aug; i++) {
        if (b_row_map[i] != INT64_MAX) {
            db_orig[b_row_map[i]] += db_aug[i];
        }
    }
}

} // namespace moreau
