/**
 * @file riccati.cpp
 * @brief Host-side implementation of batched block-tridiagonal (Riccati) KKT solver
 */

#include "moreau/kkt/riccati.hpp"
#include "moreau/cuda/status_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <unordered_map>

namespace moreau {

// ============================================================================
// Structure detection
// ============================================================================

// Helper: build greedy block boundaries starting at offset d1, verify tridiagonal
static std::vector<int32_t> try_block_decomposition(
    int64_t n, int64_t d1,
    const std::vector<int64_t>& max_row,
    const std::vector<std::vector<int64_t>>& m_cols)
{
    std::vector<int64_t> bnd = {0, d1};
    int64_t pos = d1;

    while (pos < n) {
        int64_t prev_start = bnd[bnd.size() - 2];
        int64_t prev_reach = 0;
        for (int64_t c = prev_start; c < pos; ++c)
            prev_reach = std::max(prev_reach, max_row[c]);
        int64_t min_end = prev_reach + 1;
        if (min_end > n) return {};
        int64_t next = std::max(min_end, pos + 1);
        bnd.push_back(next);
        pos = next;
    }

    int64_t nb = (int64_t)bnd.size() - 1;
    if (nb < 3) return {};

    // Verify block-tridiagonal property
    std::vector<int64_t> col_block_map(n);
    for (int64_t bi = 0; bi < nb; ++bi)
        for (int64_t c = bnd[bi]; c < bnd[bi + 1]; ++c)
            col_block_map[c] = bi;

    for (int64_t j = 0; j < n; ++j) {
        int64_t bj = col_block_map[j];
        for (int64_t i : m_cols[j]) {
            int64_t bi = col_block_map[i];
            int64_t diff = (bi > bj) ? (bi - bj) : (bj - bi);
            if (diff > 1) return {};
        }
    }

    std::vector<int32_t> block_sizes(nb);
    for (int64_t i = 0; i < nb; ++i)
        block_sizes[i] = (int32_t)(bnd[i + 1] - bnd[i]);
    return block_sizes;
}

// Greedy search over first-block sizes; returns the block-tridiagonal partition
// with the smallest max block, or empty if none exists for this ordering.
static std::vector<int32_t> greedy_block_search(
    int64_t n,
    const std::vector<int64_t>& max_row,
    const std::vector<std::vector<int64_t>>& m_cols)
{
    std::vector<int32_t> best;
    int32_t best_max_block = INT32_MAX;
    for (int64_t d1 = 1; d1 < n; ++d1) {
        auto blocks = try_block_decomposition(n, d1, max_row, m_cols);
        if (blocks.empty()) continue;

        int32_t mb = *std::max_element(blocks.begin(), blocks.end());
        if (mb < best_max_block) {
            best_max_block = mb;
            best = std::move(blocks);
        }
        // Uniform interior (max_block == d1) is likely optimal — stop searching.
        if (best_max_block <= d1) break;
    }
    return best;
}

// Reverse Cuthill–McKee ordering of the adjacency `m_cols`. Returns `new_to_old`
// (new_to_old[bp] = original column at band position bp). Self-loops ignored.
// A bandwidth-reducing reorder: for a scrambled time-stepped problem (MPC/LQR)
// it recovers a banded — hence block-tridiagonal — column order.
static std::vector<int32_t> rcm_order(
    int64_t n, const std::vector<std::vector<int64_t>>& m_cols)
{
    std::vector<int32_t> deg(n);
    for (int64_t i = 0; i < n; ++i) {
        int32_t d = 0;
        for (int64_t j : m_cols[i]) if (j != i) ++d;
        deg[i] = d;
    }

    std::vector<char> visited(n, 0);
    std::vector<int32_t> order;
    order.reserve(n);

    auto bfs_from = [&](int32_t start) {
        std::vector<int32_t> q;
        size_t head = 0;
        visited[start] = 1;
        q.push_back(start);
        while (head < q.size()) {
            int32_t node = q[head++];
            order.push_back(node);
            std::vector<int32_t> nbrs;
            for (int64_t j : m_cols[node]) {
                if (j != node && !visited[j]) {
                    visited[j] = 1;
                    nbrs.push_back((int32_t)j);
                }
            }
            // Cuthill–McKee: enqueue neighbours in ascending degree order.
            std::sort(nbrs.begin(), nbrs.end(),
                      [&](int32_t a, int32_t b) { return deg[a] < deg[b]; });
            for (int32_t nb : nbrs) q.push_back(nb);
        }
    };

    // One sweep per connected component, starting from a min-degree node
    // (an endpoint of a chain — the right seed for a banded recovery).
    for (;;) {
        int32_t start = -1;
        int32_t start_deg = INT32_MAX;
        for (int64_t i = 0; i < n; ++i) {
            if (!visited[i] && deg[i] < start_deg) { start_deg = deg[i]; start = (int32_t)i; }
        }
        if (start < 0) break;
        bfs_from(start);
    }

    std::reverse(order.begin(), order.end());  // CM -> RCM
    return order;
}

std::vector<int32_t> detect_block_tridiagonal(
    int64_t n, int64_t m,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones,
    bool allow_permute,
    std::vector<int32_t>* out_perm)
{
    if (out_perm) out_perm->clear();  // empty == identity column order

    // Riccati supports only zero + nonneg cones.
    if (cones.numSocCones > 0 || cones.numExpCones > 0 ||
        cones.numPowerCones > 0 || cones.numGenPowerCones > 0 ||
        cones.numPsdCones > 0 || cones.numXCones > 0)
        return {};
    if (n < 6) return {};
    // Riccati targets MPC/LQR with dynamics (zero cones). Without zero cones,
    // the block structure is trivial and cuDSS is more robust.
    if (cones.numZeroCones == 0) return {};

    // The A'A clique below pushes nnz_r^2 entries per row of A, so a single
    // dense row (e.g. a sum-to-one or factor-model row in a portfolio QP) would
    // explode host memory. Bail to cuDSS before building the adjacency. Uses the
    // device-independent block bound so detection stays a pure host computation.
    const int64_t max_bandwidth = 2 * (int64_t)kRiccatiMaxSmemBlock;
    for (int64_t r = 0; r < m; ++r) {
        const int64_t lo = A_ro[r], hi = A_ro[r + 1];
        if (hi - lo < 2) continue;
        if (allow_permute) {
            // A reorder can move scattered columns adjacent, so column span no
            // longer precludes banding. Guard only against dense rows that blow
            // up the clique (and cannot be banded under any order regardless).
            if (hi - lo >= max_bandwidth) return {};
        } else {
            // Fixed order: a wide column span already precludes a contiguous
            // block-tridiagonal decomposition, so bail early.
            int64_t cmin = A_ci[lo], cmax = A_ci[lo];
            for (int64_t p = lo + 1; p < hi; ++p) {
                const int64_t c = A_ci[p];
                if (c < cmin) cmin = c;
                else if (c > cmax) cmax = c;
            }
            if (cmax - cmin >= max_bandwidth) return {};
        }
    }

    // Build symmetric adjacency: M = P + A'A
    std::vector<std::vector<int64_t>> m_cols(n);

    for (int64_t i = 0; i < n; ++i) {
        for (int64_t p = P_ro[i]; p < P_ro[i + 1]; ++p) {
            int64_t j = P_ci[p];
            m_cols[i].push_back(j);
            if (i != j) m_cols[j].push_back(i);
        }
    }

    for (int64_t r = 0; r < m; ++r) {
        for (int64_t p1 = A_ro[r]; p1 < A_ro[r + 1]; ++p1) {
            int64_t ci = A_ci[p1];
            for (int64_t p2 = A_ro[r]; p2 < A_ro[r + 1]; ++p2) {
                int64_t cj = A_ci[p2];
                if (ci != cj) m_cols[ci].push_back(cj);
            }
        }
    }

    auto sort_unique = [](std::vector<std::vector<int64_t>>& adj) {
        for (auto& col_rows : adj) {
            std::sort(col_rows.begin(), col_rows.end());
            col_rows.erase(std::unique(col_rows.begin(), col_rows.end()), col_rows.end());
        }
    };
    sort_unique(m_cols);

    auto max_rows_of = [](int64_t nn, const std::vector<std::vector<int64_t>>& adj) {
        std::vector<int64_t> mr(nn);
        for (int64_t j = 0; j < nn; ++j) mr[j] = adj[j].empty() ? j : adj[j].back();
        return mr;
    };

    auto max_block_of = [](const std::vector<int32_t>& b) -> int32_t {
        return b.empty() ? INT32_MAX : *std::max_element(b.begin(), b.end());
    };

    // First try the given column order (identity permutation).
    std::vector<int64_t> max_row = max_rows_of(n, m_cols);
    std::vector<int32_t> best = greedy_block_search(n, max_row, m_cols);
    int32_t natural_max = max_block_of(best);

    // Fast path: the given order already yields a decomposition whose blocks fit
    // the (device-independent) shared-memory bound — return it at zero extra
    // cost. Larger/empty results fall through; the factory's fits_smem gate has
    // the final say on the returned block sizes.
    if (natural_max <= kRiccatiMaxSmemBlock) return best;
    if (!allow_permute) return best;

    // The given order has no small-block decomposition (a scrambled order forces
    // one giant block). Recover a band with a bandwidth-reducing (RCM) reorder
    // and re-run the contiguous-block search in the permuted column space.
    std::vector<int32_t> new_to_old = rcm_order(n, m_cols);
    std::vector<int32_t> old_to_new(n);
    for (int64_t bp = 0; bp < n; ++bp) old_to_new[new_to_old[bp]] = (int32_t)bp;

    std::vector<std::vector<int64_t>> m_cols_p(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t ip = old_to_new[i];
        for (int64_t j : m_cols[i]) m_cols_p[ip].push_back(old_to_new[j]);
    }
    sort_unique(m_cols_p);

    std::vector<int64_t> max_row_p = max_rows_of(n, m_cols_p);
    std::vector<int32_t> best_p = greedy_block_search(n, max_row_p, m_cols_p);

    // Use the reorder only if it produced a strictly tighter band.
    if (max_block_of(best_p) < natural_max) {
        if (out_perm) *out_perm = std::move(new_to_old);
        return best_p;
    }
    return best;  // reorder did not help; factory gates on the given-order result
}

// ============================================================================
// Constructor
// ============================================================================

RiccatiKKTData::RiccatiKKTData(
    int64_t n_, int64_t m_, int64_t batchSize_,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP_,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA_,
    const Cones& cones,
    const std::vector<int32_t>& block_sizes,
    const std::vector<int32_t>& perm,
    cudaStream_t stream)
    : n(n_), m(m_), batchSize(batchSize_),
      nblocks(block_sizes.size()),
      h_block_sizes(block_sizes),
      nnzP(nnzP_), nnzA(nnzA_)
{
    // Resolve the column permutation. `perm` (new_to_old: band position -> original
    // column) is empty for the identity order. old_to_new maps an original column
    // to its band position and is applied wherever a column index feeds the block
    // structure / Schur assembly below; the x-vector boundaries map back through
    // new_to_old on the device.
    has_perm_ = !perm.empty();
    std::vector<int32_t> old_to_new(n);
    if (has_perm_) {
        assert((int64_t)perm.size() == n);
        h_new_to_old = perm;
        for (int64_t bp = 0; bp < n; ++bp) old_to_new[perm[bp]] = (int32_t)bp;
    } else {
        for (int64_t i = 0; i < n; ++i) old_to_new[i] = (int32_t)i;
    }

    // Block offsets
    h_block_offsets.resize(nblocks + 1);
    h_block_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks; ++i)
        h_block_offsets[i + 1] = h_block_offsets[i] + h_block_sizes[i];
    assert(h_block_offsets[nblocks] == (int32_t)n);

    max_block = *std::max_element(h_block_sizes.begin(), h_block_sizes.end());

    // Col-to-block map
    h_col_to_block.resize(n);
    for (int64_t bi = 0; bi < nblocks; ++bi)
        for (int32_t c = h_block_offsets[bi]; c < h_block_offsets[bi + 1]; ++c)
            h_col_to_block[c] = (int32_t)bi;

    // D offsets
    h_D_offsets.resize(nblocks + 1);
    h_D_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks; ++i)
        h_D_offsets[i + 1] = h_D_offsets[i] + (int64_t)h_block_sizes[i] * h_block_sizes[i];
    D_total_elems = h_D_offsets[nblocks];

    // L offsets: L[i] has shape (block_sizes[i+1], block_sizes[i]) for i=0..nblocks-2
    h_L_offsets.resize(nblocks);  // nblocks entries; only 0..nblocks-2 used
    h_L_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks - 1; ++i) {
        h_L_offsets[i + 1] = h_L_offsets[i] +
                (int64_t)h_block_sizes[i + 1] * h_block_sizes[i];
    }
    L_total_elems = (nblocks > 1) ? h_L_offsets[nblocks - 1] : 0;

    // Upload block structure
    auto upload_i32 = [&](device_unique_ptr<int32_t>& ptr, const int32_t* data, size_t count) {
        int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * count));
        CUDA_THROW(cudaMemcpy(tmp, data, sizeof(int32_t) * count, cudaMemcpyHostToDevice));
        ptr.reset(tmp);
    };
    auto upload_i64 = [&](device_unique_ptr<int64_t>& ptr, const int64_t* data, size_t count) {
        int64_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int64_t) * count));
        CUDA_THROW(cudaMemcpy(tmp, data, sizeof(int64_t) * count, cudaMemcpyHostToDevice));
        ptr.reset(tmp);
    };

    upload_i32(d_block_sizes, h_block_sizes.data(), nblocks);
    upload_i32(d_block_offsets, h_block_offsets.data(), nblocks + 1);
    upload_i32(d_col_to_block, h_col_to_block.data(), n);
    upload_i64(d_D_offsets, h_D_offsets.data(), nblocks + 1);
    upload_i64(d_L_offsets, h_L_offsets.data(), nblocks);
    if (has_perm_) upload_i32(d_new_to_old, h_new_to_old.data(), n);

    // Dense block data
    auto alloc_double = [](device_unique_ptr<double>& ptr, size_t count) {
        double* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * count));
        CUDA_THROW(cudaMemset(tmp, 0, sizeof(double) * count));
        ptr.reset(tmp);
    };

    alloc_double(d_D, batchSize * D_total_elems);
    if (L_total_elems > 0) alloc_double(d_L, batchSize * L_total_elems);
    alloc_double(d_S, batchSize * D_total_elems);
    alloc_double(d_work, batchSize * (int64_t)max_block * max_block);
    alloc_double(d_h_inv, batchSize * m);

    // Build CSC structure from CSR input
    {
        std::vector<int64_t> h_colptr(n + 1, 0);
        std::vector<int32_t> h_rowval(nnzA);
        std::vector<int32_t> h_csc_to_csr_map(nnzA);

        // Count entries per column (binned by band position, not original column)
        for (int64_t r = 0; r < m; ++r)
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
                h_colptr[old_to_new[A_ci[p]] + 1]++;
        for (int64_t j = 0; j < n; ++j)
            h_colptr[j + 1] += h_colptr[j];

        // Fill CSC arrays
        std::vector<int64_t> cursor(h_colptr.begin(), h_colptr.end());
        std::vector<int32_t> h_csr_to_csc_map(nnzA);
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p) {
                int64_t j = old_to_new[A_ci[p]];  // band position
                int64_t pos = cursor[j];
                h_rowval[pos] = (int32_t)r;
                h_csc_to_csr_map[pos] = (int32_t)p;
                h_csr_to_csc_map[p] = (int32_t)pos;
                cursor[j]++;
            }
        }

        upload_i64(d_A_colptr, h_colptr.data(), n + 1);
        upload_i32(d_A_rowval, h_rowval.data(), std::max(nnzA, (int64_t)1));
        {
            int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * std::max(nnzA, (int64_t)1)));
            CUDA_THROW(cudaMemcpy(tmp, h_csc_to_csr_map.data(), sizeof(int32_t) * std::max(nnzA, (int64_t)1), cudaMemcpyHostToDevice));
            d_csc_to_csr.reset(tmp);
        }

        // CSR view
        std::vector<int32_t> h_csr_row_start(m + 1);
        std::vector<int32_t> h_csr_cols(std::max(nnzA, (int64_t)1));
        for (int64_t r = 0; r <= m; ++r)
            h_csr_row_start[r] = (int32_t)A_ro[r];
        for (int64_t r = 0; r < m; ++r)
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
                h_csr_cols[p] = old_to_new[A_ci[p]];  // band position

        upload_i32(d_a_csr_row_start, h_csr_row_start.data(), m + 1);
        upload_i32(d_a_csr_cols, h_csr_cols.data(), std::max(nnzA, (int64_t)1));
        upload_i32(d_a_csr_to_csc, h_csr_to_csc_map.data(), std::max(nnzA, (int64_t)1));
    }

    // A and P value storage
    {
        double* tmp;
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * batchSize * std::max(nnzA, (int64_t)1)));
        d_A_values.reset(tmp);
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * batchSize * std::max(nnzP, (int64_t)1)));
        d_P_values.reset(tmp);
    }

    // Fused AHA+P scatter map: precompute gather indices for M = P + A'*H^{-1}*A
    // For each output element in D/L, collect AHA pairs and the P CSR index (if any).
    {
        // Build P lookup: map from (D/L linear offset) -> P CSR index
        // D element at offset k (0..D_total-1): D_data[k]
        // L element at offset k (0..L_total-1): L_data[k]
        // We encode D offsets directly, L offsets as (D_total_elems + l_offset)
        std::unordered_map<int64_t, int32_t> p_offset_to_idx;

        for (int64_t i = 0; i < n; ++i) {
            int64_t i_bp = old_to_new[i];  // band position of original column i
            int32_t bi = h_col_to_block[i_bp];
            int32_t li = (int32_t)(i_bp - h_block_offsets[bi]);
            for (int64_t p = P_ro[i]; p < P_ro[i + 1]; ++p) {
                int64_t j = old_to_new[P_ci[p]];  // band position
                int32_t bj = h_col_to_block[j];
                int32_t lj = j - h_block_offsets[bj];

                if (bi == bj) {
                    // D block entry (li, lj) col-major
                    int32_t sz = h_block_sizes[bi];
                    int64_t off = h_D_offsets[bi] + li + (int64_t)lj * sz;
                    p_offset_to_idx[off] = (int32_t)p;
                    if (li != lj) {
                        // Symmetric: also (lj, li)
                        int64_t off_sym = h_D_offsets[bi] + lj + (int64_t)li * sz;
                        p_offset_to_idx[off_sym] = (int32_t)p;
                    }
                } else if (bi + 1 == bj) {
                    // L[bi] entry: row=lj (in block bj=bi+1), col=li (in block bi)
                    int32_t rows_L = h_block_sizes[bi + 1];
                    int64_t off = D_total_elems + h_L_offsets[bi] + lj + (int64_t)li * rows_L;
                    p_offset_to_idx[off] = (int32_t)p;
                } else if (bj + 1 == bi) {
                    // L[bj] entry: row=li (in block bi=bj+1), col=lj (in block bj)
                    int32_t rows_L = h_block_sizes[bj + 1];
                    int64_t off = D_total_elems + h_L_offsets[bj] + li + (int64_t)lj * rows_L;
                    p_offset_to_idx[off] = (int32_t)p;
                }
            }
        }

        // Build CSC representation on host for fast column lookups
        std::vector<int32_t> csc_col_start(n + 1, 0);
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p) {
                csc_col_start[old_to_new[A_ci[p]] + 1]++;
            }
        }
        for (int64_t c = 0; c < n; ++c) csc_col_start[c + 1] += csc_col_start[c];
        std::vector<int32_t> csc_rows(nnzA), csc_csr_idx(nnzA);
        std::vector<int32_t> csc_fill(n, 0);
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p) {
                int32_t c = old_to_new[A_ci[p]];  // band position
                int32_t pos = csc_col_start[c] + csc_fill[c]++;
                csc_rows[pos] = (int32_t)r;
                csc_csr_idx[pos] = (int32_t)p;
            }
        }

        // For each output element in D and L, collect AHA pairs and P index
        n_aha_outputs = D_total_elems + L_total_elems;
        std::vector<int32_t> h_pair_ptr(n_aha_outputs + 1);
        std::vector<int2> h_pair_ij;
        std::vector<int32_t> h_pair_row;
        std::vector<int32_t> h_p_val_idx(n_aha_outputs, -1);

        int64_t out_idx = 0;

        // D blocks
        for (int64_t blk = 0; blk < nblocks; ++blk) {
            int32_t sz = h_block_sizes[blk];
            for (int32_t lj = 0; lj < sz; ++lj) {
                int32_t col_j = h_block_offsets[blk] + lj;
                for (int32_t li = 0; li < sz; ++li) {
                    int32_t col_i = h_block_offsets[blk] + li;
                    h_pair_ptr[out_idx] = (int32_t)h_pair_ij.size();

                    // Look up P contribution for this D element
                    int64_t d_off = h_D_offsets[blk] + li + (int64_t)lj * sz;
                    auto it = p_offset_to_idx.find(d_off);
                    if (it != p_offset_to_idx.end()) {
                        h_p_val_idx[out_idx] = it->second;
                    }

                    // Intersect rows with nonzeros in both col_i and col_j
                    int32_t pi = csc_col_start[col_i], pi_end = csc_col_start[col_i + 1];
                    int32_t pj = csc_col_start[col_j], pj_end = csc_col_start[col_j + 1];
                    while (pi < pi_end && pj < pj_end) {
                        if (csc_rows[pi] < csc_rows[pj]) { ++pi; }
                        else if (csc_rows[pi] > csc_rows[pj]) { ++pj; }
                        else {
                            h_pair_ij.push_back(make_int2(csc_csr_idx[pi], csc_csr_idx[pj]));
                            h_pair_row.push_back(csc_rows[pi]);
                            ++pi; ++pj;
                        }
                    }
                    out_idx++;
                }
            }
        }

        // L blocks
        for (int64_t blk = 0; blk < nblocks - 1; ++blk) {
            int32_t sz_cur = h_block_sizes[blk];
            int32_t sz_next = h_block_sizes[blk + 1];
            for (int32_t lj = 0; lj < sz_cur; ++lj) {
                int32_t col_j = h_block_offsets[blk] + lj;
                for (int32_t li = 0; li < sz_next; ++li) {
                    int32_t col_i = h_block_offsets[blk + 1] + li;
                    h_pair_ptr[out_idx] = (int32_t)h_pair_ij.size();

                    // Look up P contribution for this L element
                    int64_t l_off = h_L_offsets[blk] + li + (int64_t)lj * sz_next;
                    auto it = p_offset_to_idx.find(D_total_elems + l_off);
                    if (it != p_offset_to_idx.end()) {
                        h_p_val_idx[out_idx] = it->second;
                    }

                    int32_t pi = csc_col_start[col_i], pi_end = csc_col_start[col_i + 1];
                    int32_t pj = csc_col_start[col_j], pj_end = csc_col_start[col_j + 1];
                    while (pi < pi_end && pj < pj_end) {
                        if (csc_rows[pi] < csc_rows[pj]) { ++pi; }
                        else if (csc_rows[pi] > csc_rows[pj]) { ++pj; }
                        else {
                            h_pair_ij.push_back(make_int2(csc_csr_idx[pi], csc_csr_idx[pj]));
                            h_pair_row.push_back(csc_rows[pi]);
                            ++pi; ++pj;
                        }
                    }
                    out_idx++;
                }
            }
        }
        assert(out_idx == n_aha_outputs);
        h_pair_ptr[n_aha_outputs] = (int32_t)h_pair_ij.size();
        n_aha_pairs = (int64_t)h_pair_ij.size();

        // Upload to device
        if (n_aha_outputs > 0) {
            {
                int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * (n_aha_outputs + 1)));
                CUDA_THROW(cudaMemcpy(tmp, h_pair_ptr.data(), sizeof(int32_t) * (n_aha_outputs + 1), cudaMemcpyHostToDevice));
                d_aha_pair_ptr.reset(tmp);
            }
            {
                int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * n_aha_outputs));
                CUDA_THROW(cudaMemcpy(tmp, h_p_val_idx.data(), sizeof(int32_t) * n_aha_outputs, cudaMemcpyHostToDevice));
                d_aha_p_val_idx.reset(tmp);
            }
            if (n_aha_pairs > 0) {
                {
                    int2* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int2) * n_aha_pairs));
                    CUDA_THROW(cudaMemcpy(tmp, h_pair_ij.data(), sizeof(int2) * n_aha_pairs, cudaMemcpyHostToDevice));
                    d_aha_pair_ij.reset(tmp);
                }
                {
                    int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * n_aha_pairs));
                    CUDA_THROW(cudaMemcpy(tmp, h_pair_row.data(), sizeof(int32_t) * n_aha_pairs, cudaMemcpyHostToDevice));
                    d_aha_pair_row.reset(tmp);
                }
            }
        }
    }

    // Workspace vectors
    alloc_double(d_schur_rhs, batchSize * n);
    alloc_double(d_lhsx, batchSize * n);
    alloc_double(d_lhsz, batchSize * m);
    alloc_double(d_rhsx, batchSize * n);
    alloc_double(d_rhsz, batchSize * m);

    // Second workspace for multi-RHS solve2 (reserved for future use)
    alloc_double(d_lhsx2, batchSize * n);
    alloc_double(d_lhsz2, batchSize * m);
    alloc_double(d_rhsx2, batchSize * n);
    alloc_double(d_rhsz2, batchSize * m);

    // cuBLAS pointer arrays (3 arrays, each batchSize)
    {
        double** tmp;
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double*) * batchSize)); d_ptr_A.reset(tmp);
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double*) * batchSize)); d_ptr_B.reset(tmp);
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double*) * batchSize)); d_ptr_C.reset(tmp);
    }

    cublas_handle = nullptr;

    // Create cuSOLVER handle and pre-allocate workspace
    CUSOLVER_THROW(cusolverDnCreate(&cusolver_handle));
    {
        CUSOLVER_THROW(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                    max_block, nullptr, max_block, &cusolver_work_size));
        double* tmp;
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * cusolver_work_size));
        d_cusolver_work.reset(tmp);
        int* tmp_info;
        CUDA_THROW(cudaMalloc(&tmp_info, sizeof(int) * batchSize));
        d_cusolver_info.reset(tmp_info);
    }

    // Pre-compute pointer arrays for S, L, work blocks
    // This eliminates setup_ptrs kernel launches in the cuBLAS factorize/solve path
    {
        int64_t work_stride = (int64_t)max_block * max_block;

        // S block pointers: d_S_block_ptrs[i * batchSize + b] = d_S + b * D_total_elems + h_D_offsets[i]
        std::vector<double*> h_S_ptrs(nblocks * batchSize);
        for (int64_t i = 0; i < nblocks; ++i)
            for (int64_t b = 0; b < batchSize; ++b)
                h_S_ptrs[i * batchSize + b] = d_S.get() + b * D_total_elems + h_D_offsets[i];
        {
            double** tmp2; CUDA_THROW(cudaMalloc(&tmp2, sizeof(double*) * nblocks * batchSize));
            CUDA_THROW(cudaMemcpy(tmp2, h_S_ptrs.data(), sizeof(double*) * nblocks * batchSize, cudaMemcpyHostToDevice));
            d_S_block_ptrs.reset(tmp2);
        }

        // L block pointers: d_L_block_ptrs[i * batchSize + b] = d_L + b * L_total_elems + h_L_offsets[i]
        if (nblocks > 1) {
            std::vector<double*> h_L_ptrs((nblocks - 1) * batchSize);
            for (int64_t i = 0; i < nblocks - 1; ++i)
                for (int64_t b = 0; b < batchSize; ++b)
                    h_L_ptrs[i * batchSize + b] = d_L.get() + b * L_total_elems + h_L_offsets[i];
            double** tmp2; CUDA_THROW(cudaMalloc(&tmp2, sizeof(double*) * (nblocks - 1) * batchSize));
            CUDA_THROW(cudaMemcpy(tmp2, h_L_ptrs.data(), sizeof(double*) * (nblocks - 1) * batchSize, cudaMemcpyHostToDevice));
            d_L_block_ptrs.reset(tmp2);
        }

        // Work block pointers: d_work_block_ptrs[b] = d_work + b * work_stride
        std::vector<double*> h_work_ptrs(batchSize);
        for (int64_t b = 0; b < batchSize; ++b)
            h_work_ptrs[b] = d_work.get() + b * work_stride;
        {
            double** tmp2; CUDA_THROW(cudaMalloc(&tmp2, sizeof(double*) * batchSize));
            CUDA_THROW(cudaMemcpy(tmp2, h_work_ptrs.data(), sizeof(double*) * batchSize, cudaMemcpyHostToDevice));
            d_work_block_ptrs.reset(tmp2);
        }
    }

    populated_ = false;

    cudaStreamSynchronize(stream);
}

// ============================================================================
// Populate
// ============================================================================

void RiccatiKKTData::populate(CSR& P, CSR& A, cudaStream_t stream) {
    if (nnzP > 0) {
        CUDA_THROW(cudaMemcpyAsync(d_P_values.get(), P.values(),
                       sizeof(double) * batchSize * nnzP,
                       cudaMemcpyDeviceToDevice, stream));
    }
    if (nnzA > 0) {
        // Store A values in CSR order (matching input)
        CUDA_THROW(cudaMemcpyAsync(d_A_values.get(), A.values(),
                       sizeof(double) * batchSize * nnzA,
                       cudaMemcpyDeviceToDevice, stream));
    }
    populated_ = true;
}

// ============================================================================
// Update H
// ============================================================================

void RiccatiKKTData::update_H(const Cones& cones, const double* /*mu_data*/, cudaStream_t stream) {
    compute_h_inv_kernel(
        d_h_inv.get(), cones.nonneg_w.data(),
        cones.numZeroCones, cones.numNonnegCones,
        m, batchSize, reg_eps_, stream);
}

// ============================================================================
// Assemble and factorize
// ============================================================================

bool RiccatiKKTData::assemble_and_factorize(cudaStream_t stream) {
    // 1. Fused assembly: M = P + A'*H^{-1}*A (writes all D/L elements, no atomics)
    if (n_aha_outputs > 0) {
        assemble_fused(
            d_D.get(), d_L.get(),
            d_A_values.get(), d_h_inv.get(), d_P_values.get(),
            d_aha_pair_ptr.get(),
            d_aha_pair_ij.get(), d_aha_pair_row.get(),
            d_aha_p_val_idx.get(),
            n_aha_outputs, D_total_elems, L_total_elems,
            nnzA, nnzP, m, batchSize, stream);
    } else {
        // No AHA scatter map — fall back to zeroing
        zero_blocks_kernel(
            d_D.get(), d_L.get() ? d_L.get() : d_D.get(),
            D_total_elems, L_total_elems,
            batchSize, stream);
    }

    // 2. Factorize: D -> S copy, block Cholesky with cuBLAS/cuSOLVER
    return riccati_factorize_blocks(
        d_S.get(), d_D.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(), d_ptr_C.get(),
        h_D_offsets.data(), h_L_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        nblocks, max_block, batchSize,
        cublas_handle, cusolver_handle, d_cusolver_info.get(),
        stream,
        d_S_block_ptrs.get(), d_L_block_ptrs.get(), d_work_block_ptrs.get(),
        reg_eps_);
}

// ============================================================================
// Solve (Schur complement)
// ============================================================================

void RiccatiKKTData::solve_schur(
    const double* rx, const double* rz,
    double* sol_x, double* sol_z,
    cudaStream_t stream)
{
    const int32_t* nto = has_perm_ ? d_new_to_old.get() : nullptr;
    // 1. Form Schur RHS: lhsx = rx + A' * diag(h_inv) * rz
    form_schur_rhs_kernel(
        d_lhsx.get(), rx, rz, d_h_inv.get(),
        d_A_values.get(), d_A_colptr.get(), d_A_rowval.get(),
        d_csc_to_csr.get(), nto,
        n, m, nnzA, batchSize, stream);

    // 2. Solve block-tridiagonal system: M * x = schur_rhs (in-place on lhsx)
    riccati_solve_blocks(
        d_lhsx.get(), d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        n, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());

    // 3. Recover z = h_inv * (A*x - rz)
    recover_z_kernel(
        d_lhsz.get(), d_lhsx.get(), rz, d_h_inv.get(),
        d_A_values.get(),
        d_a_csr_row_start.get(), d_a_csr_cols.get(), d_a_csr_to_csc.get(),
        n, m, nnzA, batchSize, stream);

    // Note: KKT-level iterative refinement removed — with reg_eps matching
    // the static regularization constant (1e-8), the Schur complement solve
    // achieves the same accuracy as cuDSS and converges in the same number of iterations.

    // Copy to output if needed. sol_x: band order -> original column order.
    if (has_perm_) {
        permute_scatter(sol_x, d_lhsx.get(), d_new_to_old.get(), n, batchSize, stream);
    } else if (sol_x != d_lhsx.get()) {
        CUDA_THROW(cudaMemcpyAsync(sol_x, d_lhsx.get(),
                       sizeof(double) * batchSize * n,
                       cudaMemcpyDeviceToDevice, stream));
    }
    if (sol_z != d_lhsz.get()) {
        CUDA_THROW(cudaMemcpyAsync(sol_z, d_lhsz.get(),
                       sizeof(double) * batchSize * m,
                       cudaMemcpyDeviceToDevice, stream));
    }
}

// ============================================================================
// Public: solve
// ============================================================================

void RiccatiKKTData::solve(const double* rhs, double* sol, cudaStream_t stream) {
    const int32_t* nto = has_perm_ ? d_new_to_old.get() : nullptr;
    // Fused: form_schur_rhs reads directly from interleaved [n+m] buffer
    form_schur_rhs_interleaved_kernel(
        d_lhsx.get(), rhs, d_h_inv.get(),
        d_A_values.get(), d_A_colptr.get(), d_A_rowval.get(),
        d_csc_to_csr.get(), nto,
        n, m, nnzA, batchSize, 1.0, stream);

    // Solve block-tridiagonal system
    riccati_solve_blocks(
        d_lhsx.get(), d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        n, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());

    // Fused: recover_z and pack x,z into interleaved output
    // rz is at offset n in the interleaved rhs
    recover_z_and_pack_kernel(
        sol, d_lhsx.get(), rhs,
        d_h_inv.get(), d_A_values.get(),
        d_a_csr_row_start.get(), d_a_csr_cols.get(), nto,
        n, m, nnzA, batchSize, stream);
}

void RiccatiKKTData::solve2(const double* rhs, double* sol, cudaStream_t stream) {
    const int32_t* nto = has_perm_ ? d_new_to_old.get() : nullptr;
    int64_t N = n + m;
    const double* rhs1 = rhs;
    const double* rhs2 = rhs + N * batchSize;
    double* sol1 = sol;
    double* sol2 = sol + N * batchSize;

    // Fused: form Schur RHS for both from interleaved [n+m] buffers
    form_schur_rhs2_both_interleaved_kernel(
        d_lhsx.get(), d_lhsx2.get(),
        rhs1, rhs2,
        d_h_inv.get(),
        d_A_values.get(), d_A_colptr.get(), d_A_rowval.get(),
        d_csc_to_csr.get(), nto,
        n, m, nnzA, batchSize, stream);

    // Multi-RHS block-tridiagonal solve (loads L^{-1} once, solves both RHS)
    riccati_solve2_blocks(
        d_lhsx.get(), d_lhsx2.get(),
        d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        n, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());

    // Fused: recover z for both, pack both into interleaved output
    recover_z2_and_pack2_kernel(
        sol1, sol2,
        d_lhsx.get(), d_lhsx2.get(),
        rhs1, rhs2,
        d_h_inv.get(), d_A_values.get(),
        d_a_csr_row_start.get(), d_a_csr_cols.get(), nto,
        n, m, nnzA, batchSize, stream);
}

// ============================================================================
// Update methods
// ============================================================================

bool RiccatiKKTData::update(
    Cones& cones,
    const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
    ScalingStrategy scaling,
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    const BatchedVector& q, const BatchedVector& b,
    BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
    BatchedVector& x2, BatchedVector& z2,
    cudaStream_t stream)
{
    assert(populated_);

    reg_eps_ = static_regularization_enable ? static_regularization_constant : 1e-8;

    bool success = cones.update_scaling(s, z, μ, scaling, stream);
    if (!success) return false;

    update_H(cones, nullptr, stream);
    success = assemble_and_factorize(stream);
    if (!success) return false;

    const int32_t* nto = has_perm_ ? d_new_to_old.get() : nullptr;
    // Solve constant RHS [-q; b] — fused: negate rx inline via rx_sign=-1
    form_schur_rhs_signed_kernel(
        d_lhsx.get(), q.data(), b.data(), d_h_inv.get(),
        d_A_values.get(), d_A_colptr.get(), d_A_rowval.get(),
        d_csc_to_csr.get(), nto,
        n, m, nnzA, batchSize, -1.0, stream);

    riccati_solve_blocks(
        d_lhsx.get(), d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        n, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());

    recover_z_kernel(
        d_lhsz.get(), d_lhsx.get(), b.data(), d_h_inv.get(),
        d_A_values.get(),
        d_a_csr_row_start.get(), d_a_csr_cols.get(), d_a_csr_to_csc.get(),
        n, m, nnzA, batchSize, stream);

    // x2 is the constant primal solution; d_lhsx is in band order — scatter to
    // original column order (plain copy when unpermuted).
    if (has_perm_) {
        permute_scatter(x2.data(), d_lhsx.get(), d_new_to_old.get(), n, batchSize, stream);
    } else {
        CUDA_THROW(cudaMemcpyAsync(x2.data(), d_lhsx.get(), sizeof(double) * batchSize * n,
                       cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_THROW(cudaMemcpyAsync(z2.data(), d_lhsz.get(), sizeof(double) * batchSize * m,
                   cudaMemcpyDeviceToDevice, stream));

    return true;
}

bool RiccatiKKTData::update(
    Cones& cones,
    const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
    ScalingStrategy scaling,
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    cudaStream_t stream)
{
    assert(populated_);

    reg_eps_ = static_regularization_enable ? static_regularization_constant : 1e-8;

    bool success = cones.update_scaling(s, z, μ, scaling, stream);
    if (!success) return false;

    update_H(cones, nullptr, stream);
    return assemble_and_factorize(stream);
}

bool RiccatiKKTData::updateFactorOnly(
    Cones& cones,
    const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
    ScalingStrategy scaling,
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    const BatchedVector& q, const BatchedVector& b,
    BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
    BatchedVector& x2, BatchedVector& z2,
    cudaStream_t stream)
{
    assert(populated_);

    reg_eps_ = static_regularization_enable ? static_regularization_constant : 1e-8;

    // Note: caller (solver.cpp) already called cones.update_scaling() via
    // variables.scale_cones(). Do NOT call it again here — double scaling
    // would factor a different KKT system than the rest of the solver assumes.
    update_H(cones, nullptr, stream);
    if (!assemble_and_factorize(stream)) return false;

    // Prepare constant RHS in workspace (but don't solve yet)
    // Store -q in d_rhsx2 for deferred solve; b stays as-is (read directly later)
    negate_vector(d_rhsx2.get(), q.data(), batchSize * n, stream);
    CUDA_THROW(cudaMemcpyAsync(d_rhsz2.get(), b.data(),
                   sizeof(double) * batchSize * m,
                   cudaMemcpyDeviceToDevice, stream));

    return true;
}

void RiccatiKKTData::solve_combined(
    const double* affine_rhs, double* affine_sol,
    double* const_x, double* const_z,
    cudaStream_t stream)
{
    const int32_t* nto = has_perm_ ? d_new_to_old.get() : nullptr;
    // Fused: form Schur RHS for both affine (from interleaved) and constant (from separate)
    form_schur_rhs2_interleaved_kernel(
        d_lhsx.get(), d_lhsx2.get(),
        affine_rhs,
        d_rhsx2.get(), d_rhsz2.get(),
        d_h_inv.get(),
        d_A_values.get(), d_A_colptr.get(), d_A_rowval.get(),
        d_csc_to_csr.get(), nto,
        n, m, nnzA, batchSize, stream);

    // Multi-RHS block-tridiagonal solve (loads S once, solves both RHS)
    riccati_solve2_blocks(
        d_lhsx.get(), d_lhsx2.get(),
        d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        n, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());

    // Fused: recover z for both, pack affine into interleaved output, constant z to separate
    recover_z2_and_pack_kernel(
        affine_sol, d_lhsz2.get(),
        d_lhsx.get(), d_lhsx2.get(),
        affine_rhs, d_rhsz2.get(),
        d_h_inv.get(),
        d_A_values.get(),
        d_a_csr_row_start.get(), d_a_csr_cols.get(), nto,
        n, m, nnzA, batchSize, stream);

    // Constant primal solution d_lhsx2 is in band order — scatter to original
    // column order (plain copy when unpermuted).
    if (has_perm_) {
        permute_scatter(const_x, d_lhsx2.get(), d_new_to_old.get(), n, batchSize, stream);
    } else {
        CUDA_THROW(cudaMemcpyAsync(const_x, d_lhsx2.get(), sizeof(double) * batchSize * n,
                       cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_THROW(cudaMemcpyAsync(const_z, d_lhsz2.get(), sizeof(double) * batchSize * m,
                   cudaMemcpyDeviceToDevice, stream));
}

bool RiccatiKKTData::regularize_and_refactor(
    bool static_regularization_enable,
    double static_regularization_constant,
    double /* static_regularization_proportional */,
    cudaStream_t stream)
{
    reg_eps_ = static_regularization_enable ? static_regularization_constant : 1e-8;
    return assemble_and_factorize(stream);
}

size_t RiccatiKKTData::memoryUsage() const noexcept {
    return sizeof(double) * batchSize * (
        D_total_elems * 2 + L_total_elems +
        (int64_t)max_block * max_block +
        m + nnzA + nnzP +
        n * 5 + m * 4  // includes solve2 workspace
    );
}

} // namespace moreau
