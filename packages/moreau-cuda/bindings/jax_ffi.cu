/**
 * @file jax_ffi.cu
 * @brief XLA FFI handlers for JAX zero-copy integration with Moreau CUDA solver
 *
 * This provides true zero-copy GPU integration between JAX and the Moreau solver.
 * Solvers are cached internally based on problem structure hash for efficient reuse.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/diff/diff_kernels.cuh"

namespace ffi = xla::ffi;
using namespace moreau;

// =============================================================================
// Solver Cache (keyed by problem structure)
// =============================================================================

struct SolverKey {
    int64_t n, m, batch_size;
    int64_t nnzP, nnzA;
    int64_t num_zero, num_nonneg, num_exp, num_soc, num_psd;
    std::vector<int64_t> soc_dims;  // Per-cone SOC dimensions
    std::vector<int64_t> psd_dims;  // Per-cone PSD matrix dimensions
    int64_t num_power;
    std::vector<double> power_alphas;       // Power cone alpha values
    int64_t num_gen_power;
    std::vector<int64_t> gen_power_dim1s;   // Per-cone dim1
    std::vector<int64_t> gen_power_dim2s;   // Per-cone dim2
    std::vector<double> gen_power_alphas;   // Flattened alpha values
    // Direct-x cones: parallel arrays one entry per direct-x cone.
    // Per-cone descriptors:
    //   x_kinds[c]            — XConeKind enum value (Nonneg=0, SOC=1, PSD=2, Exp=3, Power=4, GenPower=5).
    //   x_indices_offsets     — prefix sum into x_indices_flat (length num_dir_cones+1).
    //   x_indices_flat        — concatenated cone indices.
    //   x_power_alphas_flat   — alpha per Power x-cone (length = #Power x-cones, in cone order).
    //   x_psd_ks_flat         — psd_k per PSD x-cone.
    //   x_gen_power_*         — alphas/dim2 per GenPower x-cone (parallel to x_kinds for those entries).
    int64_t num_dir_cones;
    std::vector<int64_t> x_kinds;
    std::vector<int64_t> x_indices_offsets;
    std::vector<int64_t> x_indices_flat;
    std::vector<double>  x_power_alphas;
    std::vector<int64_t> x_psd_ks;
    std::vector<int64_t> x_gen_power_dim1s;
    std::vector<int64_t> x_gen_power_dim2s;
    std::vector<double>  x_gen_power_alphas_flat;
    // We don't hash the actual sparsity pattern - assume same dimensions = same pattern
    // This is a reasonable assumption for repeated solves with same structure

    bool operator==(const SolverKey& other) const {
        return n == other.n && m == other.m && batch_size == other.batch_size &&
               nnzP == other.nnzP && nnzA == other.nnzA &&
               num_zero == other.num_zero && num_nonneg == other.num_nonneg &&
               num_exp == other.num_exp && num_soc == other.num_soc &&
               num_psd == other.num_psd &&
               soc_dims == other.soc_dims && psd_dims == other.psd_dims &&
               num_power == other.num_power &&
               power_alphas == other.power_alphas &&
               num_gen_power == other.num_gen_power &&
               gen_power_dim1s == other.gen_power_dim1s &&
               gen_power_dim2s == other.gen_power_dim2s &&
               gen_power_alphas == other.gen_power_alphas &&
               num_dir_cones == other.num_dir_cones &&
               x_kinds == other.x_kinds &&
               x_indices_offsets == other.x_indices_offsets &&
               x_indices_flat == other.x_indices_flat &&
               x_power_alphas == other.x_power_alphas &&
               x_psd_ks == other.x_psd_ks &&
               x_gen_power_dim1s == other.x_gen_power_dim1s &&
               x_gen_power_dim2s == other.x_gen_power_dim2s &&
               x_gen_power_alphas_flat == other.x_gen_power_alphas_flat;
    }
};

struct SolverKeyHash {
    size_t operator()(const SolverKey& k) const {
        size_t h = 0;
        auto hash_combine = [&h](size_t v) {
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(std::hash<int64_t>{}(k.n));
        hash_combine(std::hash<int64_t>{}(k.m));
        hash_combine(std::hash<int64_t>{}(k.batch_size));
        hash_combine(std::hash<int64_t>{}(k.nnzP));
        hash_combine(std::hash<int64_t>{}(k.nnzA));
        hash_combine(std::hash<int64_t>{}(k.num_zero));
        hash_combine(std::hash<int64_t>{}(k.num_nonneg));
        hash_combine(std::hash<int64_t>{}(k.num_exp));
        hash_combine(std::hash<int64_t>{}(k.num_soc));
        hash_combine(std::hash<int64_t>{}(k.num_psd));
        for (auto d : k.soc_dims) {
            hash_combine(std::hash<int64_t>{}(d));
        }
        for (auto d : k.psd_dims) {
            hash_combine(std::hash<int64_t>{}(d));
        }
        hash_combine(std::hash<int64_t>{}(k.num_power));
        for (auto a : k.power_alphas) {
            uint64_t bits;
            std::memcpy(&bits, &a, sizeof(double));
            hash_combine(std::hash<uint64_t>{}(bits));
        }
        hash_combine(std::hash<int64_t>{}(k.num_gen_power));
        for (auto d : k.gen_power_dim1s) {
            hash_combine(std::hash<int64_t>{}(d));
        }
        for (auto d : k.gen_power_dim2s) {
            hash_combine(std::hash<int64_t>{}(d));
        }
        for (auto a : k.gen_power_alphas) {
            // Hash double via its bit representation
            uint64_t bits;
            std::memcpy(&bits, &a, sizeof(double));
            hash_combine(std::hash<uint64_t>{}(bits));
        }
        // Direct-x descriptors
        hash_combine(std::hash<int64_t>{}(k.num_dir_cones));
        for (auto v : k.x_kinds)            hash_combine(std::hash<int64_t>{}(v));
        for (auto v : k.x_indices_offsets)  hash_combine(std::hash<int64_t>{}(v));
        for (auto v : k.x_indices_flat)     hash_combine(std::hash<int64_t>{}(v));
        for (auto a : k.x_power_alphas) {
            uint64_t bits; std::memcpy(&bits, &a, sizeof(double));
            hash_combine(std::hash<uint64_t>{}(bits));
        }
        for (auto v : k.x_psd_ks)           hash_combine(std::hash<int64_t>{}(v));
        for (auto v : k.x_gen_power_dim1s)  hash_combine(std::hash<int64_t>{}(v));
        for (auto v : k.x_gen_power_dim2s)  hash_combine(std::hash<int64_t>{}(v));
        for (auto a : k.x_gen_power_alphas_flat) {
            uint64_t bits; std::memcpy(&bits, &a, sizeof(double));
            hash_combine(std::hash<uint64_t>{}(bits));
        }
        return h;
    }
};

struct CachedSolver {
    std::unique_ptr<CompiledSolver> solver;
    // Store host copies of structure arrays
    std::vector<int64_t> P_row_offsets_host;
    std::vector<int64_t> P_col_indices_host;
    std::vector<int64_t> A_row_offsets_host;
    std::vector<int64_t> A_col_indices_host;
};

static std::unordered_map<SolverKey, std::unique_ptr<CachedSolver>, SolverKeyHash> g_solver_cache;
static std::mutex g_cache_mutex;

// Bundle of direct-x descriptors copied from device to host. Built once
// per FFI handler invocation and consumed by both the cache key build and
// the `cones.dir_cones` construction.
struct XConeHostBuffers {
    int64_t num_dir_cones = 0;
    int64_t total_xn = 0;
    std::vector<int64_t> x_kinds;
    std::vector<int64_t> x_indices_offsets;
    std::vector<int64_t> x_indices_flat;
    std::vector<double>  x_power_alphas;
    std::vector<int64_t> x_psd_ks;
    std::vector<int64_t> x_gen_power_dim1s;
    std::vector<int64_t> x_gen_power_dim2s;
    std::vector<double>  x_gen_power_alphas_flat;
};

static XConeHostBuffers read_xcone_metadata(
    int64_t num_dir_cones,
    const int64_t* d_x_kinds,
    const int64_t* d_x_indices_offsets,
    const int64_t* d_x_indices_flat, int64_t num_x_indices_flat,
    const double*  d_x_power_alphas, int64_t num_x_power_alphas,
    const int64_t* d_x_psd_ks, int64_t num_x_psd_ks,
    const int64_t* d_x_gen_power_dim1s,
    const int64_t* d_x_gen_power_dim2s, int64_t num_x_gen_power,
    const double*  d_x_gen_power_alphas_flat, int64_t num_x_gen_power_alphas_flat
) {
    XConeHostBuffers out;
    out.num_dir_cones = num_dir_cones;
    if (num_dir_cones <= 0) return out;
    out.x_kinds.resize(num_dir_cones);
    cudaMemcpy(out.x_kinds.data(), d_x_kinds,
               num_dir_cones * sizeof(int64_t), cudaMemcpyDeviceToHost);
    out.x_indices_offsets.resize(num_dir_cones + 1);
    cudaMemcpy(out.x_indices_offsets.data(), d_x_indices_offsets,
               (num_dir_cones + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    out.total_xn = out.x_indices_offsets.back();
    if (num_x_indices_flat > 0) {
        out.x_indices_flat.resize(num_x_indices_flat);
        cudaMemcpy(out.x_indices_flat.data(), d_x_indices_flat,
                   num_x_indices_flat * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }
    if (num_x_power_alphas > 0) {
        out.x_power_alphas.resize(num_x_power_alphas);
        cudaMemcpy(out.x_power_alphas.data(), d_x_power_alphas,
                   num_x_power_alphas * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (num_x_psd_ks > 0) {
        out.x_psd_ks.resize(num_x_psd_ks);
        cudaMemcpy(out.x_psd_ks.data(), d_x_psd_ks,
                   num_x_psd_ks * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }
    if (num_x_gen_power > 0) {
        out.x_gen_power_dim1s.resize(num_x_gen_power);
        out.x_gen_power_dim2s.resize(num_x_gen_power);
        cudaMemcpy(out.x_gen_power_dim1s.data(), d_x_gen_power_dim1s,
                   num_x_gen_power * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(out.x_gen_power_dim2s.data(), d_x_gen_power_dim2s,
                   num_x_gen_power * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }
    if (num_x_gen_power_alphas_flat > 0) {
        out.x_gen_power_alphas_flat.resize(num_x_gen_power_alphas_flat);
        cudaMemcpy(out.x_gen_power_alphas_flat.data(),
                   d_x_gen_power_alphas_flat,
                   num_x_gen_power_alphas_flat * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }
    return out;
}

// Build `Cones::dir_cones` from the host-side direct-x descriptors. Walks
// `x_kinds` in order, slicing the per-cone-kind parameter arrays via
// running counters (Power and PSD cones consume one entry per cone;
// GenPower consumes one (dim1, dim2) and a contiguous slice of alphas).
static std::vector<SupportedXConeT> build_dir_cones(const XConeHostBuffers& xb) {
    std::vector<SupportedXConeT> out;
    out.reserve(xb.num_dir_cones);
    int64_t pow_idx = 0, psd_idx = 0, gp_idx = 0, gp_alpha_off = 0;
    for (int64_t c = 0; c < xb.num_dir_cones; ++c) {
        SupportedXConeT xc;
        xc.kind = static_cast<XConeKind>(xb.x_kinds[c]);
        int64_t off = xb.x_indices_offsets[c];
        int64_t end = xb.x_indices_offsets[c + 1];
        xc.indices.assign(
            xb.x_indices_flat.begin() + off,
            xb.x_indices_flat.begin() + end);
        switch (xc.kind) {
            case XConeKind::Power:
                xc.power_alpha = xb.x_power_alphas[pow_idx++];
                break;
            case XConeKind::PSD:
                xc.psd_k = xb.x_psd_ks[psd_idx++];
                break;
            case XConeKind::GenPower: {
                int64_t dim1 = xb.x_gen_power_dim1s[gp_idx];
                int64_t dim2 = xb.x_gen_power_dim2s[gp_idx];
                ++gp_idx;
                xc.gen_power_alphas.assign(
                    xb.x_gen_power_alphas_flat.begin() + gp_alpha_off,
                    xb.x_gen_power_alphas_flat.begin() + gp_alpha_off + dim1);
                xc.gen_power_dim2 = dim2;
                gp_alpha_off += dim1;
                break;
            }
            default: break;  // Nonneg / SOC / Exp need no extra params
        }
        out.push_back(std::move(xc));
    }
    return out;
}

/**
 * @brief Get or create a cached solver for the given problem structure
 *
 * Note: Structure arrays (P_row_offsets, etc.) are on DEVICE.
 * They will be copied to host for solver creation (one-time cost per unique structure).
 */
static CachedSolver* get_or_create_solver(
    cudaStream_t stream,
    int64_t n, int64_t m, int64_t batch_size,
    const int64_t* d_P_row_offsets, const int64_t* d_P_col_indices, int64_t nnzP,
    const int64_t* d_A_row_offsets, const int64_t* d_A_col_indices, int64_t nnzA,
    int64_t num_zero, int64_t num_nonneg, int64_t num_exp, int64_t num_soc,
    const int64_t* d_soc_dims, int64_t num_soc_dims,
    int64_t num_psd, const int64_t* d_psd_dims, int64_t num_psd_dims,
    int64_t num_power,
    const double* d_power_alphas, int64_t num_power_alphas,
    int64_t num_gen_power,
    const double* d_gen_power_alphas, int64_t num_gen_power_alphas,
    const int64_t* d_gen_power_dim1s, const int64_t* d_gen_power_dim2s,
    const XConeHostBuffers& xb,
    bool enable_grad
) {
    // Sync XLA's stream before D2H copies — structure arrays are written by
    // XLA on `stream` and cudaMemcpy (default stream) would race with them.
    cudaStreamSynchronize(stream);

    // Copy SOC dims from device to host for key and cone setup
    std::vector<int64_t> soc_dims_host;
    if (num_soc_dims > 0) {
        if (d_soc_dims == nullptr) {
            throw std::invalid_argument(
                "num_soc_dims > 0 but d_soc_dims is null");
        }
        soc_dims_host.resize(num_soc_dims);
        cudaMemcpy(soc_dims_host.data(), d_soc_dims,
                   num_soc_dims * sizeof(int64_t), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < num_soc_dims; i++) {
            if (soc_dims_host[i] < 2) {
                throw std::invalid_argument(
                    "soc_dims[" + std::to_string(i) + "] = " +
                    std::to_string(soc_dims_host[i]) + ", must be >= 2");
            }
        }
    }

    // Copy PSD dims from device to host
    std::vector<int64_t> psd_dims_host;
    if (num_psd_dims > 0 && d_psd_dims != nullptr) {
        psd_dims_host.resize(num_psd_dims);
        cudaMemcpy(psd_dims_host.data(), d_psd_dims,
                   num_psd_dims * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }

    // Copy power cone alphas from device to host
    std::vector<double> power_alphas_host;
    if (num_power_alphas > 0) {
        power_alphas_host.resize(num_power_alphas);
        cudaMemcpy(power_alphas_host.data(), d_power_alphas,
                   num_power_alphas * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Copy GenPowerCone params from device to host
    std::vector<double> gen_power_alphas_host;
    std::vector<int64_t> gen_power_dim1s_host, gen_power_dim2s_host;
    if (num_gen_power > 0) {
        gen_power_dim1s_host.resize(num_gen_power);
        gen_power_dim2s_host.resize(num_gen_power);
        cudaMemcpy(gen_power_dim1s_host.data(), d_gen_power_dim1s,
                   num_gen_power * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gen_power_dim2s_host.data(), d_gen_power_dim2s,
                   num_gen_power * sizeof(int64_t), cudaMemcpyDeviceToHost);
        if (num_gen_power_alphas > 0) {
            gen_power_alphas_host.resize(num_gen_power_alphas);
            cudaMemcpy(gen_power_alphas_host.data(), d_gen_power_alphas,
                       num_gen_power_alphas * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }

    SolverKey key{n, m, batch_size, nnzP, nnzA, num_zero, num_nonneg, num_exp, num_soc, num_psd,
                  soc_dims_host, psd_dims_host,
                  num_power, power_alphas_host,
                  num_gen_power, gen_power_dim1s_host, gen_power_dim2s_host, gen_power_alphas_host,
                  xb.num_dir_cones, xb.x_kinds, xb.x_indices_offsets, xb.x_indices_flat,
                  xb.x_power_alphas, xb.x_psd_ks,
                  xb.x_gen_power_dim1s, xb.x_gen_power_dim2s, xb.x_gen_power_alphas_flat};

    std::lock_guard<std::mutex> lock(g_cache_mutex);

    auto it = g_solver_cache.find(key);
    if (it != g_solver_cache.end()) {
        return it->second.get();
    }

    // Time the construction phase
    auto construction_start = std::chrono::high_resolution_clock::now();

    // Create new solver - need to copy structure arrays from device to host
    auto cached = std::make_unique<CachedSolver>();

    // Copy structure arrays to host (one-time cost per unique problem structure)
    cached->P_row_offsets_host.resize(n + 1);
    cached->P_col_indices_host.resize(nnzP);
    cached->A_row_offsets_host.resize(m + 1);
    cached->A_col_indices_host.resize(nnzA);

    cudaMemcpy(cached->P_row_offsets_host.data(), d_P_row_offsets,
               (n + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cached->P_col_indices_host.data(), d_P_col_indices,
               nnzP * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cached->A_row_offsets_host.data(), d_A_row_offsets,
               (m + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cached->A_col_indices_host.data(), d_A_col_indices,
               nnzA * sizeof(int64_t), cudaMemcpyDeviceToHost);

    Cones cones;
    cones.numZeroCones = num_zero;
    cones.numNonnegCones = num_nonneg;
    cones.numExpCones = num_exp;
    cones.numSocCones = num_soc;
    cones.socConeDims = soc_dims_host;
    cones.numPsdCones = num_psd;
    cones.psdConeDims = psd_dims_host;
    cones.numPowerCones = num_power;
    cones.powerAlphas = power_alphas_host;
    cones.numGenPowerCones = num_gen_power;
    cones.genPowerAlphas = gen_power_alphas_host;
    cones.genPowerDim1s = gen_power_dim1s_host;
    cones.genPowerDim2s = gen_power_dim2s_host;
    cones.dir_cones = build_dir_cones(xb);
    cones.numXCones = static_cast<int64_t>(cones.dir_cones.size());

    Settings settings;
    settings.verbose = false;  // Suppress output for FFI handlers
    settings.enableGrad = enable_grad;

    cached->solver = std::make_unique<CompiledSolver>(
        n, m, batch_size,
        cached->P_row_offsets_host.data(), cached->P_col_indices_host.data(), nnzP,
        cached->A_row_offsets_host.data(), cached->A_col_indices_host.data(), nnzA,
        cones, settings
    );

    // Record construction time
    auto construction_end = std::chrono::high_resolution_clock::now();
    cached->solver->info.construction_time = std::chrono::duration<double>(construction_end - construction_start).count();

    auto* ptr = cached.get();
    g_solver_cache[key] = std::move(cached);
    return ptr;
}

// =============================================================================
// XLA FFI Handlers
// =============================================================================

/**
 * @brief Copy solution and metadata from solver to FFI output buffers.
 *
 * Shared by both cold and warm-start forward handlers.
 */
static void copy_solution_to_outputs(
    cudaStream_t stream,
    CachedSolver* cached,
    int64_t n, int64_t m, int64_t batch_size,
    ffi::ResultBuffer<ffi::F64>& x,
    ffi::ResultBuffer<ffi::F64>& z,
    ffi::ResultBuffer<ffi::F64>& s,
    ffi::ResultBuffer<ffi::F64>& status_out,
    ffi::ResultBuffer<ffi::F64>& obj_val_out,
    ffi::ResultBuffer<ffi::F64>& iterations_out,
    ffi::ResultBuffer<ffi::F64>& solve_time_out,
    ffi::ResultBuffer<ffi::F64>& setup_time_out,
    ffi::ResultBuffer<ffi::F64>& construction_time_out
) {
    // Copy solution vectors to output buffers (device-to-device, async)
    cudaMemcpyAsync(x->typed_data(), cached->solver->solution.x.data(),
                    sizeof(double) * n * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(z->typed_data(), cached->solver->solution.z.data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(s->typed_data(), cached->solver->solution.s.data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);

    // Copy metadata to output buffers
    // status is stored as int32_t - convert to double via host
    std::vector<int32_t> status_int(batch_size);
    std::vector<double> status_double(batch_size);
    cudaMemcpyAsync(status_int.data(), cached->solver->solution.status.get(),
                    sizeof(int32_t) * batch_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // Need sync before conversion
    for (int64_t i = 0; i < batch_size; ++i) {
        status_double[i] = static_cast<double>(status_int[i]);
    }
    cudaMemcpyAsync(status_out->typed_data(), status_double.data(),
                    sizeof(double) * batch_size, cudaMemcpyHostToDevice, stream);

    // obj_val is per-batch (already double)
    cudaMemcpyAsync(obj_val_out->typed_data(), cached->solver->solution.obj_val.data(),
                    sizeof(double) * batch_size, cudaMemcpyDeviceToDevice, stream);

    // iterations and timing are scalars - broadcast to all batches
    std::vector<double> iters_host(batch_size, static_cast<double>(cached->solver->solution.iterations));
    std::vector<double> solve_time_host(batch_size, cached->solver->solution.solve_time);
    std::vector<double> setup_time_host(batch_size, cached->solver->info.setup_time);
    std::vector<double> construction_time_host(batch_size, cached->solver->info.construction_time);
    cudaMemcpyAsync(iterations_out->typed_data(), iters_host.data(),
                    sizeof(double) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(solve_time_out->typed_data(), solve_time_host.data(),
                    sizeof(double) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(setup_time_out->typed_data(), setup_time_host.data(),
                    sizeof(double) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(construction_time_out->typed_data(), construction_time_host.data(),
                    sizeof(double) * batch_size, cudaMemcpyHostToDevice, stream);

    // Solution is automatically cached by CompiledSolver::solve() when enableGrad=true
}

/**
 * @brief Forward solve FFI handler - stateless, uses cached solvers
 *
 * Returns 10 outputs: x, z, s, z_x, status, obj_val, iterations,
 *                     solve_time, setup_time, construction_time.
 * `z_x` is a [batch, total_xn] buffer (zero-length on slack-only problems).
 */
ffi::Error MoreauSolveFwdImpl(
    cudaStream_t stream,
    int64_t n, int64_t m,
    int64_t num_zero, int64_t num_nonneg, int64_t num_exp, int64_t num_soc,
    int64_t num_psd,
    int64_t num_power,
    int64_t num_gen_power,
    int64_t num_dir_cones,
    ffi::Buffer<ffi::S64> P_row_offsets,
    ffi::Buffer<ffi::S64> P_col_indices,
    ffi::Buffer<ffi::S64> A_row_offsets,
    ffi::Buffer<ffi::S64> A_col_indices,
    ffi::Buffer<ffi::S64> soc_dims_buf,
    ffi::Buffer<ffi::S64> psd_dims_buf,
    ffi::Buffer<ffi::F64> power_alphas_buf,
    ffi::Buffer<ffi::F64> gen_power_alphas_buf,
    ffi::Buffer<ffi::S64> gen_power_dim1s_buf,
    ffi::Buffer<ffi::S64> gen_power_dim2s_buf,
    ffi::Buffer<ffi::S64> x_kinds_buf,
    ffi::Buffer<ffi::S64> x_indices_offsets_buf,
    ffi::Buffer<ffi::S64> x_indices_flat_buf,
    ffi::Buffer<ffi::F64> x_power_alphas_buf,
    ffi::Buffer<ffi::S64> x_psd_ks_buf,
    ffi::Buffer<ffi::S64> x_gen_power_dim1s_buf,
    ffi::Buffer<ffi::S64> x_gen_power_dim2s_buf,
    ffi::Buffer<ffi::F64> x_gen_power_alphas_buf,
    ffi::Buffer<ffi::F64> P_data,
    ffi::Buffer<ffi::F64> A_data,
    ffi::Buffer<ffi::F64> q,
    ffi::Buffer<ffi::F64> b,
    ffi::ResultBuffer<ffi::F64> x,
    ffi::ResultBuffer<ffi::F64> z,
    ffi::ResultBuffer<ffi::F64> s,
    ffi::ResultBuffer<ffi::F64> z_x_out,
    ffi::ResultBuffer<ffi::F64> status_out,
    ffi::ResultBuffer<ffi::F64> obj_val_out,
    ffi::ResultBuffer<ffi::F64> iterations_out,
    ffi::ResultBuffer<ffi::F64> solve_time_out,
    ffi::ResultBuffer<ffi::F64> setup_time_out,
    ffi::ResultBuffer<ffi::F64> construction_time_out
) {
    auto q_dims = q.dimensions();
    int64_t batch_size = (q_dims.size() == 2) ? q_dims[0] : 1;

    int64_t nnzP = P_col_indices.element_count();
    int64_t nnzA = A_col_indices.element_count();
    int64_t num_soc_dims = soc_dims_buf.element_count();
    int64_t num_psd_dims = psd_dims_buf.element_count();
    int64_t num_power_alphas = power_alphas_buf.element_count();
    int64_t num_gen_power_alphas = gen_power_alphas_buf.element_count();

    cudaStreamSynchronize(stream);
    XConeHostBuffers xb = read_xcone_metadata(
        num_dir_cones,
        x_kinds_buf.typed_data(),
        x_indices_offsets_buf.typed_data(),
        x_indices_flat_buf.typed_data(), x_indices_flat_buf.element_count(),
        x_power_alphas_buf.typed_data(), x_power_alphas_buf.element_count(),
        x_psd_ks_buf.typed_data(), x_psd_ks_buf.element_count(),
        x_gen_power_dim1s_buf.typed_data(),
        x_gen_power_dim2s_buf.typed_data(), x_gen_power_dim1s_buf.element_count(),
        x_gen_power_alphas_buf.typed_data(), x_gen_power_alphas_buf.element_count()
    );

    CachedSolver* cached = get_or_create_solver(
        stream,
        n, m, batch_size,
        P_row_offsets.typed_data(), P_col_indices.typed_data(), nnzP,
        A_row_offsets.typed_data(), A_col_indices.typed_data(), nnzA,
        num_zero, num_nonneg, num_exp, num_soc,
        soc_dims_buf.typed_data(), num_soc_dims,
        num_psd, psd_dims_buf.typed_data(), num_psd_dims,
        num_power,
        power_alphas_buf.typed_data(), num_power_alphas,
        num_gen_power,
        gen_power_alphas_buf.typed_data(), num_gen_power_alphas,
        gen_power_dim1s_buf.typed_data(), gen_power_dim2s_buf.typed_data(),
        xb,
        true  // enable_grad
    );

    // Note: get_or_create_solver already synchronized `stream` for D2H copies.
    // The solver's cuDSS handle was created on the default stream (0), so
    // setup/solve run on the default stream (as the solver expects).
    cached->solver->setup(P_data.typed_data(), A_data.typed_data());
    cached->solver->solve(q.typed_data(), b.typed_data());

    copy_solution_to_outputs(stream, cached, n, m, batch_size,
        x, z, s, status_out, obj_val_out, iterations_out,
        solve_time_out, setup_time_out, construction_time_out);

    // Emit z_x in the user/original frame. The internal `variables.z_x`
    // is in the equilibrated τ-scaled frame; `unscale_z_x` applies the
    // inverse Ruiz scaling and τ_raw division.
    if (xb.total_xn > 0) {
        moreau::unscale_z_x(
            z_x_out->typed_data(),
            cached->solver->variables.z_x.data(),
            cached->solver->data.equilibration.dinv.data(),
            cached->solver->data.equilibration.c.data(),
            cached->solver->solution.τ_raw.data(),
            cached->solver->data.cones.d_xcone_indices,
            n, xb.total_xn, batch_size, stream);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoreauSolveFwd, MoreauSolveFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("num_zero")
        .Attr<int64_t>("num_nonneg")
        .Attr<int64_t>("num_exp")
        .Attr<int64_t>("num_soc")
        .Attr<int64_t>("num_psd")
        .Attr<int64_t>("num_power")
        .Attr<int64_t>("num_gen_power")
        .Attr<int64_t>("num_dir_cones")
        .Arg<ffi::Buffer<ffi::S64>>()  // P_row_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // P_col_indices
        .Arg<ffi::Buffer<ffi::S64>>()  // A_row_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // A_col_indices
        .Arg<ffi::Buffer<ffi::S64>>()  // soc_dims
        .Arg<ffi::Buffer<ffi::S64>>()  // psd_dims
        .Arg<ffi::Buffer<ffi::F64>>()  // power_alphas
        .Arg<ffi::Buffer<ffi::F64>>()  // gen_power_alphas
        .Arg<ffi::Buffer<ffi::S64>>()  // gen_power_dim1s
        .Arg<ffi::Buffer<ffi::S64>>()  // gen_power_dim2s
        .Arg<ffi::Buffer<ffi::S64>>()  // x_kinds
        .Arg<ffi::Buffer<ffi::S64>>()  // x_indices_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // x_indices_flat
        .Arg<ffi::Buffer<ffi::F64>>()  // x_power_alphas
        .Arg<ffi::Buffer<ffi::S64>>()  // x_psd_ks
        .Arg<ffi::Buffer<ffi::S64>>()  // x_gen_power_dim1s
        .Arg<ffi::Buffer<ffi::S64>>()  // x_gen_power_dim2s
        .Arg<ffi::Buffer<ffi::F64>>()  // x_gen_power_alphas
        .Arg<ffi::Buffer<ffi::F64>>()  // P_data
        .Arg<ffi::Buffer<ffi::F64>>()  // A_data
        .Arg<ffi::Buffer<ffi::F64>>()  // q
        .Arg<ffi::Buffer<ffi::F64>>()  // b
        .Ret<ffi::Buffer<ffi::F64>>()  // x
        .Ret<ffi::Buffer<ffi::F64>>()  // z
        .Ret<ffi::Buffer<ffi::F64>>()  // s
        .Ret<ffi::Buffer<ffi::F64>>()  // z_x
        .Ret<ffi::Buffer<ffi::F64>>()  // status
        .Ret<ffi::Buffer<ffi::F64>>()  // obj_val
        .Ret<ffi::Buffer<ffi::F64>>()  // iterations
        .Ret<ffi::Buffer<ffi::F64>>()  // solve_time
        .Ret<ffi::Buffer<ffi::F64>>()  // setup_time
        .Ret<ffi::Buffer<ffi::F64>>()  // construction_time
);

/**
 * @brief Backward differentiation FFI handler
 *
 * Stateless: accepts P_data, A_data, q, b as explicit arguments (saved from
 * forward pass residuals) and reloads + re-equilibrates the solver before
 * computing backward.  This ensures correctness for chained solves where
 * a subsequent forward pass may have overwritten the cached solver state.
 */
ffi::Error MoreauSolveBwdImpl(
    cudaStream_t stream,
    int64_t n, int64_t m,
    int64_t num_zero, int64_t num_nonneg, int64_t num_exp, int64_t num_soc,
    int64_t num_psd,
    int64_t num_power,
    int64_t num_gen_power,
    int64_t num_dir_cones,
    ffi::Buffer<ffi::S64> P_row_offsets,
    ffi::Buffer<ffi::S64> P_col_indices,
    ffi::Buffer<ffi::S64> A_row_offsets,
    ffi::Buffer<ffi::S64> A_col_indices,
    ffi::Buffer<ffi::S64> soc_dims_buf,
    ffi::Buffer<ffi::S64> psd_dims_buf,
    ffi::Buffer<ffi::F64> power_alphas_buf,
    ffi::Buffer<ffi::F64> gen_power_alphas_buf,
    ffi::Buffer<ffi::S64> gen_power_dim1s_buf,
    ffi::Buffer<ffi::S64> gen_power_dim2s_buf,
    ffi::Buffer<ffi::S64> x_kinds_buf,
    ffi::Buffer<ffi::S64> x_indices_offsets_buf,
    ffi::Buffer<ffi::S64> x_indices_flat_buf,
    ffi::Buffer<ffi::F64> x_power_alphas_buf,
    ffi::Buffer<ffi::S64> x_psd_ks_buf,
    ffi::Buffer<ffi::S64> x_gen_power_dim1s_buf,
    ffi::Buffer<ffi::S64> x_gen_power_dim2s_buf,
    ffi::Buffer<ffi::F64> x_gen_power_alphas_buf,
    ffi::Buffer<ffi::F64> P_data,
    ffi::Buffer<ffi::F64> A_data,
    ffi::Buffer<ffi::F64> q_data,
    ffi::Buffer<ffi::F64> b_data,
    ffi::Buffer<ffi::F64> dx,
    ffi::Buffer<ffi::F64> dz,
    ffi::Buffer<ffi::F64> ds,
    ffi::Buffer<ffi::F64> dz_x,
    ffi::Buffer<ffi::F64> x,
    ffi::Buffer<ffi::F64> z_in,
    ffi::Buffer<ffi::F64> s_in,
    ffi::Buffer<ffi::F64> z_x_in,
    ffi::ResultBuffer<ffi::F64> dP,
    ffi::ResultBuffer<ffi::F64> dA,
    ffi::ResultBuffer<ffi::F64> dq,
    ffi::ResultBuffer<ffi::F64> db
) {
    // Determine batch size from dx shape
    auto dx_dims = dx.dimensions();
    int64_t batch_size = (dx_dims.size() == 2) ? dx_dims[0] : 1;

    int64_t nnzP = P_col_indices.element_count();
    int64_t nnzA = A_col_indices.element_count();

    // Sync XLA's stream before D2H copies for cache key lookup
    cudaStreamSynchronize(stream);

    // Copy SOC dims from device for cache key lookup
    int64_t num_soc_dims = soc_dims_buf.element_count();
    std::vector<int64_t> soc_dims_host;
    if (num_soc_dims > 0) {
        soc_dims_host.resize(num_soc_dims);
        cudaMemcpy(soc_dims_host.data(), soc_dims_buf.typed_data(),
                   num_soc_dims * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }

    // Copy PSD dims from device for cache key lookup
    int64_t num_psd_dims = psd_dims_buf.element_count();
    std::vector<int64_t> psd_dims_host;
    if (num_psd_dims > 0) {
        psd_dims_host.resize(num_psd_dims);
        cudaMemcpy(psd_dims_host.data(), psd_dims_buf.typed_data(),
                   num_psd_dims * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }

    // Copy power cone alphas from device for cache key lookup
    std::vector<double> power_alphas_host;
    int64_t num_power_alphas = power_alphas_buf.element_count();
    if (num_power_alphas > 0) {
        power_alphas_host.resize(num_power_alphas);
        cudaMemcpy(power_alphas_host.data(), power_alphas_buf.typed_data(),
                   num_power_alphas * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Copy GenPowerCone dims and alphas from device for cache key lookup
    std::vector<int64_t> gen_power_dim1s_host, gen_power_dim2s_host;
    std::vector<double> gen_power_alphas_host;
    if (num_gen_power > 0) {
        gen_power_dim1s_host.resize(num_gen_power);
        gen_power_dim2s_host.resize(num_gen_power);
        cudaMemcpy(gen_power_dim1s_host.data(), gen_power_dim1s_buf.typed_data(),
                   num_gen_power * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gen_power_dim2s_host.data(), gen_power_dim2s_buf.typed_data(),
                   num_gen_power * sizeof(int64_t), cudaMemcpyDeviceToHost);
        int64_t num_gen_power_alphas = gen_power_alphas_buf.element_count();
        if (num_gen_power_alphas > 0) {
            gen_power_alphas_host.resize(num_gen_power_alphas);
            cudaMemcpy(gen_power_alphas_host.data(), gen_power_alphas_buf.typed_data(),
                       num_gen_power_alphas * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }

    XConeHostBuffers xb = read_xcone_metadata(
        num_dir_cones,
        x_kinds_buf.typed_data(),
        x_indices_offsets_buf.typed_data(),
        x_indices_flat_buf.typed_data(), x_indices_flat_buf.element_count(),
        x_power_alphas_buf.typed_data(), x_power_alphas_buf.element_count(),
        x_psd_ks_buf.typed_data(), x_psd_ks_buf.element_count(),
        x_gen_power_dim1s_buf.typed_data(),
        x_gen_power_dim2s_buf.typed_data(), x_gen_power_dim1s_buf.element_count(),
        x_gen_power_alphas_buf.typed_data(), x_gen_power_alphas_buf.element_count()
    );

    // Get cached solver (must exist from forward pass)
    SolverKey key{n, m, batch_size, nnzP, nnzA, num_zero, num_nonneg, num_exp, num_soc, num_psd,
                  soc_dims_host, psd_dims_host,
                  num_power, power_alphas_host,
                  num_gen_power, gen_power_dim1s_host, gen_power_dim2s_host, gen_power_alphas_host,
                  xb.num_dir_cones, xb.x_kinds, xb.x_indices_offsets, xb.x_indices_flat,
                  xb.x_power_alphas, xb.x_psd_ks,
                  xb.x_gen_power_dim1s, xb.x_gen_power_dim2s, xb.x_gen_power_alphas_flat};

    CachedSolver* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_solver_cache.find(key);
        if (it == g_solver_cache.end()) {
            return ffi::Error::InvalidArgument("Solver not found - forward pass must be called first");
        }
        cached = it->second.get();
    }

    auto* diff_state = cached->solver->diff_state();
    if (!diff_state) {
        return ffi::Error::InvalidArgument("DiffState not available");
    }

    // Reload problem data and re-equilibrate from saved residuals.
    // This ensures the solver's internal P, A, q, b and equilibration
    // factors match this specific forward pass, not a later one.
    cached->solver->loadDataForBackward(
        P_data.typed_data(), A_data.typed_data(),
        q_data.typed_data(), b_data.typed_data(),
        stream
    );

    // Restore solution state from saved tensors
    cudaMemcpyAsync(diff_state->x.data(), x.typed_data(),
                    sizeof(double) * n * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(diff_state->z.data(), z_in.typed_data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(diff_state->s.data(), s_in.typed_data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);

    // Sync equilibration factors from solver data to DiffState.
    // loadDataForBackward re-equilibrated solver.data.equilibration, but
    // backward() reads from diff_state->{d,dinv,e,einv,c_scale}.
    auto& eq = cached->solver->data.equilibration;
    cudaMemcpyAsync(diff_state->d.data(), eq.d.data(),
                    sizeof(double) * n * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(diff_state->dinv.data(), eq.dinv.data(),
                    sizeof(double) * n * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(diff_state->e.data(), eq.e.data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(diff_state->einv.data(), eq.einv.data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(diff_state->c_scale.data(), eq.c.data(),
                    sizeof(double) * batch_size, cudaMemcpyDeviceToDevice, stream);

    // Direct-x dual: convert saved user-frame z_x back to the equilibrated
    // τ=1 frame the IPM/diff machinery expects (inverse of `unscale_z_x`).
    if (xb.total_xn > 0) {
        equilibrate_z_x(
            diff_state->z_x.data(), z_x_in.typed_data(),
            cached->solver->data.equilibration.dinv.data(),
            cached->solver->data.equilibration.c.data(),
            cached->solver->data.cones.d_xcone_indices,
            n, xb.total_xn, batch_size, stream);
    }

    // Synchronize before computing derived quantities
    cudaStreamSynchronize(stream);

    // Set tau = 1.0 (backward() uses state.tau for HSDE coefficients)
    diff_state->tau.setToConstant(1.0, stream);

    // Note: u, pi_u, and cone derivatives are computed internally by backward()
    // via equilibrated workspace vectors - no need to compute them here.

    // Create views for upstream gradients
    BatchedVector dx_bar(const_cast<double*>(dx.typed_data()), n, batch_size);
    BatchedVector dz_bar(const_cast<double*>(dz.typed_data()), m, batch_size);
    BatchedVector ds_bar(const_cast<double*>(ds.typed_data()), m, batch_size);

    // Compute backward pass — route through the direct-x-aware overload
    // when the user supplied a non-trivial dz_x, else the simpler call.
    if (xb.total_xn > 0) {
        BatchedVector dz_x_bar(const_cast<double*>(dz_x.typed_data()),
                               xb.total_xn, batch_size);
        moreau::backward_with_dz_x(*diff_state, dx_bar, dz_bar, ds_bar,
                                   &dz_x_bar, *cached->solver, stream);
    } else {
        moreau::backward(*diff_state, dx_bar, dz_bar, ds_bar, *cached->solver, stream);
    }

    // Copy gradients to output buffers
    cudaMemcpyAsync(dP->typed_data(), diff_state->dP_values.data(),
                    sizeof(double) * nnzP * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dA->typed_data(), diff_state->dA_values.data(),
                    sizeof(double) * nnzA * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dq->typed_data(), diff_state->dq.data(),
                    sizeof(double) * n * batch_size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(db->typed_data(), diff_state->db.data(),
                    sizeof(double) * m * batch_size, cudaMemcpyDeviceToDevice, stream);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoreauSolveBwd, MoreauSolveBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("num_zero")
        .Attr<int64_t>("num_nonneg")
        .Attr<int64_t>("num_exp")
        .Attr<int64_t>("num_soc")
        .Attr<int64_t>("num_psd")
        .Attr<int64_t>("num_power")
        .Attr<int64_t>("num_gen_power")
        .Attr<int64_t>("num_dir_cones")
        .Arg<ffi::Buffer<ffi::S64>>()  // P_row_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // P_col_indices
        .Arg<ffi::Buffer<ffi::S64>>()  // A_row_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // A_col_indices
        .Arg<ffi::Buffer<ffi::S64>>()  // soc_dims
        .Arg<ffi::Buffer<ffi::S64>>()  // psd_dims
        .Arg<ffi::Buffer<ffi::F64>>()  // power_alphas
        .Arg<ffi::Buffer<ffi::F64>>()  // gen_power_alphas
        .Arg<ffi::Buffer<ffi::S64>>()  // gen_power_dim1s
        .Arg<ffi::Buffer<ffi::S64>>()  // gen_power_dim2s
        .Arg<ffi::Buffer<ffi::S64>>()  // x_kinds
        .Arg<ffi::Buffer<ffi::S64>>()  // x_indices_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // x_indices_flat
        .Arg<ffi::Buffer<ffi::F64>>()  // x_power_alphas
        .Arg<ffi::Buffer<ffi::S64>>()  // x_psd_ks
        .Arg<ffi::Buffer<ffi::S64>>()  // x_gen_power_dim1s
        .Arg<ffi::Buffer<ffi::S64>>()  // x_gen_power_dim2s
        .Arg<ffi::Buffer<ffi::F64>>()  // x_gen_power_alphas
        .Arg<ffi::Buffer<ffi::F64>>()  // P_data (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // A_data (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // q (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // b (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // dx
        .Arg<ffi::Buffer<ffi::F64>>()  // dz
        .Arg<ffi::Buffer<ffi::F64>>()  // ds
        .Arg<ffi::Buffer<ffi::F64>>()  // dz_x
        .Arg<ffi::Buffer<ffi::F64>>()  // x (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // z (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // s (saved)
        .Arg<ffi::Buffer<ffi::F64>>()  // z_x (saved)
        .Ret<ffi::Buffer<ffi::F64>>()  // dP
        .Ret<ffi::Buffer<ffi::F64>>()  // dA
        .Ret<ffi::Buffer<ffi::F64>>()  // dq
        .Ret<ffi::Buffer<ffi::F64>>()  // db
);

/**
 * @brief Forward solve FFI handler WITH warm start
 *
 * Same as MoreauSolveFwdImpl but takes 3 additional warm-start buffers
 * (warm_x, warm_z, warm_s) and calls the warm-start overload of solve().
 *
 * Returns 9 outputs: x, z, s, status, obj_val, iterations,
 *                     solve_time, setup_time, construction_time
 */
ffi::Error MoreauSolveFwdWarmImpl(
    cudaStream_t stream,
    int64_t n, int64_t m,
    int64_t num_zero, int64_t num_nonneg, int64_t num_exp, int64_t num_soc,
    int64_t num_psd,
    int64_t num_power,
    int64_t num_gen_power,
    int64_t num_dir_cones,
    ffi::Buffer<ffi::S64> P_row_offsets,
    ffi::Buffer<ffi::S64> P_col_indices,
    ffi::Buffer<ffi::S64> A_row_offsets,
    ffi::Buffer<ffi::S64> A_col_indices,
    ffi::Buffer<ffi::S64> soc_dims_buf,
    ffi::Buffer<ffi::S64> psd_dims_buf,
    ffi::Buffer<ffi::F64> power_alphas_buf,
    ffi::Buffer<ffi::F64> gen_power_alphas_buf,
    ffi::Buffer<ffi::S64> gen_power_dim1s_buf,
    ffi::Buffer<ffi::S64> gen_power_dim2s_buf,
    ffi::Buffer<ffi::S64> x_kinds_buf,
    ffi::Buffer<ffi::S64> x_indices_offsets_buf,
    ffi::Buffer<ffi::S64> x_indices_flat_buf,
    ffi::Buffer<ffi::F64> x_power_alphas_buf,
    ffi::Buffer<ffi::S64> x_psd_ks_buf,
    ffi::Buffer<ffi::S64> x_gen_power_dim1s_buf,
    ffi::Buffer<ffi::S64> x_gen_power_dim2s_buf,
    ffi::Buffer<ffi::F64> x_gen_power_alphas_buf,
    ffi::Buffer<ffi::F64> P_data,
    ffi::Buffer<ffi::F64> A_data,
    ffi::Buffer<ffi::F64> q,
    ffi::Buffer<ffi::F64> b,
    ffi::Buffer<ffi::F64> warm_x,
    ffi::Buffer<ffi::F64> warm_z,
    ffi::Buffer<ffi::F64> warm_s,
    ffi::Buffer<ffi::F64> warm_z_x,
    ffi::ResultBuffer<ffi::F64> x,
    ffi::ResultBuffer<ffi::F64> z,
    ffi::ResultBuffer<ffi::F64> s,
    ffi::ResultBuffer<ffi::F64> z_x_out,
    ffi::ResultBuffer<ffi::F64> status_out,
    ffi::ResultBuffer<ffi::F64> obj_val_out,
    ffi::ResultBuffer<ffi::F64> iterations_out,
    ffi::ResultBuffer<ffi::F64> solve_time_out,
    ffi::ResultBuffer<ffi::F64> setup_time_out,
    ffi::ResultBuffer<ffi::F64> construction_time_out
) {
    auto q_dims = q.dimensions();
    int64_t batch_size = (q_dims.size() == 2) ? q_dims[0] : 1;

    int64_t nnzP = P_col_indices.element_count();
    int64_t nnzA = A_col_indices.element_count();
    int64_t num_soc_dims = soc_dims_buf.element_count();
    int64_t num_psd_dims = psd_dims_buf.element_count();
    int64_t num_power_alphas = power_alphas_buf.element_count();
    int64_t num_gen_power_alphas = gen_power_alphas_buf.element_count();

    cudaStreamSynchronize(stream);
    XConeHostBuffers xb = read_xcone_metadata(
        num_dir_cones,
        x_kinds_buf.typed_data(),
        x_indices_offsets_buf.typed_data(),
        x_indices_flat_buf.typed_data(), x_indices_flat_buf.element_count(),
        x_power_alphas_buf.typed_data(), x_power_alphas_buf.element_count(),
        x_psd_ks_buf.typed_data(), x_psd_ks_buf.element_count(),
        x_gen_power_dim1s_buf.typed_data(),
        x_gen_power_dim2s_buf.typed_data(), x_gen_power_dim1s_buf.element_count(),
        x_gen_power_alphas_buf.typed_data(), x_gen_power_alphas_buf.element_count()
    );

    CachedSolver* cached = get_or_create_solver(
        stream,
        n, m, batch_size,
        P_row_offsets.typed_data(), P_col_indices.typed_data(), nnzP,
        A_row_offsets.typed_data(), A_col_indices.typed_data(), nnzA,
        num_zero, num_nonneg, num_exp, num_soc,
        soc_dims_buf.typed_data(), num_soc_dims,
        num_psd, psd_dims_buf.typed_data(), num_psd_dims,
        num_power,
        power_alphas_buf.typed_data(), num_power_alphas,
        num_gen_power,
        gen_power_alphas_buf.typed_data(), num_gen_power_alphas,
        gen_power_dim1s_buf.typed_data(), gen_power_dim2s_buf.typed_data(),
        xb,
        true  // enable_grad
    );

    // Note: get_or_create_solver already synchronized `stream`.
    cached->solver->setup(P_data.typed_data(), A_data.typed_data());
    cached->solver->solve(
        q.typed_data(), b.typed_data(),
        warm_x.typed_data(), warm_z.typed_data(), warm_s.typed_data(),
        /*stream=*/0,
        xb.total_xn > 0 ? warm_z_x.typed_data() : nullptr
    );

    copy_solution_to_outputs(stream, cached, n, m, batch_size,
        x, z, s, status_out, obj_val_out, iterations_out,
        solve_time_out, setup_time_out, construction_time_out);

    if (xb.total_xn > 0) {
        moreau::unscale_z_x(
            z_x_out->typed_data(),
            cached->solver->variables.z_x.data(),
            cached->solver->data.equilibration.dinv.data(),
            cached->solver->data.equilibration.c.data(),
            cached->solver->solution.τ_raw.data(),
            cached->solver->data.cones.d_xcone_indices,
            n, xb.total_xn, batch_size, stream);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoreauSolveFwdWarm, MoreauSolveFwdWarmImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("num_zero")
        .Attr<int64_t>("num_nonneg")
        .Attr<int64_t>("num_exp")
        .Attr<int64_t>("num_soc")
        .Attr<int64_t>("num_psd")
        .Attr<int64_t>("num_power")
        .Attr<int64_t>("num_gen_power")
        .Attr<int64_t>("num_dir_cones")
        .Arg<ffi::Buffer<ffi::S64>>()  // P_row_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // P_col_indices
        .Arg<ffi::Buffer<ffi::S64>>()  // A_row_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // A_col_indices
        .Arg<ffi::Buffer<ffi::S64>>()  // soc_dims
        .Arg<ffi::Buffer<ffi::S64>>()  // psd_dims
        .Arg<ffi::Buffer<ffi::F64>>()  // power_alphas
        .Arg<ffi::Buffer<ffi::F64>>()  // gen_power_alphas
        .Arg<ffi::Buffer<ffi::S64>>()  // gen_power_dim1s
        .Arg<ffi::Buffer<ffi::S64>>()  // gen_power_dim2s
        .Arg<ffi::Buffer<ffi::S64>>()  // x_kinds
        .Arg<ffi::Buffer<ffi::S64>>()  // x_indices_offsets
        .Arg<ffi::Buffer<ffi::S64>>()  // x_indices_flat
        .Arg<ffi::Buffer<ffi::F64>>()  // x_power_alphas
        .Arg<ffi::Buffer<ffi::S64>>()  // x_psd_ks
        .Arg<ffi::Buffer<ffi::S64>>()  // x_gen_power_dim1s
        .Arg<ffi::Buffer<ffi::S64>>()  // x_gen_power_dim2s
        .Arg<ffi::Buffer<ffi::F64>>()  // x_gen_power_alphas
        .Arg<ffi::Buffer<ffi::F64>>()  // P_data
        .Arg<ffi::Buffer<ffi::F64>>()  // A_data
        .Arg<ffi::Buffer<ffi::F64>>()  // q
        .Arg<ffi::Buffer<ffi::F64>>()  // b
        .Arg<ffi::Buffer<ffi::F64>>()  // warm_x
        .Arg<ffi::Buffer<ffi::F64>>()  // warm_z
        .Arg<ffi::Buffer<ffi::F64>>()  // warm_s
        .Arg<ffi::Buffer<ffi::F64>>()  // warm_z_x
        .Ret<ffi::Buffer<ffi::F64>>()  // x
        .Ret<ffi::Buffer<ffi::F64>>()  // z
        .Ret<ffi::Buffer<ffi::F64>>()  // s
        .Ret<ffi::Buffer<ffi::F64>>()  // z_x
        .Ret<ffi::Buffer<ffi::F64>>()  // status
        .Ret<ffi::Buffer<ffi::F64>>()  // obj_val
        .Ret<ffi::Buffer<ffi::F64>>()  // iterations
        .Ret<ffi::Buffer<ffi::F64>>()  // solve_time
        .Ret<ffi::Buffer<ffi::F64>>()  // setup_time
        .Ret<ffi::Buffer<ffi::F64>>()  // construction_time
);

// C-linkage for clearing cache (useful for testing)
extern "C" {
void moreau_jax_clear_cache() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_solver_cache.clear();
}
}
