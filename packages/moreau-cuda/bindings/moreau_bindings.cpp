/**
 * @file moreau_bindings.cpp
 * @brief Python bindings for Moreau solver using nanobind
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <limits>

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/status.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/diff/diff_kernels.cuh"
#include "moreau/vector/vector_kernels.cuh"
#include "moreau/chordal/chordal_info.hpp"

namespace nb = nanobind;
using namespace moreau;

// CUDA error checking helper - throws on failure
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + #call + ": " + cudaGetErrorString(err)); \
    } \
} while(0)

/**
 * @brief RAII wrapper for CUDA device memory
 *
 * Ensures device memory is freed even if an exception is thrown.
 */
class CudaDeviceMemory {
public:
    CudaDeviceMemory() : ptr_(nullptr) {}

    explicit CudaDeviceMemory(size_t bytes) : ptr_(nullptr) {
        if (bytes > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes));
        }
    }

    ~CudaDeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move only, no copy
    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    double* get() const { return static_cast<double*>(ptr_); }
    void* ptr() const { return ptr_; }

private:
    void* ptr_;
};

// Type aliases for cleaner signatures
template<typename T>
using input_array = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

// Helper to create numpy arrays with proper ownership via capsule
template<typename T>
nb::ndarray<nb::numpy, T> make_numpy_array_1d(const T* src_ptr, size_t size, bool from_device = true) {
    T* data = new T[size];
    if (from_device) {
        CUDA_CHECK(cudaMemcpy(data, src_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(data, src_ptr, sizeof(T) * size);
    }
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[1] = {size};
    return nb::ndarray<nb::numpy, T>(data, 1, shape, owner);
}

template<typename T>
nb::ndarray<nb::numpy, T> make_numpy_array_2d(const T* src_ptr, size_t dim0, size_t dim1, bool from_device = true) {
    size_t total = dim0 * dim1;
    T* data = new T[total];
    if (from_device) {
        CUDA_CHECK(cudaMemcpy(data, src_ptr, sizeof(T) * total, cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(data, src_ptr, sizeof(T) * total);
    }
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[2] = {dim0, dim1};
    return nb::ndarray<nb::numpy, T>(data, 2, shape, owner);
}

// Helper to make int32 array from device int32_t data (for status arrays)
nb::ndarray<nb::numpy, int32_t> make_status_array(const int32_t* src_ptr, size_t size) {
    int32_t* data = new int32_t[size];
    CUDA_CHECK(cudaMemcpy(data, src_ptr, sizeof(int32_t) * size, cudaMemcpyDeviceToHost));
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
    size_t shape[1] = {size};
    return nb::ndarray<nb::numpy, int32_t>(data, 1, shape, owner);
}

/**
 * @brief Python-friendly wrapper for Moreau solver
 *
 * Handles numpy array conversions and GPU memory management.
 */
class PyMoreauSolver {
private:
    std::unique_ptr<CompiledSolver> solver_;

    int64_t n_, m_, batchSize_;
    int64_t nnzP_, nnzA_;

    // Original (user-facing) dimensions — same as n_/m_ unless chordal is active
    int64_t n_user_, m_user_;
    int64_t nnzP_user_, nnzA_user_;

    // Chordal decomposition (null if no decomposition or no PSD cones)
    std::unique_ptr<ChordalInfo> chordal_;

    int device_id_;

    // Validate shapes of numpy inputs for solve (uses user-facing dimensions)
    void validate_solve_numpy_inputs(
        const input_array<double>& P_values,
        const input_array<double>& A_values,
        const input_array<double>& q,
        const input_array<double>& b
    ) {
        if (q.ndim() != 2 || static_cast<int64_t>(q.shape(0)) != batchSize_ || static_cast<int64_t>(q.shape(1)) != n_user_) {
            throw std::invalid_argument("q must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(n_user_) + ")");
        }
        if (b.ndim() != 2 || static_cast<int64_t>(b.shape(0)) != batchSize_ || static_cast<int64_t>(b.shape(1)) != m_user_) {
            throw std::invalid_argument("b must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(m_user_) + ")");
        }
        if (nnzP_user_ > 0) {
            if (P_values.ndim() != 2 || static_cast<int64_t>(P_values.shape(0)) != batchSize_ || static_cast<int64_t>(P_values.shape(1)) != nnzP_user_) {
                throw std::invalid_argument("P_values must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(nnzP_user_) + ")");
            }
        }
        if (nnzA_user_ > 0) {
            if (A_values.ndim() != 2 || static_cast<int64_t>(A_values.shape(0)) != batchSize_ || static_cast<int64_t>(A_values.shape(1)) != nnzA_user_) {
                throw std::invalid_argument("A_values must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(nnzA_user_) + ")");
            }
        }
    }

    // Build numpy result dict from solver state after solve.
    // When include_tau_kappa is true, includes tau, kappa, and dual_obj_val.
    nb::dict make_solve_result_dict(bool include_tau_kappa = true) {
        cudaDeviceSynchronize();

        auto x_result = make_numpy_array_2d<double>(solver_->solution.x.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(n_));
        auto s_result = make_numpy_array_2d<double>(solver_->solution.s.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(m_));
        auto z_result = make_numpy_array_2d<double>(solver_->solution.z.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(m_));
        auto status_result = make_status_array(solver_->info.status_device,
            static_cast<size_t>(batchSize_));

        std::vector<double> cost_primal(batchSize_);
        solver_->solution.obj_val.gpuToCpu(cost_primal.data());

        auto iterations_result = make_numpy_array_1d<int32_t>(
            solver_->info.iterations_per_batch.data(), static_cast<size_t>(batchSize_), false);

        nb::dict result;
        result["x"] = x_result;
        result["s"] = s_result;
        result["z"] = z_result;
        result["status"] = status_result;

        // Direct-x cone duals in user (original) frame. Empty array when
        // the problem has no direct-x cones.
        int64_t total_xn = solver_->variables.totalXConeNumel();
        if (total_xn > 0) {
            // Allocate a temporary device buffer + unscale, then copy to numpy.
            CudaDeviceMemory d_z_x_user(sizeof(double) * total_xn * batchSize_);
            const auto& cones = solver_->data.cones;
            // dinv / c_scale / τ_raw read from solver state.
            moreau::unscale_z_x(
                reinterpret_cast<double*>(d_z_x_user.get()),
                solver_->variables.z_x.data(),
                solver_->data.equilibration.dinv.data(),
                solver_->data.equilibration.c.data(),
                solver_->solution.τ_raw.data(),
                cones.d_xcone_indices,
                n_, total_xn, batchSize_, 0
            );
            cudaDeviceSynchronize();
            result["z_x"] = make_numpy_array_2d<double>(
                reinterpret_cast<double*>(d_z_x_user.get()),
                static_cast<size_t>(batchSize_), static_cast<size_t>(total_xn));
        } else {
            // Return shape (batch_size, 0)
            result["z_x"] = make_numpy_array_2d<double>(
                /*data=*/nullptr,
                static_cast<size_t>(batchSize_), 0);
        }

        if (include_tau_kappa) {
            result["tau"] = make_numpy_array_1d<double>(solver_->solution.τ_raw.data(),
                static_cast<size_t>(batchSize_));
            result["kappa"] = make_numpy_array_1d<double>(solver_->solution.κ_raw.data(),
                static_cast<size_t>(batchSize_));

            std::vector<double> cost_dual(batchSize_);
            solver_->solution.obj_val_dual.gpuToCpu(cost_dual.data());

            if (batchSize_ == 1) {
                result["iterations"] = solver_->info.iterations_per_batch[0];
                result["obj_val"] = cost_primal[0];
                result["dual_obj_val"] = cost_dual[0];
            } else {
                result["iterations"] = iterations_result;
                result["obj_val"] = cost_primal;
                result["dual_obj_val"] = cost_dual;
            }
        } else {
            if (batchSize_ == 1) {
                result["obj_val"] = cost_primal[0];
                result["iterations"] = solver_->info.iterations_per_batch[0];
            } else {
                result["obj_val"] = cost_primal;
                result["iterations"] = iterations_result;
            }
        }

        result["construction_time"] = solver_->info.construction_time;
        result["setup_time"] = solver_->info.setup_time;
        result["solve_time"] = solver_->info.solve_time;

        return result;
    }

    // Copy solution to device output pointers (zero-copy output path)
    void copy_solution_to_device_output(
        uintptr_t x_out_ptr, uintptr_t z_out_ptr, uintptr_t s_out_ptr,
        uintptr_t status_out_ptr, uintptr_t obj_out_ptr,
        uintptr_t z_x_out_ptr = 0
    ) {
        cudaMemcpy(reinterpret_cast<void*>(x_out_ptr), solver_->solution.x.data(),
                   sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(z_out_ptr), solver_->solution.z.data(),
                   sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(s_out_ptr), solver_->solution.s.data(),
                   sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(status_out_ptr), solver_->solution.status.get(),
                   sizeof(int32_t) * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(obj_out_ptr), solver_->info.cost_primal.data(),
                   sizeof(double) * batchSize_, cudaMemcpyDeviceToDevice);
        // Direct-x cone duals in user/original frame, when requested.
        int64_t total_xn = solver_->variables.totalXConeNumel();
        if (z_x_out_ptr != 0 && total_xn > 0) {
            const auto& cones = solver_->data.cones;
            moreau::unscale_z_x(
                reinterpret_cast<double*>(z_x_out_ptr),
                solver_->variables.z_x.data(),
                solver_->data.equilibration.dinv.data(),
                solver_->data.equilibration.c.data(),
                solver_->solution.τ_raw.data(),
                cones.d_xcone_indices,
                n_, total_xn, batchSize_, 0
            );
        }
    }

public:
    /**
     * @brief Construct solver from problem structure
     */
    PyMoreauSolver(
        int64_t n, int64_t m, int64_t batchSize,
        input_array<int64_t> P_row_offsets,
        input_array<int64_t> P_col_indices,
        input_array<int64_t> A_row_offsets,
        input_array<int64_t> A_col_indices,
        const Cones& cones,
        Settings* settings_ptr = nullptr,
        bool enable_grad = false,
        std::optional<std::vector<bool>> b_sparsity_pattern = std::nullopt
    ) : n_(n), m_(m), batchSize_(batchSize),
        n_user_(n), m_user_(m),
        device_id_(settings_ptr ? settings_ptr->deviceId : -1) {
        // Use provided settings or default
        Settings settings = settings_ptr ? *settings_ptr : Settings{};
        settings.enableGrad = enable_grad;

        // Validate dimensions
        if (P_row_offsets.size() != static_cast<size_t>(n + 1)) {
            throw std::invalid_argument("P_row_offsets must have size n+1");
        }
        if (A_row_offsets.size() != static_cast<size_t>(m + 1)) {
            throw std::invalid_argument("A_row_offsets must have size m+1");
        }

        nnzP_ = P_col_indices.size();
        nnzA_ = A_col_indices.size();
        nnzP_user_ = nnzP_;
        nnzA_user_ = nnzA_;

        // Get pointers
        const int64_t* P_ro = P_row_offsets.data();
        const int64_t* P_ci = P_col_indices.data();
        const int64_t* A_ro = A_row_offsets.data();
        const int64_t* A_ci = A_col_indices.data();

        // Chordal decomposition: check if PSD cones have exploitable sparsity
        // If so, construct the solver with augmented dimensions.
        Cones effective_cones = cones;
        std::vector<int64_t> aug_P_ro_storage, aug_P_ci_storage;
        std::vector<int64_t> aug_A_ro_storage, aug_A_ci_storage;

        if (!cones.psdConeDims.empty()) {
            // Convert CSR A to CSC for chordal analysis
            // CSC of A(m×n): colptr[n+1], rowind[nnz]
            std::vector<int64_t> A_csc_colptr(n + 1, 0);
            std::vector<int64_t> A_csc_rowind(nnzA_);

            // Count entries per column
            for (int64_t i = 0; i < nnzA_; i++) A_csc_colptr[A_ci[i] + 1]++;
            for (int64_t j = 0; j < n; j++) A_csc_colptr[j + 1] += A_csc_colptr[j];

            // Fill row indices, recording CSR→CSC mapping
            std::vector<int64_t> csr_to_csc(nnzA_);
            std::vector<int64_t> col_pos(n, 0);
            for (int64_t row = 0; row < m; row++) {
                for (int64_t idx = A_ro[row]; idx < A_ro[row + 1]; idx++) {
                    int64_t col = A_ci[idx];
                    int64_t dest = A_csc_colptr[col] + col_pos[col]++;
                    A_csc_rowind[dest] = row;
                    csr_to_csc[idx] = dest;
                }
            }

            // Compute total SOC dim for analyze()
            int64_t total_soc = 0;
            for (auto d : cones.socConeDims) total_soc += d;

            // std::vector<bool> stores packed bits — convert to contiguous bool array
            std::unique_ptr<bool[]> b_mask_storage;
            if (b_sparsity_pattern) {
                b_mask_storage = std::make_unique<bool[]>(m);
                for (int64_t i = 0; i < m; ++i) {
                    b_mask_storage[i] = (*b_sparsity_pattern)[i];
                }
            }

            // NOTE: settings.ipm.chordalDecompositionMergeMethod is plumbed
            // through Python -> _CudaSettings but is currently NOT honored
            // on CUDA — chordal_info.cpp hardcodes a fill_in==0 parent-child
            // merge regardless. Wiring it through requires an adapter between
            // moreau::SuperNodeTree (the type analyse_psd_sparsity uses) and
            // moreau::chordal::SuperNodeTree (the data-only struct the
            // strategy classes in merge_strategy.hpp operate on); the two
            // have parallel-but-incompatible VertexSet implementations. The
            // banded-PSD parity test in test_psd_cpu_cuda_parity.py is
            // marked xfail until that adapter lands. #176 stays open.
            auto chordal = std::make_unique<ChordalInfo>(ChordalInfo::analyze(
                A_csc_colptr.data(), A_csc_rowind.data(), n, m,
                cones.psdConeDims.data(), cones.psdConeDims.size(),
                cones.numZeroCones, cones.numNonnegCones,
                total_soc, cones.numExpCones, cones.numPowerCones,
                b_mask_storage.get()
            ));

            if (chordal->is_decomposed()) {
                // Convert augmented CSC A back to CSR
                int64_t n_aug = chordal->n_aug;
                int64_t m_aug = chordal->m_aug;
                int64_t nnzA_aug = chordal->A_aug_rowind.size();

                // CSC→CSR: transpose the CSC structure, recording the permutation
                aug_A_ro_storage.resize(m_aug + 1, 0);
                aug_A_ci_storage.resize(nnzA_aug);
                std::vector<int64_t> csc_to_csr(nnzA_aug);  // maps CSC index -> CSR index
                for (int64_t i = 0; i < nnzA_aug; i++) aug_A_ro_storage[chordal->A_aug_rowind[i] + 1]++;
                for (int64_t i = 0; i < m_aug; i++) aug_A_ro_storage[i + 1] += aug_A_ro_storage[i];
                std::vector<int64_t> row_pos(m_aug, 0);
                for (int64_t col = 0; col < n_aug; col++) {
                    for (int64_t idx = chordal->A_aug_colptr[col]; idx < chordal->A_aug_colptr[col + 1]; idx++) {
                        int64_t row = chordal->A_aug_rowind[idx];
                        int64_t dest = aug_A_ro_storage[row] + row_pos[row]++;
                        aug_A_ci_storage[dest] = col;
                        csc_to_csr[idx] = dest;
                    }
                }

                // Reorder A_aug_values from CSC to CSR order (solver expects CSR)
                {
                    std::vector<double> A_aug_csr(nnzA_aug);
                    for (int64_t i = 0; i < nnzA_aug; i++) {
                        A_aug_csr[csc_to_csr[i]] = chordal->A_aug_values[i];
                    }
                    chordal->A_aug_values = std::move(A_aug_csr);
                }

                // A_orig_to_aug currently maps: original CSC index -> augmented CSC index
                // Apply csc_to_csr to get: original CSC index -> augmented CSR index
                for (auto& pos : chordal->A_orig_to_aug) {
                    pos = csc_to_csr[pos];
                }

                // Compose with csr_to_csc to get: user CSR index -> augmented CSR index
                // A_orig_to_aug[csc_idx] = aug_csr_idx
                // We want: user_csr_idx -> aug_csr_idx = A_orig_to_aug[csr_to_csc[user_csr_idx]]
                {
                    int64_t A_nnz_user = chordal->A_nnz_orig;
                    std::vector<int64_t> user_to_aug(A_nnz_user);
                    for (int64_t k = 0; k < A_nnz_user; ++k) {
                        user_to_aug[k] = chordal->A_orig_to_aug[csr_to_csc[k]];
                    }
                    chordal->A_orig_to_aug = std::move(user_to_aug);
                }

                // Augmented P: original P block + empty columns for overlap variables
                // P is symmetric n_orig × n_orig; augmented P is n_aug × n_aug with
                // the extra overlap columns/rows having no entries (zero block).
                if (nnzP_ > 0) {
                    // Build augmented P CSC from original P CSR input
                    int64_t n_orig = chordal->n_orig;
                    chordal->P_nnz_orig = nnzP_;

                    // Convert original P from CSR to CSC, recording mapping
                    chordal->P_aug_colptr.assign(n_aug + 1, 0);
                    chordal->P_aug_rowind.resize(nnzP_);
                    std::vector<int64_t> p_csr_to_csc(nnzP_);

                    // Count entries per column
                    for (int64_t i = 0; i < nnzP_; i++) chordal->P_aug_colptr[P_ci[i] + 1]++;
                    for (int64_t j = 0; j < n_aug; j++) chordal->P_aug_colptr[j + 1] += chordal->P_aug_colptr[j];

                    // Fill row indices (CSC)
                    std::vector<int64_t> p_col_pos(n_aug, 0);
                    for (int64_t row = 0; row < n_orig; row++) {
                        for (int64_t idx = P_ro[row]; idx < P_ro[row + 1]; idx++) {
                            int64_t col = P_ci[idx];
                            int64_t dest = chordal->P_aug_colptr[col] + p_col_pos[col]++;
                            chordal->P_aug_rowind[dest] = row;
                            p_csr_to_csc[idx] = dest;
                        }
                    }

                    // Now convert augmented P CSC back to CSR, recording mapping
                    int64_t nnzP_aug = nnzP_;  // same nnz, just extended with empty columns
                    aug_P_ro_storage.resize(n_aug + 1, 0);
                    aug_P_ci_storage.resize(nnzP_aug);
                    std::vector<int64_t> p_csc_to_csr(nnzP_aug);
                    for (int64_t i = 0; i < nnzP_aug; i++) aug_P_ro_storage[chordal->P_aug_rowind[i] + 1]++;
                    for (int64_t i = 0; i < n_aug; i++) aug_P_ro_storage[i + 1] += aug_P_ro_storage[i];
                    std::vector<int64_t> p_row_pos(n_aug, 0);
                    for (int64_t col = 0; col < n_aug; col++) {
                        for (int64_t idx = chordal->P_aug_colptr[col]; idx < chordal->P_aug_colptr[col + 1]; idx++) {
                            int64_t row = chordal->P_aug_rowind[idx];
                            int64_t dest = aug_P_ro_storage[row] + p_row_pos[row]++;
                            aug_P_ci_storage[dest] = col;
                            p_csc_to_csr[idx] = dest;
                        }
                    }

                    // Build P_orig_to_aug: user CSR index -> augmented CSR index
                    chordal->P_orig_to_aug.resize(nnzP_);
                    for (int64_t k = 0; k < nnzP_; ++k) {
                        chordal->P_orig_to_aug[k] = p_csc_to_csr[p_csr_to_csc[k]];
                    }

                    nnzP_ = nnzP_aug;
                } else {
                    // LP: P remains empty, just extend row offsets
                    aug_P_ro_storage.resize(n_aug + 1, 0);
                    nnzP_ = 0;
                }

                // Update effective cones: replace PSD dims with decomposed dims
                effective_cones.psdConeDims = chordal->decomposed_psd_dims;
                effective_cones.numPsdCones = chordal->decomposed_psd_dims.size();

                // Update solver dimensions
                n_ = n_aug;
                m_ = m_aug;
                nnzA_ = nnzA_aug;
                P_ro = aug_P_ro_storage.data();
                P_ci = aug_P_ci_storage.data();
                A_ro = aug_A_ro_storage.data();
                A_ci = aug_A_ci_storage.data();

                chordal_ = std::move(chordal);
            }
        }

        auto construction_start = std::chrono::high_resolution_clock::now();

        solver_ = std::make_unique<CompiledSolver>(
            n_, m_, batchSize,
            P_ro, P_ci, nnzP_,
            A_ro, A_ci, nnzA_,
            effective_cones,
            settings
        );

        auto construction_end = std::chrono::high_resolution_clock::now();
        solver_->info.construction_time = std::chrono::duration<double>(construction_end - construction_start).count();
    }

    /**
     * @brief Solve optimization problem with numpy arrays
     *
     * @param P_values Dense values for P matrix, shape (batchSize, nnzP)
     * @param A_values Dense values for A matrix, shape (batchSize, nnzA)
     * @param q Linear cost vector, shape (batchSize, n)
     * @param b Constraint RHS, shape (batchSize, m)
     * @return Dictionary with solution vectors and status
     */
    nb::dict solve(
        input_array<double> P_values,
        input_array<double> A_values,
        input_array<double> q,
        input_array<double> b
    ) {

        validate_solve_numpy_inputs(P_values, A_values, q, b);

        if (chordal_) {
            // Chordal active: augment inputs on host, solve in augmented space, reverse-map
            return solve_with_chordal(P_values, A_values, q, b);
        }

        // Standard path (no chordal)
        CudaDeviceMemory d_P(sizeof(double) * nnzP_ * batchSize_);
        CudaDeviceMemory d_A(sizeof(double) * nnzA_ * batchSize_);
        CudaDeviceMemory d_q(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_b(sizeof(double) * m_ * batchSize_);

        CUDA_CHECK(cudaMemcpy(d_P.get(), P_values.data(), sizeof(double) * nnzP_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A.get(), A_values.data(), sizeof(double) * nnzA_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q.get(), q.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        solver_->setup(d_P.get(), d_A.get());
        solver_->solve(d_q.get(), d_b.get());

        return make_solve_result_dict();
    }

    // Chordal solve path: augment on host, solve augmented, reverse-map
    nb::dict solve_with_chordal(
        const input_array<double>& P_values,
        const input_array<double>& A_values,
        const input_array<double>& q,
        const input_array<double>& b
    ) {
        // For each batch: augment P, A, q, b values
        // Note: currently only single batch (batchSize=1) for chordal
        // Multi-batch chordal would require per-batch augmentation

        // Augment P values
        std::vector<double> P_aug(nnzP_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            const double* P_orig = P_values.data() + batch * nnzP_user_;
            double* P_dst = P_aug.data() + batch * nnzP_;
            chordal_->augment_P_values(nnzP_user_ > 0 ? P_orig : nullptr, P_dst, nnzP_);
        }

        // Augment A values
        std::vector<double> A_aug(nnzA_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            const double* A_orig = A_values.data() + batch * nnzA_user_;
            double* A_dst = A_aug.data() + batch * nnzA_;
            chordal_->augment_A_values(nnzA_user_ > 0 ? A_orig : nullptr, A_dst, nnzA_);
        }

        // Augment q and b
        std::vector<double> q_aug(n_ * batchSize_, 0.0);
        std::vector<double> b_aug(m_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            chordal_->augment_q(q.data() + batch * n_user_, q_aug.data() + batch * n_);
            chordal_->augment_b(b.data() + batch * m_user_, b_aug.data() + batch * m_);
        }

        // Upload augmented data to device and solve
        CudaDeviceMemory d_P(sizeof(double) * std::max((int64_t)1, nnzP_ * batchSize_));
        CudaDeviceMemory d_A(sizeof(double) * std::max((int64_t)1, nnzA_ * batchSize_));
        CudaDeviceMemory d_q(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_b(sizeof(double) * m_ * batchSize_);

        if (nnzP_ > 0) CUDA_CHECK(cudaMemcpy(d_P.get(), P_aug.data(), sizeof(double) * nnzP_ * batchSize_, cudaMemcpyHostToDevice));
        if (nnzA_ > 0) CUDA_CHECK(cudaMemcpy(d_A.get(), A_aug.data(), sizeof(double) * nnzA_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q.get(), q_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        solver_->setup(d_P.get(), d_A.get());
        solver_->solve(d_q.get(), d_b.get());
        cudaDeviceSynchronize();

        // Download augmented solution
        std::vector<double> x_aug(n_ * batchSize_);
        std::vector<double> z_aug(m_ * batchSize_);
        std::vector<double> s_aug(m_ * batchSize_);
        CUDA_CHECK(cudaMemcpy(x_aug.data(), solver_->solution.x.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z_aug.data(), solver_->solution.z.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s_aug.data(), solver_->solution.s.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));

        // Reverse-map to original dimensions
        std::vector<double> x_orig(n_user_ * batchSize_, 0.0);
        std::vector<double> z_orig(m_user_ * batchSize_, 0.0);
        std::vector<double> s_orig(m_user_ * batchSize_, 0.0);

        for (int64_t batch = 0; batch < batchSize_; batch++) {
            // x: first n_user_ entries are original variables (rest are overlap vars)
            std::memcpy(x_orig.data() + batch * n_user_,
                       x_aug.data() + batch * n_,
                       sizeof(double) * n_user_);

            // s and z: reverse-map from augmented to original
            chordal_->reverse_s(s_aug.data() + batch * m_, s_orig.data() + batch * m_user_);
            chordal_->reverse_z(z_aug.data() + batch * m_, z_orig.data() + batch * m_user_);

            // PSD completion on z
            int64_t psd_offset = chordal_->num_zero + chordal_->num_nonneg
                               + chordal_->total_soc;
            chordal_->complete_z(z_orig.data() + batch * m_user_, psd_offset);
        }

        // Build result dict with original dimensions
        nb::dict result;

        // Upload reverse-mapped results to device so make_numpy_array_2d works
        // (it copies from device to numpy)
        CudaDeviceMemory d_x_orig(sizeof(double) * n_user_ * batchSize_);
        CudaDeviceMemory d_z_orig(sizeof(double) * m_user_ * batchSize_);
        CudaDeviceMemory d_s_orig(sizeof(double) * m_user_ * batchSize_);
        CUDA_CHECK(cudaMemcpy(d_x_orig.get(), x_orig.data(), sizeof(double) * n_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z_orig.get(), z_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s_orig.get(), s_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));

        result["x"] = make_numpy_array_2d<double>((double*)d_x_orig.get(), (size_t)batchSize_, (size_t)n_user_);
        result["z"] = make_numpy_array_2d<double>((double*)d_z_orig.get(), (size_t)batchSize_, (size_t)m_user_);
        result["s"] = make_numpy_array_2d<double>((double*)d_s_orig.get(), (size_t)batchSize_, (size_t)m_user_);

        // Copy status and other info from solver
        auto status_result = make_status_array(solver_->info.status_device,
            static_cast<size_t>(batchSize_));
        result["status"] = status_result;

        std::vector<double> cost_primal(batchSize_);
        solver_->solution.obj_val.gpuToCpu(cost_primal.data());
        auto iterations_result = make_numpy_array_1d<int32_t>(
            solver_->info.iterations_per_batch.data(), static_cast<size_t>(batchSize_), false);

        // Scalarize for single-problem case (match non-chordal path)
        if (batchSize_ == 1) {
            result["obj_val"] = cost_primal[0];
            result["iterations"] = solver_->info.iterations_per_batch[0];
        } else {
            result["obj_val"] = nb::cast(cost_primal);
            result["iterations"] = iterations_result;
        }
        result["solve_time"] = solver_->info.solve_time;
        result["construction_time"] = solver_->info.construction_time;
        result["setup_time"] = solver_->info.setup_time;

        return result;
    }

    /**
     * @brief Solve optimization problem with warm start from numpy arrays
     *
     * @param P_values Dense values for P matrix, shape (batchSize, nnzP)
     * @param A_values Dense values for A matrix, shape (batchSize, nnzA)
     * @param q Linear cost vector, shape (batchSize, n)
     * @param b Constraint RHS, shape (batchSize, m)
     * @param warm_x Warm start primal variables, shape (batchSize, n)
     * @param warm_z Warm start dual variables, shape (batchSize, m)
     * @param warm_s Warm start slack variables, shape (batchSize, m)
     * @return Dictionary with solution vectors and status
     */
    nb::dict solve_warm_start(
        input_array<double> P_values,
        input_array<double> A_values,
        input_array<double> q,
        input_array<double> b,
        input_array<double> warm_x,
        input_array<double> warm_z,
        input_array<double> warm_s,
        std::optional<input_array<double>> warm_z_x = std::nullopt
    ) {
        validate_solve_numpy_inputs(P_values, A_values, q, b);

        // Validate warm start shapes (use user-facing dimensions)
        if (warm_x.ndim() != 2 || static_cast<int64_t>(warm_x.shape(0)) != batchSize_ || static_cast<int64_t>(warm_x.shape(1)) != n_user_) {
            throw std::invalid_argument("warm_x must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(n_user_) + ")");
        }
        if (warm_z.ndim() != 2 || static_cast<int64_t>(warm_z.shape(0)) != batchSize_ || static_cast<int64_t>(warm_z.shape(1)) != m_user_) {
            throw std::invalid_argument("warm_z must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(m_user_) + ")");
        }
        if (warm_s.ndim() != 2 || static_cast<int64_t>(warm_s.shape(0)) != batchSize_ || static_cast<int64_t>(warm_s.shape(1)) != m_user_) {
            throw std::invalid_argument("warm_s must have shape (" + std::to_string(batchSize_) + ", " + std::to_string(m_user_) + ")");
        }
        // Direct-x warm-start: when total_xn_ > 0, warm_z_x must be provided
        // with shape (batchSize_, total_xn_). When zero-length or omitted,
        // the solver falls back to default direct-x dual initialization.
        const int64_t total_xn = solver_ ? solver_->variables.totalXConeNumel() : int64_t{0};
        if (warm_z_x.has_value() && warm_z_x->size() > 0) {
            if (warm_z_x->ndim() != 2 ||
                static_cast<int64_t>(warm_z_x->shape(0)) != batchSize_ ||
                static_cast<int64_t>(warm_z_x->shape(1)) != total_xn) {
                throw std::invalid_argument(
                    "warm_z_x must have shape (" + std::to_string(batchSize_) +
                    ", " + std::to_string(total_xn) + ")");
            }
        }

        if (chordal_) {
            // Mirrors the device-pointer guard at line 1124-1130: chordal
            // decomposition augments PSD slack cones and rewrites the
            // KKT topology; threading `warm_z_x` through that augmented
            // system isn't implemented. Reject loudly rather than silently
            // dropping the user's warm direct-x dual.
            if (warm_z_x.has_value() && warm_z_x->size() > 0) {
                throw std::runtime_error(
                    "warm_z_x with chordal-decomposed PSD slack cones is not "
                    "yet supported on the CUDA path; either disable chordal "
                    "decomposition or omit warm_z_x.");
            }
            return solve_warm_start_with_chordal(P_values, A_values, q, b, warm_x, warm_z, warm_s);
        }

        // Standard path (no chordal): user dims = solver dims
        CudaDeviceMemory d_P(sizeof(double) * nnzP_ * batchSize_);
        CudaDeviceMemory d_A(sizeof(double) * nnzA_ * batchSize_);
        CudaDeviceMemory d_q(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_b(sizeof(double) * m_ * batchSize_);
        CudaDeviceMemory d_warm_x(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_warm_z(sizeof(double) * m_ * batchSize_);
        CudaDeviceMemory d_warm_s(sizeof(double) * m_ * batchSize_);

        CUDA_CHECK(cudaMemcpy(d_P.get(), P_values.data(), sizeof(double) * nnzP_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A.get(), A_values.data(), sizeof(double) * nnzA_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q.get(), q.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_warm_x.get(), warm_x.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_warm_z.get(), warm_z.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_warm_s.get(), warm_s.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        // Direct-x warm dual: copy when provided + total_xn > 0, else nullptr
        // (the C++ `solve(...)` overload defaults d_warm_z_x to nullptr and
        // the solver falls back to the default direct-x dual init).
        std::unique_ptr<CudaDeviceMemory> d_warm_z_x;
        const double* d_warm_z_x_ptr = nullptr;
        if (warm_z_x.has_value() && warm_z_x->size() > 0 && total_xn > 0) {
            d_warm_z_x = std::make_unique<CudaDeviceMemory>(
                sizeof(double) * total_xn * batchSize_);
            CUDA_CHECK(cudaMemcpy(d_warm_z_x->get(), warm_z_x->data(),
                                  sizeof(double) * total_xn * batchSize_,
                                  cudaMemcpyHostToDevice));
            d_warm_z_x_ptr = d_warm_z_x->get();
        }

        solver_->setup(d_P.get(), d_A.get());
        solver_->solve(d_q.get(), d_b.get(), d_warm_x.get(), d_warm_z.get(), d_warm_s.get(),
                       /*stream=*/0, d_warm_z_x_ptr);

        return make_solve_result_dict();
    }

    // Warm start with chordal: augment inputs + warm start, solve augmented, reverse-map
    nb::dict solve_warm_start_with_chordal(
        const input_array<double>& P_values,
        const input_array<double>& A_values,
        const input_array<double>& q,
        const input_array<double>& b,
        const input_array<double>& warm_x,
        const input_array<double>& warm_z,
        const input_array<double>& warm_s
    ) {
        // Augment P values
        std::vector<double> P_aug(nnzP_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            const double* P_orig = P_values.data() + batch * nnzP_user_;
            double* P_dst = P_aug.data() + batch * nnzP_;
            chordal_->augment_P_values(nnzP_user_ > 0 ? P_orig : nullptr, P_dst, nnzP_);
        }

        // Augment A values
        std::vector<double> A_aug(nnzA_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            const double* A_orig = A_values.data() + batch * nnzA_user_;
            double* A_dst = A_aug.data() + batch * nnzA_;
            chordal_->augment_A_values(nnzA_user_ > 0 ? A_orig : nullptr, A_dst, nnzA_);
        }

        // Augment q and b
        std::vector<double> q_aug(n_ * batchSize_, 0.0);
        std::vector<double> b_aug(m_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            chordal_->augment_q(q.data() + batch * n_user_, q_aug.data() + batch * n_);
            chordal_->augment_b(b.data() + batch * m_user_, b_aug.data() + batch * m_);
        }

        // Augment warm start: x gets zero-padded, z/s use forward augmentation
        std::vector<double> warm_x_aug(n_ * batchSize_, 0.0);
        std::vector<double> warm_z_aug(m_ * batchSize_, 0.0);
        std::vector<double> warm_s_aug(m_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            // x: first n_user_ entries are original, rest are overlap vars (zero)
            std::memcpy(warm_x_aug.data() + batch * n_,
                       warm_x.data() + batch * n_user_,
                       sizeof(double) * n_user_);
            // z and s: forward augmentation (user-dim → augmented-dim).
            // augment_b is the correct forward map: it scatters original rows
            // to their augmented positions and zeros overlap rows.
            chordal_->augment_b(warm_z.data() + batch * m_user_, warm_z_aug.data() + batch * m_);
            chordal_->augment_b(warm_s.data() + batch * m_user_, warm_s_aug.data() + batch * m_);
        }

        // Upload augmented data to device and solve
        CudaDeviceMemory d_P(sizeof(double) * std::max((int64_t)1, nnzP_ * batchSize_));
        CudaDeviceMemory d_A(sizeof(double) * std::max((int64_t)1, nnzA_ * batchSize_));
        CudaDeviceMemory d_q(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_b(sizeof(double) * m_ * batchSize_);
        CudaDeviceMemory d_warm_x(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_warm_z(sizeof(double) * m_ * batchSize_);
        CudaDeviceMemory d_warm_s(sizeof(double) * m_ * batchSize_);

        if (nnzP_ > 0) CUDA_CHECK(cudaMemcpy(d_P.get(), P_aug.data(), sizeof(double) * nnzP_ * batchSize_, cudaMemcpyHostToDevice));
        if (nnzA_ > 0) CUDA_CHECK(cudaMemcpy(d_A.get(), A_aug.data(), sizeof(double) * nnzA_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q.get(), q_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(), b_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_warm_x.get(), warm_x_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_warm_z.get(), warm_z_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_warm_s.get(), warm_s_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        solver_->setup(d_P.get(), d_A.get());
        solver_->solve(d_q.get(), d_b.get(), d_warm_x.get(), d_warm_z.get(), d_warm_s.get());
        cudaDeviceSynchronize();

        // Download and reverse-map solution (same as solve_with_chordal)
        std::vector<double> x_aug(n_ * batchSize_);
        std::vector<double> z_aug(m_ * batchSize_);
        std::vector<double> s_aug(m_ * batchSize_);
        CUDA_CHECK(cudaMemcpy(x_aug.data(), solver_->solution.x.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z_aug.data(), solver_->solution.z.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s_aug.data(), solver_->solution.s.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));

        std::vector<double> x_orig(n_user_ * batchSize_, 0.0);
        std::vector<double> z_orig(m_user_ * batchSize_, 0.0);
        std::vector<double> s_orig(m_user_ * batchSize_, 0.0);

        for (int64_t batch = 0; batch < batchSize_; batch++) {
            std::memcpy(x_orig.data() + batch * n_user_,
                       x_aug.data() + batch * n_,
                       sizeof(double) * n_user_);
            chordal_->reverse_s(s_aug.data() + batch * m_, s_orig.data() + batch * m_user_);
            chordal_->reverse_z(z_aug.data() + batch * m_, z_orig.data() + batch * m_user_);
            int64_t psd_offset = chordal_->num_zero + chordal_->num_nonneg
                               + chordal_->total_soc;
            chordal_->complete_z(z_orig.data() + batch * m_user_, psd_offset);
        }

        // Build result with original dimensions
        nb::dict result;
        CudaDeviceMemory d_x_orig(sizeof(double) * n_user_ * batchSize_);
        CudaDeviceMemory d_z_orig(sizeof(double) * m_user_ * batchSize_);
        CudaDeviceMemory d_s_orig(sizeof(double) * m_user_ * batchSize_);
        CUDA_CHECK(cudaMemcpy(d_x_orig.get(), x_orig.data(), sizeof(double) * n_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z_orig.get(), z_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s_orig.get(), s_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));

        result["x"] = make_numpy_array_2d<double>((double*)d_x_orig.get(), (size_t)batchSize_, (size_t)n_user_);
        result["z"] = make_numpy_array_2d<double>((double*)d_z_orig.get(), (size_t)batchSize_, (size_t)m_user_);
        result["s"] = make_numpy_array_2d<double>((double*)d_s_orig.get(), (size_t)batchSize_, (size_t)m_user_);

        auto status_result = make_status_array(solver_->info.status_device, static_cast<size_t>(batchSize_));
        result["status"] = status_result;
        std::vector<double> cost_primal(batchSize_);
        solver_->solution.obj_val.gpuToCpu(cost_primal.data());
        if (batchSize_ == 1) {
            result["obj_val"] = cost_primal[0];
            result["iterations"] = solver_->info.iterations_per_batch[0];
        } else {
            result["obj_val"] = nb::cast(cost_primal);
            result["iterations"] = make_numpy_array_1d<int32_t>(
                solver_->info.iterations_per_batch.data(), static_cast<size_t>(batchSize_), false);
        }
        result["solve_time"] = solver_->info.solve_time;
        result["construction_time"] = solver_->info.construction_time;
        result["setup_time"] = solver_->info.setup_time;
        return result;
    }

    /**
     * @brief Get problem dimensions
     */
    nb::dict get_dimensions() const {
        nb::dict dims;
        dims["n"] = n_;
        dims["m"] = m_;
        dims["batch_size"] = batchSize_;
        dims["nnzP"] = nnzP_;
        dims["nnzA"] = nnzA_;
        return dims;
    }

    /**
     * @brief Get memory usage in bytes
     */
    size_t memory_usage() const {
        return solver_->memoryUsage();
    }

    // Helper: download device arrays, augment via chordal, re-upload for solve.
    // Augments P, A, q, b from user dims to solver dims.
    // Returns device memory objects that must stay alive until solve completes.
    struct AugmentedDeviceBuffers {
        CudaDeviceMemory d_P, d_A, d_q, d_b;
        CudaDeviceMemory d_warm_x, d_warm_z, d_warm_s;
    };

    AugmentedDeviceBuffers chordal_augment_device_inputs(
        uintptr_t P_ptr, uintptr_t A_ptr, uintptr_t q_ptr, uintptr_t b_ptr,
        uintptr_t warm_x_ptr = 0, uintptr_t warm_z_ptr = 0, uintptr_t warm_s_ptr = 0
    ) {
        AugmentedDeviceBuffers bufs;

        // Download user-dim inputs from device to host
        std::vector<double> P_host(nnzP_user_ * batchSize_);
        std::vector<double> A_host(nnzA_user_ * batchSize_);
        std::vector<double> q_host(n_user_ * batchSize_);
        std::vector<double> b_host(m_user_ * batchSize_);

        if (nnzP_user_ > 0) CUDA_CHECK(cudaMemcpy(P_host.data(), reinterpret_cast<const void*>(P_ptr), sizeof(double) * nnzP_user_ * batchSize_, cudaMemcpyDeviceToHost));
        if (nnzA_user_ > 0) CUDA_CHECK(cudaMemcpy(A_host.data(), reinterpret_cast<const void*>(A_ptr), sizeof(double) * nnzA_user_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q_host.data(), reinterpret_cast<const void*>(q_ptr), sizeof(double) * n_user_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b_host.data(), reinterpret_cast<const void*>(b_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));

        // Augment
        std::vector<double> P_aug(nnzP_ * batchSize_, 0.0);
        std::vector<double> A_aug(nnzA_ * batchSize_, 0.0);
        std::vector<double> q_aug(n_ * batchSize_, 0.0);
        std::vector<double> b_aug(m_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            chordal_->augment_P_values(nnzP_user_ > 0 ? P_host.data() + batch * nnzP_user_ : nullptr,
                                       P_aug.data() + batch * nnzP_, nnzP_);
            chordal_->augment_A_values(nnzA_user_ > 0 ? A_host.data() + batch * nnzA_user_ : nullptr,
                                       A_aug.data() + batch * nnzA_, nnzA_);
            chordal_->augment_q(q_host.data() + batch * n_user_, q_aug.data() + batch * n_);
            chordal_->augment_b(b_host.data() + batch * m_user_, b_aug.data() + batch * m_);
        }

        // Upload augmented data
        bufs.d_P = CudaDeviceMemory(sizeof(double) * std::max((int64_t)1, nnzP_ * batchSize_));
        bufs.d_A = CudaDeviceMemory(sizeof(double) * std::max((int64_t)1, nnzA_ * batchSize_));
        bufs.d_q = CudaDeviceMemory(sizeof(double) * n_ * batchSize_);
        bufs.d_b = CudaDeviceMemory(sizeof(double) * m_ * batchSize_);

        if (nnzP_ > 0) CUDA_CHECK(cudaMemcpy(bufs.d_P.get(), P_aug.data(), sizeof(double) * nnzP_ * batchSize_, cudaMemcpyHostToDevice));
        if (nnzA_ > 0) CUDA_CHECK(cudaMemcpy(bufs.d_A.get(), A_aug.data(), sizeof(double) * nnzA_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bufs.d_q.get(), q_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bufs.d_b.get(), b_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        // Augment warm start if provided
        if (warm_x_ptr != 0) {
            std::vector<double> wx_host(n_user_ * batchSize_);
            std::vector<double> wz_host(m_user_ * batchSize_);
            std::vector<double> ws_host(m_user_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(wx_host.data(), reinterpret_cast<const void*>(warm_x_ptr), sizeof(double) * n_user_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(wz_host.data(), reinterpret_cast<const void*>(warm_z_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws_host.data(), reinterpret_cast<const void*>(warm_s_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));

            std::vector<double> wx_aug(n_ * batchSize_, 0.0);
            std::vector<double> wz_aug(m_ * batchSize_, 0.0);
            std::vector<double> ws_aug(m_ * batchSize_, 0.0);
            for (int64_t batch = 0; batch < batchSize_; batch++) {
                std::memcpy(wx_aug.data() + batch * n_, wx_host.data() + batch * n_user_, sizeof(double) * n_user_);
                chordal_->adjoint_reverse_z(wz_host.data() + batch * m_user_, wz_aug.data() + batch * m_);
                chordal_->adjoint_reverse_s(ws_host.data() + batch * m_user_, ws_aug.data() + batch * m_);
            }

            bufs.d_warm_x = CudaDeviceMemory(sizeof(double) * n_ * batchSize_);
            bufs.d_warm_z = CudaDeviceMemory(sizeof(double) * m_ * batchSize_);
            bufs.d_warm_s = CudaDeviceMemory(sizeof(double) * m_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(bufs.d_warm_x.get(), wx_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(bufs.d_warm_z.get(), wz_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(bufs.d_warm_s.get(), ws_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        }

        return bufs;
    }

    // Helper: reverse-map augmented solution from solver to user-dim device output pointers
    void chordal_reverse_solution_to_device(
        uintptr_t x_out_ptr, uintptr_t z_out_ptr, uintptr_t s_out_ptr,
        uintptr_t status_out_ptr, uintptr_t obj_out_ptr,
        uintptr_t z_x_out_ptr = 0
    ) {
        cudaDeviceSynchronize();

        // Download augmented solution
        std::vector<double> x_aug(n_ * batchSize_);
        std::vector<double> z_aug(m_ * batchSize_);
        std::vector<double> s_aug(m_ * batchSize_);
        CUDA_CHECK(cudaMemcpy(x_aug.data(), solver_->solution.x.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z_aug.data(), solver_->solution.z.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s_aug.data(), solver_->solution.s.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));

        // Reverse-map to original dimensions
        std::vector<double> x_orig(n_user_ * batchSize_, 0.0);
        std::vector<double> z_orig(m_user_ * batchSize_, 0.0);
        std::vector<double> s_orig(m_user_ * batchSize_, 0.0);
        for (int64_t batch = 0; batch < batchSize_; batch++) {
            std::memcpy(x_orig.data() + batch * n_user_, x_aug.data() + batch * n_, sizeof(double) * n_user_);
            chordal_->reverse_s(s_aug.data() + batch * m_, s_orig.data() + batch * m_user_);
            chordal_->reverse_z(z_aug.data() + batch * m_, z_orig.data() + batch * m_user_);
            int64_t psd_offset = chordal_->num_zero + chordal_->num_nonneg
                               + chordal_->total_soc;
            chordal_->complete_z(z_orig.data() + batch * m_user_, psd_offset);
        }

        // Upload reverse-mapped solution to user output pointers
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(x_out_ptr), x_orig.data(), sizeof(double) * n_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(z_out_ptr), z_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s_out_ptr), s_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));

        // Status and obj come directly from solver (not dimension-dependent)
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(status_out_ptr), solver_->solution.status.get(),
                   sizeof(int32_t) * batchSize_, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(obj_out_ptr), solver_->info.cost_primal.data(),
                   sizeof(double) * batchSize_, cudaMemcpyDeviceToDevice));

        // Direct-x duals: chordal augments only n and m (slack space); x_cone
        // indices reference user-space columns [0, n_user) which equal the
        // first n_user columns of the augmented matrix, so z_x is unchanged
        // and unscaled in the augmented frame. Without this write the caller
        // sees garbage in `z_x_out_ptr` whenever chordal + direct-x are mixed.
        int64_t total_xn = solver_->variables.totalXConeNumel();
        if (z_x_out_ptr != 0 && total_xn > 0) {
            const auto& cones = solver_->data.cones;
            moreau::unscale_z_x(
                reinterpret_cast<double*>(z_x_out_ptr),
                solver_->variables.z_x.data(),
                solver_->data.equilibration.dinv.data(),
                solver_->data.equilibration.c.data(),
                solver_->solution.τ_raw.data(),
                cones.d_xcone_indices,
                n_, total_xn, batchSize_, 0);
        }
    }

    nb::dict solve_from_device_pointers(
        uintptr_t P_ptr, uintptr_t A_ptr, uintptr_t q_ptr, uintptr_t b_ptr
    ) {

        if (chordal_) {
            auto bufs = chordal_augment_device_inputs(P_ptr, A_ptr, q_ptr, b_ptr);
            solver_->setup(bufs.d_P.get(), bufs.d_A.get());
            solver_->solve(bufs.d_q.get(), bufs.d_b.get());
            // Return numpy arrays - use the chordal numpy path's result builder
            // by downloading augmented, reverse-mapping, and building result
            cudaDeviceSynchronize();
            std::vector<double> x_aug(n_ * batchSize_), z_aug(m_ * batchSize_), s_aug(m_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(x_aug.data(), solver_->solution.x.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(z_aug.data(), solver_->solution.z.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(s_aug.data(), solver_->solution.s.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));
            std::vector<double> x_orig(n_user_ * batchSize_, 0.0), z_orig(m_user_ * batchSize_, 0.0), s_orig(m_user_ * batchSize_, 0.0);
            for (int64_t batch = 0; batch < batchSize_; batch++) {
                std::memcpy(x_orig.data() + batch * n_user_, x_aug.data() + batch * n_, sizeof(double) * n_user_);
                chordal_->reverse_s(s_aug.data() + batch * m_, s_orig.data() + batch * m_user_);
                chordal_->reverse_z(z_aug.data() + batch * m_, z_orig.data() + batch * m_user_);
                int64_t psd_offset = chordal_->num_zero + chordal_->num_nonneg
                                   + chordal_->total_soc;
                chordal_->complete_z(z_orig.data() + batch * m_user_, psd_offset);
            }
            CudaDeviceMemory d_x_orig(sizeof(double) * n_user_ * batchSize_);
            CudaDeviceMemory d_z_orig(sizeof(double) * m_user_ * batchSize_);
            CudaDeviceMemory d_s_orig(sizeof(double) * m_user_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(d_x_orig.get(), x_orig.data(), sizeof(double) * n_user_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_z_orig.get(), z_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_s_orig.get(), s_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));
            nb::dict result;
            result["x"] = make_numpy_array_2d<double>((double*)d_x_orig.get(), (size_t)batchSize_, (size_t)n_user_);
            result["z"] = make_numpy_array_2d<double>((double*)d_z_orig.get(), (size_t)batchSize_, (size_t)m_user_);
            result["s"] = make_numpy_array_2d<double>((double*)d_s_orig.get(), (size_t)batchSize_, (size_t)m_user_);
            // Direct-x duals: the non-chordal path adds `z_x` via
            // make_solve_result_dict; the chordal branch built the dict by
            // hand and omitted it, so chordal + direct-x silently dropped
            // the z_x output (callers see result.get('z_x') == None).
            // Chordal augments only slack space — z_x is unchanged — so the
            // same unscale_z_x path applies.
            int64_t total_xn = solver_->variables.totalXConeNumel();
            if (total_xn > 0) {
                CudaDeviceMemory d_z_x_user(sizeof(double) * total_xn * batchSize_);
                moreau::unscale_z_x(
                    reinterpret_cast<double*>(d_z_x_user.get()),
                    solver_->variables.z_x.data(),
                    solver_->data.equilibration.dinv.data(),
                    solver_->data.equilibration.c.data(),
                    solver_->solution.τ_raw.data(),
                    solver_->data.cones.d_xcone_indices,
                    n_, total_xn, batchSize_, 0);
                cudaDeviceSynchronize();
                result["z_x"] = make_numpy_array_2d<double>(
                    (double*)d_z_x_user.get(),
                    (size_t)batchSize_, (size_t)total_xn);
            } else {
                result["z_x"] = make_numpy_array_2d<double>(
                    (double*)nullptr, (size_t)batchSize_, 0);
            }
            auto status_result = make_status_array(solver_->info.status_device, static_cast<size_t>(batchSize_));
            result["status"] = status_result;
            std::vector<double> cost_primal(batchSize_);
            solver_->solution.obj_val.gpuToCpu(cost_primal.data());
            if (batchSize_ == 1) {
                result["obj_val"] = cost_primal[0];
                result["iterations"] = solver_->info.iterations_per_batch[0];
            } else {
                result["obj_val"] = cost_primal;
                result["iterations"] = make_numpy_array_1d<int32_t>(
                    solver_->info.iterations_per_batch.data(), static_cast<size_t>(batchSize_), false);
            }
            result["construction_time"] = solver_->info.construction_time;
            result["setup_time"] = solver_->info.setup_time;
            result["solve_time"] = solver_->info.solve_time;
            return result;
        }

        solver_->setup(reinterpret_cast<const double*>(P_ptr), reinterpret_cast<const double*>(A_ptr));
        solver_->solve(reinterpret_cast<const double*>(q_ptr), reinterpret_cast<const double*>(b_ptr));

        return make_solve_result_dict(/*include_tau_kappa=*/false);
    }

    void solve_to_device_pointers(
        uintptr_t P_ptr, uintptr_t A_ptr, uintptr_t q_ptr, uintptr_t b_ptr,
        uintptr_t x_out_ptr, uintptr_t z_out_ptr, uintptr_t s_out_ptr,
        uintptr_t status_out_ptr, uintptr_t obj_out_ptr,
        uintptr_t z_x_out_ptr = 0
    ) {

        if (chordal_) {
            auto bufs = chordal_augment_device_inputs(P_ptr, A_ptr, q_ptr, b_ptr);
            solver_->setup(bufs.d_P.get(), bufs.d_A.get());
            solver_->solve(bufs.d_q.get(), bufs.d_b.get());
            chordal_reverse_solution_to_device(x_out_ptr, z_out_ptr, s_out_ptr, status_out_ptr, obj_out_ptr, z_x_out_ptr);
            return;
        }

        solver_->setup(reinterpret_cast<const double*>(P_ptr), reinterpret_cast<const double*>(A_ptr));
        solver_->solve(reinterpret_cast<const double*>(q_ptr), reinterpret_cast<const double*>(b_ptr));

        copy_solution_to_device_output(x_out_ptr, z_out_ptr, s_out_ptr, status_out_ptr, obj_out_ptr, z_x_out_ptr);
    }

    void solve_warm_start_to_device_pointers(
        uintptr_t P_ptr, uintptr_t A_ptr, uintptr_t q_ptr, uintptr_t b_ptr,
        uintptr_t warm_x_ptr, uintptr_t warm_z_ptr, uintptr_t warm_s_ptr,
        uintptr_t x_out_ptr, uintptr_t z_out_ptr, uintptr_t s_out_ptr,
        uintptr_t status_out_ptr, uintptr_t obj_out_ptr,
        uintptr_t warm_z_x_ptr = 0,
        uintptr_t z_x_out_ptr = 0
    ) {

        if (chordal_) {
            if (warm_z_x_ptr != 0) {
                throw std::runtime_error(
                    "warm_z_x with chordal-decomposed PSD slack cones is not "
                    "yet supported on the CUDA path; either disable chordal "
                    "decomposition or omit warm_z_x.");
            }
            auto bufs = chordal_augment_device_inputs(P_ptr, A_ptr, q_ptr, b_ptr,
                                                       warm_x_ptr, warm_z_ptr, warm_s_ptr);
            solver_->setup(bufs.d_P.get(), bufs.d_A.get());
            solver_->solve(bufs.d_q.get(), bufs.d_b.get(),
                           bufs.d_warm_x.get(), bufs.d_warm_z.get(), bufs.d_warm_s.get());
            chordal_reverse_solution_to_device(x_out_ptr, z_out_ptr, s_out_ptr, status_out_ptr, obj_out_ptr, z_x_out_ptr);
            return;
        }

        solver_->setup(reinterpret_cast<const double*>(P_ptr), reinterpret_cast<const double*>(A_ptr));
        solver_->solve(reinterpret_cast<const double*>(q_ptr), reinterpret_cast<const double*>(b_ptr),
                       reinterpret_cast<const double*>(warm_x_ptr),
                       reinterpret_cast<const double*>(warm_z_ptr),
                       reinterpret_cast<const double*>(warm_s_ptr),
                       /*stream=*/0,
                       reinterpret_cast<const double*>(warm_z_x_ptr));

        copy_solution_to_device_output(x_out_ptr, z_out_ptr, s_out_ptr, status_out_ptr, obj_out_ptr, z_x_out_ptr);
    }

    /**
     * @brief Load problem data and equilibrate, without solving.
     *
     * Used by the backward pass to restore equilibrated solver state from
     * saved tensors. Avoids a full re-solve.
     *
     * @param P_ptr Device pointer to P values (size nnzP * batchSize)
     * @param A_ptr Device pointer to A values (size nnzA * batchSize)
     * @param q_ptr Device pointer to q vector (size n * batchSize)
     * @param b_ptr Device pointer to b vector (size m * batchSize)
     */
    void load_data_for_backward_from_device_pointers(
        uintptr_t P_ptr, uintptr_t A_ptr, uintptr_t q_ptr, uintptr_t b_ptr
    ) {
        solver_->loadDataForBackward(
            reinterpret_cast<const double*>(P_ptr),
            reinterpret_cast<const double*>(A_ptr),
            reinterpret_cast<const double*>(q_ptr),
            reinterpret_cast<const double*>(b_ptr)
        );
    }

    /**
     * @brief Restore full backward state from saved tensors without re-solving.
     *
     * Performs three steps:
     * 1. loadDataForBackward — reloads P/A/q/b and re-equilibrates
     * 2. Copies saved solution (x, z, s) into DiffState
     * 3. Syncs equilibration factors from solver data to DiffState
     *    (d, dinv, e, einv, c_scale) and recomputes tau, u, pi_u
     *
     * This replaces the old re-solve approach and matches what the JAX
     * backward path should also do.
     */
    void load_backward_state_from_device_pointers(
        uintptr_t P_ptr, uintptr_t A_ptr, uintptr_t q_ptr, uintptr_t b_ptr,
        uintptr_t x_ptr, uintptr_t z_ptr, uintptr_t s_ptr,
        uintptr_t z_x_ptr = 0
    ) {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use load_backward_state");
        }

        // Step 1: reload problem data and re-equilibrate
        solver_->loadDataForBackward(
            reinterpret_cast<const double*>(P_ptr),
            reinterpret_cast<const double*>(A_ptr),
            reinterpret_cast<const double*>(q_ptr),
            reinterpret_cast<const double*>(b_ptr)
        );

        auto* ds = solver_->diff_state();
        int64_t n = n_;
        int64_t m = m_;
        int64_t bs = batchSize_;

        // Step 2: copy saved solution into DiffState
        cudaMemcpy(ds->x.data(), reinterpret_cast<void*>(x_ptr),
                   sizeof(double) * n * bs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ds->z.data(), reinterpret_cast<void*>(z_ptr),
                   sizeof(double) * m * bs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ds->s.data(), reinterpret_cast<void*>(s_ptr),
                   sizeof(double) * m * bs, cudaMemcpyDeviceToDevice);

        // Direct-x dual: when supplied (in user/original frame), convert
        // to the equilibrated τ=1 frame and store in DiffState. Inverse
        // of `Variables::unscale`: z_x_eq = z_x_user * c / d[J].
        int64_t total_xn = solver_->variables.totalXConeNumel();
        if (total_xn > 0 && z_x_ptr != 0) {
            moreau::equilibrate_z_x(
                ds->z_x.data(),
                reinterpret_cast<const double*>(z_x_ptr),
                solver_->data.equilibration.dinv.data(),
                solver_->data.equilibration.c.data(),
                solver_->data.cones.d_xcone_indices,
                n, total_xn, bs, 0
            );
        }

        // Step 3: sync equilibration factors from solver data to DiffState
        cudaMemcpy(ds->d.data(), solver_->data.equilibration.d.data(),
                   sizeof(double) * n * bs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ds->dinv.data(), solver_->data.equilibration.dinv.data(),
                   sizeof(double) * n * bs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ds->e.data(), solver_->data.equilibration.e.data(),
                   sizeof(double) * m * bs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ds->einv.data(), solver_->data.equilibration.einv.data(),
                   sizeof(double) * m * bs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ds->c_scale.data(), solver_->data.equilibration.c.data(),
                   sizeof(double) * bs, cudaMemcpyDeviceToDevice);

        // Recompute derived quantities
        ds->tau.setToConstant(1.0, 0);
        moreau::compute_u_from_z_s(ds->u.data(), ds->z.data(), ds->s.data(), m, bs, 0);
        moreau::compute_cone_projection(ds->u, ds->pi_u, solver_->data.cones, 0,
            ds->work_m.data(), solver_->data.cones.totalGenPowerDim);

        cudaDeviceSynchronize();
    }

    /**
     * @brief Get solve info (iterations, solve_time) after a solve
     *
     * Must be called after solve(). Returns a dict with iterations and solve_time.
     */
    nb::dict get_solve_info() const {
        nb::dict result;

        // Get per-batch iterations
        if (batchSize_ == 1) {
            result["iterations"] = solver_->info.iterations_per_batch[0];
        } else {
            auto iterations_result = make_numpy_array_1d<int32_t>(
                solver_->info.iterations_per_batch.data(), static_cast<size_t>(batchSize_), false);
            result["iterations"] = iterations_result;
        }

        result["solve_time"] = solver_->info.solve_time;
        result["construction_time"] = solver_->info.construction_time;
        result["setup_time"] = solver_->info.setup_time;

        return result;
    }

    /**
     * @brief Run centering iterations for smoothed differentiation
     *
     * Must be called after solve() and before cache_solution_for_backward().
     * Only does work when diff_method is Smoothed.
     */
    void refine_smoothing_iterate() {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use refine_smoothing_iterate()");
        }
        solver_->refineSmoothingIterate(*solver_->diff_state(), 0);
    }

    /**
     * @brief Cache solution state for backward differentiation
     *
     * Must be called after solve() before backward().
     */
    void cache_solution_for_backward() {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use backward()");
        }
        moreau::cache_solution_for_backward(*solver_->diff_state(), *solver_, 0);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Set solution state from device pointers for backward differentiation
     *
     * This allows overriding the cached solution state with saved tensors from
     * PyTorch autograd, which is necessary for correct gradient computation when
     * multiple forward passes share the same solver.
     *
     * @param x_ptr Solution x, shape (batch_size, n)
     * @param z_ptr Solution z, shape (batch_size, m)
     * @param s_ptr Solution s, shape (batch_size, m)
     */
    void set_solution_from_device_pointers(
        uintptr_t x_ptr, uintptr_t z_ptr, uintptr_t s_ptr
    ) {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use set_solution");
        }

        if (chordal_) {
            // User passes user-dim solution; DiffState uses augmented dims.
            // Download user-dim, augment to solver dims, upload.
            std::vector<double> x_host(n_user_ * batchSize_);
            std::vector<double> z_host(m_user_ * batchSize_);
            std::vector<double> s_host(m_user_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(x_host.data(), reinterpret_cast<void*>(x_ptr), sizeof(double) * n_user_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(z_host.data(), reinterpret_cast<void*>(z_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(s_host.data(), reinterpret_cast<void*>(s_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));

            std::vector<double> x_aug(n_ * batchSize_, 0.0);
            std::vector<double> z_aug(m_ * batchSize_, 0.0);
            std::vector<double> s_aug(m_ * batchSize_, 0.0);
            for (int64_t batch = 0; batch < batchSize_; batch++) {
                std::memcpy(x_aug.data() + batch * n_, x_host.data() + batch * n_user_, sizeof(double) * n_user_);
                chordal_->augment_b(z_host.data() + batch * m_user_, z_aug.data() + batch * m_);
                chordal_->augment_b(s_host.data() + batch * m_user_, s_aug.data() + batch * m_);
            }

            CUDA_CHECK(cudaMemcpy(solver_->diff_state()->x.data(), x_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver_->diff_state()->z.data(), z_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver_->diff_state()->s.data(), s_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        } else {
            // No chordal: user dims = solver dims, direct D2D copy
            cudaMemcpy(solver_->diff_state()->x.data(), reinterpret_cast<void*>(x_ptr),
                       sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToDevice);
            cudaMemcpy(solver_->diff_state()->z.data(), reinterpret_cast<void*>(z_ptr),
                       sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
            cudaMemcpy(solver_->diff_state()->s.data(), reinterpret_cast<void*>(s_ptr),
                       sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
        }

        // Set tau = 1.0 for differentiation
        solver_->diff_state()->tau.setToConstant(1.0, 0);

        // Compute u = z - s
        moreau::compute_u_from_z_s(
            solver_->diff_state()->u.data(), solver_->diff_state()->z.data(), solver_->diff_state()->s.data(),
            m_, batchSize_, 0
        );

        // Compute cone projection Π_K*(u)
        moreau::compute_cone_projection(solver_->diff_state()->u, solver_->diff_state()->pi_u, solver_->data.cones, 0,
            solver_->diff_state()->work_m.data(), solver_->data.cones.totalGenPowerDim);

        cudaDeviceSynchronize();
    }

    /**
     * @brief Get equilibration factors to device pointers
     *
     * This copies the current equilibration factors from the solver to user-provided
     * device memory. Used to save equilibration state for later restoration.
     *
     * @param d_ptr Output d scaling vector, shape (batch_size, n)
     * @param e_ptr Output e scaling vector, shape (batch_size, m)
     * @param c_ptr Output c scaling value, shape (batch_size,)
     */
    void get_equilibration_to_device_pointers(
        uintptr_t d_ptr, uintptr_t e_ptr, uintptr_t c_ptr
    ) {
        if (!solver_) {
            throw std::runtime_error("Solver not initialized");
        }

        cudaMemcpy(reinterpret_cast<void*>(d_ptr), solver_->data.equilibration.d.data(),
                   sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(e_ptr), solver_->data.equilibration.e.data(),
                   sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(c_ptr), solver_->data.equilibration.c.data(),
                   sizeof(double) * batchSize_, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Set equilibration factors from device pointers and restore DiffState
     *
     * This restores the equilibration state for backward differentiation when
     * using saved tensors from a previous forward pass.
     *
     * @param d_ptr Input d scaling vector, shape (batch_size, n)
     * @param e_ptr Input e scaling vector, shape (batch_size, m)
     * @param c_ptr Input c scaling value, shape (batch_size,)
     */
    void set_equilibration_from_device_pointers(
        uintptr_t d_ptr, uintptr_t e_ptr, uintptr_t c_ptr
    ) {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use set_equilibration");
        }

        // Copy equilibration factors to DiffState
        cudaMemcpy(solver_->diff_state()->d.data(), reinterpret_cast<void*>(d_ptr),
                   sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(solver_->diff_state()->e.data(), reinterpret_cast<void*>(e_ptr),
                   sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(solver_->diff_state()->c_scale.data(), reinterpret_cast<void*>(c_ptr),
                   sizeof(double) * batchSize_, cudaMemcpyDeviceToDevice);

        // Compute dinv = 1/d, einv = 1/e on device using existing reciprocal kernel
        moreau::reciprocal(solver_->diff_state()->dinv.data(), solver_->diff_state()->d.data(),
                           n_ * batchSize_, 0);
        moreau::reciprocal(solver_->diff_state()->einv.data(), solver_->diff_state()->e.data(),
                           m_ * batchSize_, 0);

        cudaDeviceSynchronize();
    }

    /**
     * @brief Backward differentiation with device pointers
     *
     * @param dx_ptr Upstream gradient w.r.t. x, shape (batch_size, n)
     * @param dz_ptr Upstream gradient w.r.t. z, shape (batch_size, m)
     * @param ds_ptr Upstream gradient w.r.t. s, shape (batch_size, m)
     * @param dP_out_ptr Output gradient w.r.t. P values, shape (batch_size, nnzP)
     * @param dq_out_ptr Output gradient w.r.t. q, shape (batch_size, n)
     * @param dA_out_ptr Output gradient w.r.t. A values, shape (batch_size, nnzA)
     * @param db_out_ptr Output gradient w.r.t. b, shape (batch_size, m)
     */
    void backward_to_device_pointers(
        uintptr_t dx_ptr, uintptr_t dz_ptr, uintptr_t ds_ptr,
        uintptr_t dP_out_ptr, uintptr_t dq_out_ptr,
        uintptr_t dA_out_ptr, uintptr_t db_out_ptr,
        uintptr_t dz_x_ptr = 0
    ) {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use backward()");
        }

        if (chordal_) {
            if (dz_x_ptr != 0) {
                throw std::runtime_error(
                    "dz_x backward with chordal-decomposed PSD slack cones is "
                    "not yet supported on the CUDA path; either disable "
                    "chordal decomposition or omit dz_x.");
            }
            // Upstream grads are in user dims. Augment, run backward, reverse-map outputs.

            // Download user-dim upstream grads
            std::vector<double> dx_host(n_user_ * batchSize_);
            std::vector<double> dz_host(m_user_ * batchSize_);
            std::vector<double> ds_host(m_user_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(dx_host.data(), reinterpret_cast<void*>(dx_ptr), sizeof(double) * n_user_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dz_host.data(), reinterpret_cast<void*>(dz_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ds_host.data(), reinterpret_cast<void*>(ds_ptr), sizeof(double) * m_user_ * batchSize_, cudaMemcpyDeviceToHost));

            // Augment upstream grads
            std::vector<double> dx_aug(n_ * batchSize_, 0.0);
            std::vector<double> dz_aug(m_ * batchSize_, 0.0);
            std::vector<double> ds_aug(m_ * batchSize_, 0.0);
            for (int64_t b = 0; b < batchSize_; b++) {
                std::memcpy(dx_aug.data() + b * n_, dx_host.data() + b * n_user_, sizeof(double) * n_user_);
                chordal_->adjoint_reverse_z(dz_host.data() + b * m_user_, dz_aug.data() + b * m_);
                chordal_->adjoint_reverse_s(ds_host.data() + b * m_user_, ds_aug.data() + b * m_);
            }

            // Upload and run backward in augmented space
            CudaDeviceMemory d_dx(sizeof(double) * n_ * batchSize_);
            CudaDeviceMemory d_dz(sizeof(double) * m_ * batchSize_);
            CudaDeviceMemory d_ds(sizeof(double) * m_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(d_dx.get(), dx_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_dz.get(), dz_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ds.get(), ds_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

            BatchedVector dx_bar(d_dx.get(), n_, batchSize_);
            BatchedVector dz_bar(d_dz.get(), m_, batchSize_);
            BatchedVector ds_bar(d_ds.get(), m_, batchSize_);

            moreau::backward(*solver_->diff_state(), dx_bar, dz_bar, ds_bar, *solver_, 0);
            cudaDeviceSynchronize();

            // Download augmented output grads, reverse-map to user dims, re-upload
            std::vector<double> dP_aug(nnzP_ * batchSize_);
            std::vector<double> dq_aug(n_ * batchSize_);
            std::vector<double> dA_aug(nnzA_ * batchSize_);
            std::vector<double> db_aug(m_ * batchSize_);
            CUDA_CHECK(cudaMemcpy(dP_aug.data(), solver_->diff_state()->dP_values.data(), sizeof(double) * nnzP_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dq_aug.data(), solver_->diff_state()->dq.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dA_aug.data(), solver_->diff_state()->dA_values.data(), sizeof(double) * nnzA_ * batchSize_, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(db_aug.data(), solver_->diff_state()->db.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));

            std::vector<double> dP_orig(nnzP_user_ * batchSize_, 0.0);
            std::vector<double> dq_orig(n_user_ * batchSize_, 0.0);
            std::vector<double> dA_orig(nnzA_user_ * batchSize_, 0.0);
            std::vector<double> db_orig(m_user_ * batchSize_, 0.0);
            for (int64_t b = 0; b < batchSize_; b++) {
                chordal_->adjoint_augment_dP(dP_aug.data() + b * nnzP_, dP_orig.data() + b * nnzP_user_);
                chordal_->adjoint_augment_dq(dq_aug.data() + b * n_, dq_orig.data() + b * n_user_);
                chordal_->adjoint_augment_dA(dA_aug.data() + b * nnzA_, dA_orig.data() + b * nnzA_user_);
                chordal_->adjoint_augment_db(db_aug.data() + b * m_, db_orig.data() + b * m_user_);
            }

            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dP_out_ptr), dP_orig.data(), sizeof(double) * nnzP_user_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dq_out_ptr), dq_orig.data(), sizeof(double) * n_user_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dA_out_ptr), dA_orig.data(), sizeof(double) * nnzA_user_ * batchSize_, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(db_out_ptr), db_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));
            return;
        }

        // Standard path (no chordal): user dims = solver dims
        BatchedVector dx_bar(reinterpret_cast<double*>(dx_ptr), n_, batchSize_);
        BatchedVector dz_bar(reinterpret_cast<double*>(dz_ptr), m_, batchSize_);
        BatchedVector ds_bar(reinterpret_cast<double*>(ds_ptr), m_, batchSize_);

        // Compute backward pass — route through `backward_with_dz_x` when
        // upstream dz_x is supplied (zero-copy), else fall through to the
        // simpler signature.
        if (dz_x_ptr != 0) {
            int64_t total_xn = solver_->variables.totalXConeNumel();
            BatchedVector dz_x_bar(reinterpret_cast<double*>(dz_x_ptr),
                                   total_xn, batchSize_);
            moreau::backward_with_dz_x(
                *solver_->diff_state(), dx_bar, dz_bar, ds_bar,
                &dz_x_bar, *solver_, 0
            );
        } else {
            moreau::backward(
                *solver_->diff_state(), dx_bar, dz_bar, ds_bar,
                *solver_, 0
            );
        }
        cudaDeviceSynchronize();

        // Copy output gradients to output pointers
        cudaMemcpy(reinterpret_cast<void*>(dP_out_ptr), solver_->diff_state()->dP_values.data(),
                   sizeof(double) * nnzP_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(dq_out_ptr), solver_->diff_state()->dq.data(),
                   sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(dA_out_ptr), solver_->diff_state()->dA_values.data(),
                   sizeof(double) * nnzA_ * batchSize_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(reinterpret_cast<void*>(db_out_ptr), solver_->diff_state()->db.data(),
                   sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToDevice);
    }

    /**
     * @brief Backward differentiation with numpy arrays. `dz_x` is an
     * optional upstream gradient on direct-x cone duals (shape
     * `[batch_size, total_xn]`). Pass an empty/None array for slack-only
     * problems or to skip dz_x backprop.
     */
    nb::dict backward(
        input_array<double> dx,
        input_array<double> dz,
        input_array<double> ds,
        std::optional<input_array<double>> dz_x = std::nullopt
    ) {
        if (!solver_->diff_state()) {
            throw std::runtime_error("enable_grad must be True to use backward()");
        }

        if (chordal_) {
            if (dz_x.has_value() && dz_x->size() > 0) {
                throw std::runtime_error(
                    "Backward with dz_x is not supported on chordal-decomposed problems");
            }
            return backward_with_chordal(dx, dz, ds);
        }

        // Standard path (no chordal): user dimensions = solver dimensions
        CudaDeviceMemory d_dx(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_dz(sizeof(double) * m_ * batchSize_);
        CudaDeviceMemory d_ds(sizeof(double) * m_ * batchSize_);

        CUDA_CHECK(cudaMemcpy(d_dx.get(), dx.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dz.get(), dz.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ds.get(), ds.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        BatchedVector dx_bar(d_dx.get(), n_, batchSize_);
        BatchedVector dz_bar(d_dz.get(), m_, batchSize_);
        BatchedVector ds_bar(d_ds.get(), m_, batchSize_);

        // Optional dz_x: upload + wrap if provided.
        std::unique_ptr<CudaDeviceMemory> d_dz_x;
        std::unique_ptr<BatchedVector> dz_x_bar;
        const BatchedVector* dz_x_ptr = nullptr;
        if (dz_x.has_value() && dz_x->size() > 0) {
            int64_t total_xn = solver_->variables.totalXConeNumel();
            int64_t expected = total_xn * batchSize_;
            if (dz_x->size() != static_cast<size_t>(expected)) {
                throw std::runtime_error(
                    "dz_x size " + std::to_string(dz_x->size()) +
                    " does not match expected batch_size * total_xn = " +
                    std::to_string(expected));
            }
            d_dz_x = std::make_unique<CudaDeviceMemory>(sizeof(double) * expected);
            CUDA_CHECK(cudaMemcpy(d_dz_x->get(), dz_x->data(),
                                  sizeof(double) * expected, cudaMemcpyHostToDevice));
            dz_x_bar = std::make_unique<BatchedVector>(d_dz_x->get(), total_xn, batchSize_);
            dz_x_ptr = dz_x_bar.get();
        }

        moreau::backward_with_dz_x(
            *solver_->diff_state(),
            dx_bar, dz_bar, ds_bar, dz_x_ptr,
            *solver_, 0
        );
        cudaDeviceSynchronize();

        nb::dict result;
        result["dP_values"] = make_numpy_array_2d<double>(solver_->diff_state()->dP_values.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(nnzP_));
        result["dq"] = make_numpy_array_2d<double>(solver_->diff_state()->dq.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(n_));
        result["dA_values"] = make_numpy_array_2d<double>(solver_->diff_state()->dA_values.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(nnzA_));
        result["db"] = make_numpy_array_2d<double>(solver_->diff_state()->db.data(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(m_));

        // Debug: expose smoothed iterates in original space (debug builds only).
        // The cached iterates are in equilibrated space; we unscale them:
        //   x_orig = d * x_eq,  z_orig = e/c * z_eq,  s_orig = einv * s_eq
#ifndef NDEBUG
        auto& state = *solver_->diff_state();
        if (state.smoothing_cached) {
            // Allocate temp GPU buffers for unscaled iterates
            moreau::BatchedVector x_orig(n_, batchSize_);
            moreau::BatchedVector z_orig(m_, batchSize_);
            moreau::BatchedVector s_orig(m_, batchSize_);
            // x_orig = d * x_eq
            moreau::elementwise_mul(x_orig, state.d, state.smoothing_x, 0);
            // z_orig = e * z_eq (then divide by c)
            moreau::elementwise_mul(z_orig, state.e, state.smoothing_z, 0);
            moreau::div_per_batch(z_orig, z_orig, state.c_scale, 0);
            // s_orig = einv * s_eq
            moreau::elementwise_mul(s_orig, state.einv, state.smoothing_s, 0);
            cudaDeviceSynchronize();
            result["debug_smoothing_x"] = make_numpy_array_2d<double>(x_orig.data(),
                static_cast<size_t>(batchSize_), static_cast<size_t>(n_));
            result["debug_smoothing_z"] = make_numpy_array_2d<double>(z_orig.data(),
                static_cast<size_t>(batchSize_), static_cast<size_t>(m_));
            result["debug_smoothing_s"] = make_numpy_array_2d<double>(s_orig.data(),
                static_cast<size_t>(batchSize_), static_cast<size_t>(m_));
        }
#endif

        return result;
    }

    // Backward with chordal: augment upstream grads, run backward, reverse-map output grads
    nb::dict backward_with_chordal(
        const input_array<double>& dx,
        const input_array<double>& dz,
        const input_array<double>& ds
    ) {
        // Upstream grads are in user (original) dimensions.
        // We need to augment dz and ds (dx stays as-is for the first n_user_ entries).
        // Then run backward in augmented space, then reverse-map dP/dA/dq/db.

        // Augment dx: [dx_orig, 0...0] for overlap variables
        std::vector<double> dx_aug(n_ * batchSize_, 0.0);
        for (int64_t b = 0; b < batchSize_; b++) {
            std::memcpy(dx_aug.data() + b * n_, dx.data() + b * n_user_,
                       sizeof(double) * n_user_);
        }

        // Augment dz and ds: use adjoint of reverse mapping
        std::vector<double> dz_aug(m_ * batchSize_, 0.0);
        std::vector<double> ds_aug(m_ * batchSize_, 0.0);
        for (int64_t b = 0; b < batchSize_; b++) {
            chordal_->adjoint_reverse_z(dz.data() + b * m_user_, dz_aug.data() + b * m_);
            chordal_->adjoint_reverse_s(ds.data() + b * m_user_, ds_aug.data() + b * m_);
        }

        // Upload and run backward in augmented space
        CudaDeviceMemory d_dx(sizeof(double) * n_ * batchSize_);
        CudaDeviceMemory d_dz(sizeof(double) * m_ * batchSize_);
        CudaDeviceMemory d_ds(sizeof(double) * m_ * batchSize_);

        CUDA_CHECK(cudaMemcpy(d_dx.get(), dx_aug.data(), sizeof(double) * n_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dz.get(), dz_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ds.get(), ds_aug.data(), sizeof(double) * m_ * batchSize_, cudaMemcpyHostToDevice));

        BatchedVector dx_bar(d_dx.get(), n_, batchSize_);
        BatchedVector dz_bar(d_dz.get(), m_, batchSize_);
        BatchedVector ds_bar(d_ds.get(), m_, batchSize_);

        moreau::backward(*solver_->diff_state(), dx_bar, dz_bar, ds_bar, *solver_, 0);
        cudaDeviceSynchronize();

        // Download augmented gradients
        std::vector<double> dP_aug(nnzP_ * batchSize_);
        std::vector<double> dq_aug(n_ * batchSize_);
        std::vector<double> dA_aug(nnzA_ * batchSize_);
        std::vector<double> db_aug(m_ * batchSize_);

        CUDA_CHECK(cudaMemcpy(dP_aug.data(), solver_->diff_state()->dP_values.data(),
                   sizeof(double) * nnzP_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dq_aug.data(), solver_->diff_state()->dq.data(),
                   sizeof(double) * n_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dA_aug.data(), solver_->diff_state()->dA_values.data(),
                   sizeof(double) * nnzA_ * batchSize_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(db_aug.data(), solver_->diff_state()->db.data(),
                   sizeof(double) * m_ * batchSize_, cudaMemcpyDeviceToHost));

        // Reverse-map gradients to original dimensions
        std::vector<double> dP_orig(nnzP_user_ * batchSize_, 0.0);
        std::vector<double> dq_orig(n_user_ * batchSize_, 0.0);
        std::vector<double> dA_orig(nnzA_user_ * batchSize_, 0.0);
        std::vector<double> db_orig(m_user_ * batchSize_, 0.0);

        for (int64_t b = 0; b < batchSize_; b++) {
            chordal_->adjoint_augment_dP(dP_aug.data() + b * nnzP_, dP_orig.data() + b * nnzP_user_);
            chordal_->adjoint_augment_dq(dq_aug.data() + b * n_, dq_orig.data() + b * n_user_);
            chordal_->adjoint_augment_dA(dA_aug.data() + b * nnzA_, dA_orig.data() + b * nnzA_user_);
            chordal_->adjoint_augment_db(db_aug.data() + b * m_, db_orig.data() + b * m_user_);
        }

        // Upload to device for numpy array creation
        CudaDeviceMemory d_dP_out(sizeof(double) * std::max((int64_t)1, nnzP_user_ * batchSize_));
        CudaDeviceMemory d_dq_out(sizeof(double) * n_user_ * batchSize_);
        CudaDeviceMemory d_dA_out(sizeof(double) * std::max((int64_t)1, nnzA_user_ * batchSize_));
        CudaDeviceMemory d_db_out(sizeof(double) * m_user_ * batchSize_);

        if (nnzP_user_ > 0) CUDA_CHECK(cudaMemcpy(d_dP_out.get(), dP_orig.data(), sizeof(double) * nnzP_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dq_out.get(), dq_orig.data(), sizeof(double) * n_user_ * batchSize_, cudaMemcpyHostToDevice));
        if (nnzA_user_ > 0) CUDA_CHECK(cudaMemcpy(d_dA_out.get(), dA_orig.data(), sizeof(double) * nnzA_user_ * batchSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_db_out.get(), db_orig.data(), sizeof(double) * m_user_ * batchSize_, cudaMemcpyHostToDevice));

        nb::dict result;
        result["dP_values"] = make_numpy_array_2d<double>((double*)d_dP_out.get(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(nnzP_user_));
        result["dq"] = make_numpy_array_2d<double>((double*)d_dq_out.get(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(n_user_));
        result["dA_values"] = make_numpy_array_2d<double>((double*)d_dA_out.get(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(nnzA_user_));
        result["db"] = make_numpy_array_2d<double>((double*)d_db_out.get(),
            static_cast<size_t>(batchSize_), static_cast<size_t>(m_user_));
        return result;
    }

    // Return user-facing dimensions (original, before chordal augmentation)
    int64_t n() const { return n_user_; }
    int64_t m() const { return m_user_; }
    int64_t batch_size() const { return batchSize_; }
    int64_t nnzP() const { return nnzP_user_; }
    int64_t nnzA() const { return nnzA_user_; }
    bool enable_grad() const { return solver_->settings.enableGrad; }
    int device_id() const { return device_id_; }

    // Solver-internal dimensions (augmented if chordal is active)
    int64_t n_solver() const { return n_; }
    int64_t m_solver() const { return m_; }
    int64_t nnzP_solver() const { return nnzP_; }
    int64_t nnzA_solver() const { return nnzA_; }
};


NB_MODULE(_moreau_cuda, m) {
    m.doc() = "Moreau CUDA: GPU-accelerated conic optimization solver";

    // Disable nanobind leak warnings at interpreter shutdown
    // These are false positives - objects are properly cleaned up by GC when
    // gc.collect() is called, but at shutdown Python doesn't run GC on all
    // remaining objects before nanobind's atexit handler runs.
    nb::set_leak_warnings(false);

    // Device management functions
    m.def("device_count", []() {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }, "Get the number of available CUDA devices");

    m.def("get_device", []() {
        int device = 0;
        cudaGetDevice(&device);
        return device;
    }, "Get the current CUDA device");

    m.def("set_device", [](int device_id) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
        }
    }, nb::arg("device_id"), "Set the current CUDA device");

    m.def("get_device_name", [](int device_id) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaGetDeviceProperties failed: " + std::string(cudaGetErrorString(err)));
        }
        return std::string(prop.name);
    }, nb::arg("device_id") = 0, "Get the name of a CUDA device");

    // XConeKind enum (direct-x cone kinds)
    nb::enum_<XConeKind>(m, "XConeKind")
        .value("Nonneg",   XConeKind::Nonneg,    "Nonnegative orthant on x[indices]")
        .value("SOC",      XConeKind::SOC,        "Second-order cone on x[indices]")
        .value("PSD",      XConeKind::PSD,        "PSD triangle cone on svec(x[indices])")
        .value("Exp",      XConeKind::Exp,        "Exponential cone on x[indices] (3D)")
        .value("Power",    XConeKind::Power,      "Power cone on x[indices] (3D, with alpha)")
        .value("GenPower", XConeKind::GenPower,   "Generalized power cone on x[indices]")
        .export_values();

    // SupportedXConeT: descriptor for one direct-x cone.
    //
    // Construction is positional + keyword:
    //   SupportedXConeT(kind=XConeKind.SOC, indices=[3, 4, 5])
    //   SupportedXConeT(kind=XConeKind.PSD, indices=[..6 entries..], psd_k=3)
    //   SupportedXConeT(kind=XConeKind.Power, indices=[0, 1, 2], power_alpha=0.6)
    //   SupportedXConeT(kind=XConeKind.GenPower, indices=[0,1,2,3],
    //                   gen_power_alphas=[0.5,0.5], gen_power_dim2=2)
    nb::class_<SupportedXConeT>(m, "SupportedXConeT")
        .def(nb::init<>())
        .def("__init__",
             [](SupportedXConeT* self, XConeKind kind,
                std::vector<int64_t> indices, int64_t psd_k,
                double power_alpha,
                std::vector<double> gen_power_alphas,
                int64_t gen_power_dim2) {
                 new (self) SupportedXConeT{kind, std::move(indices), psd_k,
                                            power_alpha,
                                            std::move(gen_power_alphas),
                                            gen_power_dim2};
             },
             nb::arg("kind"), nb::arg("indices"),
             nb::arg("psd_k") = int64_t{0},
             nb::arg("power_alpha") = 0.0,
             nb::arg("gen_power_alphas") = std::vector<double>{},
             nb::arg("gen_power_dim2") = int64_t{0},
             "Create a direct-x cone descriptor.\n"
             "  psd_k          required for PSD kind\n"
             "  power_alpha    required for Power kind (alpha in (0,1))\n"
             "  gen_power_alphas / gen_power_dim2  required for GenPower kind")
        .def_rw("kind", &SupportedXConeT::kind,
            "Cone kind (Nonneg, SOC, PSD, Exp, Power, or GenPower)")
        .def_rw("indices", &SupportedXConeT::indices,
            "Indices into x constrained by this cone (distinct; non-negative)")
        .def_rw("psd_k", &SupportedXConeT::psd_k,
            "PSD matrix dimension (only used when kind == PSD)")
        .def_rw("power_alpha", &SupportedXConeT::power_alpha,
            "Power cone exponent alpha in (0,1) (only used when kind == Power)")
        .def_rw("gen_power_alphas", &SupportedXConeT::gen_power_alphas,
            "GenPower alpha weights (positive, sum to 1.0; only when kind == GenPower)")
        .def_rw("gen_power_dim2", &SupportedXConeT::gen_power_dim2,
            "GenPower tail dimension >= 1 (only used when kind == GenPower)")
        .def("numel", &SupportedXConeT::numel,
            "Number of primal indices covered by this cone")
        .def("degree", &SupportedXConeT::degree,
            "Cone degree (Nonneg: numel; SOC: 1; PSD: psd_k; asymmetric: 3 or numel)");

    // Solver status enum
    nb::enum_<SolverStatus>(m, "SolverStatus")
        .value("Unsolved", SolverStatus::Unsolved)
        .value("Solved", SolverStatus::Solved)
        .value("PrimalInfeasible", SolverStatus::PrimalInfeasible)
        .value("DualInfeasible", SolverStatus::DualInfeasible)
        .value("AlmostSolved", SolverStatus::AlmostSolved)
        .value("AlmostPrimalInfeasible", SolverStatus::AlmostPrimalInfeasible)
        .value("AlmostDualInfeasible", SolverStatus::AlmostDualInfeasible)
        .value("MaxIterations", SolverStatus::MaxIterations)
        .value("MaxTime", SolverStatus::MaxTime)
        .value("NumericalError", SolverStatus::NumericalError)
        .value("InsufficientProgress", SolverStatus::InsufficientProgress)
        .value("CallbackTerminated", SolverStatus::CallbackTerminated)
        .export_values();

    // Cones structure
    nb::class_<Cones>(m, "Cones")
        .def(nb::init<>())
        .def_rw("num_zero_cones", &Cones::numZeroCones,
            "Number of zero cone constraints (equality constraints)")
        .def_rw("num_nonneg_cones", &Cones::numNonnegCones,
            "Number of nonnegative cone constraints (>= 0)")
        .def_rw("num_exp_cones", &Cones::numExpCones,
            "Number of exponential cones (3 constraints per cone)")
        .def_ro("num_so_cones", &Cones::numSocCones,
            "Number of second-order cones (read-only, derived from soc_cone_dims)")
        .def_prop_rw("soc_cone_dims",
            [](const Cones& c) { return c.socConeDims; },
            [](Cones& c, std::vector<int64_t> dims) {
                for (size_t i = 0; i < dims.size(); i++) {
                    if (dims[i] < 2) {
                        throw std::invalid_argument(
                            "soc_cone_dims[" + std::to_string(i) + "] = " +
                            std::to_string(dims[i]) + ", must be >= 2");
                    }
                }
                c.socConeDims = std::move(dims);
                c.numSocCones = static_cast<int64_t>(c.socConeDims.size());
            },
            "Dimensions of each second-order cone (each >= 2)")
        .def_rw("num_power_cones", &Cones::numPowerCones,
            "Number of power cones (3 constraints per cone)")
        .def_rw("power_alphas", &Cones::powerAlphas,
            "Alpha parameters for power cones (one per cone)")
        .def_ro("num_psd_cones", &Cones::numPsdCones,
            "Number of PSD cones (read-only, derived from psd_cone_dims)")
        .def_prop_rw("psd_cone_dims",
            [](const Cones& c) { return c.psdConeDims; },
            [](Cones& c, std::vector<int64_t> dims) {
                for (size_t i = 0; i < dims.size(); i++) {
                    if (dims[i] < 1) {
                        throw std::invalid_argument(
                            "psd_cone_dims[" + std::to_string(i) + "] = " +
                            std::to_string(dims[i]) + ", must be >= 1");
                    }
                }
                c.psdConeDims = std::move(dims);
                c.numPsdCones = static_cast<int64_t>(c.psdConeDims.size());
            },
            "Matrix dimensions of each PSD cone (each >= 1)")
        .def_rw("num_gen_power_cones", &Cones::numGenPowerCones,
            "Number of generalized power cones")
        .def_rw("gen_power_alphas", &Cones::genPowerAlphas,
            "Flattened alpha parameters for generalized power cones")
        .def_rw("gen_power_dim1s", &Cones::genPowerDim1s,
            "dim1 (number of alphas) for each generalized power cone")
        .def_rw("gen_power_dim2s", &Cones::genPowerDim2s,
            "dim2 for each generalized power cone")
        .def_rw("dir_cones", &Cones::dir_cones,
            "Direct-x cones: each entry constrains a subvector of x directly, "
            "rather than through a slack variable.")
        .def("total_constraints", &Cones::totalConstraints,
            "Get total number of constraints across all cones")
        .def("degree", &Cones::degree,
            "Get total cone degree for complementarity measure");

    // IPMSettings structure (nested in Settings)
    nb::class_<IPMSettings>(m, "IPMSettings")
        .def(nb::init<>())
        .def_rw("max_step_fraction", &IPMSettings::maxStepFraction,
            "Maximum interior point step length (default: 0.99)")
        .def_rw("tol_gap_abs", &IPMSettings::tolGapAbs,
            "Absolute duality gap tolerance (default: 1e-8)")
        .def_rw("tol_gap_rel", &IPMSettings::tolGapRel,
            "Relative duality gap tolerance (default: 1e-8)")
        .def_rw("tol_feas", &IPMSettings::tolFeas,
            "Feasibility check tolerance (default: 1e-8)")
        .def_rw("tol_infeas_abs", &IPMSettings::tolInfeasAbs,
            "Absolute infeasibility tolerance (default: 1e-8)")
        .def_rw("tol_infeas_rel", &IPMSettings::tolInfeasRel,
            "Relative infeasibility tolerance (default: 1e-8)")
        .def_rw("tol_ktratio", &IPMSettings::tolKtRatio,
            "Kappa/tau tolerance (default: 1e-6)")
        .def_rw("reduced_tol_gap_abs", &IPMSettings::reducedTolGapAbs,
            "Reduced absolute duality gap tolerance (default: 5e-5)")
        .def_rw("reduced_tol_gap_rel", &IPMSettings::reducedTolGapRel,
            "Reduced relative duality gap tolerance (default: 5e-5)")
        .def_rw("reduced_tol_feas", &IPMSettings::reducedTolFeas,
            "Reduced feasibility tolerance (default: 1e-4)")
        .def_rw("reduced_tol_infeas_abs", &IPMSettings::reducedTolInfeasAbs,
            "Reduced absolute infeasibility tolerance (default: 5e-12)")
        .def_rw("reduced_tol_infeas_rel", &IPMSettings::reducedTolInfeasRel,
            "Reduced relative infeasibility tolerance (default: 5e-5)")
        .def_rw("reduced_tol_ktratio", &IPMSettings::reducedTolKtRatio,
            "Reduced kappa/tau tolerance (default: 1e-4)")
        .def_prop_rw("equilibration_enable",
            [](const IPMSettings &s) { return s.equilibrationSettings.enable; },
            [](IPMSettings &s, bool enable) { s.equilibrationSettings.enable = enable; },
            "Enable equilibration (default: true)")
        .def_rw("linesearch_backtrack_step", &IPMSettings::linesearchBacktrackStep,
            "Line search backtracking step (default: 0.8)")
        .def_rw("min_switch_step_length", &IPMSettings::minSwitchStepLength,
            "Minimum step size for asymmetric cones (default: 1e-1)")
        .def_rw("min_terminate_step_length", &IPMSettings::minTerminateStepLength,
            "Minimum step size for termination (default: 1e-4)")
        .def_rw("static_regularization_enable", &IPMSettings::staticRegularizationEnable,
            "Enable static regularization (default: true)")
        .def_rw("static_regularization_constant", &IPMSettings::staticRegularizationConstant,
            "Static regularization constant (default: 1e-8)")
        .def_rw("static_regularization_proportional", &IPMSettings::staticRegularizationProportional,
            "Static regularization proportional term (default: eps^2)")
        .def_rw("dynamic_regularization_enable", &IPMSettings::dynamicRegularizationEnable,
            "Enable dynamic regularization (default: true)")
        .def_rw("dynamic_regularization_eps", &IPMSettings::dynamicRegularizationEps,
            "Dynamic regularization threshold (default: 1e-13)")
        .def_rw("dynamic_regularization_delta", &IPMSettings::dynamicRegularizationDelta,
            "Dynamic regularization shift (default: 2e-7)")
        .def_rw("kkt_solver_type", &IPMSettings::kktSolverType,
            "KKT linear system solver type (default: Auto)")
        .def_rw("diff_method", &IPMSettings::diffMethod,
            "Differentiation method for backward pass (default: Auto)")
        .def_rw("diff_smoothing_mu", &IPMSettings::diffSmoothingMu,
            "Smoothing parameter mu for smoothed differentiation (default: 1e-4)")
        .def_rw("diff_smoothing_step_factor", &IPMSettings::diffSmoothingStepFactor,
            "Step factor for central-path refinement (default: 30)")
        .def_rw("max_lu_nnz", &IPMSettings::maxLuNnz,
            "cuDSS max LU fill-in limit. -1 = cuDSS default (100*nnz). "
            "Set lower to prevent OOM for dense SDP problems.")
        .def_rw("cudss_ir_steps", &IPMSettings::cudssIrSteps,
            "cuDSS iterative refinement steps (default: 2)")
        .def_rw("cudss_pivot_enable", &IPMSettings::cudssPivotEnable,
            "Enable cuDSS pivoting for numerically challenging problems (default: false)")
        .def_rw("chordal_decomposition_merge_method",
            &IPMSettings::chordalDecompositionMergeMethod,
            "Chordal merge strategy for sparse PSD: 'clique_graph' (default, "
            "matches CPU), 'parent_child', or 'none'.");

    // SolverType enum
    nb::enum_<SolverType>(m, "SolverType")
        .value("IPM", SolverType::IPM, "Interior Point Method solver")
        .export_values();

    // KKTSolverType enum
    nb::enum_<KKTSolverType>(m, "KKTSolverType")
        .value("Auto", KKTSolverType::Auto, "Auto-select (uses CuDSS)")
        .value("CuDSS", KKTSolverType::CuDSS, "Sparse LDL via cuDSS (default, works for all problems)")
        .value("Riccati", KKTSolverType::Riccati, "Block-tridiagonal Riccati (MPC/MHE structured problems)")
        .value("Woodbury", KKTSolverType::Woodbury, "Woodbury identity (diagonal P + low-rank A, e.g. portfolio)")
        .export_values();

    // DiffMethod enum
    nb::enum_<DiffMethod>(m, "DiffMethod")
        .value("Auto", DiffMethod::Auto, "Auto-select (uses Exact)")
        .value("Exact", DiffMethod::Exact, "Exact cone projection Jacobian")
        .value("Smoothed", DiffMethod::Smoothed, "Smoothed central-path derivative")
        .export_values();

    // Settings structure
    nb::class_<Settings>(m, "Settings")
        .def(nb::init<>())
        .def_rw("device_id", &Settings::deviceId,
            "CUDA device ID, -1 for current device (default: -1)")
        .def_rw("solver", &Settings::solver,
            "Solver algorithm type (default: SolverType.IPM)")
        .def_rw("max_iter", &Settings::maxIter,
            "Maximum number of iterations (default: 200)")
        .def_rw("time_limit", &Settings::timeLimit,
            "Maximum solve time in seconds (default: infinity)")
        .def_rw("verbose", &Settings::verbose,
            "Enable verbose output (default: true)")
        .def_rw("enable_grad", &Settings::enableGrad,
            "Enable gradient computation (default: false)")
        .def_rw("yolo", &Settings::yolo,
            "YOLO mode: fixed iterations, zero GPU sync (default: false)")
        .def_rw("yolo_num_iters", &Settings::yoloNumIters,
            "Number of iterations in YOLO mode (default: 15)")
        .def_rw("ipm", &Settings::ipm,
            "IPM-specific settings (IPMSettings object)");

    // Main solver class
    nb::class_<PyMoreauSolver>(m, "Solver")
        .def(nb::init<int64_t, int64_t, int64_t,
                      input_array<int64_t>, input_array<int64_t>,
                      input_array<int64_t>, input_array<int64_t>,
                      const Cones&, Settings*, bool,
                      std::optional<std::vector<bool>>>(),
            nb::arg("n"),
            nb::arg("m"),
            nb::arg("batch_size") = 1,
            nb::arg("P_row_offsets"),
            nb::arg("P_col_indices"),
            nb::arg("A_row_offsets"),
            nb::arg("A_col_indices"),
            nb::arg("cones"),
            nb::arg("settings").none() = nb::none(),
            nb::arg("enable_grad") = false,
            nb::arg("b_sparsity_pattern").none() = nb::none(),
            R"pbdoc(
                Create a Moreau solver for conic optimization problems.

                Solves problems of the form:
                    minimize    (1/2)x'Px + q'x
                    subject to  Ax + s = b
                                x in K1,  s in K2

                K2 constrains the slack s; K1 constrains x directly (direct-x
                cones). Each is a product of cones.

                Parameters:
                    n: Number of primal variables
                    m: Number of constraints
                    batch_size: Number of problems to solve in parallel (default: 1)
                    P_row_offsets: CSR row offsets for P matrix (size n+1)
                    P_col_indices: CSR column indices for P matrix
                    A_row_offsets: CSR row offsets for A matrix (size m+1)
                    A_col_indices: CSR column indices for A matrix
                    cones: Cone structure specification
                    settings: Solver settings (optional). Use settings.device_id for multi-GPU.
                    enable_grad: Enable gradient computation (optional, default: False)
            )pbdoc")
        .def("solve", &PyMoreauSolver::solve,
            nb::arg("P_values"),
            nb::arg("A_values"),
            nb::arg("q"),
            nb::arg("b"),
            R"pbdoc(
                Solve the optimization problem.

                Parameters:
                    P_values: Values for P matrix, shape (batch_size, nnzP)
                    A_values: Values for A matrix, shape (batch_size, nnzA)
                    q: Linear cost vector, shape (batch_size, n)
                    b: Constraint RHS, shape (batch_size, m)

                Returns:
                    Dictionary containing:
                        - x: Primal solution, shape (batch_size, n)
                        - s: Slack variables, shape (batch_size, m)
                        - z: Dual variables, shape (batch_size, m)
                        - tau: Homogeneous scaling factor
                        - kappa: Homogeneous slack
                        - status: Solver status (SolverStatus enum)
                        - iterations: int for single problem, array of int32 for batch (per-problem iteration count)
                        - solve_time: Solution time in seconds
                        - obj_val: Primal objective value
                        - dual_obj_val: Dual objective value
            )pbdoc")
        .def("solve_warm_start", &PyMoreauSolver::solve_warm_start,
            nb::arg("P_values"),
            nb::arg("A_values"),
            nb::arg("q"),
            nb::arg("b"),
            nb::arg("warm_x"),
            nb::arg("warm_z"),
            nb::arg("warm_s"),
            nb::arg("warm_z_x") = nb::none(),
            R"pbdoc(
                Solve the optimization problem with warm start.

                Parameters:
                    P_values: Values for P matrix, shape (batch_size, nnzP)
                    A_values: Values for A matrix, shape (batch_size, nnzA)
                    q: Linear cost vector, shape (batch_size, n)
                    b: Constraint RHS, shape (batch_size, m)
                    warm_x: Warm start primal variables, shape (batch_size, n)
                    warm_z: Warm start dual variables, shape (batch_size, m)
                    warm_z_x: Optional warm start for direct-x cone duals,
                              shape (batch_size, total_x_dim). Required when
                              the problem has direct-x cones — pass None or
                              omit when no dir_cones are present.
                    warm_s: Warm start slack variables, shape (batch_size, m)

                Returns:
                    Dictionary (same as solve)
            )pbdoc")
        .def("get_dimensions", &PyMoreauSolver::get_dimensions,
            "Get problem dimensions (n, m, batch_size, nnzP, nnzA)")
        .def("memory_usage", &PyMoreauSolver::memory_usage,
            "Get total GPU memory usage in bytes")
        .def("solve_from_device_pointers", &PyMoreauSolver::solve_from_device_pointers,
            nb::arg("P_ptr"), nb::arg("A_ptr"), nb::arg("q_ptr"), nb::arg("b_ptr"),
            "Solve with raw CUDA device pointers (from tensor.data_ptr())")
        .def("load_data_for_backward_from_device_pointers",
            &PyMoreauSolver::load_data_for_backward_from_device_pointers,
            nb::arg("P_ptr"), nb::arg("A_ptr"), nb::arg("q_ptr"), nb::arg("b_ptr"),
            "Load problem data and equilibrate without solving (for backward pass)")
        .def("load_backward_state_from_device_pointers",
            &PyMoreauSolver::load_backward_state_from_device_pointers,
            nb::arg("P_ptr"), nb::arg("A_ptr"), nb::arg("q_ptr"), nb::arg("b_ptr"),
            nb::arg("x_ptr"), nb::arg("z_ptr"), nb::arg("s_ptr"),
            nb::arg("z_x_ptr") = 0,
            "Restore full backward state (problem data + equilibration + solution) without re-solving. Pass z_x_ptr=0 for slack-only problems; otherwise must point to a [batch_size, total_xn] array of direct-x cone duals in user frame.")
        .def("solve_to_device_pointers", &PyMoreauSolver::solve_to_device_pointers,
            nb::arg("P_ptr"), nb::arg("A_ptr"), nb::arg("q_ptr"), nb::arg("b_ptr"),
            nb::arg("x_out_ptr"), nb::arg("z_out_ptr"), nb::arg("s_out_ptr"),
            nb::arg("status_out_ptr"), nb::arg("obj_out_ptr"),
            nb::arg("z_x_out_ptr") = 0,
            "Solve with device pointers, output to device pointers (full zero-copy). Pass z_x_out_ptr=0 for slack-only problems; otherwise must point to a [batch_size, total_xn] output buffer.")
        .def("solve_warm_start_to_device_pointers", &PyMoreauSolver::solve_warm_start_to_device_pointers,
            nb::arg("P_ptr"), nb::arg("A_ptr"), nb::arg("q_ptr"), nb::arg("b_ptr"),
            nb::arg("warm_x_ptr"), nb::arg("warm_z_ptr"), nb::arg("warm_s_ptr"),
            nb::arg("x_out_ptr"), nb::arg("z_out_ptr"), nb::arg("s_out_ptr"),
            nb::arg("status_out_ptr"), nb::arg("obj_out_ptr"),
            nb::arg("warm_z_x_ptr") = 0,
            nb::arg("z_x_out_ptr") = 0,
            "Solve with warm start using device pointers (full zero-copy). Pass warm_z_x_ptr=0 for problems without direct-x cones; z_x_out_ptr to a [batch_size, total_xn] buffer to receive direct-x cone duals.")
        .def_prop_ro("n", &PyMoreauSolver::n)
        .def_prop_ro("m", &PyMoreauSolver::m)
        .def_prop_ro("batch_size", &PyMoreauSolver::batch_size)
        .def_prop_ro("nnzP", &PyMoreauSolver::nnzP)
        .def_prop_ro("nnzA", &PyMoreauSolver::nnzA)
        .def_prop_ro("enable_grad", &PyMoreauSolver::enable_grad)
        .def_prop_ro("device_id", &PyMoreauSolver::device_id)
        .def_prop_ro("n_solver", &PyMoreauSolver::n_solver,
            "Solver-internal n (augmented if chordal is active)")
        .def_prop_ro("m_solver", &PyMoreauSolver::m_solver,
            "Solver-internal m (augmented if chordal is active)")
        .def_prop_ro("nnzP_solver", &PyMoreauSolver::nnzP_solver,
            "Solver-internal nnzP (augmented if chordal is active)")
        .def_prop_ro("nnzA_solver", &PyMoreauSolver::nnzA_solver,
            "Solver-internal nnzA (augmented if chordal is active)")
        .def("get_solve_info", &PyMoreauSolver::get_solve_info,
            "Get solve info (iterations, solve_time, construction_time, setup_time) after a solve")
        .def("refine_smoothing_iterate", &PyMoreauSolver::refine_smoothing_iterate,
            "Run centering iterations for smoothed differentiation (must be called after solve)")
        .def("cache_solution_for_backward", &PyMoreauSolver::cache_solution_for_backward,
            "Cache solution state for backward differentiation (must be called after solve)")
        .def("backward", &PyMoreauSolver::backward,
            nb::arg("dx"), nb::arg("dz"), nb::arg("ds"),
            nb::arg("dz_x") = nb::none(),
            R"pbdoc(
                Compute gradients via backward differentiation.

                Requires enable_grad=True. Solution is automatically cached after solve().

                Parameters:
                    dx: Upstream gradient w.r.t. x, shape (batch_size, n)
                    dz: Upstream gradient w.r.t. z, shape (batch_size, m)
                    ds: Upstream gradient w.r.t. s, shape (batch_size, m)
                    dz_x: Optional upstream gradient w.r.t. direct-x cone duals,
                          shape (batch_size, total_xn). Pass None for slack-only
                          problems or to skip direct-x dual gradients.

                Returns:
                    Dictionary containing:
                        - dP_values: Gradient w.r.t. P matrix values, shape (batch_size, nnzP)
                        - dq: Gradient w.r.t. q vector, shape (batch_size, n)
                        - dA_values: Gradient w.r.t. A matrix values, shape (batch_size, nnzA)
                        - db: Gradient w.r.t. b vector, shape (batch_size, m)
            )pbdoc")
        .def("backward_to_device_pointers", &PyMoreauSolver::backward_to_device_pointers,
            nb::arg("dx_ptr"), nb::arg("dz_ptr"), nb::arg("ds_ptr"),
            nb::arg("dP_out_ptr"), nb::arg("dq_out_ptr"), nb::arg("dA_out_ptr"), nb::arg("db_out_ptr"),
            nb::arg("dz_x_ptr") = 0,
            "Backward differentiation with device pointers (zero-copy for PyTorch). Pass dz_x_ptr=0 for slack-only or to skip dz_x backprop; otherwise must point to a [batch_size, total_xn] device buffer of upstream gradients on direct-x cone duals.")
        .def("set_solution_from_device_pointers", &PyMoreauSolver::set_solution_from_device_pointers,
            nb::arg("x_ptr"), nb::arg("z_ptr"), nb::arg("s_ptr"),
            "Set solution state from device pointers for backward differentiation")
        .def("get_equilibration_to_device_pointers", &PyMoreauSolver::get_equilibration_to_device_pointers,
            nb::arg("d_ptr"), nb::arg("e_ptr"), nb::arg("c_ptr"),
            "Get equilibration factors to device pointers (for saving state)")
        .def("set_equilibration_from_device_pointers", &PyMoreauSolver::set_equilibration_from_device_pointers,
            nb::arg("d_ptr"), nb::arg("e_ptr"), nb::arg("c_ptr"),
            "Set equilibration factors from device pointers (for restoring state)");

    // Version info
    m.attr("__version__") = "0.4.0";
}
