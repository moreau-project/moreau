// vector/kernels.cu
#include "moreau/vector/vector_kernels.cuh"
#include "moreau/vector/vector.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/cuda/utils.cuh"
#include "moreau/residuals/residuals_kernels.cuh"
#include "moreau/profiling/profiler.hpp"
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>

__global__ void setConstantKernel(double* __restrict__ data, double value, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

/**
 * @brief Batch set constant kernel - sets multiple arrays to the same value in one launch
 *
 * Each block handles one array. Thread grid-strides over the array elements.
 * This replaces N separate setConstantKernel calls with 1 kernel launch.
 *
 * @param ptrs Array of N pointers to arrays to initialize
 * @param sizes Array of N sizes for each array
 * @param value The constant value to set
 * @param num_arrays Number of arrays (N)
 */
__global__ void batchSetConstantKernel(
    double** ptrs,
    const int64_t* __restrict__ sizes,
    double value,
    int64_t num_arrays
) {
    int64_t array_idx = blockIdx.x;
    if (array_idx >= num_arrays) return;

    double* data = ptrs[array_idx];
    int64_t size = sizes[array_idx];

    // Grid-stride loop over elements
    for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
        data[i] = value;
    }
}

/**
 * @brief Reciprocal kernel - computes out = 1.0 / in
 *
 * Replaces setToConstant(1.0) + elementwise_div pattern
 */
__global__ void reciprocalKernel(double* __restrict__ out, const double* __restrict__ in, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0 / in[idx];
    }
}

// Scale array by scalar
__global__ void scale_by_scalar_kernel(
    double* __restrict__ array,
    double scale,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (idx >= n || b >= batch_size) return;
    array[b * n + idx] *= scale;
}

// Compute infinity norm using warp-shuffle + shared memory reduction
__global__ void compute_inf_norm_kernel(
    const double* __restrict__ array,
    double* __restrict__ result,
    int64_t n,
    int64_t batch_size)
{
    int tid = threadIdx.x;
    int64_t b = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = (i < n) ? fabs(array[b * n + i]) : 0.0;

    // Warp-level reduction first (no shared memory needed within warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // Inter-warp reduction via shared memory (only lane 0 of each warp participates)
    __shared__ double warp_results[8]; // max 256/32 = 8 warps
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        val = (lane < num_warps) ? warp_results[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) {
            atomicMaxDouble(&result[b], val);
        }
    }
}


// Compute mean using warp-shuffle + shared memory reduction
__global__ void compute_mean_kernel(
    const double* __restrict__ array,
    double* __restrict__ partial_sums,
    int64_t n,
    int64_t batch_size)
{
    int tid = threadIdx.x;
    int64_t b = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = (i < n) ? array[b * n + i] : 0.0;

    // Warp-level reduction first
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ double warp_results[8]; // max 256/32 = 8 warps
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        val = (lane < num_warps) ? warp_results[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            atomicAdd(&partial_sums[b], val);
        }
    }
}

namespace moreau {

void setConstantDevice(double* data, double value, int64_t size, cudaStream_t stream) {
    if (size <= 0) return;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    MOREAU_KERNEL_LAUNCH(setConstantKernel, numBlocks, blockSize, 0, stream,
        data, value, size);
}

void batchSetConstant(
    const std::vector<double*>& ptrs,
    const std::vector<int64_t>& sizes,
    double value,
    cudaStream_t stream
) {
    if (ptrs.empty()) return;

    int64_t num_arrays = ptrs.size();

    // Thread-local cached device pointers (grow-only, reused across calls)
    thread_local double** d_ptrs = nullptr;
    thread_local int64_t* d_sizes = nullptr;
    thread_local int64_t cached_capacity = 0;

    // Grow cache if needed (never shrink)
    if (num_arrays > cached_capacity) {
        if (d_ptrs) cudaFree(d_ptrs);
        if (d_sizes) cudaFree(d_sizes);

        cudaError_t err = cudaMalloc(&d_ptrs, num_arrays * sizeof(double*));
        if (err != cudaSuccess) {
            d_ptrs = nullptr;
            d_sizes = nullptr;
            cached_capacity = 0;
            throw std::runtime_error(std::string("batchSetConstant: cudaMalloc for d_ptrs failed: ") + cudaGetErrorString(err));
        }

        err = cudaMalloc(&d_sizes, num_arrays * sizeof(int64_t));
        if (err != cudaSuccess) {
            cudaFree(d_ptrs);
            d_ptrs = nullptr;
            d_sizes = nullptr;
            cached_capacity = 0;
            throw std::runtime_error(std::string("batchSetConstant: cudaMalloc for d_sizes failed: ") + cudaGetErrorString(err));
        }

        cached_capacity = num_arrays;
    }

    // Copy to device (async, no sync needed)
    cudaMemcpyAsync(d_ptrs, ptrs.data(), num_arrays * sizeof(double*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sizes, sizes.data(), num_arrays * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Find max size to determine thread count
    int64_t maxSize = *std::max_element(sizes.begin(), sizes.end());
    int threadsPerBlock = (maxSize < 256) ? ((maxSize + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(batchSetConstantKernel, num_arrays, threadsPerBlock, 0, stream,
        d_ptrs, d_sizes, value, num_arrays);

    // No sync or free needed - device pointers are cached for reuse
    // Stream ordering ensures memcpy completes before kernel reads
}

void reciprocal(double* out, const double* in, int64_t size, cudaStream_t stream) {
    if (size <= 0) return;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    MOREAU_KERNEL_LAUNCH(reciprocalKernel, numBlocks, blockSize, 0, stream,
        out, in, size);
}

void scale_by_scalar(
    BatchedVector& d_array,
    double scale,
    cudaStream_t stream)
{
    if (d_array.n() == 0 || d_array.batchSize() == 0) return;
    dim3 blk(256, 1, 1);
    dim3 grd((d_array.n() + blk.x - 1) / blk.x, d_array.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(scale_by_scalar_kernel, grd, blk, 0, stream,
        d_array.data(), scale, d_array.n(), d_array.batchSize());
}

void compute_mean(
    const BatchedVector& d_array,
    BatchedVector& d_mean_result,
    cudaStream_t stream
) {
    assert(d_mean_result.n() == 1 && "d_mean_result must have length 1 per batch");
    assert(d_mean_result.batchSize() == d_array.batchSize());

    if (d_array.n() == 0) {
        setConstantDevice(d_mean_result.data(), 0.0, d_mean_result.batchSize(), stream);
        return;
    }

    // DON'T use memset - initialize with a kernel instead
    // Or use a separate initialization kernel that's guaranteed to finish first
    
    dim3 blk(256, 1, 1);
    dim3 grd((d_array.n() + blk.x - 1) / blk.x, d_array.batchSize(), 1);
    
    // First, initialize output to zero
    setConstantDevice(d_mean_result.data(), 0.0, d_array.batchSize(), stream);
    
    // Now compute the sum
    MOREAU_KERNEL_LAUNCH(compute_mean_kernel, grd, blk, 0, stream,
        d_array.data(), d_mean_result.data(), d_array.n(), d_array.batchSize());

    // Divide by n to get mean
    scale_by_scalar(d_mean_result, 1.0 / d_array.n(), stream);
}

void compute_inf_norm(
    const BatchedVector& d_array,
    BatchedVector& d_norm_result,
    cudaStream_t stream
) {
    cudaMemsetAsync(d_norm_result.data(), 0, sizeof(double) * d_array.batchSize(), stream);

    if (d_array.n() == 0 || d_array.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((d_array.n() + blk.x - 1) / blk.x, d_array.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(compute_inf_norm_kernel, grd, blk, 0, stream,
        d_array.data(), d_norm_result.data(), d_array.n(), d_array.batchSize());
}

// AXPBY kernel: y = a*x + b*y
__global__ void axpby_kernel(
    double* __restrict__ y,
    double a,
    const double* __restrict__ x,
    double b,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    y[global_idx] = a * x[global_idx] + b * y[global_idx];
}

void axpby(
    BatchedVector& y,
    double a,
    const BatchedVector& x,
    double b,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());

    if (y.n() == 0 || y.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(axpby_kernel, grd, blk, 0, stream,
        y.data(), a, x.data(), b, y.n(), y.batchSize());
}

// WAXPBY kernel: z = a*x + b*y
__global__ void waxpby_kernel(
    double* __restrict__ z,
    double a,
    const double* __restrict__ x,
    double b,
    const double* __restrict__ y,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    z[global_idx] = a * x[global_idx] + b * y[global_idx];
}

void waxpby(
    BatchedVector& z,
    double a,
    const BatchedVector& x,
    double b,
    const BatchedVector& y,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.n() == z.n());
    assert(x.batchSize() == y.batchSize() && x.batchSize() == z.batchSize());

    if (z.n() == 0 || z.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((z.n() + blk.x - 1) / blk.x, z.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(waxpby_kernel, grd, blk, 0, stream,
        z.data(), a, x.data(), b, y.data(), z.n(), z.batchSize());
}

// Dot product kernel using warp-shuffle + shared memory reduction
__global__ void dot_batched_kernel(
    double* __restrict__ result,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int64_t n)
{
    int tid = threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t base = batch_idx * n;
    double val = (i < n) ? (x[base + i] * y[base + i]) : 0.0;

    // Warp-level reduction first
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ double warp_results[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        val = (lane < num_warps) ? warp_results[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            atomicAdd(&result[batch_idx], val);
        }
    }
}

void dot_batched(
    BatchedVector& result,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(result.n() == 1 && result.batchSize() == x.batchSize());

    // Initialize result to zero
    cudaMemsetAsync(result.data(), 0, sizeof(double) * result.batchSize(), stream);

    if (x.n() == 0 || x.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((x.n() + blk.x - 1) / blk.x, x.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(dot_batched_kernel, grd, blk, 0, stream,
        result.data(), x.data(), y.data(), x.n());
}

// Quadratic form: result = x' * P * y using cuSPARSE
void quad_form_batched(
    BatchedVector& result,
    const void* P_csr_void,
    const BatchedVector& x,
    const BatchedVector& y,
    BatchedVector& temp,
    void*& buffer,
    size_t& buffer_size,
    void*& aligned_buffer_y,
    void*& aligned_buffer_temp,
    size_t& aligned_buffer_size,
    cusparseHandle_t cusparse_handle,
    cudaStream_t stream)
{
    const CSR* P_csr = static_cast<const CSR*>(P_csr_void);

    int64_t n = x.n();
    int64_t batchSize = x.batchSize();

    // We need to compute result = x' * P * y for each batch
    // Step 1: Compute temp = P * y (SpMV for each batch)
    // Step 2: Compute result = x' * temp (dot product for each batch)

    // Use cuSPARSE handle from Solver (no more static singleton)
    cusparseSetStream(cusparse_handle, stream);

    // For multi-batch operations with odd n, we need aligned buffers for cuSPARSE
    // cuSPARSE requires 16-byte alignment, which means n must be even (sizeof(double)=8)
    // If n is already even, batch*n offsets are naturally aligned
    bool needs_alignment = (batchSize > 1) && (n % 2 == 1);

    double* aligned_y = nullptr;
    double* aligned_temp = nullptr;
    int64_t n_padded = n;

    if (needs_alignment) {
        // Round n up to nearest even number for aligned stride
        n_padded = (n + 1) & ~1;

        // Allocate aligned staging buffers for ALL batches with padded stride
        // This allows bulk copy with stride conversion using cudaMemcpy2D
        size_t required_aligned_size = sizeof(double) * n_padded * batchSize;

        // Allocate or reuse cached aligned buffers
        if (required_aligned_size > aligned_buffer_size) {
            if (aligned_buffer_y) cudaFree(aligned_buffer_y);
            if (aligned_buffer_temp) cudaFree(aligned_buffer_temp);
            cudaMalloc(&aligned_buffer_y, required_aligned_size);
            cudaMalloc(&aligned_buffer_temp, required_aligned_size);
            aligned_buffer_size = required_aligned_size;
        }

        aligned_y = static_cast<double*>(aligned_buffer_y);
        aligned_temp = static_cast<double*>(aligned_buffer_temp);

        // Bulk copy: Copy ALL batches at once, converting from n-stride to n_padded-stride
        // This is much more efficient than copying batch-by-batch
        cudaMemcpy2DAsync(
            aligned_y,                      // dst
            n_padded * sizeof(double),      // dst pitch (aligned stride)
            y.data(),                       // src
            n * sizeof(double),             // src pitch (original stride)
            n * sizeof(double),             // width (bytes per batch)
            batchSize,                      // height (number of batches)
            cudaMemcpyDeviceToDevice,
            stream
        );
    }

    // For each batch, compute temp[batch] = P * y[batch]
    double alpha_spmv = 1.0;
    double beta_spmv = 0.0;

    for (int64_t batch = 0; batch < batchSize; batch++) {
        double* y_ptr;
        double* temp_ptr;

        if (needs_alignment) {
            // Use aligned buffers with padded stride (already copied in bulk above)
            y_ptr = aligned_y + batch * n_padded;
            temp_ptr = aligned_temp + batch * n_padded;
        } else {
            // Even n or single batch: use original data directly
            y_ptr = const_cast<double*>(y.data() + batch * n);
            temp_ptr = temp.data() + batch * n;
        }

        // Create dense vector descriptors for this batch
        cusparseDnVecDescr_t vecY, vecTemp;
        cusparseCreateDnVec(&vecY, n, y_ptr, CUDA_R_64F);
        cusparseCreateDnVec(&vecTemp, n, temp_ptr, CUDA_R_64F);

        // Set batch for P matrix
        const_cast<CSR*>(P_csr)->setCusparseBatch(batch);

        // Compute required buffer size (only on first batch)
        if (batch == 0) {
            size_t requiredBufferSize = 0;
            cusparseSpMV_bufferSize(
                cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_spmv,
                P_csr->cusparseDescriptor(),
                vecY,
                &beta_spmv,
                vecTemp,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &requiredBufferSize
            );

            if (requiredBufferSize > buffer_size) {
                if (buffer) cudaFree(buffer);
                cudaMalloc(&buffer, requiredBufferSize);
                buffer_size = requiredBufferSize;
            }
        }

        cusparseSpMV(
            cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_spmv,
            P_csr->cusparseDescriptor(),
            vecY,
            &beta_spmv,
            vecTemp,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            buffer
        );

        cusparseDestroyDnVec(vecY);
        cusparseDestroyDnVec(vecTemp);
    }

    if (needs_alignment) {
        // Bulk copy: Copy ALL batch results back, converting from n_padded-stride to n-stride
        cudaMemcpy2DAsync(
            temp.data(),                    // dst
            n * sizeof(double),             // dst pitch (original stride)
            aligned_temp,                   // src
            n_padded * sizeof(double),      // src pitch (aligned stride)
            n * sizeof(double),             // width (bytes per batch)
            batchSize,                      // height (number of batches)
            cudaMemcpyDeviceToDevice,
            stream
        );
    }

    // Note: aligned buffers are cached and will be reused on subsequent calls
    // Caller is responsible for cleanup (typically in destructor)

    // Step 2: Compute result = x' * temp (dot product for each batch)
    dot_batched(result, x, temp, stream);
}

// ============================================================================
// Fused quadratic form kernel: result = x'Py in one launch
// Each thread computes one row of P*y (SpMV) and accumulates x[row]*(P*y)[row]
// into a shared-memory reduction → single atomicAdd per block
// Eliminates the O(n) temp vector traffic entirely
// ============================================================================
__global__ void fused_quad_form_kernel(
    double* __restrict__ result,
    const int64_t* __restrict__ rowOffsets,
    const int64_t* __restrict__ colIndices,
    const double* __restrict__ values,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int64_t n,
    int64_t nnz,
    int64_t batchSize)
{
    int tid = threadIdx.x;
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (b >= batchSize) return;

    double dot_val = 0.0;
    if (row < n) {
        int64_t row_start = rowOffsets[row];
        int64_t row_end = rowOffsets[row + 1];
        int64_t val_base = b * nnz;
        int64_t vec_base = b * n;

        // Compute (P*y)[row]
        double py_row = 0.0;
        for (int64_t idx = row_start; idx < row_end; ++idx) {
            py_row += values[val_base + idx] * y[vec_base + colIndices[idx]];
        }

        // Accumulate x[row] * (P*y)[row]
        dot_val = x[vec_base + row] * py_row;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_val += __shfl_down_sync(0xffffffff, dot_val, offset);
    }

    __shared__ double warp_results[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        warp_results[warp_id] = dot_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        dot_val = (lane < num_warps) ? warp_results[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_val += __shfl_down_sync(0xffffffff, dot_val, offset);
        }
        if (lane == 0) {
            atomicAdd(&result[b], dot_val);
        }
    }
}

// Quadratic form for symmetric matrices: result = x' * P * y
// P is stored as full symmetric matrix in CSR format
// Uses fused kernel that computes SpMV + dot product in one launch
void quad_form_symmetric_batched(
    BatchedVector& result,
    const void* P_csr_void,
    const BatchedVector& x,
    const BatchedVector& y,
    BatchedVector& temp,
    cudaStream_t stream)
{
    const CSR* P_csr = static_cast<const CSR*>(P_csr_void);

    int64_t n = x.n();
    int64_t batchSize = x.batchSize();

    assert(result.n() == 1 && result.batchSize() == batchSize);
    assert(x.n() == n && x.batchSize() == batchSize);
    assert(y.n() == n && y.batchSize() == batchSize);
    assert(P_csr->rows() == n && P_csr->cols() == n);

    if (n == 0 || P_csr->nnz() == 0 || batchSize == 0) {
        cudaMemsetAsync(result.data(), 0, sizeof(double) * batchSize, stream);
        return;
    }

    // Initialize result to zero (fused kernel uses atomicAdd)
    cudaMemsetAsync(result.data(), 0, sizeof(double) * batchSize, stream);

    dim3 blk(256, 1, 1);
    dim3 grd((n + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_quad_form_kernel, grd, blk, 0, stream,
        result.data(),
        P_csr->rowOffsets(), P_csr->colIndices(), P_csr->values(),
        x.data(), y.data(),
        n, P_csr->nnz(), batchSize);
}

// Element-wise division kernel: z = x / y
__global__ void elementwise_div_kernel(
    double* __restrict__ z,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    z[global_idx] = x[global_idx] / y[global_idx];
}

void elementwise_div(
    BatchedVector& z,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.n() == z.n());
    assert(x.batchSize() == y.batchSize() && x.batchSize() == z.batchSize());

    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (z.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((z.n() + blk.x - 1) / blk.x, z.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(elementwise_div_kernel, grd, blk, 0, stream,
        z.data(), x.data(), y.data(), z.n(), z.batchSize());
}

// Safe element-wise division: z = x / max(|y|, eps) * sign(y)
// Prevents NaN from near-zero denominators (e.g., tau_den in IPM)
__global__ void safe_elementwise_div_kernel(
    double* __restrict__ z,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double eps,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    double denom = y[global_idx];
    double abs_denom = fabs(denom);
    double safe_denom = (abs_denom > eps) ? denom : copysign(eps, denom);
    z[global_idx] = x[global_idx] / safe_denom;
}

void safe_elementwise_div(
    BatchedVector& z,
    const BatchedVector& x,
    const BatchedVector& y,
    double eps,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.n() == z.n());
    assert(x.batchSize() == y.batchSize() && x.batchSize() == z.batchSize());

    if (z.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((z.n() + blk.x - 1) / blk.x, z.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(safe_elementwise_div_kernel, grd, blk, 0, stream,
        z.data(), x.data(), y.data(), eps, z.n(), z.batchSize());
}

// Element-wise multiplication kernel: z = x * y
__global__ void elementwise_mul_kernel(
    double* __restrict__ z,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    z[global_idx] = x[global_idx] * y[global_idx];
}

void elementwise_mul(
    BatchedVector& z,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.n() == z.n());
    assert(x.batchSize() == y.batchSize() && x.batchSize() == z.batchSize());

    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (z.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((z.n() + blk.x - 1) / blk.x, z.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(elementwise_mul_kernel, grd, blk, 0, stream,
        z.data(), x.data(), y.data(), z.n(), z.batchSize());
}

// Multiply vector by per-batch scalar: y = x * scalar[batch_idx]
__global__ void mul_per_batch_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    const double* __restrict__ scalars,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    y[global_idx] = x[global_idx] * scalars[batch_idx];
}

void mul_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream)
{
    assert(y.n() == x.n());
    assert(y.batchSize() == x.batchSize());
    assert(scalars.n() == 1 && scalars.batchSize() == y.batchSize());

    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (y.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(mul_per_batch_kernel, grd, blk, 0, stream,
        y.data(), x.data(), scalars.data(), y.n(), y.batchSize());
}

// Scale vector by per-batch scalar: y = scalar[batch_idx] * x
__global__ void scale_per_batch_kernel(
    double* __restrict__ y,
    const double* __restrict__ scalars,
    const double* __restrict__ x,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    y[global_idx] = scalars[batch_idx] * x[global_idx];
}

void scale_per_batch(
    BatchedVector& y,
    const BatchedVector& scalars,
    const BatchedVector& x,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(scalars.n() == 1 && scalars.batchSize() == x.batchSize());

    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (y.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(scale_per_batch_kernel, grd, blk, 0, stream,
        y.data(), scalars.data(), x.data(), y.n(), y.batchSize());
}

// Divide vector by per-batch scalar: y = x / scalar[batch_idx]
__global__ void div_per_batch_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    const double* __restrict__ scalars,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    y[global_idx] = x[global_idx] / scalars[batch_idx];
}

void div_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(scalars.n() == 1 && scalars.batchSize() == x.batchSize());

    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (y.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(div_per_batch_kernel, grd, blk, 0, stream,
        y.data(), x.data(), scalars.data(), y.n(), y.batchSize());
}

// Add per-batch scalar: y = x + scalar[batch_idx]
__global__ void add_scalar_per_batch_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    const double* __restrict__ scalars,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    y[global_idx] = x[global_idx] + scalars[batch_idx];
}

void add_scalar_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(scalars.n() == 1 && scalars.batchSize() == x.batchSize());
    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (y.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(add_scalar_per_batch_kernel, grd, blk, 0, stream,
        y.data(), x.data(), scalars.data(), y.n(), y.batchSize());
}

// Element-wise minimum with per-batch scalar: y = min(x, scalar[batch_idx])
__global__ void min_scalar_per_batch_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    const double* __restrict__ scalars,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    double x_val = x[global_idx];
    double scalar = scalars[batch_idx];
    y[global_idx] = (x_val < scalar) ? x_val : scalar;
}

void min_scalar_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(scalars.n() == 1 && scalars.batchSize() == x.batchSize());
    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (y.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(min_scalar_per_batch_kernel, grd, blk, 0, stream,
        y.data(), x.data(), scalars.data(), y.n(), y.batchSize());
}

__global__ void axpy_batched_kernel(
    double* __restrict__ y,
    const double* __restrict__ alpha,
    const double* __restrict__ x,
    int64_t n,
    int64_t batchSize
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;

    if (i >= n || batch >= batchSize) return;

    int64_t idx = batch * n + i;
    // y += alpha[batch] * x
    y[idx] += alpha[batch] * x[idx];
}

void axpy_batched(
    BatchedVector& y,
    const BatchedVector& alpha,
    const BatchedVector& x,
    cudaStream_t stream
) {
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(alpha.n() == 1 && alpha.batchSize() == x.batchSize());
    // Early return for zero-size vectors to avoid CUDA error from gridDim.x=0
    if (y.n() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(axpy_batched_kernel, grd, blk, 0, stream,
        y.data(), alpha.data(), x.data(), y.n(), y.batchSize());
}

// Scalar minus vector kernel: y = scalar - x
__global__ void scalar_minus_vector_kernel(
    double* __restrict__ y,
    double scalar,
    const double* __restrict__ x,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    y[global_idx] = scalar - x[global_idx];
}

void scalar_minus_vector(
    BatchedVector& y,
    double scalar,
    const BatchedVector& x,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());

    if (y.n() == 0 || y.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((y.n() + blk.x - 1) / blk.x, y.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(scalar_minus_vector_kernel, grd, blk, 0, stream,
        y.data(), scalar, x.data(), y.n(), y.batchSize());
}

// Compute adaptive centering target: mu_step[b] = min(mu[b] * factor, target)
__global__ void compute_mu_step_kernel(
    double* __restrict__ mu_step,
    const double* __restrict__ mu,
    double factor,
    double target,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    double val = mu[idx] * factor;
    mu_step[idx] = (val < target) ? val : target;
}

void compute_mu_step(
    BatchedVector& mu_step,
    const BatchedVector& mu,
    double factor,
    double target,
    cudaStream_t stream)
{
    assert(mu_step.n() == 1 && mu.n() == 1);
    assert(mu_step.batchSize() == mu.batchSize());

    if (mu.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((mu.batchSize() + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(compute_mu_step_kernel, grd, blk, 0, stream,
        mu_step.data(), mu.data(), factor, target, mu.batchSize());
}

// Sanitize alpha: replace NaN/negative with 0.0
__global__ void sanitize_alpha_kernel(
    double* __restrict__ alpha,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    double val = alpha[idx];
    if (isnan(val) || val < 0.0) {
        alpha[idx] = 0.0;
    }
}

void sanitize_alpha(
    BatchedVector& alpha,
    cudaStream_t stream)
{
    assert(alpha.n() == 1);

    if (alpha.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((alpha.batchSize() + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(sanitize_alpha_kernel, grd, blk, 0, stream,
        alpha.data(), alpha.batchSize());
}

// Zero the step direction for batches that contain NaN.
// This prevents NaN propagation through add_step (since 0*NaN = NaN).
// Each thread handles one batch: scans x, z, s for NaN and zeros all if found.
__global__ void zero_nan_step_kernel(
    double* __restrict__ step_x,
    double* __restrict__ step_z,
    double* __restrict__ step_s,
    int64_t n,
    int64_t m,
    int64_t batchSize)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    bool has_nan = false;
    for (int64_t i = 0; i < n && !has_nan; ++i) {
        if (isnan(step_x[b * n + i])) has_nan = true;
    }
    for (int64_t i = 0; i < m && !has_nan; ++i) {
        if (isnan(step_z[b * m + i])) has_nan = true;
        if (isnan(step_s[b * m + i])) has_nan = true;
    }
    if (has_nan) {
        for (int64_t i = 0; i < n; ++i) step_x[b * n + i] = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            step_z[b * m + i] = 0.0;
            step_s[b * m + i] = 0.0;
        }
    }
}

void zero_nan_step(
    BatchedVector& step_x,
    BatchedVector& step_z,
    BatchedVector& step_s,
    int64_t n,
    int64_t m,
    cudaStream_t stream)
{
    int64_t batchSize = step_x.batchSize();
    if (batchSize == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((batchSize + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(zero_nan_step_kernel, grd, blk, 0, stream,
        step_x.data(), step_z.data(), step_s.data(),
        n, m, batchSize);
}

// ============================================================================
// FUSED KERNELS - reduce kernel launch overhead
// ============================================================================

/**
 * @brief Fused dot product + accumulate kernel
 * Computes dest[batch] += alpha * sum(x[i] * y[i]) for each batch
 * Uses shared memory reduction like dot_batched_kernel
 */
__global__ void dot_add_kernel(
    double* __restrict__ dest,
    double alpha,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int64_t n)
{
    int tid = threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t base = batch_idx * n;
    double val = (i < n) ? (x[base + i] * y[base + i]) : 0.0;

    // Warp-level reduction first
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ double warp_results[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        val = (lane < num_warps) ? warp_results[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            atomicAdd(&dest[batch_idx], alpha * val);
        }
    }
}

void dot_add(
    BatchedVector& dest,
    double alpha,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.batchSize() == y.batchSize());
    assert(dest.n() == 1 && dest.batchSize() == x.batchSize());

    // NOTE: dest is NOT zeroed - caller must initialize if needed
    // This allows accumulation across multiple calls

    if (x.n() == 0 || x.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((x.n() + blk.x - 1) / blk.x, x.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(dot_add_kernel, grd, blk, 0, stream,
        dest.data(), alpha, x.data(), y.data(), x.n());
}

/**
 * @brief Fused z = x1 + tau * x2 kernel (tau is per-batch scalar)
 */
__global__ void axpby_scaled_kernel(
    double* __restrict__ z,
    const double* __restrict__ x1,
    const double* __restrict__ tau,
    const double* __restrict__ x2,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    z[global_idx] = x1[global_idx] + tau[batch_idx] * x2[global_idx];
}

void axpby_scaled(
    BatchedVector& z,
    const BatchedVector& x1,
    const BatchedVector& tau,
    const BatchedVector& x2,
    cudaStream_t stream)
{
    assert(x1.n() == x2.n() && x1.n() == z.n());
    assert(x1.batchSize() == x2.batchSize() && x1.batchSize() == z.batchSize());
    assert(tau.n() == 1 && tau.batchSize() == z.batchSize());

    if (z.n() == 0 || z.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((z.n() + blk.x - 1) / blk.x, z.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(axpby_scaled_kernel, grd, blk, 0, stream,
        z.data(), x1.data(), tau.data(), x2.data(), z.n(), z.batchSize());
}

/**
 * @brief Multi-dot kernel for computing multiple dot products and accumulating
 *
 * Thread block computes partial sums for all dot products simultaneously.
 * Uses register tiling to avoid multiple shared memory reductions.
 */
__global__ void multi_dot_add_kernel_2(
    double* __restrict__ dest,
    double alpha0, const double* __restrict__ x0, const double* __restrict__ y0, int64_t n0,
    double alpha1, const double* __restrict__ x1, const double* __restrict__ y1, int64_t n1,
    int64_t batch_size)
{
    int tid = threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t base0 = batch_idx * n0;
    int64_t base1 = batch_idx * n1;

    double v0 = (i < n0) ? (x0[base0 + i] * y0[base0 + i]) : 0.0;
    double v1 = (i < n1) ? (x1[base1 + i] * y1[base1 + i]) : 0.0;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, offset);
        v1 += __shfl_down_sync(0xffffffff, v1, offset);
    }

    __shared__ double w0[8], w1[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) { w0[warp_id] = v0; w1[warp_id] = v1; }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        v0 = (lane < num_warps) ? w0[lane] : 0.0;
        v1 = (lane < num_warps) ? w1[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, offset);
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
        }
        if (lane == 0) {
            atomicAdd(&dest[batch_idx], alpha0 * v0 + alpha1 * v1);
        }
    }
}

__global__ void multi_dot_add_kernel_4(
    double* __restrict__ dest,
    double alpha0, const double* __restrict__ x0, const double* __restrict__ y0, int64_t n0,
    double alpha1, const double* __restrict__ x1, const double* __restrict__ y1, int64_t n1,
    double alpha2, const double* __restrict__ x2, const double* __restrict__ y2, int64_t n2,
    double alpha3, const double* __restrict__ x3, const double* __restrict__ y3, int64_t n3,
    int64_t batch_size)
{
    int tid = threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t base0 = batch_idx * n0;
    int64_t base1 = batch_idx * n1;
    int64_t base2 = batch_idx * n2;
    int64_t base3 = batch_idx * n3;

    double v0 = (i < n0) ? (x0[base0 + i] * y0[base0 + i]) : 0.0;
    double v1 = (i < n1) ? (x1[base1 + i] * y1[base1 + i]) : 0.0;
    double v2 = (i < n2) ? (x2[base2 + i] * y2[base2 + i]) : 0.0;
    double v3 = (i < n3) ? (x3[base3 + i] * y3[base3 + i]) : 0.0;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, offset);
        v1 += __shfl_down_sync(0xffffffff, v1, offset);
        v2 += __shfl_down_sync(0xffffffff, v2, offset);
        v3 += __shfl_down_sync(0xffffffff, v3, offset);
    }

    __shared__ double w0[8], w1[8], w2[8], w3[8];
    int warp_id = tid / 32;
    int lane = tid % 32;

    if (lane == 0) { w0[warp_id] = v0; w1[warp_id] = v1; w2[warp_id] = v2; w3[warp_id] = v3; }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        v0 = (lane < num_warps) ? w0[lane] : 0.0;
        v1 = (lane < num_warps) ? w1[lane] : 0.0;
        v2 = (lane < num_warps) ? w2[lane] : 0.0;
        v3 = (lane < num_warps) ? w3[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, offset);
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
            v2 += __shfl_down_sync(0xffffffff, v2, offset);
            v3 += __shfl_down_sync(0xffffffff, v3, offset);
        }
        if (lane == 0) {
            atomicAdd(&dest[batch_idx],
                alpha0 * v0 + alpha1 * v1 +
                alpha2 * v2 + alpha3 * v3);
        }
    }
}

void multi_dot_add_2(
    BatchedVector& dest,
    double alpha0, const BatchedVector& x0, const BatchedVector& y0,
    double alpha1, const BatchedVector& x1, const BatchedVector& y1,
    cudaStream_t stream)
{
    assert(dest.n() == 1);
    assert(x0.n() == y0.n() && x1.n() == y1.n());
    assert(x0.batchSize() == dest.batchSize() && x1.batchSize() == dest.batchSize());

    if (dest.batchSize() == 0) return;

    int64_t max_n = std::max(x0.n(), x1.n());
    if (max_n == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((max_n + blk.x - 1) / blk.x, dest.batchSize(), 1);

    MOREAU_KERNEL_LAUNCH(multi_dot_add_kernel_2, grd, blk, 0, stream,
        dest.data(),
        alpha0, x0.data(), y0.data(), x0.n(),
        alpha1, x1.data(), y1.data(), x1.n(),
        dest.batchSize());
}

void multi_dot_add_4(
    BatchedVector& dest,
    double alpha0, const BatchedVector& x0, const BatchedVector& y0,
    double alpha1, const BatchedVector& x1, const BatchedVector& y1,
    double alpha2, const BatchedVector& x2, const BatchedVector& y2,
    double alpha3, const BatchedVector& x3, const BatchedVector& y3,
    cudaStream_t stream)
{
    assert(dest.n() == 1);

    if (dest.batchSize() == 0) return;

    int64_t max_n = std::max({x0.n(), x1.n(), x2.n(), x3.n()});
    if (max_n == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((max_n + blk.x - 1) / blk.x, dest.batchSize(), 1);

    MOREAU_KERNEL_LAUNCH(multi_dot_add_kernel_4, grd, blk, 0, stream,
        dest.data(),
        alpha0, x0.data(), y0.data(), x0.n(),
        alpha1, x1.data(), y1.data(), x1.n(),
        alpha2, x2.data(), y2.data(), x2.n(),
        alpha3, x3.data(), y3.data(), x3.n(),
        dest.batchSize());
}

/**
 * @brief Fused z = a*x + tau*y kernel (tau is per-batch scalar)
 */
__global__ void waxpby_scaled_kernel(
    double* __restrict__ z,
    double a,
    const double* __restrict__ x,
    const double* __restrict__ tau,
    const double* __restrict__ y,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch_idx = blockIdx.y;
    if (idx >= n || batch_idx >= batch_size) return;

    int64_t global_idx = batch_idx * n + idx;
    z[global_idx] = a * x[global_idx] + tau[batch_idx] * y[global_idx];
}

void waxpby_scaled(
    BatchedVector& z,
    double a,
    const BatchedVector& x,
    const BatchedVector& tau,
    const BatchedVector& y,
    cudaStream_t stream)
{
    assert(x.n() == y.n() && x.n() == z.n());
    assert(x.batchSize() == y.batchSize() && x.batchSize() == z.batchSize());
    assert(tau.n() == 1 && tau.batchSize() == z.batchSize());

    if (z.n() == 0 || z.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((z.n() + blk.x - 1) / blk.x, z.batchSize(), 1);
    MOREAU_KERNEL_LAUNCH(waxpby_scaled_kernel, grd, blk, 0, stream,
        z.data(), a, x.data(), tau.data(), y.data(), z.n(), z.batchSize());
}

// ============================================================================
// Triple subtraction kernel: z0=x0-y0, z1=x1-y1, z2=x2-y2
// Fuses 3 waxpby(z, 1.0, x, -1.0, y) into 1 kernel
// ============================================================================
__global__ void triple_sub_kernel(
    double* __restrict__ z0, const double* __restrict__ x0, const double* __restrict__ y0, int64_t n0,
    double* __restrict__ z1, const double* __restrict__ x1, const double* __restrict__ y1, int64_t n1,
    double* __restrict__ z2, const double* __restrict__ x2, const double* __restrict__ y2, int64_t n2,
    int64_t batchSize)
{
    int64_t b = blockIdx.y;
    if (b >= batchSize) return;

    // Each thread handles one element from each of the 3 vectors
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vector 0 (typically size m)
    if (idx < n0) {
        z0[b * n0 + idx] = x0[b * n0 + idx] - y0[b * n0 + idx];
    }

    // Vector 1 (typically size n)
    if (idx < n1) {
        z1[b * n1 + idx] = x1[b * n1 + idx] - y1[b * n1 + idx];
    }

    // Vector 2 (typically size n)
    if (idx < n2) {
        z2[b * n2 + idx] = x2[b * n2 + idx] - y2[b * n2 + idx];
    }
}

void triple_sub(
    BatchedVector& z0, const BatchedVector& x0, const BatchedVector& y0,
    BatchedVector& z1, const BatchedVector& x1, const BatchedVector& y1,
    BatchedVector& z2, const BatchedVector& x2, const BatchedVector& y2,
    cudaStream_t stream)
{
    int64_t batchSize = z0.batchSize();
    assert(z1.batchSize() == batchSize && z2.batchSize() == batchSize);

    // Find max n to cover all vectors
    int64_t max_n = std::max({z0.n(), z1.n(), z2.n()});

    if (max_n == 0 || batchSize == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((max_n + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(triple_sub_kernel, grd, blk, 0, stream,
        z0.data(), x0.data(), y0.data(), z0.n(),
        z1.data(), x1.data(), y1.data(), z1.n(),
        z2.data(), x2.data(), y2.data(), z2.n(),
        batchSize);
}

// ============================================================================
// Fused per-batch scalar kernel: dest += a*x + b*y
// For accumulating two scalar-per-batch values in one launch
// ============================================================================
__global__ void axpby2_scalar_kernel(
    double* __restrict__ dest,
    double a, const double* __restrict__ x,
    double b, const double* __restrict__ y,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    dest[idx] += a * x[idx] + b * y[idx];
}

void axpby2_scalar(
    BatchedVector& dest,
    double a, const BatchedVector& x,
    double b, const BatchedVector& y,
    cudaStream_t stream)
{
    assert(dest.n() == 1 && x.n() == 1 && y.n() == 1);
    assert(dest.batchSize() == x.batchSize() && dest.batchSize() == y.batchSize());

    if (dest.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((dest.batchSize() + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(axpby2_scalar_kernel, grd, blk, 0, stream,
        dest.data(), a, x.data(), b, y.data(), dest.batchSize());
}

// ============================================================================
// Fused tau_num initialization: dest = a - b / c (per-batch scalars)
// Replaces memcpy + elementwise_div + axpby pattern
// ============================================================================
__global__ void sub_div_scalar_kernel(
    double* __restrict__ dest,
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ c,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    dest[idx] = a[idx] - b[idx] / c[idx];
}

void sub_div_scalar(
    BatchedVector& dest,
    const BatchedVector& a,
    const BatchedVector& b,
    const BatchedVector& c,
    cudaStream_t stream)
{
    assert(dest.n() == 1 && a.n() == 1 && b.n() == 1 && c.n() == 1);
    assert(dest.batchSize() == a.batchSize());
    assert(dest.batchSize() == b.batchSize());
    assert(dest.batchSize() == c.batchSize());

    if (dest.batchSize() == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((dest.batchSize() + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(sub_div_scalar_kernel, grd, blk, 0, stream,
        dest.data(), a.data(), b.data(), c.data(), dest.batchSize());
}

// ============================================================================
// Fused step update kernel: x += alpha*dx, s += alpha*ds, z += alpha*dz,
// tau += alpha*dtau, kappa += alpha*dkappa — all in one launch
// Replaces 3x axpy_batched + update_tau_kappa_kernel (4 launches -> 1)
// ============================================================================
__global__ void apply_step_kernel(
    double* __restrict__ x, const double* __restrict__ dx, int64_t n,
    double* __restrict__ s, const double* __restrict__ ds,
    double* __restrict__ z, const double* __restrict__ dz, int64_t m,
    double* __restrict__ tau, const double* __restrict__ dtau,
    double* __restrict__ kappa, const double* __restrict__ dkappa,
    const double* __restrict__ alpha,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (b >= batchSize) return;

    double a = alpha[b];

    // Update x (size n)
    if (idx < n) {
        x[b * n + idx] += a * dx[b * n + idx];
    }

    // Update s and z (size m)
    if (idx < m) {
        s[b * m + idx] += a * ds[b * m + idx];
        z[b * m + idx] += a * dz[b * m + idx];
    }

    // Thread 0 updates tau and kappa scalars
    if (idx == 0) {
        tau[b] += a * dtau[b];
        kappa[b] += a * dkappa[b];
    }
}

void apply_step(
    BatchedVector& x, const BatchedVector& dx,
    BatchedVector& s, const BatchedVector& ds,
    BatchedVector& z, const BatchedVector& dz,
    BatchedVector& tau, const BatchedVector& dtau,
    BatchedVector& kappa, const BatchedVector& dkappa,
    const BatchedVector& alpha,
    cudaStream_t stream)
{
    int64_t n = x.n();
    int64_t m = s.n();
    int64_t batchSize = x.batchSize();

    if (batchSize == 0) return;

    int64_t max_dim = std::max(n, m);
    if (max_dim == 0) max_dim = 1; // at least 1 thread for tau/kappa

    dim3 blk(256, 1, 1);
    dim3 grd((max_dim + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(apply_step_kernel, grd, blk, 0, stream,
        x.data(), dx.data(), n,
        s.data(), ds.data(),
        z.data(), dz.data(), m,
        tau.data(), dtau.data(),
        kappa.data(), dkappa.data(),
        alpha.data(), batchSize);
}

// Direct-x cone dual step update: z_x += α · dz_x (per-batch α).
__global__ void apply_step_z_x_kernel(
    double* __restrict__ z_x, const double* __restrict__ dz_x,
    const double* __restrict__ alpha,
    int64_t totalXConeNumel,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (b >= batchSize) return;
    if (idx >= totalXConeNumel) return;
    double a = alpha[b];
    z_x[b * totalXConeNumel + idx] += a * dz_x[b * totalXConeNumel + idx];
}

void apply_step_z_x(
    BatchedVector& z_x, const BatchedVector& dz_x,
    const BatchedVector& alpha,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    int64_t batchSize = z_x.batchSize();
    if (batchSize == 0 || totalXConeNumel <= 0) return;
    dim3 blk(256, 1, 1);
    dim3 grd((totalXConeNumel + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(apply_step_z_x_kernel, grd, blk, 0, stream,
        z_x.data(), dz_x.data(), alpha.data(),
        totalXConeNumel, batchSize);
}

// ============================================================================
// Fused warm-start scaling: x*=dinv, z*=einv*c, s*=e  (4→1 kernel)
// ============================================================================
__global__ void fused_warm_start_scaling_kernel(
    double* __restrict__ x,
    double* __restrict__ z,
    double* __restrict__ s,
    const double* __restrict__ dinv,
    const double* __restrict__ einv,
    const double* __restrict__ e,
    const double* __restrict__ c,
    int64_t n,
    int64_t m,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    double c_val = c[batch];

    if (idx < n) {
        int64_t gi = batch * n + idx;
        x[gi] *= dinv[gi];
    }

    if (idx < m) {
        int64_t gm = batch * m + idx;
        z[gm] *= einv[gm] * c_val;
        s[gm] *= e[gm];
    }
}

void fused_warm_start_scaling(
    BatchedVector& x,
    BatchedVector& z,
    BatchedVector& s,
    const BatchedVector& dinv,
    const BatchedVector& einv,
    const BatchedVector& e,
    const BatchedVector& c,
    cudaStream_t stream)
{
    int64_t n = x.n();
    int64_t m = z.n();
    int64_t batchSize = x.batchSize();

    int64_t max_dim = (n > m) ? n : m;
    if (max_dim == 0) return;

    dim3 blk(256, 1, 1);
    dim3 grd((max_dim + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_warm_start_scaling_kernel, grd, blk, 0, stream,
        x.data(), z.data(), s.data(),
        dinv.data(), einv.data(), e.data(), c.data(),
        n, m, batchSize);
}

// Build per-batch normalization scale: τ for optimal, κ for infeasible
// SolverStatus enum: PrimalInfeasible=2, DualInfeasible=3,
//                    AlmostPrimalInfeasible=5, AlmostDualInfeasible=6
__global__ void build_normalization_scale_kernel(
    double* __restrict__ scale,
    const double* __restrict__ tau,
    const double* __restrict__ kappa,
    const int32_t* __restrict__ status,
    int64_t batch_size)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int32_t s = status[b];
    // Infeasible statuses: 2 (PrimalInfeasible), 3 (DualInfeasible),
    //                      5 (AlmostPrimalInfeasible), 6 (AlmostDualInfeasible)
    bool is_infeasible = (s == 2 || s == 3 || s == 5 || s == 6);
    scale[b] = is_infeasible ? kappa[b] : tau[b];
}

void build_normalization_scale(
    BatchedVector& scale,
    const BatchedVector& tau,
    const BatchedVector& kappa,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream)
{
    assert(scale.n() == 1 && scale.batchSize() == batchSize);
    assert(tau.n() == 1 && tau.batchSize() == batchSize);
    assert(kappa.n() == 1 && kappa.batchSize() == batchSize);

    dim3 blk(256, 1, 1);
    dim3 grd((batchSize + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(build_normalization_scale_kernel, grd, blk, 0, stream,
        scale.data(), tau.data(), kappa.data(), status, batchSize);
}

// Set objective values to NaN for infeasible batches (matching CPU solver behavior)
__global__ void set_infeasible_obj_nan_kernel(
    double* __restrict__ obj_val,
    double* __restrict__ obj_val_dual,
    const int32_t* __restrict__ status,
    int64_t batch_size)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int32_t s = status[b];
    bool is_infeasible = (s == 2 || s == 3 || s == 5 || s == 6);
    if (is_infeasible) {
        obj_val[b] = __longlong_as_double(0x7FF8000000000000LL); // NaN
        obj_val_dual[b] = __longlong_as_double(0x7FF8000000000000LL);
    }
}

void set_infeasible_obj_nan(
    BatchedVector& obj_val,
    BatchedVector& obj_val_dual,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream)
{
    dim3 blk(256, 1, 1);
    dim3 grd((batchSize + blk.x - 1) / blk.x, 1, 1);
    MOREAU_KERNEL_LAUNCH(set_infeasible_obj_nan_kernel, grd, blk, 0, stream,
        obj_val.data(), obj_val_dual.data(), status, batchSize);
}

} // namespace moreau
