#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>

/**
 * @brief cusolverDnDsyevd with its workspace zeroed first.
 *
 * For an all-zero matrix of order >= 2, cusolverDnDsyevd in cuSOLVER 11.7
 * (CUDA 12.9) returns eigenvalues that depend on what its workspace held before
 * the call: zero when the workspace is finite, NaN with info = 0 when an
 * earlier call on a non-finite matrix left NaN there. Nonzero matrices aren't
 * affected, and neither is cuSOLVER 12.0 (CUDA 13). Every syevd call in Moreau
 * goes through this wrapper, so that its result never depends on the calls
 * before it. The memset runs on the handle's stream and costs a few
 * microseconds.
 *
 * Takes the arguments of cusolverDnDsyevd.
 */
inline cusolverStatus_t dsyevd_zeroed_workspace(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, double* A, int lda, double* W, double* work, int lwork, int* info
) {
    cudaStream_t stream = nullptr;
    cusolverStatus_t status = cusolverDnGetStream(handle, &stream);
    if (status != CUSOLVER_STATUS_SUCCESS) return status;
    if (cudaMemsetAsync(work, 0, sizeof(double) * lwork, stream) != cudaSuccess)
        return CUSOLVER_STATUS_EXECUTION_FAILED;
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}
