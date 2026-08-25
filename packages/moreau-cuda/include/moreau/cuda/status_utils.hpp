#pragma once

#include <stdexcept>
#include <string>
#include <cublas_v2.h>
#include <cusolverDn.h>

/**
 * Status-check helpers for cuBLAS/cuSOLVER calls, mirroring CUDA_THROW in
 * utils.hpp. API status codes are host-side launch/configuration errors
 * (not numerical convergence info), so failures always throw.
 */

inline void cublas_throw(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS error at ") + file + ":" + std::to_string(line) +
            ": " + cublasGetStatusString(status));
    }
}

inline void cusolver_throw(cusolverStatus_t status, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuSOLVER error at ") + file + ":" + std::to_string(line) +
            ": status " + std::to_string(status));
    }
}

#define CUBLAS_THROW(val) cublas_throw((val), __FILE__, __LINE__)
#define CUSOLVER_THROW(val) cusolver_throw((val), __FILE__, __LINE__)
