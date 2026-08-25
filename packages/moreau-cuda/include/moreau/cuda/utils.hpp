#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <memory>

inline void cuda_throw(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(result));
    }
}

inline void cuda_log(cudaError_t result, const char* file, int line) {
    // Suppress "driver shutting down" errors during cleanup - these are expected
    // when Python garbage collector destroys CUDA objects after driver shutdown begins
    if (result != cudaSuccess && result != cudaErrorCudartUnloading) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << ": " << cudaGetErrorString(result) << std::endl;
    }
}

#define CUDA_THROW(val) cuda_throw((val), __FILE__, __LINE__)
#define CUDA_LOG(val)   cuda_log((val), __FILE__, __LINE__)

struct cuda_deleter {
    void operator()(void* p) const noexcept { if (p) CUDA_LOG(cudaFree(p)); }
};
template<class T> using device_unique_ptr = std::unique_ptr<T, cuda_deleter>;

/**
 * @brief Create a device_unique_ptr with cudaMalloc
 *
 * @tparam T Type to allocate (can be pointer type for arrays of pointers)
 * @param count Number of elements
 * @return device_unique_ptr<T> Owning pointer
 */
template<class T>
device_unique_ptr<T> make_device_unique(size_t count = 1) {
    T* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, sizeof(T) * count);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    return device_unique_ptr<T>(ptr);
}