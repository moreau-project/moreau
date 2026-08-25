#pragma once

#include <moreau/cuda/utils.hpp>

// Kernel-launch wrapper. In debug builds, checks cudaGetLastError() after the
// launch and throws via CUDA_THROW so a bad launch reports file:line.
#ifndef NDEBUG
#define MOREAU_KERNEL_LAUNCH(kernel, blocks, threads, shared, stream, ...) \
    do { \
        kernel<<<blocks, threads, shared, stream>>>(__VA_ARGS__); \
        CUDA_THROW(cudaGetLastError()); \
    } while (0)
#else
#define MOREAU_KERNEL_LAUNCH(kernel, blocks, threads, shared, stream, ...) \
    kernel<<<blocks, threads, shared, stream>>>(__VA_ARGS__)
#endif
