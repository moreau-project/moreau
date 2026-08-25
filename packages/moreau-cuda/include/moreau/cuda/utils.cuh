#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Custom atomicAdd for double (for compatibility with older CUDA architectures < 6.0)
// Modern CUDA (6.0+) has native support, but we define this for safety
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Use native atomicAdd for double on compute capability 6.0+
#else
static __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Helper for atomic max on doubles
static __device__ void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double current = __longlong_as_double(assumed);
        if (val <= current) return;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// Helper for atomic min on doubles
static __device__ void atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double current = __longlong_as_double(assumed);
        if (val >= current) return;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

