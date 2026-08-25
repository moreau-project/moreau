#pragma once

/**
 * @file debug.hpp
 * @brief Centralized debug utilities for moreau-cuda.
 *
 * All debug output is double-gated:
 *   1. Compile-time: compiled out when NDEBUG is defined (release builds).
 *   2. Runtime: requires MOREAU_DEBUG env var to be set (and != "0").
 *
 * In release builds (NDEBUG defined), isDebugEnabled() is constexpr false,
 * so `if (isDebugEnabled()) { ... }` is dead-code-eliminated by the compiler.
 * No #ifndef NDEBUG wrapper is needed at call sites.
 */

#include <cstdio>

#ifdef NDEBUG

inline constexpr bool isDebugEnabled() { return false; }
static_assert(!isDebugEnabled(), "Debug must be disabled in release builds");

#define MOREAU_DEBUG_PRINT(fmt, ...)

#else

#include <cstdlib>
#include <mutex>
#include <string>

inline bool isDebugEnabled() {
    static std::once_flag flag;
    static bool enabled = false;
    std::call_once(flag, []() {
        const char* env = std::getenv("MOREAU_DEBUG");
        enabled = (env && std::string(env) != "0");
    });
    return enabled;
}

#define MOREAU_DEBUG_PRINT(fmt, ...) \
    do { if (isDebugEnabled()) fprintf(stderr, fmt "\n", ##__VA_ARGS__); } while(0)

#endif
