/**
 * @file cones.hpp
 * @brief CPU-only cone structure for active-set solver
 *
 * Lightweight version of the CUDA Cones struct containing only
 * the fields needed by the CPU solvers.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace moreau {

struct Cones {
    int64_t numZeroCones = 0;
    int64_t numNonnegCones = 0;
    int64_t numExpCones = 0;
    int64_t numSocCones = 0;
    int64_t numPowerCones = 0;
    int64_t numGenPowerCones = 0;

    // SOC cone dimensions (active-set checks these are empty)
    std::vector<int64_t> socConeDims;
    std::vector<int64_t> socConeDimsOriginal;

    // Power cone alphas (active-set checks these are empty)
    std::vector<double> powerAlphas;

    // Generalized power cone alphas (active-set checks these are empty)
    std::vector<double> genPowerAlphas;

    int64_t totalConstraints() const {
        int64_t total = numZeroCones + numNonnegCones + 3 * numExpCones + 3 * numPowerCones;
        for (auto d : socConeDims) total += d;
        return total;
    }
};

} // namespace moreau
