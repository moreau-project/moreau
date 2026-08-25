// Microbenchmark: time Cones::update_scaling for SOC- and GenPow-heavy
// workloads.
//
// Measures just the scaling step with cudaEvent timers, averaged over many
// iterations, across a few cone-shape scenarios. The scaling kernel is the
// unit of work parallelized by the block-per-cone path for dim > 32.

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

struct Scenario {
    const char* name;
    std::vector<int64_t> socDims;
    // Gen-power cones: each entry is (dim1, dim2). Alphas are uniform 1/dim1.
    std::vector<std::pair<int64_t, int64_t>> genPowDims;
    int64_t batchSize;
};

void fill_interior(double* out, int64_t dim, std::mt19937_64& rng, double margin) {
    std::normal_distribution<double> normal(0.0, 1.0);
    double norm_sq = 0.0;
    for (int64_t i = 1; i < dim; i++) {
        out[i] = normal(rng);
        norm_sq += out[i] * out[i];
    }
    out[0] = std::sqrt(norm_sq) + margin;
}

// For GenPowerCone interior sampling we simply set all z entries positive;
// the scaling kernel only requires z_i > 0 for i<dim1 and zeta = phi - ||w||²
// > 0 — easy to hit with small w and z≈1.
void fill_genpow_interior(double* out, int64_t dim1, int64_t dim2,
                          std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.5, 1.5);
    std::normal_distribution<double> w(0.0, 0.01);
    for (int64_t i = 0; i < dim1; i++) out[i] = u(rng);
    for (int64_t i = 0; i < dim2; i++) out[dim1 + i] = w(rng);
}

double time_scaling(const Scenario& sc, int iters, int warmup) {
    int64_t totalSocDim = 0;
    for (auto d : sc.socDims) totalSocDim += d;
    int64_t totalGenPowDim = 0;
    for (auto& p : sc.genPowDims) totalGenPowDim += p.first + p.second;
    int64_t m = totalSocDim + totalGenPowDim;

    Cones cones{};
    cones.socConeDims = sc.socDims;
    cones.numSocCones = static_cast<int64_t>(sc.socDims.size());
    if (!sc.genPowDims.empty()) {
        cones.numGenPowerCones = static_cast<int64_t>(sc.genPowDims.size());
        for (auto& p : sc.genPowDims) {
            int64_t dim1 = p.first, dim2 = p.second;
            cones.genPowerDim1s.push_back(dim1);
            cones.genPowerDim2s.push_back(dim2);
            for (int64_t i = 0; i < dim1; i++) {
                cones.genPowerAlphas.push_back(1.0 / static_cast<double>(dim1));
            }
        }
    }
    cones.initialize(sc.batchSize, /*stream=*/0);

    BatchedVector s(m, sc.batchSize);
    BatchedVector z(m, sc.batchSize);
    BatchedVector mu(1, sc.batchSize);

    std::mt19937_64 rng(12345);
    std::vector<double> s_host(m * sc.batchSize);
    std::vector<double> z_host(m * sc.batchSize);
    std::vector<double> mu_host(sc.batchSize, 1.0);

    for (int64_t b = 0; b < sc.batchSize; b++) {
        int64_t offset = 0;
        for (auto d : sc.socDims) {
            fill_interior(&s_host[b * m + offset], d, rng, 0.5);
            fill_interior(&z_host[b * m + offset], d, rng, 0.5);
            offset += d;
        }
        for (auto& p : sc.genPowDims) {
            int64_t dim = p.first + p.second;
            fill_genpow_interior(&s_host[b * m + offset], p.first, p.second, rng);
            fill_genpow_interior(&z_host[b * m + offset], p.first, p.second, rng);
            offset += dim;
        }
    }

    cudaMemcpy(s.data(), s_host.data(), sizeof(double) * m * sc.batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(z.data(), z_host.data(), sizeof(double) * m * sc.batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(mu.data(), mu_host.data(), sizeof(double) * sc.batchSize, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cones.update_scaling(s, z, mu, ScalingStrategy::PrimalDual, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++) {
        cones.update_scaling(s, z, mu, ScalingStrategy::PrimalDual, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;  // avg ms per update_scaling call
}

} // namespace

int main() {
    const int iters = 500;
    const int warmup = 50;

    auto genpow_uniform = [](int64_t count, int64_t dim1, int64_t dim2) {
        std::vector<std::pair<int64_t, int64_t>> v;
        v.reserve(count);
        for (int64_t i = 0; i < count; i++) v.emplace_back(dim1, dim2);
        return v;
    };

    std::vector<Scenario> scenarios = {
        // Multistage-portfolio shape: a handful of high-dim SOCs, small batch.
        {"SOC 1x dim=5000, batch=1",    {5000},                           {}, 1},
        {"SOC 1x dim=5000, batch=8",    {5000},                           {}, 8},
        {"SOC 10x dim=1000, batch=1",   std::vector<int64_t>(10, 1000),   {}, 1},
        {"SOC 10x dim=1000, batch=8",   std::vector<int64_t>(10, 1000),   {}, 8},
        {"SOC 4x dim=500 + 4x dim=3, batch=32",
         {3,3,3,3,500,500,500,500}, {}, 32},
        {"SOC 1000x dim=3, batch=1",    std::vector<int64_t>(1000, 3),    {}, 1},
        {"SOC 1000x dim=3, batch=8",    std::vector<int64_t>(1000, 3),    {}, 8},
        // GenPow scenarios: similar scaling structure, sparse-path > dim=32.
        {"GenPow 1x (dim1=500,dim2=500), batch=1",
         {}, genpow_uniform(1, 500, 500), 1},
        {"GenPow 1x (dim1=500,dim2=500), batch=8",
         {}, genpow_uniform(1, 500, 500), 8},
        {"GenPow 10x (dim1=100,dim2=100), batch=1",
         {}, genpow_uniform(10, 100, 100), 1},
        {"GenPow 10x (dim1=100,dim2=100), batch=8",
         {}, genpow_uniform(10, 100, 100), 8},
        {"GenPow 4x (40,24) + 4x (2,1), batch=32",
         {}, {{40,24},{40,24},{40,24},{40,24},{2,1},{2,1},{2,1},{2,1}}, 32},
        {"GenPow 1000x (2,1), batch=1",
         {}, genpow_uniform(1000, 2, 1), 1},
    };

    std::printf("%-48s %12s\n", "scenario", "us/call");
    std::printf("%-48s %12s\n", "--------", "-------");
    for (const auto& sc : scenarios) {
        double ms = time_scaling(sc, iters, warmup);
        std::printf("%-48s %12.2f\n", sc.name, ms * 1e3);
    }

    // compute_cone_derivative (backward-pass) timings. Reuses SOC scenarios only.
    std::printf("\n--- compute_cone_derivative (SOC-only scenarios) ---\n");
    std::printf("%-48s %12s\n", "scenario", "us/call");
    std::printf("%-48s %12s\n", "--------", "-------");
    for (const auto& sc : scenarios) {
        if (sc.socDims.empty() || !sc.genPowDims.empty()) continue;

        int64_t totalSocDim = 0;
        for (auto d : sc.socDims) totalSocDim += d;
        int64_t m = totalSocDim;

        Cones cones{};
        cones.socConeDims = sc.socDims;
        cones.numSocCones = static_cast<int64_t>(sc.socDims.size());
        cones.initialize(sc.batchSize, 0);

        BatchedVector u(m, sc.batchSize);
        std::mt19937_64 rng(54321);
        std::vector<double> u_host(m * sc.batchSize);
        for (int64_t b = 0; b < sc.batchSize; b++) {
            int64_t offset = 0;
            for (auto d : sc.socDims) {
                fill_interior(&u_host[b * m + offset], d, rng, 0.5);
                offset += d;
            }
        }
        cudaMemcpy(u.data(), u_host.data(),
                   sizeof(double) * m * sc.batchSize, cudaMemcpyHostToDevice);

        ConeDerivatives derivs(cones, sc.batchSize, 0);

        for (int i = 0; i < warmup; i++) {
            compute_cone_derivative(u, derivs, cones, 0);
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; i++) {
            compute_cone_derivative(u, derivs, cones, 0);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::printf("%-48s %12.2f\n", sc.name, (ms / iters) * 1e3);
    }
    return 0;
}
