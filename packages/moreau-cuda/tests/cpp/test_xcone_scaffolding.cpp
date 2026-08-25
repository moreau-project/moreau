// test_xcone_scaffolding.cpp
//
// Tests for the direct-x cone metadata scaffolding on CUDA:
// - `Cones.x_cones` stores user-supplied direct-x specs.
// - `SupportedXConeT` constructs cleanly.
// - `Cones::initialize()` populates the direct-x metadata layout.
//
// These tests are the CUDA counterpart of the Rust-side
// `compositexcone::tests` module, adapted to the C++ API shape.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/cones/cones.hpp"
#include "moreau/kkt/kkt.hpp"
#include "moreau/kkt/kkt_kernels.cuh"

#include <vector>

using namespace moreau;

TEST(XConeScaffoldingTest, DefaultConesHaveNoXCones) {
    Cones cones{};
    EXPECT_EQ(cones.numXCones, 0);
    EXPECT_TRUE(cones.x_cones.empty());
    EXPECT_EQ(cones.totalXConeNumel, 0);
    EXPECT_EQ(cones.numSparseXSoc, 0);
}

TEST(XConeScaffoldingTest, SupportedXConeTFieldsRoundTrip) {
    SupportedXConeT spec;
    spec.kind = XConeKind::SOC;
    spec.indices = {3, 4, 5, 6};

    EXPECT_EQ(spec.kind, XConeKind::SOC);
    EXPECT_EQ(spec.numel(), 4);
    EXPECT_EQ(spec.degree(), 1);  // SOC always degree 1

    SupportedXConeT nn;
    nn.kind = XConeKind::Nonneg;
    nn.indices = {0, 2, 7};
    EXPECT_EQ(nn.numel(), 3);
    EXPECT_EQ(nn.degree(), 3);  // Nonneg: degree == numel
}

TEST(XConeScaffoldingTest, ConesStoresUserXCones) {
    Cones cones{};
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 1}});
    cones.x_cones.push_back(SupportedXConeT{XConeKind::SOC, {2, 3, 4}});

    ASSERT_EQ(cones.x_cones.size(), 2u);
    EXPECT_EQ(cones.x_cones[0].kind, XConeKind::Nonneg);
    EXPECT_EQ(cones.x_cones[0].indices.size(), 2u);
    EXPECT_EQ(cones.x_cones[1].kind, XConeKind::SOC);
    EXPECT_EQ(cones.x_cones[1].indices.size(), 3u);
}

TEST(XConeScaffoldingTest, InitializeAcceptsDenseSOCXCones) {
    // Dense SOC (dim ≤ 4) is fully supported on CUDA: scaling +
    // step-math kernels and direct-x KKT assembly all handle it.
    Cones cones{};
    cones.numNonnegCones = 1;
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::SOC, {0, 1, 2}});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(/*batchSize=*/1, stream));

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, InitializeAcceptsSparseSOCXCones) {
    // Sparse SOC (dim > 4) uses the rank-2 u/v expansion; wired
    // end-to-end in scaling, refresh, and step-math.
    Cones cones{};
    cones.numNonnegCones = 1;
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::SOC, {0, 1, 2, 3, 4}});  // dim = 5

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(/*batchSize=*/1, stream));

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, InitializeAcceptsLargeSOCXCones) {
    // SOC kernels are streaming (no stack-array dim cap), so large dims
    // initialize without issue.
    Cones cones{};
    std::vector<int64_t> indices;
    for (int64_t i = 0; i < 64; ++i) indices.push_back(i);  // dim = 64
    cones.x_cones.push_back(SupportedXConeT{XConeKind::SOC, indices});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(/*batchSize=*/1, stream));

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, InitializeAllocatesNonnegXConeArrays) {
    // Nonneg direct-x cones should pass through initialize(), allocating
    // and uploading the d_xcone_* metadata arrays plus the batched
    // working storage. Verifies the arrays round-trip the user x_cones.
    Cones cones{};
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 2}});
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {4}});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(/*batchSize=*/2, stream));
    EXPECT_EQ(cones.numXCones, 2);
    EXPECT_EQ(cones.totalXConeNumel, 3);
    EXPECT_EQ(cones.totalXConeHsEntries, 3);
    EXPECT_EQ(cones.numSparseXSoc, 0);

    auto download = [&](int64_t* dptr, size_t n) {
        std::vector<int64_t> host(n);
        cudaMemcpy(host.data(), dptr, sizeof(int64_t) * n,
                   cudaMemcpyDeviceToHost);
        return host;
    };
    EXPECT_EQ(download(cones.d_xcone_kinds, 2),
              (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(download(cones.d_xcone_dims, 2),
              (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(download(cones.d_xcone_numel_offsets, 3),
              (std::vector<int64_t>{0, 2, 3}));
    EXPECT_EQ(download(cones.d_xcone_hs_offsets, 3),
              (std::vector<int64_t>{0, 2, 3}));
    EXPECT_EQ(download(cones.d_xcone_indices, 3),
              (std::vector<int64_t>{0, 2, 4}));

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, InitializePsdXConeMetadataLayout) {
    // PSD direct-x cones plumb cleanly through Cones::initialize() and the
    // derived metadata arrays (svec dims, matrix dims, Hs entries, device
    // kind/dim arrays) are laid out correctly for the forward kernels to
    // consume.
    Cones cones{};
    cones.numZeroCones = 1;
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::PSD, {0, 1, 2}, /*psd_k=*/2});
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::PSD, {3, 4, 5, 6, 7, 8}, /*psd_k=*/3});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(/*batchSize=*/2, stream));

    EXPECT_EQ(cones.numXCones, 2);
    EXPECT_EQ(cones.numXPsdCones, 2);
    // svec dims: 2*(2+1)/2=3, 3*(3+1)/2=6 → 9 total
    EXPECT_EQ(cones.totalXConeNumel, 9);
    EXPECT_EQ(cones.totalXPsdSvecDim, 9);
    // matrix dims: 2 + 3 = 5
    EXPECT_EQ(cones.totalXPsdMatDim, 5);
    // matsq dims: 4 + 9 = 13
    EXPECT_EQ(cones.totalXPsdMatSqDim, 13);
    // Hs entries (svec×svec upper-tri): 3*4/2=6, 6*7/2=21 → 27 total
    EXPECT_EQ(cones.totalXPsdHsEntries, 27);
    EXPECT_EQ(cones.totalXConeHsEntries, 27);

    auto download = [&](int64_t* dptr, size_t n) {
        std::vector<int64_t> host(n);
        cudaMemcpy(host.data(), dptr, sizeof(int64_t) * n,
                   cudaMemcpyDeviceToHost);
        return host;
    };
    EXPECT_EQ(download(cones.d_xcone_kinds, 2),
              (std::vector<int64_t>{2, 2}));  // both PSD
    EXPECT_EQ(download(cones.d_xcone_dims, 2),
              (std::vector<int64_t>{3, 6}));
    EXPECT_EQ(download(cones.d_xcone_psd_idx, 2),
              (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(download(cones.d_xcone_psd_k, 2),
              (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(download(cones.d_xcone_psd_svec_offsets, 3),
              (std::vector<int64_t>{0, 3, 9}));
    EXPECT_EQ(download(cones.d_xcone_psd_mat_offsets, 3),
              (std::vector<int64_t>{0, 2, 5}));
    EXPECT_EQ(download(cones.d_xcone_psd_matsq_offsets, 3),
              (std::vector<int64_t>{0, 4, 13}));
    EXPECT_EQ(download(cones.d_xcone_psd_hs_offsets, 3),
              (std::vector<int64_t>{0, 6, 27}));

    // Workspace allocations must be sized to the totals.
    EXPECT_EQ(cones.xcone_psd_lambda.n(), cones.totalXPsdMatDim);
    EXPECT_EQ(cones.xcone_psd_R.n(), cones.totalXPsdMatSqDim);
    EXPECT_EQ(cones.xcone_psd_Rinv.n(), cones.totalXPsdMatSqDim);

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, InitializePsdXConeRejectsBadShape) {
    // psd_k must match indices.size() == k(k+1)/2.
    Cones cones{};
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::PSD, {0, 1, 2, 3}, /*psd_k=*/3});  // 4 != 6

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_THROW(cones.initialize(1, stream), std::invalid_argument);

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, MixedPsdAndSocXConeMetadata) {
    // Mixed direct-x stack: SOC and PSD coexist, distinct PSD-only
    // offsets are populated only for the PSD cones.
    Cones cones{};
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::SOC, {0, 1, 2}});                 // dim=3
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::PSD, {3, 4, 5}, /*psd_k=*/2});    // svec=3
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::Nonneg, {6, 7}});                 // dim=2

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(1, stream));
    EXPECT_EQ(cones.numXCones, 3);
    EXPECT_EQ(cones.numXPsdCones, 1);
    EXPECT_EQ(cones.totalXPsdSvecDim, 3);
    EXPECT_EQ(cones.totalXPsdMatDim, 2);
    EXPECT_EQ(cones.totalXPsdMatSqDim, 4);

    auto download = [&](int64_t* dptr, size_t n) {
        std::vector<int64_t> host(n);
        cudaMemcpy(host.data(), dptr, sizeof(int64_t) * n,
                   cudaMemcpyDeviceToHost);
        return host;
    };
    // psd_idx maps cone index → x-PSD-only index (-1 if not PSD).
    EXPECT_EQ(download(cones.d_xcone_psd_idx, 3),
              (std::vector<int64_t>{-1, 0, -1}));
    EXPECT_EQ(download(cones.d_xcone_kinds, 3),
              (std::vector<int64_t>{1, 2, 0}));  // SOC, PSD, Nonneg

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, InitializeIsUnchangedWhenXConesEmpty) {
    // Sanity: the scaffolding must be inert when no direct-x cones are
    // declared. A plain LP on CUDA should still construct + initialize
    // cleanly with the new fields in place.
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;
    ASSERT_TRUE(cones.x_cones.empty());

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_NO_THROW(cones.initialize(/*batchSize=*/1, stream));
    EXPECT_EQ(cones.numXCones, 0);

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, KKTIndexMapForNonnegXCone) {
    // n=3 with a nonneg x-cone on {0, 2}. P has diagonals at positions
    // (0,0) and (2,2); (1,1) gets filled by the εI fallback. We build a
    // KKTData directly (bypassing Cones::initialize() which would throw)
    // and verify that HXConeHsIndex() points at each Pdiag[i].
    //
    // P structure (upper tri CSR): row 0 -> col 0; row 1 -> (none, εI); row 2 -> col 2
    // A structure: 1x3 zero row (just to exercise a realistic m>0)
    std::vector<int64_t> P_ro = {0, 1, 1, 2};
    std::vector<int64_t> P_ci = {0, 2};
    std::vector<int64_t> A_ro = {0, 0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.numZeroCones = 1;
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 2}});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    KKTData kkt(/*n=*/3, /*m=*/1, /*batch=*/1,
                P_ro.data(), P_ci.data(), /*nnzP=*/2,
                A_ro.data(), A_ci.data(), /*nnzA=*/0,
                cones, /*cudss_ir_steps=*/0,
                /*cudss_pivot_enable=*/false, stream);

    // Read back the KKT index map for direct-x cone entries.
    std::vector<int64_t> xcone_diag(2);
    ASSERT_NE(kkt.HXConeHsIndex(), nullptr);
    cudaMemcpy(xcone_diag.data(), kkt.HXConeHsIndex(),
               sizeof(int64_t) * 2, cudaMemcpyDeviceToHost);

    // Read back Pdiag so we can cross-check against the expected slots.
    std::vector<int64_t> Pdiag(3);
    ASSERT_NE(kkt.PDiagIndex(), nullptr);
    cudaMemcpy(Pdiag.data(), kkt.PDiagIndex(),
               sizeof(int64_t) * 3, cudaMemcpyDeviceToHost);

    EXPECT_EQ(xcone_diag[0], Pdiag[0]);
    EXPECT_EQ(xcone_diag[1], Pdiag[2]);

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, KKTSparseSOCAddsExpansionColumns) {
    // Sparse SOC x-cone (dim=5) should add 2 expansion cols/rows (u, v)
    // at positions n+m+p_slack = 5+0+0 = 5 and 6 (p = 2 total). The KKT
    // matrix should have dimension N = n + m + p = 7. Each of the 5
    // (1,1) rows should contain the corresponding x-cone diag + two
    // expansion-col entries (v at col 5, u at col 6).
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.x_cones.push_back(
        SupportedXConeT{XConeKind::SOC, {0, 1, 2, 3, 4}});  // dim=5 -> sparse

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    KKTData kkt(/*n=*/5, /*m=*/0, /*batch=*/1,
                P_ro.data(), P_ci.data(), /*nnzP=*/5,
                A_ro.data(), A_ci.data(), /*nnzA=*/0,
                cones, /*cudss_ir_steps=*/0,
                /*cudss_pivot_enable=*/false, stream);

    EXPECT_EQ(kkt.n, 5);
    EXPECT_EQ(kkt.p, 2);  // 2 expansion rows/cols for 1 sparse x-cone
    const int64_t N = kkt.n + kkt.m + kkt.p;
    EXPECT_EQ(N, 7);

    std::vector<int64_t> h_rowOff((size_t)(N + 1));
    std::vector<int64_t> h_colIdx((size_t)kkt.matrix().nnz());
    kkt.matrix().indicesGpuToCpu(h_rowOff.data(), h_colIdx.data());

    // Each of the 5 (1,1) rows should end with cols 5 and 6 (v, u
    // expansion) — verify by checking the last two colIdx entries of
    // each row are {5, 6} in order.
    for (int64_t i = 0; i < 5; ++i) {
        int64_t end = h_rowOff[i + 1];
        ASSERT_GE(end - h_rowOff[i], 3)
            << "Row " << i << " must have at least diag + 2 exp cols";
        EXPECT_EQ(h_colIdx[end - 2], 5) << "Row " << i << " v col";
        EXPECT_EQ(h_colIdx[end - 1], 6) << "Row " << i << " u col";
    }

    // Expansion diag rows (rows 5 and 6) have one entry each.
    EXPECT_EQ(h_rowOff[6] - h_rowOff[5], 1);
    EXPECT_EQ(h_colIdx[h_rowOff[5]], 5);  // v diag at (5, 5)
    EXPECT_EQ(h_rowOff[7] - h_rowOff[6], 1);
    EXPECT_EQ(h_colIdx[h_rowOff[6]], 6);  // u diag at (6, 6)

    // Verify u/v device arrays are populated (5 entries each, one per row).
    ASSERT_NE(kkt.HXConeVIndex(), nullptr);
    ASSERT_NE(kkt.HXConeUIndex(), nullptr);
    std::vector<int64_t> v_slots(5), u_slots(5);
    cudaMemcpy(v_slots.data(), kkt.HXConeVIndex(),
               sizeof(int64_t) * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_slots.data(), kkt.HXConeUIndex(),
               sizeof(int64_t) * 5, cudaMemcpyDeviceToHost);
    for (int64_t p = 0; p < 5; ++p) {
        // v slot for cone-local position p should map to row p, col 5
        EXPECT_EQ(h_colIdx[v_slots[(size_t)p]], 5);
        EXPECT_EQ(h_colIdx[u_slots[(size_t)p]], 6);
    }

    // exp_diag array: 2 entries per sparse x-cone = [v diag slot, u diag slot]
    ASSERT_NE(kkt.HXConeExpDiagIndex(), nullptr);
    std::vector<int64_t> exp_diag(2);
    cudaMemcpy(exp_diag.data(), kkt.HXConeExpDiagIndex(),
               sizeof(int64_t) * 2, cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_colIdx[exp_diag[0]], 5);
    EXPECT_EQ(h_colIdx[exp_diag[1]], 6);

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, RefreshXConeHsKernelWritesBaselinePlusHs) {
    // Construct a KKT for a n=3 problem with a nonneg x-cone on {0, 2}.
    // Manually set xcone_Hs = [h0, h1] and px_baseline = [b0, b1], run
    // the refresh kernel, and verify KKT.values at the two mapped slots.
    std::vector<int64_t> P_ro = {0, 1, 1, 2};
    std::vector<int64_t> P_ci = {0, 2};
    std::vector<int64_t> A_ro = {0, 0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.numZeroCones = 1;
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 2}});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    const int64_t batchSize = 2;
    KKTData kkt(/*n=*/3, /*m=*/1, /*batch=*/batchSize,
                P_ro.data(), P_ci.data(), /*nnzP=*/2,
                A_ro.data(), A_ci.data(), /*nnzA=*/0,
                cones, /*cudss_ir_steps=*/0,
                /*cudss_pivot_enable=*/false, stream);

    const int64_t nnzKKT = kkt.matrix().nnz();
    const int64_t totalHs = 2;  // two diag entries

    // Read back the x-cone slot map so we know which KKT positions to
    // verify against.
    std::vector<int64_t> slots(totalHs);
    cudaMemcpy(slots.data(), kkt.HXConeHsIndex(),
               sizeof(int64_t) * totalHs, cudaMemcpyDeviceToHost);

    // Zero-initialize KKT.values so unrelated slots stay at 0.
    std::vector<double> zero_vals(batchSize * nnzKKT, 0.0);
    cudaMemcpy(kkt.matrix().values(), zero_vals.data(),
               sizeof(double) * batchSize * nnzKKT, cudaMemcpyHostToDevice);

    // Upload per-batch baselines + Hs. Baseline is batch-major; entry
    // index minor. Batch 0: b = [1.0, 2.0], h = [0.5, 0.25].
    //                 Batch 1: b = [3.0, 4.0], h = [0.1, 0.2].
    std::vector<double> baseline_host = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> hs_host       = {0.5, 0.25, 0.1, 0.2};
    double *d_baseline = nullptr, *d_hs = nullptr;
    cudaMalloc(&d_baseline, sizeof(double) * batchSize * totalHs);
    cudaMalloc(&d_hs, sizeof(double) * batchSize * totalHs);
    cudaMemcpy(d_baseline, baseline_host.data(),
               sizeof(double) * batchSize * totalHs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hs, hs_host.data(),
               sizeof(double) * batchSize * totalHs, cudaMemcpyHostToDevice);

    refresh_xcone_hs(kkt.matrix().values(), d_hs, d_baseline,
                     kkt.HXConeHsIndex(), batchSize, nnzKKT, totalHs, stream);
    cudaStreamSynchronize(stream);

    // Read back KKT.values and verify the two slots per batch equal
    // baseline + hs, leaving other entries untouched (0).
    std::vector<double> out(batchSize * nnzKKT);
    cudaMemcpy(out.data(), kkt.matrix().values(),
               sizeof(double) * batchSize * nnzKKT, cudaMemcpyDeviceToHost);
    for (int64_t b = 0; b < batchSize; ++b) {
        for (int64_t k = 0; k < totalHs; ++k) {
            double expected = baseline_host[(size_t)(b * totalHs + k)] +
                              hs_host[(size_t)(b * totalHs + k)];
            EXPECT_DOUBLE_EQ(out[(size_t)(b * nnzKKT + slots[(size_t)k])],
                             expected)
                << "batch=" << b << " cone_entry=" << k;
        }
    }

    cudaFree(d_baseline);
    cudaFree(d_hs);
    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, UpdateXConesNonnegScalingKernel) {
    // Verifies the nonneg x-cone NT-scaling kernel: given x[J] and z_x
    // for a single cone with 3 indices, the kernel must write
    //   w[p]   = sqrt(x[J[p]] / z[p])
    //   lam[p] = sqrt(x[J[p]] * z[p])
    //   Hs[p]  = x[J[p]] / z[p]
    // We set up the metadata device arrays by hand (Cones::initialize()
    // still throws for non-empty x_cones, so we bypass it).
    const int64_t batchSize = 2;
    const int64_t n = 5;
    const int64_t numXCones = 1;
    const int64_t totalXConeNumel = 3;
    const int64_t totalXConeHsEntries = 3;   // nonneg: k diag entries

    // Cone covers x[1], x[3], x[4] as a direct-x nonneg cone.
    const int64_t kinds_host[] = {0 /* Nonneg */};
    const int64_t dims_host[]  = {3};
    const int64_t numel_offsets_host[] = {0, 3};
    const int64_t hs_offsets_host[]    = {0, 3};
    const int64_t indices_host[]       = {1, 3, 4};

    // x = [10, 4, 99, 9, 16] per batch; z_x = [1, 1, 4] per batch.
    // Direct-x nonneg NT (CPU convention, swapped args vs slack):
    //   w   = sqrt(z/x) = sqrt([1/4, 1/9, 4/16])    = [0.5, 1/3, 0.5]
    //   lam = sqrt(x*z) = sqrt([4*1, 9*1, 16*4])    = [2, 3, 8]
    //   Hs  = z/x       =       [1/4, 1/9, 4/16]    = [0.25, 1/9, 0.25]
    std::vector<double> x_host(batchSize * n);
    for (int64_t b = 0; b < batchSize; ++b) {
        x_host[b * n + 0] = 10.0;
        x_host[b * n + 1] = 4.0;
        x_host[b * n + 2] = 99.0;
        x_host[b * n + 3] = 9.0;
        x_host[b * n + 4] = 16.0;
    }
    std::vector<double> z_host(batchSize * totalXConeNumel);
    for (int64_t b = 0; b < batchSize; ++b) {
        z_host[b * totalXConeNumel + 0] = 1.0;
        z_host[b * totalXConeNumel + 1] = 1.0;
        z_host[b * totalXConeNumel + 2] = 4.0;
    }

    auto upload = [](const void* src, size_t bytes) -> void* {
        void* d = nullptr;
        cudaMalloc(&d, bytes);
        cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice);
        return d;
    };
    int64_t* d_kinds  = static_cast<int64_t*>(upload(kinds_host,  sizeof(kinds_host)));
    int64_t* d_dims   = static_cast<int64_t*>(upload(dims_host,   sizeof(dims_host)));
    int64_t* d_numoff = static_cast<int64_t*>(upload(numel_offsets_host, sizeof(numel_offsets_host)));
    int64_t* d_hsoff  = static_cast<int64_t*>(upload(hs_offsets_host, sizeof(hs_offsets_host)));
    int64_t* d_idx    = static_cast<int64_t*>(upload(indices_host, sizeof(indices_host)));
    double* d_x = static_cast<double*>(upload(x_host.data(), x_host.size() * sizeof(double)));
    double* d_z = static_cast<double*>(upload(z_host.data(), z_host.size() * sizeof(double)));

    double *d_w = nullptr, *d_lam = nullptr, *d_hs = nullptr;
    cudaMalloc(&d_w,   sizeof(double) * batchSize * totalXConeNumel);
    cudaMalloc(&d_lam, sizeof(double) * batchSize * totalXConeNumel);
    cudaMalloc(&d_hs,  sizeof(double) * batchSize * totalXConeHsEntries);

    update_xcones_nonneg_scaling(
        d_x, d_z, d_kinds, d_dims, d_numoff, d_hsoff, d_idx,
        d_w, d_lam, d_hs,
        /*kkt_values=*/nullptr, /*xcone_px_baseline=*/nullptr,
        /*H_xcone_hs_idx=*/nullptr, /*nnzKKT=*/0,
        batchSize, n, numXCones, totalXConeNumel, totalXConeHsEntries,
        /*stream=*/0);
    cudaDeviceSynchronize();

    std::vector<double> w_out(batchSize * totalXConeNumel);
    std::vector<double> lam_out(batchSize * totalXConeNumel);
    std::vector<double> hs_out(batchSize * totalXConeHsEntries);
    cudaMemcpy(w_out.data(),   d_w,   w_out.size()   * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(lam_out.data(), d_lam, lam_out.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs_out.data(),  d_hs,  hs_out.size()  * sizeof(double), cudaMemcpyDeviceToHost);

    const double w_exp[]   = {0.5, 1.0 / 3.0, 0.5};
    const double lam_exp[] = {2.0, 3.0, 8.0};
    const double hs_exp[]  = {0.25, 1.0 / 9.0, 0.25};
    for (int64_t b = 0; b < batchSize; ++b) {
        for (int64_t p = 0; p < 3; ++p) {
            EXPECT_DOUBLE_EQ(w_out[b * 3 + p], w_exp[p]) << "w b=" << b << " p=" << p;
            EXPECT_DOUBLE_EQ(lam_out[b * 3 + p], lam_exp[p]) << "lam b=" << b << " p=" << p;
            EXPECT_DOUBLE_EQ(hs_out[b * 3 + p], hs_exp[p]) << "hs b=" << b << " p=" << p;
        }
    }

    cudaFree(d_kinds); cudaFree(d_dims); cudaFree(d_numoff);
    cudaFree(d_hsoff); cudaFree(d_idx);
    cudaFree(d_x); cudaFree(d_z);
    cudaFree(d_w); cudaFree(d_lam); cudaFree(d_hs);
}

TEST(XConeScaffoldingTest, KKTInitAndRefreshXConePxBaseline) {
    // End-to-end for the init + refresh pair:
    //   1. Populate KKT.values at the two x-cone nonneg slots with
    //      hand-picked P values.
    //   2. Snapshot: kkt.init_xcone_px_baseline(cones) -> xcone_px_baseline
    //      should equal those values per batch.
    //   3. Refresh: with cones.xcone_Hs set to known per-batch values,
    //      KKT.values[slot] should become baseline + Hs.
    std::vector<int64_t> P_ro = {0, 1, 1, 2};
    std::vector<int64_t> P_ci = {0, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 2}});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    const int64_t batchSize = 2;
    KKTData kkt(/*n=*/3, /*m=*/0, /*batch=*/batchSize,
                P_ro.data(), P_ci.data(), /*nnzP=*/2,
                A_ro.data(), A_ci.data(), /*nnzA=*/0,
                cones, /*cudss_ir_steps=*/0,
                /*cudss_pivot_enable=*/false, stream);
    cones.initialize(batchSize, stream);

    // Read back slot map + populate the two KKT slots per batch with
    // known P-contribution values.
    std::vector<int64_t> slots(2);
    cudaMemcpy(slots.data(), kkt.HXConeHsIndex(),
               sizeof(int64_t) * 2, cudaMemcpyDeviceToHost);

    const int64_t nnzKKT = kkt.matrix().nnz();
    std::vector<double> vals(batchSize * nnzKKT, 0.0);
    // Batch 0: slot[0] = 7.0, slot[1] = 11.0
    // Batch 1: slot[0] = 2.0, slot[1] = 3.0
    vals[0 * nnzKKT + slots[0]] = 7.0;
    vals[0 * nnzKKT + slots[1]] = 11.0;
    vals[1 * nnzKKT + slots[0]] = 2.0;
    vals[1 * nnzKKT + slots[1]] = 3.0;
    cudaMemcpy(kkt.matrix().values(), vals.data(),
               sizeof(double) * batchSize * nnzKKT,
               cudaMemcpyHostToDevice);

    // Step 2: init baseline.
    kkt.init_xcone_px_baseline(cones, stream);
    cudaStreamSynchronize(stream);

    // Set xcone_Hs to known deltas per batch.
    const int64_t totalHs = 2;
    std::vector<double> hs_host = {0.5, 0.25, 0.1, 0.2};
    cudaMemcpy(cones.xcone_Hs.data(), hs_host.data(),
               sizeof(double) * batchSize * totalHs,
               cudaMemcpyHostToDevice);

    // Step 3: refresh.
    kkt.refresh_xcone_hs(cones, stream);
    cudaStreamSynchronize(stream);

    std::vector<double> out(batchSize * nnzKKT);
    cudaMemcpy(out.data(), kkt.matrix().values(),
               sizeof(double) * batchSize * nnzKKT,
               cudaMemcpyDeviceToHost);

    // Expected: KKT.values[slot] = baseline + hs per (batch, slot).
    EXPECT_DOUBLE_EQ(out[0 * nnzKKT + slots[0]], 7.0 + 0.5);
    EXPECT_DOUBLE_EQ(out[0 * nnzKKT + slots[1]], 11.0 + 0.25);
    EXPECT_DOUBLE_EQ(out[1 * nnzKKT + slots[0]], 2.0 + 0.1);
    EXPECT_DOUBLE_EQ(out[1 * nnzKKT + slots[1]], 3.0 + 0.2);

    cudaStreamDestroy(stream);
}

TEST(XConeScaffoldingTest, KKTIndexMapForDenseSOCXCone) {
    // Dense SOC x-cone (dim=3) on interleaved indices {1, 0, 2}.
    //
    // The SOC scaling kernel gathers x via `d_xcone_indices` in
    // cone-internal order, so Hs[pos] is a cone-internal (rr, cc)
    // upper-tri entry (column-major):
    //   pos 0: (rr=0,cc=0)
    //   pos 1: (rr=0,cc=1)   pos 2: (rr=1,cc=1)
    //   pos 3: (rr=0,cc=2)   pos 4: (rr=1,cc=2)   pos 5: (rr=2,cc=2)
    // It must scatter into the KKT (1,1) block at the *primal* indices
    // (indices[rr], indices[cc]) — NOT at sorted position `pos`. For
    // indices = {1, 0, 2}: cone-pos 0→x1, 1→x0, 2→x2, so
    //   Hs0 (0,0)→x(1,1)→KKT(1,1)   Hs1 (0,1)→x(1,0)→KKT(0,1)
    //   Hs2 (1,1)→x(0,0)→KKT(0,0)   Hs3 (0,2)→x(1,2)→KKT(1,2)
    //   Hs4 (1,2)→x(0,2)→KKT(0,2)   Hs5 (2,2)→x(2,2)→KKT(2,2)
    // (The earlier sorted-position expectation predated the interleaved-
    // index KKT-layout fix and silently mis-scattered Hs for unsorted
    // cones; `KKTIndexMapInterleavedSOCXCone` end-to-end in
    // test_xcone_endtoend pins the numerical consequence.)
    //
    // P has just the diagonal so the x-cone footprint contributes all
    // off-diagonal upper-tri positions — a good structural stress test.
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.x_cones.push_back(SupportedXConeT{XConeKind::SOC, {1, 0, 2}});

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    KKTData kkt(/*n=*/3, /*m=*/0, /*batch=*/1,
                P_ro.data(), P_ci.data(), /*nnzP=*/3,
                A_ro.data(), A_ci.data(), /*nnzA=*/0,
                cones, /*cudss_ir_steps=*/0,
                /*cudss_pivot_enable=*/false, stream);

    // Read back the full KKT CSR structure and the x-cone Hs slot map.
    const int64_t N = 3;  // n + m + p = 3 + 0 + 0
    std::vector<int64_t> h_rowOff(N + 1);
    const int64_t nnzKKT = kkt.matrix().nnz();
    std::vector<int64_t> h_colIdx(nnzKKT);
    kkt.matrix().indicesGpuToCpu(h_rowOff.data(), h_colIdx.data());

    std::vector<int64_t> hs_slots(6);
    ASSERT_NE(kkt.HXConeHsIndex(), nullptr);
    cudaMemcpy(hs_slots.data(), kkt.HXConeHsIndex(),
               sizeof(int64_t) * 6, cudaMemcpyDeviceToHost);

    // Expected (row, col) pairs in Hs order — cone-internal (rr,cc)
    // mapped through indices = {1, 0, 2} to KKT (min,max) positions.
    struct RC { int64_t row, col; };
    std::vector<RC> expected = {
        {1, 1}, {0, 1}, {0, 0},
        {1, 2}, {0, 2}, {2, 2},
    };
    for (size_t e = 0; e < expected.size(); ++e) {
        int64_t slot = hs_slots[e];
        ASSERT_GE(slot, 0);
        ASSERT_LT(slot, nnzKKT);
        // Find which row the slot belongs to.
        int64_t row = -1;
        for (int64_t r = 0; r < N; ++r) {
            if (slot >= h_rowOff[r] && slot < h_rowOff[r + 1]) {
                row = r;
                break;
            }
        }
        EXPECT_EQ(row, expected[e].row) << "Hs entry " << e;
        EXPECT_EQ(h_colIdx[slot], expected[e].col) << "Hs entry " << e;
    }

    cudaStreamDestroy(stream);
}
