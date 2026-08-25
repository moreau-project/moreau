// test_equilibration.cpp
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include "moreau/equilibration/equilibration.hpp"
#include "moreau/equilibration/equilibration_kernels.cuh"
#include "moreau/cones/cones.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class EquilibrationDataTest : public ::testing::Test {
protected:
    void SetUp() override {
        n = 10;
        m = 5;
        batchSize = 4;
        cudaStreamCreate(&stream);
    }
    
    void TearDown() override {
        cudaStreamDestroy(stream);
    }

    int n, m, batchSize;
    cudaStream_t stream;
};

TEST_F(EquilibrationDataTest, ConstructorAllocates) {
    EquilibrationData equil(n, m, batchSize);
    
    EXPECT_NE(equil.d.data(), nullptr);
    EXPECT_NE(equil.dinv.data(), nullptr);
    EXPECT_NE(equil.e.data(), nullptr);
    EXPECT_NE(equil.einv.data(), nullptr);
    EXPECT_NE(equil.c.data(), nullptr);
    EXPECT_NE(equil.dwork.data(), nullptr);
    EXPECT_NE(equil.ework.data(), nullptr);
    
    EXPECT_EQ(equil.d.n(), n);
    EXPECT_EQ(equil.d.batchSize(), batchSize);
    EXPECT_EQ(equil.dinv.n(), n);
    EXPECT_EQ(equil.dinv.batchSize(), batchSize);
    EXPECT_EQ(equil.e.n(), m);
    EXPECT_EQ(equil.e.batchSize(), batchSize);
    EXPECT_EQ(equil.einv.n(), m);
    EXPECT_EQ(equil.einv.batchSize(), batchSize);
    EXPECT_EQ(equil.c.n(), 1);
    EXPECT_EQ(equil.c.batchSize(), batchSize);
    EXPECT_EQ(equil.dwork.n(), n);
    EXPECT_EQ(equil.dwork.batchSize(), batchSize);
    EXPECT_EQ(equil.ework.n(), m);
    EXPECT_EQ(equil.ework.batchSize(), batchSize);
}

TEST_F(EquilibrationDataTest, ResetToIdentity) {
    EquilibrationData equil(n, m, batchSize);
    
    equil.reset(stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_d(batchSize * n);
    std::vector<double> h_dinv(batchSize * n);
    std::vector<double> h_e(batchSize * m);
    std::vector<double> h_einv(batchSize * m);
    std::vector<double> h_c(batchSize);
    std::vector<double> h_dwork(batchSize * n);
    std::vector<double> h_ework(batchSize * m);
    
    equil.gpuToCpu(h_d.data(), h_dinv.data(), h_e.data(), h_einv.data(), h_c.data(), h_dwork.data(), h_ework.data(), stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_d[i], 1.0) << "d[" << i << "] != 1.0";
        EXPECT_DOUBLE_EQ(h_dinv[i], 1.0) << "dinv[" << i << "] != 1.0";
    }
    
    for (int i = 0; i < batchSize * m; i++) {
        EXPECT_DOUBLE_EQ(h_e[i], 1.0) << "e[" << i << "] != 1.0";
        EXPECT_DOUBLE_EQ(h_einv[i], 1.0) << "einv[" << i << "] != 1.0";
    }
    
    for (int i = 0; i < batchSize; i++) {
        EXPECT_DOUBLE_EQ(h_c[i], 1.0) << "c[" << i << "] != 1.0";
    }

    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_dwork[i], 1.0) << "dwork[" << i << "] != 1.0";
    }
    for (int i = 0; i < batchSize * m; i++) {
        EXPECT_DOUBLE_EQ(h_ework[i], 1.0) << "ework[" << i << "] != 1.0";
    }
}

TEST_F(EquilibrationDataTest, FullRoundtrip) {
    EquilibrationData equil(n, m, batchSize);
    
    std::vector<double> h_d(batchSize * n);
    std::vector<double> h_dinv(batchSize * n);
    std::vector<double> h_e(batchSize * m);
    std::vector<double> h_einv(batchSize * m);
    std::vector<double> h_c(batchSize);
    std::vector<double> h_dwork(batchSize * n);
    std::vector<double> h_ework(batchSize * m);
    
    for (int i = 0; i < batchSize * n; i++) {
        h_d[i] = 2.0 + i * 0.1;
        h_dinv[i] = 1.0 / h_d[i];
    }
    
    for (int i = 0; i < batchSize * m; i++) {
        h_e[i] = 3.0 + i * 0.2;
        h_einv[i] = 1.0 / h_e[i];
    }
    
    for (int i = 0; i < batchSize; i++) {
        h_c[i] = 0.5 + i * 0.05;
    }

    for (int i = 0; i < batchSize * n; i++) {
        h_dwork[i] = 1.0 + i * 0.1;
    }
    for (int i = 0; i < batchSize * m; i++) {
        h_ework[i] = 1.0 + i * 0.2;
    }
    
    equil.cpuToGpu(h_d.data(), h_dinv.data(), h_e.data(), h_einv.data(), h_c.data(), h_dwork.data(), h_ework.data(), stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_d_out(batchSize * n);
    std::vector<double> h_dinv_out(batchSize * n);
    std::vector<double> h_e_out(batchSize * m);
    std::vector<double> h_einv_out(batchSize * m);
    std::vector<double> h_c_out(batchSize);
    std::vector<double> h_dwork_out(batchSize * n);
    std::vector<double> h_ework_out(batchSize * m);
    
    equil.gpuToCpu(h_d_out.data(), h_dinv_out.data(), 
                   h_e_out.data(), h_einv_out.data(), h_c_out.data(), h_dwork_out.data(), h_ework_out.data(), stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_d[i], h_d_out[i]);
        EXPECT_DOUBLE_EQ(h_dinv[i], h_dinv_out[i]);
    }
    
    for (int i = 0; i < batchSize * m; i++) {
        EXPECT_DOUBLE_EQ(h_e[i], h_e_out[i]);
        EXPECT_DOUBLE_EQ(h_einv[i], h_einv_out[i]);
    }
    
    for (int i = 0; i < batchSize; i++) {
        EXPECT_DOUBLE_EQ(h_c[i], h_c_out[i]);
    }

    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_dwork[i], h_dwork_out[i]);
    }
    for (int i = 0; i < batchSize * m; i++) {
        EXPECT_DOUBLE_EQ(h_ework[i], h_ework_out[i]);
    }
}

TEST_F(EquilibrationDataTest, PartialTransferD) {
    EquilibrationData equil(n, m, batchSize);
    
    std::vector<double> h_d(batchSize * n, 2.5);
    equil.cpuToGpu(h_d.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_d_out(batchSize * n);
    equil.gpuToCpu(h_d_out.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_d_out[i], 2.5);
    }
}

TEST_F(EquilibrationDataTest, PartialTransferE) {
    EquilibrationData equil(n, m, batchSize);
    
    std::vector<double> h_e(batchSize * m, 4.5);
    equil.cpuToGpu(nullptr, nullptr, h_e.data(), nullptr, nullptr, nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_e_out(batchSize * m);
    equil.gpuToCpu(nullptr, nullptr, h_e_out.data(), nullptr, nullptr, nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize * m; i++) {
        EXPECT_DOUBLE_EQ(h_e_out[i], 4.5);
    }
}

TEST_F(EquilibrationDataTest, PartialTransferC) {
    EquilibrationData equil(n, m, batchSize);
    
    std::vector<double> h_c(batchSize, 0.75);
    equil.cpuToGpu(nullptr, nullptr, nullptr, nullptr, h_c.data(), nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_c_out(batchSize);
    equil.gpuToCpu(nullptr, nullptr, nullptr, nullptr, h_c_out.data(), nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize; i++) {
        EXPECT_DOUBLE_EQ(h_c_out[i], 0.75);
    }
}

TEST_F(EquilibrationDataTest, PartialTransferMultiple) {
    EquilibrationData equil(n, m, batchSize);
    
    std::vector<double> h_d(batchSize * n, 1.5);
    std::vector<double> h_c(batchSize, 0.8);
    
    equil.cpuToGpu(h_d.data(), nullptr, nullptr, nullptr, h_c.data(), nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_d_out(batchSize * n);
    std::vector<double> h_c_out(batchSize);
    
    equil.gpuToCpu(h_d_out.data(), nullptr, nullptr, nullptr, h_c_out.data(), nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_d_out[i], 1.5);
    }
    
    for (int i = 0; i < batchSize; i++) {
        EXPECT_DOUBLE_EQ(h_c_out[i], 0.8);
    }
}

TEST_F(EquilibrationDataTest, MemoryUsage) {
    EquilibrationData equil(n, m, batchSize);

    // Memory layout: d(n) + dinv(n) + e(m) + einv(m) + c(1) + dwork(n) + ework(m) + mean_P_row_norm(1)
    size_t expected = sizeof(double) * batchSize * (
        2 * n + 2 * m + 1 + n + m + 1  // d, dinv, e, einv, c, dwork, ework, mean_P_row_norm
    );
    EXPECT_EQ(equil.memoryUsage(), expected);
}

TEST_F(EquilibrationDataTest, StreamSupport) {
    EquilibrationData equil(n, m, batchSize);
    cudaStream_t test_stream;
    cudaStreamCreate(&test_stream);
    
    std::vector<double> h_d(batchSize * n, 1.5);
    std::vector<double> h_c(batchSize, 0.8);
    
    equil.cpuToGpu(h_d.data(), nullptr, nullptr, nullptr, h_c.data(), nullptr, nullptr, test_stream);
    cudaStreamSynchronize(test_stream);
    
    std::vector<double> h_d_out(batchSize * n);
    std::vector<double> h_c_out(batchSize);
    
    equil.gpuToCpu(h_d_out.data(), nullptr, nullptr, nullptr, h_c_out.data(), nullptr, nullptr, test_stream);
    cudaStreamSynchronize(test_stream);
    
    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_d_out[i], 1.5);
    }
    
    for (int i = 0; i < batchSize; i++) {
        EXPECT_DOUBLE_EQ(h_c_out[i], 0.8);
    }
    
    cudaStreamDestroy(test_stream);
}

TEST_F(EquilibrationDataTest, VarLenSocRectificationUsesConeDimensions) {
    const int64_t n = 2;
    const int64_t m = 7;
    const int64_t batchSize = 1;
    const int64_t nnzP = 4;
    const int64_t nnzA = 7;

    Cones cones;
    cones.numSocCones = 2;
    cones.socConeDims = {2, 5};
    cones.initialize(batchSize);

    CSR P(n, n, nnzP, batchSize);
    CSR A(m, n, nnzA, batchSize);

    std::vector<int64_t> P_row_offsets = {0, 1, 2, 3, 4};
    std::vector<int64_t> P_col_indices = {0, 1, 0, 1};
    std::vector<double> P_values = {1.0, 1.0, 1.0, 1.0};

    std::vector<int64_t> A_row_offsets = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> A_col_indices = {0, 0, 0, 0, 0, 0, 0};
    std::vector<double> A_values = {10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    P.indicesCpuToGpu(P_row_offsets.data(), P_col_indices.data(), stream);
    P.valuesCpuToGpu(P_values.data(), stream);
    A.indicesCpuToGpu(A_row_offsets.data(), A_col_indices.data(), stream);
    A.valuesCpuToGpu(A_values.data(), stream);

    BatchedVector q(n, batchSize);
    BatchedVector b(m, batchSize);

    std::vector<double> q_values = {1.0, 1.0};
    std::vector<double> b_values(m, 2.0);
    q.cpuToGpu(q_values.data(), stream);
    b.cpuToGpu(b_values.data(), stream);

    std::vector<int64_t> h_matrixPRowOf = moreau::computeRowOfFromCSR(P_row_offsets.data(), n, nnzP);
    std::vector<int64_t> h_matrixARowOf = moreau::computeRowOfFromCSR(A_row_offsets.data(), m, nnzA);
    int64_t* matrixPRowOf = moreau::allocateAndCopyRowOfToGPU(h_matrixPRowOf, stream);
    int64_t* matrixARowOf = moreau::allocateAndCopyRowOfToGPU(h_matrixARowOf, stream);

    EquilibrationData equil(n, m, batchSize);
    EquilibrationSettings settings;
    settings.max_iter = 1;

    equilibration(equil, P, matrixPRowOf, A, matrixARowOf, q, b, settings, cones, stream);
    cudaStreamSynchronize(stream);

    std::vector<double> computed_e(m);
    equil.gpuToCpu(nullptr, nullptr, computed_e.data(), nullptr, nullptr, nullptr, nullptr, stream);
    cudaStreamSynchronize(stream);

    double expected_soc0_mean = (1.0 / std::sqrt(10.0) + 1.0) / 2.0;  // first SOC dim=2
    EXPECT_NEAR(computed_e[0], expected_soc0_mean, 1e-12);
    EXPECT_NEAR(computed_e[1], expected_soc0_mean, 1e-12);

    // With correct variable-dimension rectification, SOC dim=5 should remain at 1.0.
    for (int64_t i = 2; i < m; i++) {
        EXPECT_NEAR(computed_e[i], 1.0, 1e-12);
    }

    cudaFree(matrixPRowOf);
    cudaFree(matrixARowOf);
}

TEST_F(EquilibrationDataTest, SmallProblem) {
    EquilibrationData equil(1, 1, 1);
    equil.reset(stream);
    cudaStreamSynchronize(stream);
    
    double h_d, h_dinv, h_e, h_einv, h_c, h_dwork, h_ework;
    equil.gpuToCpu(&h_d, &h_dinv, &h_e, &h_einv, &h_c, &h_dwork, &h_ework, stream);
    cudaStreamSynchronize(stream);
    
    EXPECT_DOUBLE_EQ(h_d, 1.0);
    EXPECT_DOUBLE_EQ(h_dinv, 1.0);
    EXPECT_DOUBLE_EQ(h_e, 1.0);
    EXPECT_DOUBLE_EQ(h_einv, 1.0);
    EXPECT_DOUBLE_EQ(h_c, 1.0);
    EXPECT_DOUBLE_EQ(h_dwork, 1.0);
    EXPECT_DOUBLE_EQ(h_ework, 1.0);
}

TEST_F(EquilibrationDataTest, AccessUnderlyingVectors) {
    EquilibrationData equil(n, m, batchSize);
    
    EXPECT_EQ(equil.d.n(), n);
    EXPECT_EQ(equil.d.batchSize(), batchSize);
    EXPECT_EQ(equil.dinv.n(), n);
    EXPECT_EQ(equil.dinv.batchSize(), batchSize);
    EXPECT_EQ(equil.e.n(), m);
    EXPECT_EQ(equil.e.batchSize(), batchSize);
    EXPECT_EQ(equil.einv.n(), m);
    EXPECT_EQ(equil.einv.batchSize(), batchSize);
    EXPECT_EQ(equil.c.n(), 1);
    EXPECT_EQ(equil.c.batchSize(), batchSize);
}

TEST_F(EquilibrationDataTest, DirectVectorAccess) {
    EquilibrationData equil(n, m, batchSize);
    
    std::vector<double> h_d(batchSize * n, 3.5);
    equil.d.cpuToGpu(h_d.data(), stream);
    cudaStreamSynchronize(stream);
    
    std::vector<double> h_d_out(batchSize * n);
    equil.d.gpuToCpu(h_d_out.data(), stream);
    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < batchSize * n; i++) {
        EXPECT_DOUBLE_EQ(h_d_out[i], 3.5);
    }
}

TEST_F(EquilibrationDataTest, ClarabelBatchedExample) {
    int n = 5;
    int m = 13;
    int batchSize = 2;
    int nnzP = 5;
    int nnzA = 14;
    
    CSR P(n, n, nnzP, batchSize);
    CSR A(m, n, nnzA, batchSize);
    BatchedVector q(n, batchSize);
    BatchedVector b(m, batchSize);
    
    std::vector<int64_t> P_rowOffsets = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_colIndices = {0, 1, 2, 3, 4};
    std::vector<double> P_values_batch0 = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    std::vector<int64_t> A_rowOffsets = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14
    };
    std::vector<int64_t> A_colIndices = {
        0, 1, 2, 3, 0, 1, 2, 2, 0, 1, 3, 4, 0, 1
    };
    std::vector<double> A_values_batch0 = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0
    };
    
    std::vector<double> q_values_batch0 = {1.0, -1.0, 0.5, -0.5, 0.0};
    std::vector<double> b_values_batch0 = {
        1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0
    };

    std::vector<double> P_values_batch1 = {5.0, 4.0, 3.0, 2.0, 1.0};
    std::vector<double> A_values_batch1 = {
        5.0, 2.5, 1.0, 2.0, 1.0, 1.5, 1.0, 2.0, 9.0, 10.0, 9.0, 1.0, 6.0, 7.0
    };
    std::vector<double> q_values_batch1 = {2.0, -2.0, 1.5, -1.5, -1.0};
    std::vector<double> b_values_batch1 = {
        9.0, 1.0, 7.0, 2.0, 0.0, -4.0, -3.0, 1.0, 0.0, 0.0, 15.0, -2.0, 0.0
    };
    
    // Transfer indices first
    P.indicesCpuToGpu(P_rowOffsets.data(), P_colIndices.data(), stream);
    A.indicesCpuToGpu(A_rowOffsets.data(), A_colIndices.data(), stream);
    
    // Prepare sequential values
    std::vector<double> P_values_sequential(nnzP * batchSize);
    for (int i = 0; i < nnzP; i++) {
        P_values_sequential[i] = P_values_batch0[i];
        P_values_sequential[nnzP + i] = P_values_batch1[i];
    }
    P.valuesCpuToGpu(P_values_sequential.data(), stream);
    
    std::vector<double> A_values_sequential(nnzA * batchSize);
    for (int i = 0; i < nnzA; i++) {
        A_values_sequential[i] = A_values_batch0[i];
        A_values_sequential[nnzA + i] = A_values_batch1[i];
    }
    A.valuesCpuToGpu(A_values_sequential.data(), stream);
    
    std::vector<double> q_values_sequential(n * batchSize);
    for (int i = 0; i < n; i++) {
        q_values_sequential[i] = q_values_batch0[i];
        q_values_sequential[n + i] = q_values_batch1[i];
    }
    q.cpuToGpu(q_values_sequential.data(), stream);
    
    std::vector<double> b_values_sequential(m * batchSize);
    for (int i = 0; i < m; i++) {
        b_values_sequential[i] = b_values_batch0[i];
        b_values_sequential[m + i] = b_values_batch1[i];
    }
    b.cpuToGpu(b_values_sequential.data(), stream);
    
    // CRITICAL: Synchronize after all data uploads before computing row_of
    cudaStreamSynchronize(stream);
    
    Cones cones;
    cones.numZeroCones = 2;
    cones.numNonnegCones = 2;
    cones.numExpCones = 1;
    cones.socConeDims = {3};
    cones.numSocCones = 1;
    cones.numPowerCones = 1;
    cones.powerAlphas = {0.6};
    
    EquilibrationData equil(n, m, batchSize);
    
    EquilibrationSettings settings;
    settings.max_iter = 10;
    settings.scale_min = 1e-4;
    settings.scale_max = 1e4;

    // Compute row_of arrays and transfer to GPU
    std::vector<int64_t> h_matrixPRowOf = moreau::computeRowOfFromCSR(P_rowOffsets.data(), n, nnzP);
    std::vector<int64_t> h_matrixARowOf = moreau::computeRowOfFromCSR(A_rowOffsets.data(), m, nnzA);
    int64_t* matrixPRowOf = moreau::allocateAndCopyRowOfToGPU(h_matrixPRowOf, stream);
    int64_t* matrixARowOf = moreau::allocateAndCopyRowOfToGPU(h_matrixARowOf, stream);
    
    // CRITICAL: Synchronize after row_of uploads
    cudaStreamSynchronize(stream);

    // Call equilibration
    equilibration(equil, P, matrixPRowOf, A, matrixARowOf, q, b, settings, cones, stream);
    
    // CRITICAL: Synchronize after equilibration completes
    cudaStreamSynchronize(stream);
    
    // Expected values for BATCH 0
    std::vector<double> expected_dwork_batch0 = {
        0.28747406964430816, 0.4969392301500449, 1.0000000000000002, 1.0, 1.0
    };

    std::vector<double> expected_ework_batch0 = {
        1.0, 1.0, 1.0, 1.0, 0.8532019802740828, 0.952095403787599,
        1.285959156111361, 0.9612987854586279, 1.0040861455952865,
        1.0375486590863268, 0.9376187517799576, 0.9151142335323873,
        1.189472671308199
    };

    std::vector<double> expected_d_batch0 = {
        0.2874740696443081, 0.2672612419124244, 0.30955563293443084,
        0.26808304200578786, 0.23978076221594932
    };

    std::vector<double> expected_e_batch0 = {
        3.4711185413527454, 1.86727692801687, 1.0755339944662083,
        0.9314677524441021, 0.5932447229340823, 0.5932447229340823,
        0.5932447229340823, 0.38808761926390384, 0.3880876192639038,
        0.3880876192639039, 0.3178999433547183, 0.31789994335471816,
        0.3178999433547182
    };

    std::vector<double> expected_dinv_batch0 = {
        3.4785746110503144, 3.7416573867739413, 3.2304370962999633,
        3.7301874542978743, 4.1704763583134685
    };

    std::vector<double> expected_einv_batch0 = {
        0.2880915728133806, 0.5355392041725937, 0.9297707047337949,
        1.0735744714468907, 1.6856449983308388, 1.6856449983308388,
        1.6856449983308388, 2.5767377013900283, 2.576737701390029,
        2.576737701390028, 3.145643844560811, 3.145643844560812,
        3.1456438445608113
    };

    double expected_c_batch0 = 3.478574611050313;
    
    // Expected values for BATCH 1
    std::vector<double> expected_d_batch1 = {
        0.3333333333333333, 0.31622776601683794, 0.5773502691896258,
        0.3333333333333333, 1.0
    };
    
    std::vector<double> expected_e_batch1 = {
        0.5996556925785107, 1.2631997821947059, 1.7301935498381285,
        1.4977983789203222, 2.276015416667009, 2.2760154166670095,
        2.2760154166670095, 0.5050812192826687, 0.5050812192826687,
        0.5050812192826687, 0.5949766531133347, 0.5949766531133348,
        0.5949766531133348
    };
    
    std::vector<double> expected_dinv_batch1 = {
        3.0, 3.162277660168379, 1.732050807568877, 3.0, 1.0
    };
    
    std::vector<double> expected_einv_batch1 = {
        1.6676236253174128, 0.7916404151547447, 0.5779700196510136,
        0.6676466032236216, 0.4393643350027907, 0.4393643350027906,
        0.4393643350027906, 1.979879595246542, 1.979879595246542,
        1.979879595246542, 1.6807382184952961, 1.680738218495296,
        1.680738218495296
    };

    double expected_c_batch1 = 1.0;

    std::vector<double> expected_dwork_batch1 = {
        0.5555555555555556, 0.4, 1.0000000000000002, 0.2222222222222222, 1.0
    };

    std::vector<double> expected_ework_batch1 = {
        1.0, 1.0, 1.0, 1.0, 0.7603014559961291, 1.081610904054473,
        1.3154686750970412, 0.5834485335584269, 1.5152436578480062,
        1.5972070563081895, 1.7849299593400043, 0.5949766531133348,
        1.3174957890507928
    };
    
    // Get computed values
    std::vector<double> computed_d(n * batchSize);
    std::vector<double> computed_dinv(n * batchSize);
    std::vector<double> computed_e(m * batchSize);
    std::vector<double> computed_einv(m * batchSize);
    std::vector<double> computed_c(batchSize);
    std::vector<double> computed_dwork(n * batchSize);
    std::vector<double> computed_ework(m * batchSize);

    equil.gpuToCpu(computed_d.data(), computed_dinv.data(),
                   computed_e.data(), computed_einv.data(),
                   computed_c.data(), computed_dwork.data(),
                   computed_ework.data(), stream);
    cudaStreamSynchronize(stream);
    
    // Verify BATCH 0
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_d[i], expected_d_batch0[i], 1e-10) 
            << "Batch 0: d[" << i << "] mismatch";
    }
    
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_dinv[i], expected_dinv_batch0[i], 1e-10)
            << "Batch 0: dinv[" << i << "] mismatch";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_e[i], expected_e_batch0[i], 1e-10)
            << "Batch 0: e[" << i << "] mismatch";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_einv[i], expected_einv_batch0[i], 1e-10)
            << "Batch 0: einv[" << i << "] mismatch";
    }
    
    EXPECT_NEAR(computed_c[0], expected_c_batch0, 1e-10) << "Batch 0: c mismatch";
    
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_d[i] * computed_dinv[i], 1.0, 1e-10)
            << "Batch 0: d[" << i << "] * dinv[" << i << "] != 1";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_e[i] * computed_einv[i], 1.0, 1e-10)
            << "Batch 0: e[" << i << "] * einv[" << i << "] != 1";
    }

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_dwork[i], expected_dwork_batch0[i], 1e-10)
            << "Batch 0: dwork[" << i << "] mismatch";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_ework[i], expected_ework_batch0[i], 1e-10)
            << "Batch 0: ework[" << i << "] mismatch";
    }
    
    // Verify BATCH 1
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_d[n + i], expected_d_batch1[i], 1e-10) 
            << "Batch 1: d[" << i << "] mismatch";
    }
    
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_dinv[n + i], expected_dinv_batch1[i], 1e-10)
            << "Batch 1: dinv[" << i << "] mismatch";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_e[m + i], expected_e_batch1[i], 1e-10)
            << "Batch 1: e[" << i << "] mismatch";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_einv[m + i], expected_einv_batch1[i], 1e-10)
            << "Batch 1: einv[" << i << "] mismatch";
    }
    
    EXPECT_NEAR(computed_c[1], expected_c_batch1, 1e-10) << "Batch 1: c mismatch";
    
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_d[n + i] * computed_dinv[n + i], 1.0, 1e-10)
            << "Batch 1: d[" << i << "] * dinv[" << i << "] != 1";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_e[m + i] * computed_einv[m + i], 1.0, 1e-10)
            << "Batch 1: e[" << i << "] * einv[" << i << "] != 1";
    }

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(computed_dwork[n + i], expected_dwork_batch1[i], 1e-10)
            << "Batch 1: dwork[" << i << "] mismatch";
    }
    
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(computed_ework[m + i], expected_ework_batch1[i], 1e-10)
            << "Batch 1: ework[" << i << "] mismatch";
    }
    
    // Verify scaled matrices and vectors
    std::vector<double> expected_P_scaled_batch0 = {
        0.28747406964430816, 0.4969392301500449, 1.0000000000000002, 1.0, 1.0
    };
    
    std::vector<double> expected_q_scaled_batch0 = {
        1.0, -0.9296881706343354, 0.5384061827166609, -0.4662734317872343, 0.0
    };
    
    std::vector<double> expected_A_scaled_batch0 = {
        0.9978565733004885, 0.9981015015524106, 0.9988128191984512,
        0.9988428344220363, 0.8527123739843532, 0.9513079284561298,
        1.2854957198500965, 0.9610776689220335, 1.0040861455952867,
        1.0372077909530708, 0.9374594225480058, 0.9147154887120174,
        1.1880438759263865, 1.1894726713081991
    };
    
    std::vector<double> expected_b_scaled_batch0 = {
        3.4711185413527454, 1.86727692801687, 0.5377669972331042,
        0.46573387622205104, 0.0, 0.5932447229340823, 1.1864894458681645,
        0.38808761926390384, 0.0, 0.0, 0.3178999433547183,
        0.31789994335471816, 0.0
    };
    
    std::vector<double> expected_P_scaled_batch1 = {
        0.5555555555555556, 0.4, 1.0000000000000002, 0.2222222222222222, 1.0
    };
    
    std::vector<double> expected_q_scaled_batch1 = {
        0.6666666666666666, -0.6324555320336759, 0.8660254037844388, -0.5, -1.0
    };
    
    std::vector<double> expected_A_scaled_batch1 = {
        0.9994261542975177, 0.9986471128909701, 0.9989277117491978,
        0.9985322526135483, 0.7586718055556695, 1.0796089059487368,
        1.3140581134924363, 0.5832175558309464, 1.5152436578480062,
        1.5972070563081895, 1.7849299593400043, 0.5949766531133348,
        1.1899533062266696, 1.3170369649234352
    };
    
    std::vector<double> expected_b_scaled_batch1 = {
        5.396901233206598, 1.2631997821947059, 12.1113548488669,
        2.9955967578406444, 0.0, -9.104061666668038, -6.828046250001028,
        0.5050812192826687, 0.0, 0.0, 8.924649796700022,
        -1.1899533062266696, 0.0
    };
    
    std::vector<double> scaled_P(nnzP * batchSize);
    std::vector<double> scaled_A(nnzA * batchSize);
    std::vector<double> scaled_q(n * batchSize);
    std::vector<double> scaled_b(m * batchSize);
    
    P.valuesGpuToCpu(scaled_P.data(), stream);
    A.valuesGpuToCpu(scaled_A.data(), stream);
    q.gpuToCpu(scaled_q.data(), stream);
    b.gpuToCpu(scaled_b.data(), stream);
    cudaStreamSynchronize(stream);
    
    // Verify scaled P batch 0
    for (int i = 0; i < nnzP; i++) {
        EXPECT_NEAR(scaled_P[i], expected_P_scaled_batch0[i], 1e-8)
            << "Batch 0: scaled P[" << i << "] mismatch";
    }
    
    // Verify scaled q batch 0
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(scaled_q[i], expected_q_scaled_batch0[i], 1e-8)
            << "Batch 0: scaled q[" << i << "] mismatch";
    }
    
    // Verify scaled A batch 0
    for (int i = 0; i < nnzA; i++) {
        EXPECT_NEAR(scaled_A[i], expected_A_scaled_batch0[i], 1e-8)
            << "Batch 0: scaled A[" << i << "] mismatch";
    }
    
    // Verify scaled b batch 0
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(scaled_b[i], expected_b_scaled_batch0[i], 1e-8)
            << "Batch 0: scaled b[" << i << "] mismatch";
    }
    
    // Verify scaled P batch 1
    for (int i = 0; i < nnzP; i++) {
        EXPECT_NEAR(scaled_P[nnzP + i], expected_P_scaled_batch1[i], 1e-8)
            << "Batch 1: scaled P[" << i << "] mismatch";
    }
    
    // Verify scaled q batch 1
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(scaled_q[n + i], expected_q_scaled_batch1[i], 1e-8)
            << "Batch 1: scaled q[" << i << "] mismatch";
    }
    
    // Verify scaled A batch 1
    for (int i = 0; i < nnzA; i++) {
        EXPECT_NEAR(scaled_A[nnzA + i], expected_A_scaled_batch1[i], 1e-8)
            << "Batch 1: scaled A[" << i << "] mismatch";
    }
    
    // Verify scaled b batch 1
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(scaled_b[m + i], expected_b_scaled_batch1[i], 1e-8)
            << "Batch 1: scaled b[" << i << "] mismatch";
    }
    
    // Cleanup
    cudaFree(matrixPRowOf);
    cudaFree(matrixARowOf);
}

TEST(EquilibrationKernels, ComputeBatchRowNormsP_FullSymmetricBatch) {
    using namespace moreau;
    int n = 3, batchSize = 2, nnzP = 9;  // Full 3x3 symmetric matrix

    // Full symmetric P matrix (both upper and lower triangle stored)
    // Row 0: P[0,0]=1, P[0,1]=-4, P[0,2]=2
    // Row 1: P[1,0]=-4, P[1,1]=3, P[1,2]=-1
    // Row 2: P[2,0]=2, P[2,1]=-1, P[2,2]=5
    int64_t h_rowOffsets[] = {0, 3, 6, 9};
    int64_t h_col_indices[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    // Batch 0: diag(1,3,5), off-diag symmetric (-4, 2, -1)
    // Batch 1: diag(2,-5,-1), off-diag symmetric (0, 7, 6)
    double h_values[] = {
        // Batch 0: row 0: 1,-4,2; row 1: -4,3,-1; row 2: 2,-1,5
        1, -4, 2, -4, 3, -1, 2, -1, 5,
        // Batch 1: row 0: 2,0,7; row 1: 0,-5,6; row 2: 7,6,-1
        2, 0, 7, 0, -5, 6, 7, 6, -1
    };

    moreau::CSR P(n, n, nnzP, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    P.indicesCpuToGpu(h_rowOffsets, h_col_indices, stream);
    P.valuesCpuToGpu(h_values, stream);

    std::vector<int64_t> h_row_of = moreau::computeRowOfFromCSR(h_rowOffsets, n, nnzP);
    int64_t* d_row_of = moreau::allocateAndCopyRowOfToGPU(h_row_of, stream);
    cudaStreamSynchronize(stream);

    moreau::BatchedVector d_row_norms(n, batchSize);
    d_row_norms.setToConstant(0.0, stream);
    cudaStreamSynchronize(stream);

    moreau::compute_batch_row_norms_P(P, d_row_of, d_row_norms, stream);

    std::vector<double> h_row_norms(n * batchSize);
    d_row_norms.gpuToCpu(h_row_norms.data(), stream);
    cudaStreamSynchronize(stream);

    // Batch 0: row 0 max(|1|,|-4|,|2|)=4, row 1 max(|-4|,|3|,|-1|)=4, row 2 max(|2|,|-1|,|5|)=5
    EXPECT_NEAR(h_row_norms[0 * n + 0], 4.0, 1e-12);
    EXPECT_NEAR(h_row_norms[0 * n + 1], 4.0, 1e-12);
    EXPECT_NEAR(h_row_norms[0 * n + 2], 5.0, 1e-12);

    // Batch 1: row 0 max(|2|,|0|,|7|)=7, row 1 max(|0|,|-5|,|6|)=6, row 2 max(|7|,|6|,|-1|)=7
    EXPECT_NEAR(h_row_norms[1 * n + 0], 7.0, 1e-12);
    EXPECT_NEAR(h_row_norms[1 * n + 1], 6.0, 1e-12);
    EXPECT_NEAR(h_row_norms[1 * n + 2], 7.0, 1e-12);

    cudaFree(d_row_of);
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeBatchColNormsANoReset_RectangularBatch) {
    using namespace moreau;
    int nrow = 2, ncol = 3, batchSize = 2, nnzA = 5;

    int64_t h_rowOffsets[] = {0, 3, 5};
    int64_t h_col_indices[] = {0, 1, 2, 1, 2};
    double h_values[] = {1, -4, 2, 3, -1, 2, 0, 7, -5, 6};

    moreau::CSR A(nrow, ncol, nnzA, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    A.indicesCpuToGpu(h_rowOffsets, h_col_indices, stream);
    A.valuesCpuToGpu(h_values, stream);

    moreau::BatchedVector d_col_norms(ncol, batchSize);
    d_col_norms.setToConstant(0.0, stream);
    cudaStreamSynchronize(stream);

    moreau::compute_batch_col_norms_A_noreset(A, d_col_norms, stream);

    std::vector<double> h_col_norms(ncol * batchSize);
    d_col_norms.gpuToCpu(h_col_norms.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_col_norms[0 * ncol + 0], 1.0, 1e-12);
    EXPECT_NEAR(h_col_norms[0 * ncol + 1], 4.0, 1e-12);
    EXPECT_NEAR(h_col_norms[0 * ncol + 2], 2.0, 1e-12);

    EXPECT_NEAR(h_col_norms[1 * ncol + 0], 2.0, 1e-12);
    EXPECT_NEAR(h_col_norms[1 * ncol + 1], 5.0, 1e-12);
    EXPECT_NEAR(h_col_norms[1 * ncol + 2], 7.0, 1e-12);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeBatchRowNormsA_SimpleBatch) {
    using namespace moreau;
    int n = 3, batchSize = 2, nnzA = 6;

    int64_t h_rowOffsets[] = {0, 3, 5, 6};
    int64_t h_col_indices[] = {0, 1, 2, 1, 2, 2};
    double h_values[] = {1, -4, 2, 3, -1, 5, 2, 0, 7, -5, 6, -1};

    moreau::CSR A(n, n, nnzA, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    A.indicesCpuToGpu(h_rowOffsets, h_col_indices, stream);
    A.valuesCpuToGpu(h_values, stream);

    std::vector<int64_t> h_row_of = moreau::computeRowOfFromCSR(h_rowOffsets, n, nnzA);
    int64_t* d_row_of = moreau::allocateAndCopyRowOfToGPU(h_row_of, stream);
    cudaStreamSynchronize(stream);

    moreau::BatchedVector d_row_norms(n, batchSize);
    d_row_norms.setToConstant(0.0, stream);
    cudaStreamSynchronize(stream);

    moreau::compute_batch_row_norms_A(A, d_row_of, d_row_norms, stream);

    std::vector<double> h_row_norms(n * batchSize);
    d_row_norms.gpuToCpu(h_row_norms.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_row_norms[0 * n + 0], 4.0, 1e-12);
    EXPECT_NEAR(h_row_norms[0 * n + 1], 3.0, 1e-12);
    EXPECT_NEAR(h_row_norms[0 * n + 2], 5.0, 1e-12);

    EXPECT_NEAR(h_row_norms[1 * n + 0], 7.0, 1e-12);
    EXPECT_NEAR(h_row_norms[1 * n + 1], 6.0, 1e-12);
    EXPECT_NEAR(h_row_norms[1 * n + 2], 1.0, 1e-12);

    cudaFree(d_row_of);
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ProcessDworkAndEwork) {
    constexpr int n = 3;
    constexpr int m = 2;
    constexpr int batch_size = 2;

    std::vector<double> h_dwork = {4.0, 0.0, 16.0, 0.0, 25.0, 100.0};
    std::vector<double> h_ework = {9.0, 0.0, 36.0, 0.0};
    std::vector<double> h_d = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> h_e = {1.0, 1.0, 1.0, 1.0};

    double scale_min = 0.5;
    double scale_max = 2.0;

    moreau::BatchedVector dwork(n, batch_size);
    moreau::BatchedVector ework(m, batch_size);
    moreau::BatchedVector d(n, batch_size);
    moreau::BatchedVector e(m, batch_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    dwork.cpuToGpu(h_dwork.data(), stream);
    ework.cpuToGpu(h_ework.data(), stream);
    d.cpuToGpu(h_d.data(), stream);
    e.cpuToGpu(h_e.data(), stream);
    cudaStreamSynchronize(stream);

    moreau::process_dwork(dwork, d, scale_min, scale_max, n, batch_size, stream);
    moreau::process_ework(ework, e, scale_min, scale_max, m, batch_size, stream);

    std::vector<double> h_dwork_out(n * batch_size);
    std::vector<double> h_ework_out(m * batch_size);
    std::vector<double> h_d_out(n * batch_size);
    std::vector<double> h_e_out(m * batch_size);

    dwork.gpuToCpu(h_dwork_out.data(), stream);
    ework.gpuToCpu(h_ework_out.data(), stream);
    d.gpuToCpu(h_d_out.data(), stream);
    e.gpuToCpu(h_e_out.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_dwork_out[0 * n + 0], 0.5, 1e-12);
    EXPECT_NEAR(h_dwork_out[0 * n + 1], 1.0, 1e-12);
    EXPECT_NEAR(h_dwork_out[0 * n + 2], 0.5, 1e-12);

    EXPECT_NEAR(h_dwork_out[1 * n + 0], 1.0, 1e-12);
    EXPECT_NEAR(h_dwork_out[1 * n + 1], 0.5, 1e-12);
    EXPECT_NEAR(h_dwork_out[1 * n + 2], 0.5, 1e-12);

    EXPECT_NEAR(h_d_out[0 * n + 0], 1.0 * 0.5, 1e-12);
    EXPECT_NEAR(h_d_out[0 * n + 1], 1.0 * 1.0, 1e-12);
    EXPECT_NEAR(h_d_out[0 * n + 2], 1.0 * 0.5, 1e-12);
    EXPECT_NEAR(h_d_out[1 * n + 0], 1.0 * 1.0, 1e-12);
    EXPECT_NEAR(h_d_out[1 * n + 1], 1.0 * 0.5, 1e-12);
    EXPECT_NEAR(h_d_out[1 * n + 2], 1.0 * 0.5, 1e-12);

    EXPECT_NEAR(h_ework_out[0 * m + 0], 0.5, 1e-12);
    EXPECT_NEAR(h_ework_out[0 * m + 1], 1.0, 1e-12);
    EXPECT_NEAR(h_ework_out[1 * m + 0], 0.5, 1e-12);
    EXPECT_NEAR(h_ework_out[1 * m + 1], 1.0, 1e-12);

    EXPECT_NEAR(h_e_out[0 * m + 0], 1.0 * h_ework_out[0 * m + 0], 1e-12);
    EXPECT_NEAR(h_e_out[0 * m + 1], 1.0 * h_ework_out[0 * m + 1], 1e-12);
    EXPECT_NEAR(h_e_out[1 * m + 0], 1.0 * h_ework_out[1 * m + 0], 1e-12);
    EXPECT_NEAR(h_e_out[1 * m + 1], 1.0 * h_ework_out[1 * m + 1], 1e-12);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, TestScaleDataBatch) {
    using namespace moreau;

    constexpr int n = 3;
    constexpr int m = 2;
    constexpr int batch_size = 2;
    constexpr int nnzP = 3;
    constexpr int nnzA = 6;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CSR P(n, n, nnzP, batch_size);

    int64_t P_rowOffsets[] = {0, 1, 2, 3};
    int64_t P_colIndices[] = {0, 1, 2};

    std::vector<double> P_values(nnzP * batch_size);
    P_values[0] = 2.0;  P_values[1] = 3.0;  P_values[2] = 4.0;
    P_values[3] = 1.0;  P_values[4] = 5.0;  P_values[5] = -1.0;

    P.indicesCpuToGpu(P_rowOffsets, P_colIndices, stream);
    P.valuesCpuToGpu(P_values.data(), stream);

    std::vector<int64_t> h_P_row_of = computeRowOfFromCSR(P_rowOffsets, P.n(), nnzP);
    int64_t* d_P_row_of = allocateAndCopyRowOfToGPU(h_P_row_of, stream);

    CSR A(m, n, nnzA, batch_size);

    int64_t A_rowOffsets[] = {0, 3, 6};
    int64_t A_colIndices[] = {0, 1, 2, 0, 1, 2};

    std::vector<double> A_values(nnzA * batch_size);
    A_values[0] = 1.0;  A_values[1] = 2.0;  A_values[2] = 3.0;
    A_values[3] = 4.0;  A_values[4] = 5.0;  A_values[5] = 6.0;
    A_values[6] = 7.0;  A_values[7] = 8.0;  A_values[8] = 9.0;
    A_values[9] = 10.0; A_values[10] = 11.0; A_values[11] = 12.0;

    A.indicesCpuToGpu(A_rowOffsets, A_colIndices, stream);
    A.valuesCpuToGpu(A_values.data(), stream);

    std::vector<int64_t> h_A_row_of = computeRowOfFromCSR(A_rowOffsets, A.n(), nnzA);
    int64_t* d_A_row_of = allocateAndCopyRowOfToGPU(h_A_row_of, stream);

    BatchedVector q(n, batch_size);
    BatchedVector b(m, batch_size);

    std::vector<double> q_values(n * batch_size);
    q_values[0] = 1.0;  q_values[1] = 2.0;  q_values[2] = 3.0;
    q_values[3] = -2.0; q_values[4] = 5.0;  q_values[5] = 1.0;

    std::vector<double> b_values(m * batch_size);
    b_values[0] = 7.0;  b_values[1] = 9.0;
    b_values[2] = -4.0; b_values[3] = 2.5;

    q.cpuToGpu(q_values.data(), stream);
    b.cpuToGpu(b_values.data(), stream);

    BatchedVector d(n, batch_size);
    BatchedVector e(m, batch_size);

    std::vector<double> d_values(n * batch_size);
    d_values[0] = 2.0;  d_values[1] = 0.5;  d_values[2] = -1.0;
    d_values[3] = 0.25; d_values[4] = 1.0;  d_values[5] = 2.0;

    std::vector<double> e_values(m * batch_size);
    e_values[0] = 4.0;  e_values[1] = -2.0;
    e_values[2] = 1.5;  e_values[3] = 3.0;

    d.cpuToGpu(d_values.data(), stream);
    e.cpuToGpu(e_values.data(), stream);
    cudaStreamSynchronize(stream);

    moreau::scale_data(P, d_P_row_of, A, d_A_row_of, q, b, &d, &e, stream);

    std::vector<double> P_scaled(nnzP * batch_size);
    std::vector<double> A_scaled(nnzA * batch_size);
    std::vector<double> q_scaled(n * batch_size);
    std::vector<double> b_scaled(m * batch_size);

    P.valuesGpuToCpu(P_scaled.data(), stream);
    A.valuesGpuToCpu(A_scaled.data(), stream);
    q.gpuToCpu(q_scaled.data(), stream);
    b.gpuToCpu(b_scaled.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(P_scaled[0], 8.0, 1e-12);
    EXPECT_NEAR(P_scaled[1], 0.75, 1e-12);
    EXPECT_NEAR(P_scaled[2], 4.0, 1e-12);

    EXPECT_NEAR(A_scaled[0], 8.0, 1e-12);
    EXPECT_NEAR(A_scaled[1], 4.0, 1e-12);
    EXPECT_NEAR(A_scaled[2], -12.0, 1e-12);
    EXPECT_NEAR(A_scaled[3], -16.0, 1e-12);
    EXPECT_NEAR(A_scaled[4], -5.0, 1e-12);
    EXPECT_NEAR(A_scaled[5], 12.0, 1e-12);

    EXPECT_NEAR(q_scaled[0], 2.0, 1e-12);
    EXPECT_NEAR(q_scaled[1], 1.0, 1e-12);
    EXPECT_NEAR(q_scaled[2], -3.0, 1e-12);

    EXPECT_NEAR(b_scaled[0], 28.0, 1e-12);
    EXPECT_NEAR(b_scaled[1], -18.0, 1e-12);

    EXPECT_NEAR(P_scaled[3], 0.0625, 1e-12);
    EXPECT_NEAR(P_scaled[4], 5.0, 1e-12);
    EXPECT_NEAR(P_scaled[5], -4.0, 1e-12);

    EXPECT_NEAR(A_scaled[6], 2.625, 1e-12);
    EXPECT_NEAR(A_scaled[7], 12.0, 1e-12);
    EXPECT_NEAR(A_scaled[8], 27.0, 1e-12);
    EXPECT_NEAR(A_scaled[9], 7.5, 1e-12);
    EXPECT_NEAR(A_scaled[10], 33.0, 1e-12);
    EXPECT_NEAR(A_scaled[11], 72.0, 1e-12);

    EXPECT_NEAR(q_scaled[3], -0.5, 1e-12);
    EXPECT_NEAR(q_scaled[4], 5.0, 1e-12);
    EXPECT_NEAR(q_scaled[5], 2.0, 1e-12);

    EXPECT_NEAR(b_scaled[2], -6.0, 1e-12);
    EXPECT_NEAR(b_scaled[3], 7.5, 1e-12);

    cudaFree(d_P_row_of);
    cudaFree(d_A_row_of);
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ScaleByScalar_SingleBatch) {
    using namespace moreau;

    int n = 5;
    int batchSize = 1;

    BatchedVector vec(n, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    scale_by_scalar(vec, 2.0, stream);

    std::vector<double> h_output(n);
    vec.gpuToCpu(h_output.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_output[0], 2.0, 1e-12);
    EXPECT_NEAR(h_output[1], 4.0, 1e-12);
    EXPECT_NEAR(h_output[2], 6.0, 1e-12);
    EXPECT_NEAR(h_output[3], 8.0, 1e-12);
    EXPECT_NEAR(h_output[4], 10.0, 1e-12);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ScaleByScalar_MultipleBatches) {
    using namespace moreau;

    int n = 4;
    int batchSize = 3;

    BatchedVector vec(n, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    scale_by_scalar(vec, 0.5, stream);

    std::vector<double> h_output(n * batchSize);
    vec.gpuToCpu(h_output.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_output[0], 0.5, 1e-12);
    EXPECT_NEAR(h_output[1], 1.0, 1e-12);
    EXPECT_NEAR(h_output[2], 1.5, 1e-12);
    EXPECT_NEAR(h_output[3], 2.0, 1e-12);

    EXPECT_NEAR(h_output[4], 2.5, 1e-12);
    EXPECT_NEAR(h_output[5], 3.0, 1e-12);
    EXPECT_NEAR(h_output[6], 3.5, 1e-12);
    EXPECT_NEAR(h_output[7], 4.0, 1e-12);

    EXPECT_NEAR(h_output[8], 4.5, 1e-12);
    EXPECT_NEAR(h_output[9], 5.0, 1e-12);
    EXPECT_NEAR(h_output[10], 5.5, 1e-12);
    EXPECT_NEAR(h_output[11], 6.0, 1e-12);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ScaleByScalar_NegativeScalar) {
    using namespace moreau;

    int n = 3;
    int batchSize = 2;

    BatchedVector vec(n, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {
        2.0, -4.0, 6.0,
        -1.0, 3.0, -5.0
    };
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    scale_by_scalar(vec, -0.5, stream);

    std::vector<double> h_output(n * batchSize);
    vec.gpuToCpu(h_output.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_output[0], -1.0, 1e-12);
    EXPECT_NEAR(h_output[1], 2.0, 1e-12);
    EXPECT_NEAR(h_output[2], -3.0, 1e-12);

    EXPECT_NEAR(h_output[3], 0.5, 1e-12);
    EXPECT_NEAR(h_output[4], -1.5, 1e-12);
    EXPECT_NEAR(h_output[5], 2.5, 1e-12);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ScaleByScalar_ZeroScalar) {
    using namespace moreau;

    int n = 4;
    int batchSize = 1;

    BatchedVector vec(n, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {10.0, 20.0, 30.0, 40.0};
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    scale_by_scalar(vec, 0.0, stream);

    std::vector<double> h_output(n);
    vec.gpuToCpu(h_output.data(), stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(h_output[i], 0.0, 1e-12);
    }
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeMean_SingleBatch) {
    using namespace moreau;

    int n = 5;
    int batchSize = 1;

    BatchedVector vec(n, batchSize);
    BatchedVector mean_result(1, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {2.0, 4.0, 6.0, 8.0, 10.0};
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);
    
    // Check for errors after upload
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error after cpuToGpu: " << cudaGetErrorString(err);

    compute_mean(vec, mean_result, stream);
    
    // Check for errors after kernel
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error after compute_mean: " << cudaGetErrorString(err);
    
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error after sync: " << cudaGetErrorString(err);

    double h_mean;
    mean_result.gpuToCpu(&h_mean, stream);
    cudaStreamSynchronize(stream);

    std::cout << "Computed mean: " << h_mean << " (expected: 6.0)" << std::endl;
    EXPECT_NEAR(h_mean, 6.0, 1e-10);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeMean_MultipleBatches) {
    using namespace moreau;

    int n = 4;
    int batchSize = 3;

    BatchedVector vec(n, batchSize);
    BatchedVector mean_result(1, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {
        1.0, 2.0, 3.0, 4.0,
        8.0, 4.0, 0.0, 4.0,
        10.0, 20.0, 30.0, 40.0
    };
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    compute_mean(vec, mean_result, stream);
    cudaStreamSynchronize(stream);

    std::vector<double> h_means(batchSize);
    mean_result.gpuToCpu(h_means.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_means[0], 2.5, 1e-10);
    EXPECT_NEAR(h_means[1], 4.0, 1e-10);
    EXPECT_NEAR(h_means[2], 25.0, 1e-10);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeMean_WithNegativeValues) {
    using namespace moreau;

    int n = 6;
    int batchSize = 2;

    BatchedVector vec(n, batchSize);
    BatchedVector mean_result(1, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {
        3.0, -3.0, 6.0, -6.0, 9.0, -9.0,
        -2.0, -4.0, -6.0, -8.0, -10.0, -12.0
    };
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    compute_mean(vec, mean_result, stream);

    std::vector<double> h_means(batchSize);
    mean_result.gpuToCpu(h_means.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_means[0], 0.0, 1e-10);
    EXPECT_NEAR(h_means[1], -7.0, 1e-10);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeMean_LargeVector) {
    using namespace moreau;

    int n = 1000;
    int batchSize = 2;

    BatchedVector vec(n, batchSize);
    BatchedVector mean_result(1, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input(n * batchSize);

    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0;
    }

    for (int i = 0; i < n; i++) {
        h_input[n + i] = (double)(i + 1);
    }

    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    compute_mean(vec, mean_result, stream);
    cudaStreamSynchronize(stream);

    std::vector<double> h_means(batchSize);
    mean_result.gpuToCpu(h_means.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_means[0], 1.0, 1e-10);
    EXPECT_NEAR(h_means[1], 500.5, 1e-8);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeMean_SingleElement) {
    using namespace moreau;

    int n = 1;
    int batchSize = 3;

    BatchedVector vec(n, batchSize);
    BatchedVector mean_result(1, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input = {42.0, -7.5, 0.0};
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    compute_mean(vec, mean_result, stream);

    std::vector<double> h_means(batchSize);
    mean_result.gpuToCpu(h_means.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_means[0], 42.0, 1e-12);
    EXPECT_NEAR(h_means[1], -7.5, 1e-12);
    EXPECT_NEAR(h_means[2], 0.0, 1e-12);
    
    cudaStreamDestroy(stream);
}

TEST(EquilibrationKernels, ComputeMean_AllZeros) {
    using namespace moreau;

    int n = 10;
    int batchSize = 2;

    BatchedVector vec(n, batchSize);
    BatchedVector mean_result(1, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> h_input(n * batchSize, 0.0);
    vec.cpuToGpu(h_input.data(), stream);
    cudaStreamSynchronize(stream);

    compute_mean(vec, mean_result, stream);

    std::vector<double> h_means(batchSize);
    mean_result.gpuToCpu(h_means.data(), stream);
    cudaStreamSynchronize(stream);

    EXPECT_NEAR(h_means[0], 0.0, 1e-12);
    EXPECT_NEAR(h_means[1], 0.0, 1e-12);
    
    cudaStreamDestroy(stream);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
