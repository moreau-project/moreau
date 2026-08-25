// test_matrix.cpp
#include <gtest/gtest.h>
#include "moreau/matrix/csr.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace moreau;

class BatchedCSRMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple test dimensions
        n = 4;
        m = 4;
        nnz = 6;
        batchSize = 2;
    }

    void TearDown() override {
        // Clean up is automatic with RAII
    }

    // Helper function to create a simple CSR matrix for testing
    // Matrix: [2 0 1 0]
    //         [0 3 0 0]
    //         [0 0 4 5]
    //         [0 0 0 6]
    // Indices are shared across batches, only values differ per batch
    void createSimpleCSRData(std::vector<int64_t>& rowOffsets,
                             std::vector<int64_t>& colIndices,
                             std::vector<double>& values,
                             int batch) {
        // Row offsets: shared structure across all batches
        rowOffsets[0] = 0;
        rowOffsets[1] = 2;  // Row 0 has 2 elements
        rowOffsets[2] = 3;  // Row 1 has 1 element
        rowOffsets[3] = 5;  // Row 2 has 2 elements
        rowOffsets[4] = 6;  // Row 3 has 1 element

        // Column indices: shared structure across all batches
        colIndices[0] = 0;  // Row 0, col 0
        colIndices[1] = 2;  // Row 0, col 2
        colIndices[2] = 1;  // Row 1, col 1
        colIndices[3] = 2;  // Row 2, col 2
        colIndices[4] = 3;  // Row 2, col 3
        colIndices[5] = 3;  // Row 3, col 3

        // Values: different for each batch
        for (int b = 0; b < batch; b++) {
            double offset = b * 10.0;
            values[b * nnz + 0] = 2.0 + offset;
            values[b * nnz + 1] = 1.0 + offset;
            values[b * nnz + 2] = 3.0 + offset;
            values[b * nnz + 3] = 4.0 + offset;
            values[b * nnz + 4] = 5.0 + offset;
            values[b * nnz + 5] = 6.0 + offset;
        }
    }

    int n, m, nnz, batchSize;
};

// Test 1: Constructor allocates memory
TEST_F(BatchedCSRMatrixTest, ConstructorAllocates) {
    ASSERT_NO_THROW({
        CSR mat(n, m, nnz, batchSize);
        EXPECT_NE(mat.rowOffsets(), nullptr);
        EXPECT_NE(mat.colIndices(), nullptr);
        EXPECT_NE(mat.values(), nullptr);
        EXPECT_EQ(mat.n()   , n);
        EXPECT_EQ(mat.m()   , m);
        EXPECT_EQ(mat.nnz() , nnz);
        EXPECT_EQ(mat.batchSize() , batchSize);
    });
}

// Test 2: Memory ownership and lifecycle
TEST_F(BatchedCSRMatrixTest, WrapConstructorDoesNotAllocate) {
    // Test that the CSR matrix properly manages its own memory
    CSR mat(n, m, nnz, batchSize);

    // Verify dimensions
    EXPECT_EQ(mat.n(), n);
    EXPECT_EQ(mat.m(), m);
    EXPECT_EQ(mat.nnz(), nnz);
    EXPECT_EQ(mat.batchSize(), batchSize);

    // Verify pointers are allocated
    EXPECT_NE(mat.rowOffsets(), nullptr);
    EXPECT_NE(mat.colIndices(), nullptr);
    EXPECT_NE(mat.values(), nullptr);

    // Store pointers to verify they're consistent
    auto ro = mat.rowOffsets();
    auto ci = mat.colIndices();
    auto v = mat.values();

    // Verify pointers remain stable
    EXPECT_EQ(mat.rowOffsets(), ro);
    EXPECT_EQ(mat.colIndices(), ci);
    EXPECT_EQ(mat.values(), v);
}

// Test 3: Indices CPU to GPU transfer
TEST_F(BatchedCSRMatrixTest, IndicesCpuToGpu) {
    CSR mat(n, m, nnz, batchSize);

    // Create host data - indices shared across batches
    std::vector<int64_t> h_rowOffsets(n + 1);
    std::vector<int64_t> h_colIndices(nnz);
    std::vector<double> h_values(batchSize * nnz);  // Values per batch

    createSimpleCSRData(h_rowOffsets, h_colIndices, h_values, batchSize);

    // Copy to GPU
    ASSERT_NO_THROW({
        mat.indicesCpuToGpu(h_rowOffsets.data(), h_colIndices.data());
    });

    // Verify by copying back
    std::vector<int64_t> h_rowOffsets_out(n + 1);
    std::vector<int64_t> h_colIndices_out(nnz);

    mat.indicesGpuToCpu(h_rowOffsets_out.data(), h_colIndices_out.data());

    // Need to synchronize before checking results
    cudaDeviceSynchronize();

    EXPECT_EQ(h_rowOffsets, h_rowOffsets_out);
    EXPECT_EQ(h_colIndices, h_colIndices_out);
}

// Test 4: Values CPU to GPU transfer
TEST_F(BatchedCSRMatrixTest, ValuesCpuToGpu) {
    CSR mat(n, m, nnz, batchSize);

    // Create host data
    std::vector<double> h_values(batchSize * nnz);
    std::iota(h_values.begin(), h_values.end(), 1.0);  // Fill with 1.0, 2.0, 3.0, ...

    // Copy to GPU
    ASSERT_NO_THROW({
        mat.valuesCpuToGpu(h_values.data());
    });

    // Verify by copying back
    std::vector<double> h_values_out(batchSize * nnz);
    mat.valuesGpuToCpu(h_values_out.data());
    
    cudaDeviceSynchronize();
    
    for (size_t i = 0; i < h_values.size(); i++) {
        EXPECT_DOUBLE_EQ(h_values[i], h_values_out[i]);
    }
}

// Test 5: Full roundtrip test
TEST_F(BatchedCSRMatrixTest, FullRoundtrip) {
    CSR mat(n, m, nnz, batchSize);

    // Create host data - indices shared, values per batch
    std::vector<int64_t> h_rowOffsets(n + 1);
    std::vector<int64_t> h_colIndices(nnz);
    std::vector<double> h_values(batchSize * nnz);

    createSimpleCSRData(h_rowOffsets, h_colIndices, h_values, batchSize);

    // Upload
    mat.indicesCpuToGpu(h_rowOffsets.data(), h_colIndices.data());
    mat.valuesCpuToGpu(h_values.data());

    // Download
    std::vector<int64_t> h_rowOffsets_out(n + 1);
    std::vector<int64_t> h_colIndices_out(nnz);
    std::vector<double> h_values_out(batchSize * nnz);

    mat.indicesGpuToCpu(h_rowOffsets_out.data(), h_colIndices_out.data());
    mat.valuesGpuToCpu(h_values_out.data());

    cudaDeviceSynchronize();

    // Verify
    EXPECT_EQ(h_rowOffsets, h_rowOffsets_out);
    EXPECT_EQ(h_colIndices, h_colIndices_out);

    for (size_t i = 0; i < h_values.size(); i++) {
        EXPECT_DOUBLE_EQ(h_values[i], h_values_out[i]);
    }
}

// Test 6: Stream support
TEST_F(BatchedCSRMatrixTest, StreamSupport) {
    CSR mat(n, m, nnz, batchSize);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Create host data - indices shared, values per batch
    std::vector<int64_t> h_rowOffsets(n + 1);
    std::vector<int64_t> h_colIndices(nnz);
    std::vector<double> h_values(batchSize * nnz);

    createSimpleCSRData(h_rowOffsets, h_colIndices, h_values, batchSize);

    // Upload with stream
    ASSERT_NO_THROW({
        mat.indicesCpuToGpu(h_rowOffsets.data(), h_colIndices.data(), stream);
        mat.valuesCpuToGpu(h_values.data(), stream);
    });

    // Synchronize stream
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Download with stream
    std::vector<int64_t> h_rowOffsets_out(n + 1);
    std::vector<int64_t> h_colIndices_out(nnz);
    std::vector<double> h_values_out(batchSize * nnz);

    mat.indicesGpuToCpu(h_rowOffsets_out.data(), h_colIndices_out.data(), stream);
    mat.valuesGpuToCpu(h_values_out.data(), stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Verify
    EXPECT_EQ(h_rowOffsets, h_rowOffsets_out);
    EXPECT_EQ(h_colIndices, h_colIndices_out);
    for (size_t i = 0; i < h_values.size(); i++) {
        EXPECT_DOUBLE_EQ(h_values[i], h_values_out[i]);
    }

    cudaStreamDestroy(stream);
}

// Test 7: Large batch size
TEST_F(BatchedCSRMatrixTest, LargeBatchSize) {
    int largeBatchSize = 128;
    CSR mat(n, m, nnz, largeBatchSize);

    std::vector<int64_t> h_rowOffsets(n + 1);  // Shared across batches
    std::vector<int64_t> h_colIndices(nnz);    // Shared across batches
    std::vector<double> h_values(largeBatchSize * nnz);  // Per batch

    createSimpleCSRData(h_rowOffsets, h_colIndices, h_values, largeBatchSize);

    ASSERT_NO_THROW({
        mat.indicesCpuToGpu(h_rowOffsets.data(), h_colIndices.data());
        mat.valuesCpuToGpu(h_values.data());
    });

    std::vector<double> h_values_out(largeBatchSize * nnz);
    mat.valuesGpuToCpu(h_values_out.data());
    cudaDeviceSynchronize();

    for (size_t i = 0; i < h_values.size(); i++) {
        EXPECT_DOUBLE_EQ(h_values[i], h_values_out[i]);
    }
}

// Test 8: Zero matrix (edge case)
TEST_F(BatchedCSRMatrixTest, ZeroMatrix) {
    int zeroNnz = 0;
    CSR mat(n, m, zeroNnz, batchSize);

    std::vector<int64_t> h_rowOffsets(n + 1, 0);  // Shared across batches

    ASSERT_NO_THROW({
        mat.indicesCpuToGpu(h_rowOffsets.data(), nullptr);
    });
}

// Test 9: Multiple updates to values
TEST_F(BatchedCSRMatrixTest, MultipleValueUpdates) {
    CSR mat(n, m, nnz, batchSize);

    std::vector<int64_t> h_rowOffsets(n + 1);  // Shared across batches
    std::vector<int64_t> h_colIndices(nnz);    // Shared across batches
    std::vector<double> h_values(batchSize * nnz);  // Per batch

    createSimpleCSRData(h_rowOffsets, h_colIndices, h_values, batchSize);

    // Upload structure once
    mat.indicesCpuToGpu(h_rowOffsets.data(), h_colIndices.data());

    // Update values multiple times
    for (int iter = 0; iter < 5; iter++) {
        // Modify values
        for (auto& val : h_values) {
            val *= 2.0;
        }

        mat.valuesCpuToGpu(h_values.data());

        std::vector<double> h_values_out(batchSize * nnz);
        mat.valuesGpuToCpu(h_values_out.data());
        cudaDeviceSynchronize();

        for (size_t i = 0; i < h_values.size(); i++) {
            EXPECT_DOUBLE_EQ(h_values[i], h_values_out[i]);
        }
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}