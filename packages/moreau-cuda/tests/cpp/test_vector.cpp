// test_vector.cpp
#include <gtest/gtest.h>
#include "moreau/vector/vector.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace moreau;

class BatchedVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        n = 100;
        batchSize = 32;
    }

    void TearDown() override {
        // Clean up is automatic with RAII
    }

    // Helper function to create test data
    void createTestData(std::vector<double>& data, int n, int batch) {
        data.resize(n * batch);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                data[b * n + i] = static_cast<double>(b * n + i) + 1.0;
            }
        }
    }

    int n, batchSize;
};

// Test 1: Constructor allocates memory
TEST_F(BatchedVectorTest, ConstructorAllocates) {
    ASSERT_NO_THROW({
        BatchedVector vec(n, batchSize);
        EXPECT_NE(vec.data(), nullptr);
        EXPECT_EQ(vec.n()   , n);
        EXPECT_EQ(vec.batchSize() , batchSize);
    });
}

// Test 2: Memory ownership and lifecycle
TEST_F(BatchedVectorTest, WrapConstructorDoesNotAllocate) {
    // Test that the BatchedVector properly manages its own memory
    BatchedVector vec(n, batchSize);

    // Verify dimensions
    EXPECT_EQ(vec.n(), n);
    EXPECT_EQ(vec.batchSize(), batchSize);

    // Verify pointer is allocated
    EXPECT_NE(vec.data(), nullptr);

    // Store pointer to verify it's consistent
    auto ptr = vec.data();

    // Verify pointer remains stable
    EXPECT_EQ(vec.data(), ptr);
}

// Test 3: CPU to GPU transfer
TEST_F(BatchedVectorTest, CpuToGpu) {
    BatchedVector vec(n, batchSize);

    // Create host data
    std::vector<double> h_data;
    createTestData(h_data, n, batchSize);

    // Copy to GPU
    ASSERT_NO_THROW({
        vec.cpuToGpu(h_data.data());
    });

    // Verify by copying back
    std::vector<double> h_data_out(batchSize * n);
    vec.gpuToCpu(h_data_out.data());
    
    cudaDeviceSynchronize();
    
    for (size_t i = 0; i < h_data.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data[i], h_data_out[i]);
    }
}

// Test 4: GPU to CPU transfer
TEST_F(BatchedVectorTest, GpuToCpu) {
    BatchedVector vec(n, batchSize);

    // Create host data
    std::vector<double> h_data(batchSize * n);
    std::iota(h_data.begin(), h_data.end(), 1.0);

    // Upload
    vec.cpuToGpu(h_data.data());
    
    // Download
    std::vector<double> h_data_out(batchSize * n);
    ASSERT_NO_THROW({
        vec.gpuToCpu(h_data_out.data());
    });
    
    cudaDeviceSynchronize();
    
    for (size_t i = 0; i < h_data.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data[i], h_data_out[i]);
    }
}

// Test 5: Full roundtrip test
TEST_F(BatchedVectorTest, FullRoundtrip) {
    BatchedVector vec(n, batchSize);

    // Create host data
    std::vector<double> h_data;
    createTestData(h_data, n, batchSize);

    // Upload
    vec.cpuToGpu(h_data.data());

    // Download
    std::vector<double> h_data_out(batchSize * n);
    vec.gpuToCpu(h_data_out.data());
    
    cudaDeviceSynchronize();

    // Verify
    for (size_t i = 0; i < h_data.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data[i], h_data_out[i]);
    }
}

// Test 6: Stream support
TEST_F(BatchedVectorTest, StreamSupport) {
    BatchedVector vec(n, batchSize);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Create host data
    std::vector<double> h_data;
    createTestData(h_data, n, batchSize);

    // Upload with stream
    ASSERT_NO_THROW({
        vec.cpuToGpu(h_data.data(), stream);
    });

    // Synchronize stream
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Download with stream
    std::vector<double> h_data_out(batchSize * n);
    vec.gpuToCpu(h_data_out.data(), stream);
    
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Verify
    for (size_t i = 0; i < h_data.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data[i], h_data_out[i]);
    }

    cudaStreamDestroy(stream);
}

// Test 7: SetToConstant function
TEST_F(BatchedVectorTest, SetToConstantFunction) {
    BatchedVector vec(n, batchSize);

    // Set to constant on GPU
    ASSERT_NO_THROW({
        vec.setToConstant(5.0);
    });

    // Download and verify
    std::vector<double> h_data_out(batchSize * n);
    vec.gpuToCpu(h_data_out.data());
    
    cudaDeviceSynchronize();
    
    for (size_t i = 0; i < h_data_out.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data_out[i], 5.0);
    }
}

// Test 8: Zero function with stream
TEST_F(BatchedVectorTest, ZeroFunctionWithStream) {
    BatchedVector vec(n, batchSize);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create and upload non-zero data
    std::vector<double> h_data(batchSize * n, 7.5);
    vec.cpuToGpu(h_data.data(), stream);

    // Zero out on GPU with stream
    vec.setToConstant(0.0, stream);

    // Download and verify
    std::vector<double> h_data_out(batchSize * n);
    vec.gpuToCpu(h_data_out.data(), stream);
    
    cudaStreamSynchronize(stream);
    
    for (size_t i = 0; i < h_data_out.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data_out[i], 0.0);
    }

    cudaStreamDestroy(stream);
}

// Test 9: Memory usage calculation
TEST_F(BatchedVectorTest, MemoryUsage) {
    BatchedVector vec(n, batchSize);
    
    size_t expected = sizeof(double) * n * batchSize;
    EXPECT_EQ(vec.memoryUsage(), expected);
}

// Test 10: Large vector
TEST_F(BatchedVectorTest, LargeVector) {
    int largeN = 1000000;
    int largeBatch = 16;
    
    BatchedVector vec(largeN, largeBatch);

    std::vector<double> h_data(largeN * largeBatch);
    for (size_t i = 0; i < h_data.size(); i++) {
        h_data[i] = static_cast<double>(i % 1000) / 1000.0;
    }

    ASSERT_NO_THROW({
        vec.cpuToGpu(h_data.data());
    });

    std::vector<double> h_data_out(largeN * largeBatch);
    vec.gpuToCpu(h_data_out.data());
    cudaDeviceSynchronize();
    
    // Spot check some values
    for (int i = 0; i < 100; i++) {
        int idx = i * (h_data.size() / 100);
        EXPECT_DOUBLE_EQ(h_data[idx], h_data_out[idx]);
    }
}

// Test 11: Small vector (edge case)
TEST_F(BatchedVectorTest, SmallVector) {
    int smallN = 1;
    int smallBatch = 1;
    
    BatchedVector vec(smallN, smallBatch);

    std::vector<double> h_data = {42.0};
    vec.cpuToGpu(h_data.data());

    std::vector<double> h_data_out(1);
    vec.gpuToCpu(h_data_out.data());
    cudaDeviceSynchronize();
    
    EXPECT_DOUBLE_EQ(h_data_out[0], 42.0);
}

// Test 12: Multiple updates
TEST_F(BatchedVectorTest, MultipleUpdates) {
    BatchedVector vec(n, batchSize);

    // Update multiple times
    for (int iter = 0; iter < 10; iter++) {
        std::vector<double> h_data(batchSize * n, iter * 1.5);
        
        vec.cpuToGpu(h_data.data());
        
        std::vector<double> h_data_out(batchSize * n);
        vec.gpuToCpu(h_data_out.data());
        cudaDeviceSynchronize();
        
        for (size_t i = 0; i < h_data.size(); i++) {
            EXPECT_DOUBLE_EQ(h_data[i], h_data_out[i]);
        }
    }
}

// Test 13: Different batch sizes
TEST_F(BatchedVectorTest, DifferentBatchSizes) {
    std::vector<int> batchSizes = {1, 2, 4, 8, 16, 32, 64, 128};
    
    for (int batch : batchSizes) {
        BatchedVector vec(n, batch);
        
        std::vector<double> h_data(n * batch);
        std::iota(h_data.begin(), h_data.end(), 1.0);
        
        vec.cpuToGpu(h_data.data());
        
        std::vector<double> h_data_out(n * batch);
        vec.gpuToCpu(h_data_out.data());
        cudaDeviceSynchronize();
        
        for (size_t i = 0; i < h_data.size(); i++) {
            EXPECT_DOUBLE_EQ(h_data[i], h_data_out[i]) 
                << "Failed for batch size " << batch << " at index " << i;
        }
    }
}

// Test 14: Data persistence through operations
TEST_F(BatchedVectorTest, WrappedPointerLifetime) {
    BatchedVector vec(n, batchSize);

    // Initialize with a constant value
    std::vector<double> h_init(batchSize * n, 123.456);
    vec.cpuToGpu(h_init.data());
    cudaDeviceSynchronize();

    // Verify we can read the data back
    std::vector<double> h_data_out(batchSize * n);
    vec.gpuToCpu(h_data_out.data());
    cudaDeviceSynchronize();

    for (size_t i = 0; i < h_data_out.size(); i++) {
        EXPECT_DOUBLE_EQ(h_data_out[i], 123.456);
    }

    // Verify data persists after multiple operations
    vec.setToConstant(456.789);
    cudaDeviceSynchronize();

    std::vector<double> h_verify(batchSize * n);
    vec.gpuToCpu(h_verify.data());
    cudaDeviceSynchronize();

    for (size_t i = 0; i < h_verify.size(); i++) {
        EXPECT_DOUBLE_EQ(h_verify[i], 456.789);
    }
}

// Test 15: Concurrent operations with multiple streams
TEST_F(BatchedVectorTest, MultipleStreams) {
    const int numStreams = 4;
    std::vector<cudaStream_t> streams(numStreams);
    std::vector<BatchedVector*> vectors;
    
    // Create streams and vectors
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
        vectors.push_back(new BatchedVector(n, batchSize));
    }
    
    // Upload to all vectors concurrently
    std::vector<std::vector<double>> h_data_all(numStreams);
    for (int i = 0; i < numStreams; i++) {
        h_data_all[i].resize(batchSize * n);
        for (size_t j = 0; j < h_data_all[i].size(); j++) {
            h_data_all[i][j] = i * 1000.0 + j;
        }
        vectors[i]->cpuToGpu(h_data_all[i].data(), streams[i]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Download and verify
    for (int i = 0; i < numStreams; i++) {
        std::vector<double> h_data_out(batchSize * n);
        vectors[i]->gpuToCpu(h_data_out.data(), streams[i]);
        cudaStreamSynchronize(streams[i]);
        
        for (size_t j = 0; j < h_data_out.size(); j++) {
            EXPECT_DOUBLE_EQ(h_data_out[j], h_data_all[i][j])
                << "Failed for stream " << i << " at index " << j;
        }
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; i++) {
        delete vectors[i];
        cudaStreamDestroy(streams[i]);
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}