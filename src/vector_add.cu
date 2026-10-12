#include <cuda_runtime.h>
#include <array>
#include <cstdio>

constexpr int kElementCount = 5;

__global__ void vectorAdd(const int* a, const int* b, int* result, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        result[index] = a[index] * b[index];
    }
}

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

int main() {
    const std::array<int, kElementCount> hostA{1, 2, 3, 4, 5};
    const std::array<int, kElementCount> hostB{10, 20, 30, 40, 50};
    std::array<int, kElementCount> hostResult{};

    const std::size_t bytes = sizeof(int) * kElementCount;
    int* deviceA = nullptr;
    int* deviceB = nullptr;
    int* deviceResult = nullptr;

    if (!checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(deviceA)") ||
        !checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(deviceB)") ||
        !checkCuda(cudaMalloc(&deviceResult, bytes), "cudaMalloc(deviceResult)")) {
        return 1;
    }

    if (!checkCuda(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(hostA to device)") ||
        !checkCuda(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(hostB to device)")) {
        return 1;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks = (kElementCount + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(deviceA, deviceB, deviceResult, kElementCount);

    if (!checkCuda(cudaGetLastError(), "vectorAdd kernel launch") ||
        !checkCuda(cudaDeviceSynchronize(), "vectorAdd kernel execution") ||
        !checkCuda(cudaMemcpy(hostResult.data(), deviceResult, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(device to host)")) {
        return 1;
    }

    for (int i = 0; i < kElementCount; ++i) {
        if (hostResult[i] != hostA[i] * hostB[i]) {
            std::fprintf(stderr, "Incorrect result at index %d\n", i);
            return 1;
        }
        std::printf("%d * %d = %d\n", hostA[i], hostB[i], hostResult[i]);
    }

    if (!checkCuda(cudaFree(deviceA), "cudaFree(deviceA)") ||
        !checkCuda(cudaFree(deviceB), "cudaFree(deviceB)") ||
        !checkCuda(cudaFree(deviceResult), "cudaFree(deviceResult)")) {
        return 1;
    }

    return 0;
}
