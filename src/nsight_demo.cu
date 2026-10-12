#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    float *da, *db, *dc;

    check(cudaMalloc(&da, bytes));
    check(cudaMalloc(&db, bytes));
    check(cudaMalloc(&dc, bytes));

    // Copy the inputs from CPU memory to GPU memory.
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    // Repeat the kernel so its launches are easy to find in the timeline.
    for (int k = 0; k < 100; ++k) {
        vector_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    // Copy the output back and verify every element.
    check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    for (float value : c) {
        if (value != 3.0f) {
            std::fprintf(stderr, "FAIL: incorrect result\n");
            return 1;
        }
    }

    check(cudaFree(da));
    check(cudaFree(db));
    check(cudaFree(dc));
    std::printf("PASS: all results are 3.0\n");
    return 0;
}
