#include "cuda_runtime_api.h"

__global__ void MatmulPracticeKernel(float* M_device, float* N_device, float* P_device, int width){
    int py = threadIdx.y;
    int px = threadIdx.x;

    float pValue = 0;
    for (int i = 0; i < width; i ++){
        float M_element = M_device[py * width + i];
        float N_element = N_device[i * width + px];
        // pValue += M_device[py * width + i] * N_device[i * width + px];
        pValue += M_element * N_element;
    }
    P_device[py * width + py] = pValue;
}

void MatmulPracticeOnDevice(float* M_host, float* N_host, float* P_host, int width){
    float* M_device;
    float* N_device;
    float* P_device;

    int size = width * width * sizeof(float);
    cudaMalloc((void**)&M_device, size);
    cudaMalloc((void**)&N_device, size);
    cudaMalloc((void**)&P_device, size);

    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1);
    dim3 blockDim(width, width);
    MatmulPracticeKernel <<<gridDim, blockDim>>> (M_device, N_device, P_device, width);

    cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost);

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
