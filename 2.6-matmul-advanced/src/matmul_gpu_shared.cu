#include "cuda_runtime_api.h"

#define TILE_WIDTH 32
/* 
    使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
*/
__global__ void MatmulSharedKernel(float *M_device, float *N_device, float *P_device, int width){
    __shared__ float M_deviceShared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_deviceShared[TILE_WIDTH][TILE_WIDTH];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float P_element = 0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / TILE_WIDTH; m ++) {
        M_deviceShared[ty][tx] = M_device[y * width + (m * TILE_WIDTH + tx)];
        N_deviceShared[ty][tx] = N_device[(m * TILE_WIDTH + ty)* width + tx];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k ++) {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }
        __syncthreads();
    }

    P_device[y * width + x] = P_element;
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width){
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    cudaMalloc((void**)&M_device, size);
    cudaMalloc((void**)&N_device, size);

    /* 分配M, N拷贝到GPU上*/
    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice);

    /* 分配P在GPU上的空间*/
    float *P_device;
    cudaMalloc((void**)&P_device, size);

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    MatmulSharedKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, size);

    /* 将结果从device拷贝回host*/
    cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost);

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}