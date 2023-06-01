#include "cuda_runtime_api.h"
#include "utils.hpp"

/* 
    使用Tiling优化的matmul的函数实现
*/
__global__ void MatmulTileKernel(float *M_device, float *N_device, float *P_device, int width){
    /* 
        对于x, y多个根据block和thread进行索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;

    /* 对于每一个P的元素，我们只需要循环遍历width次M和N中的元素就可以了*/
    for (int k = 0; k < width; k ++){
        P_element += M_device[y * width + k] * N_device[k * width + x];
    }

    P_device[y * width + x] = P_element;
}

/*
    使用Tiling技术
    一个tile处理的就是block, 将一个矩阵分为多个小的tile，这些tile之间的执行独立，并且可以并行
*/
void MatmulTileOnDevice(float *M_host, float *N_host, float* P_host, int width, int tile_width){
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc(&M_device, size));
    CUDA_CHECK(cudaMalloc(&N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void**)&P_device, size));

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimBlock(tile_width, tile_width);
    dim3 dimGrid(width / tile_width, width / tile_width);
    MatmulTileKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, size);

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK(); 

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}

