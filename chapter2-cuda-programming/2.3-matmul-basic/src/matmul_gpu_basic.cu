#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"

/* matmul的函数实现*/
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width){
    /* 
        我们设定每一个thread负责P中的一个坐标的matmul
        所以一共有width * width个thread并行处理P的计算
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;

    /* 对于每一个P的元素，我们只需要循环遍历width次M和N中的元素就可以了*/
    for (int k = 0; k < width; k ++){
        float M_element = M_device[y * width + k];
        float N_element = N_device[k * width + x];
        P_element += M_element * N_element;
    }

    P_device[y * width + x] = P_element;
}

/*
    CUDA中使用block对矩阵中某一片区域进行集中计算。这个类似于loop中的tile
    感兴趣的同学可以试着改一下blockSize，也就是tileSize，看看速度会发生什么样子的变化
    当blockSize达到一个数量的时候，这个程序会出错。下一个案例中我们会分析
*/
void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize){
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;

    cudaMalloc(&M_device, size);
    cudaMalloc(&N_device, size);

    /* 分配M, N拷贝到GPU上*/
    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice);

    /* 分配P在GPU上的空间*/
    float *P_device;
    cudaMalloc(&P_device, size);

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：将一个矩阵切分成多个blockSize * blockSize的大小 */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    MatmulKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);

    /* 将结果从device拷贝回host*/
    cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}

