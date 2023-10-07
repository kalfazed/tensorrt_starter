#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "utils.hpp"

void cublasSgemmOnDevice(float *M_host, float *N_host, float* P_host, int width){
    /* 为cublas创建handle来控制cublas的上下文 */
    /* 这里需要注意一点的是，跟使用cuda library一样，第一次使用cublas相关的library需要做一些初始化，所以需要warmup */
    cublasHandle_t handle;
    cublasCreate(&handle);

    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    float *P_device;

    /* 分配P在GPU上的空间*/
    CUDA_CHECK(cudaMalloc(&M_device, size));
    CUDA_CHECK(cudaMalloc(&N_device, size));
    CUDA_CHECK(cudaMalloc(&P_device, size));

    /* 把M的数据从CPU拷贝到GPU上 */
    cublasSetVector(
            width * width, sizeof(float), 
            M_host, 1, 
            M_device, 1
    );

    /* 把N的数据从CPU拷贝到GPU上 */
    cublasSetVector(
            width * width, sizeof(float), 
            N_host, 1, 
            N_device, 1
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    /* 
     * 开始做gemm计算，默认公式是P = alpha * M * N + beta * P
     * 使用cublas的重点是需要理解每一个参数所代表的意思是什么
     * 这里也需要注意一点，就是cublas做gemm的时候的rotation
     * cublas做gemm计算的时候是column major，主要是起初为了兼容fortran语言，
     * 所以，我们如果按照正常方式传入参数做P = M * N的话cublas做的gemm实际上是P^T = M^T * N^T。
     * 因此为了得到正确的输出，我们可以做P^T = N^T * M^T，这样就等同于P = M * N了 
    */

    float alpha = 1;
    float beta  = 0;
    cublasSgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, //这个参数是用来决定是否进行rotation
            width, width, width,
            &alpha, 
            N_device, width, 
            M_device, width, 
            &beta, 
            P_device, width);

    CUDA_CHECK(cudaDeviceSynchronize());

    /* 把P的数据从GPU拷贝到CPU上 */
    cublasGetVector(
            width * width, sizeof(float),
            P_device, 1,
            P_host, 1);
    /* 注意，由于cublass是列优先存储，所以这里计算得到的P_host是转置的 */

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);

    /* 释放管理cublass的handler */
    cublasDestroy(handle);
}

