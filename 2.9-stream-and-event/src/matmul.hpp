#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include <cuda_runtime.h>

__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width);
__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize);

void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize);

void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize, bool staticMem);
void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize, bool staticMem, cudaStream_t stream);

void MatmulSingleStream(float* M_input, float* N_input, float* P_output, int width, int blockSize, bool staticMem, int count);
void MatmulMultiStream(float* M_input, float* N_input, float* P_output, int width, int blockSize, bool staticMem, int count);
void MatmulMultiStream2(float* M_input, float* N_input, float* P_output, int width, int blockSize, bool staticMem, int count);

extern void MatmulOnHost(float *M_host, float *N_host, float* P_host, int width);

#endif //__MATMUL_HPP__
