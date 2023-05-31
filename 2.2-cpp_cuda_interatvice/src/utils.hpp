#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <cuda_runtime.h>
#include <system_error>

#define CUDACHECK(call) {                                                  \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                           \
    }                                                                      \
}

#endif //__LOGER__HPP__
