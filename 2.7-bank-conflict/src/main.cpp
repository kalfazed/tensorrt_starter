#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.hpp"
#include "timer.hpp"
#include "matmul.hpp"


int seed;
int main() {
    cudaSharedMemConfig smemConfig;
    CUDA_CHECK(cudaDeviceGetSharedMemConfig(&smemConfig));
    if (smemConfig == 0) {
        LOG("Bank Size is default");
    }else if(smemConfig == 1) {
        LOG("Bank Size is 4 bytes (32 bits)");
    }else {
        LOG("Bank Size is 8 bytes (64 bits)");
    }
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    /*
     * 这里使用更大的bank size可以减少bank conflict发生的几率，从而提升带宽
    */

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    // cudaFuncCachePreferNone;      默认的分配原则
    // cudaFuncCachePreferShared;    Shared memory: 48KB, L1 cache: 16KB
    // cudaFuncCachePreferL1;        Shared memory: 16KB, L1 cache: 48KB
    // cudaFuncCachePreferEqual;     Shared memory: 32KB, L1 cache: 32KB
    return 0;
}
