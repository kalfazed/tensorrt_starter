#include <stdio.h>
#include <cuda_runtime.h>
#include <string>

#include "utils.hpp"

int main(){
    int count;
    int index = 0;
    cudaGetDeviceCount(&count);
    while (index < count) {
        cudaSetDevice(index);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, index);
        LOG("%-40s",             "*********************Architecture related**********************");
        LOG("%-40s%d%s",         "Device id: ",                   index, "");
        LOG("%-40s%s%s",         "Device name: ",                 prop.name, "");
        LOG("%-40s%.1f%s",       "Device compute capability: ",   prop.major + (float)prop.minor / 10, "");
        LOG("%-40s%.2f%s",       "GPU global meory size: ",       (float)prop.totalGlobalMem / (1<<30), "GB");
        LOG("%-40s%.2f%s",       "L2 cache size: ",               (float)prop.l2CacheSize / (1<<20), "MB");
        LOG("%-40s%.2f%s",       "Shared memory per block: ",     (float)prop.sharedMemPerBlock / (1<<10), "KB");
        LOG("%-40s%.2f%s",       "Shared memory per SM: ",        (float)prop.sharedMemPerMultiprocessor / (1<<10), "KB");
        LOG("%-40s%.2f%s",       "Device clock rate: ",           prop.clockRate*1E-6, "GHz");
        LOG("%-40s%.2f%s",       "Device memory clock rate: ",    prop.memoryClockRate*1E-6, "Ghz");
        LOG("%-40s%d%s",         "Number of SM: ",                prop.multiProcessorCount, "");
        LOG("%-40s%d%s",         "Warp size: ",                   prop.warpSize, "");

        LOG("%-40s",             "*********************Parameter related************************");
        LOG("%-40s%d%s",         "Max block numbers: ",           prop.maxBlocksPerMultiProcessor, "");
        LOG("%-40s%d%s",         "Max threads per block: ",       prop.maxThreadsPerBlock, "");
        LOG("%-40s%d:%d:%d%s",   "Max block dimension size:",     prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], "");
        LOG("%-40s%d:%d:%d%s",   "Max grid dimension size: ",     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], "");
        index ++;
        printf("\n");
    }
    return 0;
}
