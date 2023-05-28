#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "matmul.hpp"
#include "bits/stdc++.h"

#define MAXSIZE 1 << 12
#define TILESIZE 32

void matmul_test(){
    int width = MAXSIZE;
    int tile_width = TILESIZE;
    long int size = width * width;
    float* M_host = (float*)malloc(size * sizeof(float));
    float* N_host = (float*)malloc(size * sizeof(float));
    float* P_host = (float*)malloc(size * sizeof(float));
    for (long int i = 0; i < size; i++){
        M_host[i] = 1;
        N_host[i] = 1;
    }
    auto start = std::chrono::steady_clock::now();
    // MatmulOnHost(M_host, N_host, P_host, width);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elasped_time = end - start;
    // std::cout << "elasped time in host without optimization: " << std::fixed << std::setprecision(8) << elasped_time.count() << std::endl;
    /* 274.432s*/

    start = std::chrono::steady_clock::now();
    MatmulOnDevice(M_host, N_host, P_host, width);
    end = std::chrono::steady_clock::now();
    elasped_time = end - start;
    std::cout << "elasped time in device warmup:               " << std::fixed << std::setprecision(8) << elasped_time.count() << std::endl;

    start = std::chrono::steady_clock::now();
    MatmulOnDevice(M_host, N_host, P_host, width);
    end = std::chrono::steady_clock::now();
    elasped_time = end - start;
    std::cout << "elasped time in device without optimization: " << std::fixed << std::setprecision(8) << elasped_time.count() << std::endl;
    /* gridDim: (1, 1) */
    /* blockDim: (MAXSIZE, MAXSIZE) */
    /* 0.0735947s*/


    start = std::chrono::steady_clock::now();
    MatmulTileOnDevice(M_host, N_host, P_host, width, tile_width);
    end = std::chrono::steady_clock::now();
    elasped_time = end - start;
    std::cout << "elasped time in device using tiling:         " << std::fixed << std::setprecision(8) << elasped_time.count() << std::endl;
    /* gridDim: (MAXSIZE / TILESIZE, MAXSIZE / TILESIZE) */
    /* blockDim: (TILESIZE, TILESIZE) */
    /* TILESIZE = 32: 0.0452454s*/

    start = std::chrono::steady_clock::now();
    MatmulSharedOnDevice(M_host, N_host, P_host, width);
    end = std::chrono::steady_clock::now();
    elasped_time = end - start;
    std::cout << "elasped time in device using shared memory: " << std::fixed << std::setprecision(8) << elasped_time.count() << std::endl;
    /* gridDim: (MAXSIZE / TILESIZE, MAXSIZE / TILESIZE) */
    /* blockDim: (TILESIZE, TILESIZE) */
    /* TILESIZE = 32: 0.0452454s*/

    start = std::chrono::steady_clock::now();
    // MatmulPracticeOnDevice(M_host, N_host, P_host, width);
    MatmulOnDevice(M_host, N_host, P_host, width);
    end = std::chrono::steady_clock::now();
    elasped_time = end - start;
    std::cout << "elasped time in device with pratice:        " << std::fixed << std::setprecision(8) << elasped_time.count() << std::endl;

    free(M_host); M_host = NULL;
    free(N_host); N_host = NULL;
    free(P_host); P_host = NULL;
}


int main(int argc, char const *argv[])
{
    matmul_test();
    return 0;
}
