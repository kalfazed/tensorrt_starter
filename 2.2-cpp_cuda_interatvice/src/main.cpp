#include <stdio.h>
#include <cuda_runtime.h>
#include "print_index.hpp"
#include "utils.hpp"

void print_one_dim(int inputSize, int blockSize){
    int gridSize = inputSize / blockSize;

    dim3 block(blockSize);
    dim3 grid(gridSize);

    // print_idx_device(block, grid);
    // print_dim_device(block, grid);
    // print_thread_idx_per_block_device(block, grid);
    print_thread_idx_device(block, grid);

    CUDACHECK(cudaDeviceSynchronize());
}

void print_two_dim(int inputSize, int blockSize){
    int gridSize = inputSize / blockSize;

    dim3 block(blockSize, blockSize);
    dim3 grid(gridSize, gridSize);

    // print_idx_device(block, grid);
    // print_dim_device(block, grid);
    // print_thread_idx_per_block_device(block, grid);
    print_thread_idx_device(block, grid);

    CUDACHECK(cudaDeviceSynchronize());
}


int main(){
    int inputSize;
    int blockSize;

    /* one-dimention test */
    // inputSize = 32;
    // blockSize = 4;
    // print_one_dim(inputSize, blockSize);
        
    /* one-dimention test */
    inputSize = 8;
    blockSize = 4;
    print_two_dim(inputSize, blockSize);
    return 0;
}
