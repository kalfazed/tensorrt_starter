#ifndef __PRINT_INDEX_HPP
#define __PRINT_INDEX_HPP

#include <cuda_runtime.h>
void print_idx_device(dim3 grid, dim3 block);
void print_dim_device(dim3 grid, dim3 block);
void print_thread_idx_per_block_device(dim3 grid, dim3 block);
void print_thread_idx_device(dim3 grid, dim3 block);

#endif //__PRINT_INDEX_HPP
