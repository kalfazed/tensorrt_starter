#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width);
void MatmulTileOnDevice(float *M_host, float *N_host, float* P_host, int width, int tile_width);
void MatmulSharedOnDevice(float *M_host, float *N_host, float* P_host, int width);
void MatmulPracticeOnDevice(float* M_host, float* N_host, float* P_host, int width);
extern void MatmulOnHost(float *M_host, float *N_host, float* P_host, int width);

#endif //__MATMUL_HPP__