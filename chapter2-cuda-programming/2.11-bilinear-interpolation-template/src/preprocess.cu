#include "cuda_runtime_api.h"
#include "stdio.h"
#include <iostream>

#include "utils.hpp"

template <typename T>
__global__ void resize_bilinear_BGR2RGB_shift_kernel(
    T* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h) 
{

    // resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = round((float)y * scaled_h);
    int src_x1 = round((float)x * scaled_w);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y1 > srcH || src_x1 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = (float)y * scaled_h - src_y1;
        float tw   = (float)x * scaled_w - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);
        float a1_2 = (1.0 - tw) * th;
        float a2_1 = tw * (1.0 - th);
        float a2_2 = tw * th;

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;

        // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
        y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
        x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = (y * tarW  + x) * 3;

        // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB
        tar[tarIdx + 0] = round(
                          a1_1 * src[srcIdx1_1 + 2] + 
                          a1_2 * src[srcIdx1_2 + 2] +
                          a2_1 * src[srcIdx2_1 + 2] +
                          a2_2 * src[srcIdx2_2 + 2]);

        tar[tarIdx + 1] = round(
                          a1_1 * src[srcIdx1_1 + 1] + 
                          a1_2 * src[srcIdx1_2 + 1] +
                          a2_1 * src[srcIdx2_1 + 1] +
                          a2_2 * src[srcIdx2_2 + 1]);

        tar[tarIdx + 2] = round(
                          a1_1 * src[srcIdx1_1 + 0] + 
                          a1_2 * src[srcIdx1_2 + 0] +
                          a2_1 * src[srcIdx2_1 + 0] +
                          a2_2 * src[srcIdx2_2 + 0]);
    }
}

template <typename T>
void resize_bilinear_gpu(
    T* d_tar, uint8_t* d_src, 
    int tarW, int tarH, 
    int srcW, int srcH)
{
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(tarW / 16 + 1, tarH / 16 + 1, 1);
    
    //scaled resize
    float scaled_h = (float)srcH / tarH;
    float scaled_w = (float)srcW / tarW;
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    scaled_h = scale;
    scaled_w = scale;
    
    resize_bilinear_BGR2RGB_shift_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
}

template __global__ void resize_bilinear_BGR2RGB_shift_kernel<uint8_t>(uint8_t* tar, uint8_t* src, int tarW, int tarH, int srcW, int srcH, float scaled_w, float scaled_h);
template __global__ void resize_bilinear_BGR2RGB_shift_kernel<float>(float* tar, uint8_t* src, int tarW, int tarH, int srcW, int srcH, float scaled_w, float scaled_h);
template void resize_bilinear_gpu<uint8_t>(uint8_t* d_tar, uint8_t* d_src, int tarW, int tarH, int srcW, int srcH);
template void resize_bilinear_gpu<float>(float* d_tar, uint8_t* d_src, int tarW, int tarH, int srcW, int srcH);
