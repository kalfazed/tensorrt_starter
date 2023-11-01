#include "cuda_runtime_api.h"
#include "stdio.h"
#include <iostream>
#include "trt_preprocess.hpp"

namespace preprocess{

TransInfo    trans;
AffineMatrix affine_matrix;

void warpaffine_init(int srcH, int srcW, int tarH, int tarW){
    trans.src_h = srcH;
    trans.src_w = srcW;
    trans.tar_h = tarH;
    trans.tar_w = tarW;
    affine_matrix.init(trans);
}

__host__ __device__ void affine_transformation(
    float trans_matrix[6], 
    int src_x, int src_y, 
    float* tar_x, float* tar_y)
{
    *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
    *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
}

__global__ void nearest_BGR2RGB_nhwc2nchw_norm_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH,
    float scaled_w, float scaled_h,
    float* d_mean, float* d_std) 
{
    // nearest neighbour -- resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // nearest neighbour -- 计算最近坐标
    int src_y = floor((float)y * scaled_h);
    int src_x = floor((float)x * scaled_w);

    if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH) {
        // nearest neighbour -- 对于越界的部分，不进行计算
    } else {
        // nearest neighbour -- 计算tar中对应坐标的索引
        int tarIdx  = y * tarW + x;
        int tarArea = tarW * tarH;

        // nearest neighbour -- 计算src中最近邻坐标的索引
        int srcIdx = (src_y * srcW + src_x) * 3;

        // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw + norm
        tar[tarIdx + tarArea * 0] = (src[srcIdx + 2] / 255.0f - d_mean[2]) / d_std[2];
        tar[tarIdx + tarArea * 1] = (src[srcIdx + 1] / 255.0f - d_mean[1]) / d_std[1];
        tar[tarIdx + tarArea * 2] = (src[srcIdx + 0] / 255.0f - d_mean[0]) / d_std[0];
    }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_norm_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h,
    float* d_mean, float* d_std) 
{

    // bilinear interpolation -- resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = ((y + 0.5) * scaled_h - 0.5) - src_y1;
        float tw   = ((x + 0.5) * scaled_w - 0.5) - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
        float a1_2 = tw * (1.0 - th);          //左下
        float a2_1 = (1.0 - tw) * th;          //右上
        float a2_2 = tw * th;                  //左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = y * tarW  + x;
        int tarArea   = tarW * tarH;

        // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB + NHWC2NCHW normalization
        // 注意，这里tar和src进行遍历的方式是不一样的
        tar[tarIdx + tarArea * 0] = 
            (round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f - d_mean[2]) / d_std[2];

        tar[tarIdx + tarArea * 1] = 
            (round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f - d_mean[1]) / d_std[1];

        tar[tarIdx + tarArea * 2] = 
            (round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f - d_mean[0]) / d_std[0];

    }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h,
    float* d_mean, float* d_std) 
{
    // resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = (float)y * scaled_h - src_y1;
        float tw   = (float)x * scaled_w - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);  // 右下
        float a1_2 = tw * (1.0 - th);          // 左下
        float a2_1 = (1.0 - tw) * th;          // 右上
        float a2_2 = tw * th;                  // 左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  // 左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  // 右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  // 左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  // 右下

        // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
        y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
        x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = (y * tarW  + x) * 3;
        int tarArea   = tarW * tarH;

        // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw
        tar[tarIdx + tarArea * 0] = 
            (round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f - d_mean[2]) / d_std[2];

        tar[tarIdx + tarArea * 1] = 
            (round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f - d_mean[1]) / d_std[1];

        tar[tarIdx + tarArea * 2] = 
            (round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f - d_mean[0]) / d_std[0];
    }
}


__global__ void nearest_BGR2RGB_nhwc2nchw_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH,
    float scaled_w, float scaled_h)
{
    // nearest neighbour -- resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // nearest neighbour -- 计算最近坐标
    int src_y = floor((float)y * scaled_h);
    int src_x = floor((float)x * scaled_w);

    if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH) {
        // nearest neighbour -- 对于越界的部分，不进行计算
    } else {
        // nearest neighbour -- 计算tar中对应坐标的索引
        int tarIdx  = y * tarW + x;
        int tarArea = tarW * tarH;

        // nearest neighbour -- 计算src中最近邻坐标的索引
        int srcIdx = (src_y * srcW + src_x) * 3;

        // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw + norm
        tar[tarIdx + tarArea * 0] = src[srcIdx + 2] / 255.0f;
        tar[tarIdx + tarArea * 1] = src[srcIdx + 1] / 255.0f;
        tar[tarIdx + tarArea * 2] = src[srcIdx + 0] / 255.0f;
    }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h)
{

    // bilinear interpolation -- resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = ((y + 0.5) * scaled_h - 0.5) - src_y1;
        float tw   = ((x + 0.5) * scaled_w - 0.5) - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
        float a1_2 = tw * (1.0 - th);          //左下
        float a2_1 = (1.0 - tw) * th;          //右上
        float a2_2 = tw * th;                  //左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = y * tarW  + x;
        int tarArea   = tarW * tarH;

        // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB + NHWC2NCHW normalization
        // 注意，这里tar和src进行遍历的方式是不一样的
        tar[tarIdx + tarArea * 0] = 
            round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f;

        tar[tarIdx + tarArea * 1] = 
            round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f;

        tar[tarIdx + tarArea * 2] = 
            round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f;

    }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_shift_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h)
{
    // resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = (float)y * scaled_h - src_y1;
        float tw   = (float)x * scaled_w - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);  // 右下
        float a1_2 = tw * (1.0 - th);          // 左下
        float a2_1 = (1.0 - tw) * th;          // 右上
        float a2_2 = tw * th;                  // 左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  // 左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  // 右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  // 左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  // 右下

        // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
        y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
        x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = y * tarW  + x;
        int tarArea   = tarW * tarH;

        // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw
        tar[tarIdx + tarArea * 0] = 
            round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f;

        tar[tarIdx + tarArea * 1] = 
            round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f;

        tar[tarIdx + tarArea * 2] = 
            round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f;
    }
}

__global__ void warpaffine_BGR2RGB_kernel(
    float* tar, uint8_t* src, 
    TransInfo trans,
    AffineMatrix affine_matrix)
{
    float src_x, src_y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    affine_transformation(affine_matrix.reverse, x + 0.5, y + 0.5, &src_x, &src_y);

    int src_x1 = floor(src_x - 0.5);
    int src_y1 = floor(src_y - 0.5);
    int src_x2 = src_x1 + 1;
    int src_y2 = src_y1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y1 > trans.src_h || src_x1 > trans.src_w) {
    } else {
        float tw   = src_x - src_x1;
        float th   = src_y - src_y1;

        float a1_1 = (1.0 - tw) * (1.0 - th);
        float a1_2 = tw * (1.0 - th);
        float a2_1 = (1.0 - tw) * th;
        float a2_2 = tw * th;

        int srcIdx1_1 = (src_y1 * trans.src_w + src_x1) * 3;
        int srcIdx1_2 = (src_y1 * trans.src_w + src_x2) * 3;
        int srcIdx2_1 = (src_y2 * trans.src_w + src_x1) * 3;
        int srcIdx2_2 = (src_y2 * trans.src_w + src_x2) * 3;

        int tarIdx    = y * trans.tar_w  + x;
        int tarArea   = trans.tar_w * trans.tar_h;

        tar[tarIdx + tarArea * 0] = 
            round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f;

        tar[tarIdx + tarArea * 1] = 
            round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f;

        tar[tarIdx + tarArea * 2] = 
            round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f;
    }
}

void resize_bilinear_gpu(
    float* d_tar, uint8_t* d_src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float* d_mean, float* d_std,
    tactics tac) 
{
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(tarW / 32 + 1, tarH / 32 + 1, 1);
   
    //scaled resize
    float scaled_h = (float)srcH / tarH;
    float scaled_w = (float)srcW / tarW;
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    switch (tac) {
    case tactics::GPU_NEAREST:
        nearest_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dimGrid, dimBlock>>>
                (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean, d_std);
        break;
    case tactics::GPU_NEAREST_CENTER:
        nearest_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dimGrid, dimBlock>>>
                (d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
        break;
    case tactics::GPU_BILINEAR:
        bilinear_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean, d_std);
        break;
    case tactics::GPU_BILINEAR_CENTER:
        bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel 
                <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
        break;
    default:
        LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
        exit(1);
    }
}

void resize_bilinear_gpu(
    float* d_tar, uint8_t* d_src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    tactics tac) 
{
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(tarW / 32 + 1, tarH / 32 + 1, 1);
   
    //scaled resize
    float scaled_h = (float)srcH / tarH;
    float scaled_w = (float)srcW / tarW;
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    switch (tac) {
    case tactics::GPU_NEAREST:
        nearest_BGR2RGB_nhwc2nchw_kernel <<<dimGrid, dimBlock>>>
                (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
        break;
    case tactics::GPU_NEAREST_CENTER:
        nearest_BGR2RGB_nhwc2nchw_kernel <<<dimGrid, dimBlock>>>
                (d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale);
        break;
    case tactics::GPU_BILINEAR:
        bilinear_BGR2RGB_nhwc2nchw_kernel <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
        break;
    case tactics::GPU_BILINEAR_CENTER:
        bilinear_BGR2RGB_nhwc2nchw_shift_kernel <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale);
        break;
    case tactics::GPU_WARP_AFFINE:
        warpaffine_init(srcH, srcW, tarH, tarW);
        warpaffine_BGR2RGB_kernel <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, trans, affine_matrix);
        break;
    default:
        LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
        exit(1);
    }
}

} // namespace preprocess
