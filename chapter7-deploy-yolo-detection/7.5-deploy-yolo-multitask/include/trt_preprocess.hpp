#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "opencv2/opencv.hpp"
#include "trt_timer.hpp"

namespace preprocess{

enum class tactics : int32_t {
    CPU_NEAREST         = 0,
    CPU_BILINEAR        = 1,
    GPU_NEAREST         = 2,
    GPU_NEAREST_CENTER  = 3,
    GPU_BILINEAR        = 4,
    GPU_BILINEAR_CENTER = 5,
    GPU_WARP_AFFINE     = 6,
};

struct TransInfo{
    int src_w = 0;
    int src_h = 0;
    int tar_w = 0;
    int tar_h = 0;
    TransInfo() = default;
    TransInfo(int srcW, int srcH, int tarW, int tarH):
        src_w(srcW), src_h(srcH), tar_w(tarW), tar_h(tarH){}
};

struct AffineMatrix{
    float forward[6];
    float reverse[6];
    float forward_scale;
    float reverse_scale;

    void calc_forward_matrix(TransInfo trans){
        forward[0] = forward_scale;
        forward[1] = 0;
        forward[2] = - forward_scale * trans.src_w * 0.5 + trans.tar_w * 0.5;
        forward[3] = 0;
        forward[4] = forward_scale;
        forward[5] = - forward_scale * trans.src_h * 0.5 + trans.tar_h * 0.5;
    };

    void calc_reverse_matrix(TransInfo trans){
        reverse[0] = reverse_scale;
        reverse[1] = 0;
        reverse[2] = - reverse_scale * trans.tar_w * 0.5 + trans.src_w * 0.5;
        reverse[3] = 0;
        reverse[4] = reverse_scale;
        reverse[5] = - reverse_scale * trans.tar_h * 0.5 + trans.src_h * 0.5;
    };

    void init(TransInfo trans){
        float scaled_w = (float)trans.tar_w / trans.src_w;
        float scaled_h = (float)trans.tar_h / trans.src_h;
        forward_scale = (scaled_w < scaled_h ? scaled_w : scaled_h);
        reverse_scale = 1 / forward_scale;
    
        calc_forward_matrix(trans);
        calc_reverse_matrix(trans);
    }
};

// 对结构体设置default instance
extern  TransInfo    trans;
extern  AffineMatrix affine_matrix;
void    warpaffine_init(int srcH, int srcW, int tarH, int tarW);

cv::Mat preprocess_resize_cpu(cv::Mat &src, const int& tarH, const int& tarW, float* mean, float* std, tactics tac);
cv::Mat preprocess_resize_cpu(cv::Mat &src, const int& tarH, const int& tarW, tactics tac);
void    preprocess_resize_gpu(cv::Mat &h_src, float* d_tar, const int& tarH, const int& tarW, float* mean, float* std, tactics tac);
void    preprocess_resize_gpu(cv::Mat &h_src, float* d_tar, const int& tarH, const int& tarW, tactics tac);
void    resize_bilinear_gpu(float* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW, float* mean, float* std, tactics tac);
void    resize_bilinear_gpu(float* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW, tactics tac);

__host__ __device__ void affine_transformation(float* trans_matrix, int src_x, int src_y, float* tar_x, float* tar_y);

};  // namespace preprocess


#endif //__PREPROCESS_HPP__
