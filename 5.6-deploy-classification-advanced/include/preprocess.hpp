#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "opencv2/opencv.hpp"
#include "timer.hpp"

cv::Mat preprocess_resize_cpu(cv::Mat &src, const int& tarH, const int& tarW, Timer timer, int tactis);
void preprocess_resize_gpu(cv::Mat &h_src, float* d_tar, const int& tarH, const int& tarW, float* mean, float* std, int tactis);
void resize_bilinear_gpu(float* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW, float* mean, float* std, int tactis);

#endif //__PREPROCESS_HPP__
