#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "opencv2/opencv.hpp"
#include "timer.hpp"

cv::Mat preprocess_cpu(cv::Mat &src, const int& tarH, const int& tarW, Timer timer, int tactis);
cv::Mat preprocess_gpu(cv::Mat &h_src, const int& tarH, const int& tarW, Timer timer, int tactis);
void resize_bilinear_gpu(uint8_t* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW, int tactis);

#endif //__PREPROCESS_HPP__
