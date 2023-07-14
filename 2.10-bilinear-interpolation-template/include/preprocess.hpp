#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "opencv2/opencv.hpp"
#include "timer.hpp"

cv::Mat preprocess_cpu(cv::Mat &src, const int& tarH, const int& tarW, Timer timer);
template<typename T> cv::Mat preprocess_gpu(cv::Mat &h_src, const int& tarH, const int& tarW, Timer timer);
template<typename T> void resize_bilinear_gpu(T* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW);
#endif //__PREPROCESS_HPP__
