#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

void preprocess_cv_cvtcolor(cv::Mat src, cv::Mat tar);
void preprocess_cv_mat_at(cv::Mat src, cv::Mat tar);
void preprocess_cv_mat_iterator(cv::Mat src, cv::Mat tar);
void preprocess_cv_mat_data(cv::Mat src, cv::Mat tar);
void preprocess_cv_pointer(cv::Mat src, cv::Mat tar);

void preprocess_cv_mat_at(cv::Mat src, float* tar, float* mean, float* std);
void preprocess_cv_mat_iterator(cv::Mat src, float* tar, float* mean, float* std);
void preprocess_cv_mat_data(cv::Mat src, float* tar, float* mean, float* std);
void preprocess_cv_pointer(cv::Mat src, float* tar, float* mean, float* std);

#endif //__PREPROCESS_HPP__