#include <iostream>
#include <memory>

#include "model.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "preprocess.hpp"

using namespace std;

void bgr2rgb_cpu_speed_test(){
    Timer   timer;
    string  imagePath = "data/fox.png";
    string  savePath  = "";
    cv::Mat src       = cv::imread(imagePath);
    cv::Mat tar(src.rows, src.cols, CV_8UC3);

    LOG("Starting cpu bgr2rgb speed test...");

    timer.start_cpu();
    preprocess_cv_cvtcolor(src, tar);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::cvtcolor takes");
    savePath  = "data/fox-cvtcolor.png";
    cv::imwrite(savePath, tar);

    timer.start_cpu();
    preprocess_cv_mat_at(src, tar);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::Mat::at takes");
    savePath  = "data/fox-mat-at.png";
    cv::imwrite(savePath, tar);

    timer.start_cpu();
    preprocess_cv_mat_iterator(src, tar);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::MatIterator_ takes");
    savePath  = "data/fox-mat-iterator.png";
    cv::imwrite(savePath, tar);

    timer.start_cpu();
    preprocess_cv_mat_data(src, tar);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::Mat data takes");
    savePath  = "data/fox-mat-data.png";
    cv::imwrite(savePath, tar);

    timer.start_cpu();
    preprocess_cv_pointer(src, tar);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using pointer takes");
    savePath  = "data/fox-pointer.png";
    cv::imwrite(savePath, tar);
}

void bgr2rgb_norm_hwc2chw_cpu_speed_test(){
    Timer   timer;
    string  imagePath = "data/fox.png";
    cv::Mat src       = cv::imread(imagePath);
    int     size      = src.cols * src.rows * src.channels();
    float*  tar       = (float*)malloc(size * sizeof(float));
    int     width     = 224;
    int     height    = 224;
    int     channel   = 3;
    int     classes   = 1000;
    float   mean[3]   = {0.406, 0.456, 0.485};
    float   std[3]    = {0.225, 0.224, 0.229};

    cv::resize(src, src, cv::Size(width, height));
    LOG("Starting cpu bgr2rgb + normalization + hwc2chw speed test...");


    timer.start_cpu();
    preprocess_cv_mat_at(src, tar, mean, std);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::Mat::at takes");

    timer.start_cpu();
    preprocess_cv_mat_iterator(src, tar, mean, std);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::MatIterator_ takes");

    timer.start_cpu();
    preprocess_cv_mat_data(src, tar, mean, std);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using cv::Mat data takes");

    timer.start_cpu();
    int area = src.rows * src.cols;
    int offset_ch0 = area * 0;
    int offset_ch1 = area * 1;
    int offset_ch2 = area * 2;

    for (int i = 0; i < src.rows; i ++) {
        cv::Vec3b* src_ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j ++) {
            tar[offset_ch2++] = (src_ptr[j][0] / 255.0f - mean[0]) / std[0];
            tar[offset_ch1++] = (src_ptr[j][1] / 255.0f - mean[1]) / std[1];
            tar[offset_ch0++] = (src_ptr[j][2] / 255.0f - mean[2]) / std[2];
        }
    } 
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using pointer takes");

    timer.start_cpu();
    int image_area = src.cols * src.rows;
    unsigned char* pimage = src.data;
    float* phost_b = tar + image_area * 0;
    float* phost_g = tar + image_area * 1;
    float* phost_r = tar + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Using pointer(trtpo) takes");
}

int main(int argc, char const *argv[])
{
    // bgr2rgb_cpu_speed_test();
    bgr2rgb_norm_hwc2chw_cpu_speed_test();
}
