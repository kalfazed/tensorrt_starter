#include "preprocess.hpp"

void preprocess_cv_cvtcolor(cv::Mat src, cv::Mat tar){
    cv::cvtColor(src, tar, cv::COLOR_RGB2BGR);
}

void preprocess_cv_mat_at(cv::Mat src, cv::Mat tar){
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            tar.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(i, j)[0];
            tar.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(i, j)[1];
            tar.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(i, j)[2];
        }
    }
}

void preprocess_cv_mat_at(cv::Mat src, float* tar, float* mean, float* std){
    float* ptar_ch0 = tar + src.rows * src.cols * 0;
    float* ptar_ch1 = tar + src.rows * src.cols * 1;
    float* ptar_ch2 = tar + src.rows * src.cols * 2;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            (*ptar_ch2++) = (src.at<cv::Vec3b>(i, j)[0] / 255.0f - mean[0]) / std[0];
            (*ptar_ch1++) = (src.at<cv::Vec3b>(i, j)[1] / 255.0f - mean[1]) / std[1];
            (*ptar_ch0++) = (src.at<cv::Vec3b>(i, j)[2] / 255.0f - mean[2]) / std[2];
        }
    }
}

void preprocess_cv_mat_iterator(cv::Mat src, cv::Mat tar){
    cv::MatIterator_<cv::Vec3b> src_it = src.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> tar_it = tar.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> end    = src.end<cv::Vec3b>();
    for (; src_it != end; src_it++, tar_it++) {
        (*tar_it)[2] = (*src_it)[0];
        (*tar_it)[1] = (*src_it)[1];
        (*tar_it)[0] = (*src_it)[2];
    }
}

void preprocess_cv_mat_iterator(cv::Mat src, float* tar, float* mean, float* std){
    float* ptar_ch0 = tar + src.rows * src.cols * 0;
    float* ptar_ch1 = tar + src.rows * src.cols * 1;
    float* ptar_ch2 = tar + src.rows * src.cols * 2;
    cv::MatIterator_<cv::Vec3b> it     = src.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> end    = src.end<cv::Vec3b>();

    for (; it != end; it++) {
        (*ptar_ch2++) = ((*it)[0] / 255.0f - mean[0]) / std[0];
        (*ptar_ch1++) = ((*it)[1] / 255.0f - mean[1]) / std[1];
        (*ptar_ch0++) = ((*it)[2] / 255.0f - mean[2]) / std[2];
    }
}

void preprocess_cv_mat_data(cv::Mat src, cv::Mat tar){
    int height   = src.rows;
    int width    = src.cols;
    int channels = src.channels();

    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            int index = i * width * channels + j * channels;
            tar.data[index + 2] = src.data[index + 0];
            tar.data[index + 1] = src.data[index + 1];
            tar.data[index + 0] = src.data[index + 2];
        }
    }
}

void preprocess_cv_mat_data(cv::Mat src, float* tar, float* mean, float* std){
    float* ptar_ch0 = tar + src.rows * src.cols * 0;
    float* ptar_ch1 = tar + src.rows * src.cols * 1;
    float* ptar_ch2 = tar + src.rows * src.cols * 2;
    int height      = src.rows;
    int width       = src.cols;
    int channels    = src.channels();

    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            int index = i * width * channels + j * channels;
            (*ptar_ch2++) = (src.data[index + 0] / 255.0f - mean[0]) / std[0];
            (*ptar_ch1++) = (src.data[index + 1] / 255.0f - mean[1]) / std[1];
            (*ptar_ch0++) = (src.data[index + 2] / 255.0f - mean[2]) / std[2];
        }
    }
}

void preprocess_cv_pointer(cv::Mat src, cv::Mat tar){
    for (int i = 0; i < src.rows; i ++) {
        cv::Vec3b* src_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tar_ptr = tar.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j ++) {
            tar_ptr[j][2] = src_ptr[j][0];
            tar_ptr[j][1] = src_ptr[j][1];
            tar_ptr[j][0] = src_ptr[j][2];
        }
    }
}

void preprocess_cv_pointer(cv::Mat src, float* tar, float* mean, float* std){
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
}
